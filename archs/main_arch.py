import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from archs.restormer_utils import Restormer_Ushaped, TransformerBlock
from archs.transformer import Transformer_2D
from archs.arch_utils import Incremental_Modulation,DeformableConv2d_guide, ResUnet
from archs.arch_utils import get_nonspade_norm_layer, BaseNetwork, RRDB, make_layer, get_keys


class SR_Branch(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=48, gc=32, k=2, heads = [1,2,4,8,8,8]):

        super(SR_Branch, self).__init__()
        self.size = int(256 / (2 ** k))
        dim_mapping = {8: nf * 2 ** 3, 16: nf * 2 ** 3, 32: nf * 2 ** 3,
                       64: nf * 2 ** 2, 128: nf * 2 ** 1,
                       256: nf}
        head_mapping = {8: heads[5], 16: heads[4], 32: heads[3], 64: heads[2], 128: heads[1], 256: heads[0]}

        self.conv_first = nn.Conv2d(in_nc, dim_mapping[self.size], 3, 1, 1, bias=True)
        self.trunk_first = make_layer(functools.partial(RRDB, nf=dim_mapping[self.size], gc=gc), 2)
        self.k = k


        modules = []
        modules_ali = []
        modules_mod = []
        for i in range(self.k,-1,-1):
            size = int(256 / (2 ** i))
            nf_ = dim_mapping[size]
            if i != 0:
                nf__ = dim_mapping[int(size*2)]
            else:
                nf__ = nf_
            h_ = head_mapping[size]
            modules.append(nn.Sequential(*[TransformerBlock(dim=nf_*2, num_heads=h_, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', is_deform=True) for i in range(2)]))
            modules.append(nn.Conv2d(nf_ * 2, nf_, 3, 1, 1, bias=True))
            modules.append(nn.Upsample(scale_factor=2, mode='nearest'))
            modules.append(nn.Conv2d(nf_, nf__, 3, 1, 1, bias=True))

            modules_ali.append(DeformableConv2d_guide(nf_, nf_, kernel_size=3, stride=1, padding=1, dilation=1, bias=False))
            modules_mod.append(Incremental_Modulation(norm_nc=nf_, guide_nc=1, norm_type='instance'))

        self.modules_ = nn.ModuleList(modules)
        self.modules_ali = nn.ModuleList(modules_ali)
        self.modules_mod = nn.ModuleList(modules_mod)


        self.out = nn.Sequential(*[
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(nf, int(nf / 2), 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(int(nf / 2), out_nc, 3, 1, 1, bias=True)
        ])

    def forward(self, x, ref_features, return_feature=True):
        if ref_features[0].shape[-1] > ref_features[-1].shape[-1]:
            ref_features = ref_features[::-1]  # invert ref_features
        if len(ref_features) > self.k + 1:
            off = len(ref_features) - self.k - 1
            ref_features = ref_features[off:]
        features = []

        x = self.conv_first(x)  # 16×16×64
        trunk = self.trunk_first(x)
        fea = x + trunk
        fea_dists = []
        for i in range(self.k + 1):
            ref_fea_ = ref_features[i]

            features.append(fea)
            ref_fea = ref_fea_
            fea_cos_dist = torch.sum(
                torch.mul(F.normalize(fea, dim=1), F.normalize(ref_fea_, dim=1)), dim=1, keepdim=True)
            fea_dists.append(fea_cos_dist)
            fea = self.modules_ali[i](fea, ref_fea)
            fc = torch.cat((fea, ref_fea), dim=1)
            trunk = self.modules_[4 * i + 1](self.modules_[4 * i](fc))
            trunk = self.modules_mod[i](trunk, fea_cos_dist)
            fea = fea + trunk
            if i != self.k:
                fea = self.modules_[4 * i + 2](fea)
                fea = self.modules_[4 * i + 3](fea)
        out = self.out(fea)
        if return_feature:
            return out, features, fea_dists
        return out


class CompositeModel(nn.Module):
    '''
    Abbreviations:
    k: sampling ratio
    tar_na: non-aligned target HR image
    tar_a: aligned target HR image
    rs_cms: cross-modality synthesis result
    rs_sr: super-resolution result
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        k = int(np.log2(config['output_size']) - np.log2(config['input_size']))
        self.cms = Restormer_Ushaped(dim=config['dim_restormer'])
        self.sr = SR_Branch(k=k, in_nc=config['input_nc'], out_nc=config['input_nc'])
        self.resunet = ResUnet(1,1, cfg='A', init_func='kaiming', init_to_identity=True)
        self.spatial_transform = Transformer_2D().cuda()
        self.load_weights()

    def forward(self, lr, ref, tar_na):
        rs_cms, features_cms = self.cms(ref)
        deformation = self.resunet(tar_na, ref)
        tar_a = self.spatial_transform(tar_na, deformation)
        rs_sr, features, fea_dist = self.sr(lr, features_cms, return_feature=True)
        return rs_sr, rs_cms, features, features_cms, fea_dist, tar_a

    def forward_cms(self, ref, tar_na):
        rs_cms, features_cms = self.cms(ref)
        deformation = self.resunet(tar_na, ref)
        tar_a = self.spatial_transform(tar_na, deformation)
        return rs_cms, features_cms, tar_a, deformation

    def forward_test(self, lr, ref):
        rs_cms, features_cms = self.cms(ref)
        rs_sr, features, fea_dist = self.sr(lr, features_cms, return_feature=True)
        return rs_sr, rs_cms

    def forward_test_cms(self, ref):
        rs_cms, features_cms = self.cms(ref)
        return rs_cms
    def load_weights(self):
        if self.config['use_pretrained_cms']:
            if self.config['cms_path'] is not None:
                print('Loading pretrained CMS from checkpoint: {}'.format(self.config['cms_path']))
                ckpt = torch.load(self.config['cms_path'], map_location='cpu')
                self.cms.load_state_dict(get_keys(ckpt, 'cms'), strict=True)
                self.resunet.load_state_dict(get_keys(ckpt, 'resunet'), strict=True)
            else:
                raise ValueError('no str found for the path of pretrained cms branch')
            if self.config['sr_path'] is not None:
                print('Loading the SR Branch from checkpoint: {}'.format(self.config['sr_path']))
                ckpt = torch.load(self.config['sr_path'], map_location='cpu')
                self.sr.load_state_dict(get_keys(ckpt, 'sr'), strict=False)
            else:
                print('No pretrained model for the SR Branch! Training SR Branch from begining!')
        else:
            if self.config['sr_path'] is not None:
                print('Loading the composite network from checkpoint: {}'.format(self.config['sr_path']))
                ckpt = torch.load(self.config['sr_path'], map_location='cpu')
                self.sr.load_state_dict(get_keys(ckpt, 'sr'), strict=False)
                self.cms.load_state_dict(get_keys(ckpt, 'cms'), strict=True)
                self.resunet.load_state_dict(get_keys(ckpt, 'resunet'), strict=True)
            else:
                print('No pretrained model! Training from begining!')


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)
        # self.load_weights()

    def forward(self, img):
        return self.model(img)

class DiffProjDiscriminator(BaseNetwork):
    def __init__(self):
        super().__init__()

        nf = 64
        input_nc = 1
        label_nc = 1

        norm_layer = get_nonspade_norm_layer(norm_type='spectralinstance')

        # bottom-up pathway
        self.enc1 = nn.Sequential(
            norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.enc2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf, nf * 2, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.enc3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.enc4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 8, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.enc5 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))

        # top-down pathway
        self.lat2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 4, kernel_size=1)),
            nn.LeakyReLU(0.2, True))
        self.lat3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 4, kernel_size=1)),
            nn.LeakyReLU(0.2, True))
        self.lat4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 8, nf * 4, kernel_size=1)),
            nn.LeakyReLU(0.2, True))
        self.lat5 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 8, nf * 4, kernel_size=1)),
            nn.LeakyReLU(0.2, True))

        # upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # final layers
        self.final2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.final3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.final4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.final5 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True))
        # true/false prediction and semantic alignment prediction
        self.tf = nn.Conv2d(nf * 2, 1, kernel_size=1)
        self.tf_ = nn.Conv2d(nf * 2, 1, kernel_size=1)
        self.seg = nn.Conv2d(nf * 2, nf * 2, kernel_size=1)
        self.embedding = nn.Conv2d(label_nc, nf * 2, kernel_size=1)

    def forward(self, fake_and_real_img, segmap):
        # bottom-up pathway
        feat11 = self.enc1(fake_and_real_img)
        feat12 = self.enc2(feat11)
        feat13 = self.enc3(feat12)
        feat14 = self.enc4(feat13)
        feat15 = self.enc5(feat14)
        # top-down pathway and lateral connections
        feat25 = self.lat5(feat15)
        feat24 = self.up(feat25) + self.lat4(feat14)
        feat23 = self.up(feat24) + self.lat3(feat13)
        feat22 = self.up(feat23) + self.lat2(feat12)
        # final prediction layers
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        feat34 = self.final4(feat24)
        feat35 = self.final5(feat25)
        # Patch-based True/False prediction
        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        pred4 = self.tf(feat34)
        pred5 = self.tf_(feat35)
        seg2 = self.seg(feat32)
        seg3 = self.seg(feat33)
        seg4 = self.seg(feat34)

        # intermediate features for discriminator feature matching loss
        feats = [feat12, feat13, feat14, feat15]

        # segmentation map embedding
        segemb = self.embedding(segmap)
        segemb = F.avg_pool2d(segemb, kernel_size=2, stride=2)
        segemb2 = F.avg_pool2d(segemb, kernel_size=2, stride=2)
        segemb3 = F.avg_pool2d(segemb2, kernel_size=2, stride=2)
        segemb4 = F.avg_pool2d(segemb3, kernel_size=2, stride=2)

        # semantics embedding discriminator score
        pred2 += torch.mul(segemb2, seg2).sum(dim=1, keepdim=True)
        pred3 += torch.mul(segemb3, seg3).sum(dim=1, keepdim=True)
        pred4 += torch.mul(segemb4, seg4).sum(dim=1, keepdim=True)

        # concat results from multiple resolutions
        results = [pred2, pred3, pred4]

        return [feats, results], pred5


class DiffProjDiscriminator2(BaseNetwork):
    def __init__(self):
        super().__init__()

        nf = 64
        input_nc = 1
        label_nc = 1

        norm_layer = get_nonspade_norm_layer(norm_type='spectralinstance')

        # bottom-up pathway
        self.enc1 = nn.Sequential(
            norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.enc2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf, nf * 2, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.enc3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.enc4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 8, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.enc5 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True))

        # top-down pathway
        self.lat2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 4, kernel_size=1)),
            nn.LeakyReLU(0.2, True))
        self.lat3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 4, kernel_size=1)),
            nn.LeakyReLU(0.2, True))
        self.lat4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 8, nf * 4, kernel_size=1)),
            nn.LeakyReLU(0.2, True))
        self.lat5 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 8, nf * 4, kernel_size=1)),
            nn.LeakyReLU(0.2, True))

        # upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # final layers
        self.final2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.final3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True))
        self.final4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, True))

        # true/false prediction and semantic alignment prediction
        self.tf = nn.Conv2d(nf * 2, 1, kernel_size=1)
        self.seg = nn.Conv2d(nf * 2, nf * 2, kernel_size=1)
        self.embedding = nn.Conv2d(label_nc, nf * 2, kernel_size=1)

    def forward(self, fake_and_real_img, segmap):
        # bottom-up pathway
        feat11 = self.enc1(fake_and_real_img)
        feat12 = self.enc2(feat11)
        feat13 = self.enc3(feat12)
        feat14 = self.enc4(feat13)
        feat15 = self.enc5(feat14)
        # top-down pathway and lateral connections
        feat25 = self.lat5(feat15)
        feat24 = self.up(feat25) + self.lat4(feat14)
        feat23 = self.up(feat24) + self.lat3(feat13)
        feat22 = self.up(feat23) + self.lat2(feat12)
        # final prediction layers
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        feat34 = self.final4(feat24)

        # Patch-based True/False prediction
        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        pred4 = self.tf(feat34)

        seg2 = self.seg(feat32)
        seg3 = self.seg(feat33)
        seg4 = self.seg(feat34)

        # intermediate features for discriminator feature matching loss
        feats = [feat12, feat13, feat14, feat15]

        # segmentation map embedding
        segemb = self.embedding(segmap)
        segemb = F.avg_pool2d(segemb, kernel_size=2, stride=2)
        segemb2 = F.avg_pool2d(segemb, kernel_size=2, stride=2)
        segemb3 = F.avg_pool2d(segemb2, kernel_size=2, stride=2)
        segemb4 = F.avg_pool2d(segemb3, kernel_size=2, stride=2)

        # semantics embedding discriminator score
        pred2 += torch.mul(segemb2, seg2).sum(dim=1, keepdim=True)
        pred3 += torch.mul(segemb3, seg3).sum(dim=1, keepdim=True)
        pred4 += torch.mul(segemb4, seg4).sum(dim=1, keepdim=True)

        # concat results from multiple resolutions
        results = [pred2, pred3, pred4]

        return [feats, results]