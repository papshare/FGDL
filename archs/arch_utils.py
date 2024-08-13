import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from archs.layers import DownBlock, Conv, ResnetTransformer
from torch.nn import init
import torch.nn.utils.spectral_norm as spectral_norm


# The number of filters in each block of the encoding part (down-sampling).
ndf = {'A': [32, 64, 64, 64, 64, 64, 64], }
# The number of filters in each block of the decoding part (up-sampling).
# If len(ndf[cfg]) > len(nuf[cfg]) - then the deformation field is up-sampled to match the input size.
nuf = {'A': [64, 64, 64, 64, 64, 64, 32], }
# Indicate if res-blocks are used in the down-sampling path.
use_down_resblocks = {'A': True, }
# indicate the number of res-blocks applied on the encoded features.
resnet_nblocks = {'A': 3, }
# Indicate if the a final refinement layer is applied on the before deriving the deformation field
refine_output = {'A': True, }
# The activation used in the down-sampling path.
down_activation = {'A': 'leaky_relu', }
# The activation used in the up-sampling path.
up_activation = {'A': 'leaky_relu', }


class ResUnet(torch.nn.Module):
    def __init__(self, nc_a, nc_b, cfg, init_func, init_to_identity):
        super(ResUnet, self).__init__()
        act = down_activation[cfg]
        # ------------ Down-sampling path
        self.ndown_blocks = len(ndf[cfg])
        self.nup_blocks = len(nuf[cfg])
        assert self.ndown_blocks >= self.nup_blocks
        in_nf = nc_a + nc_b
        conv_num = 1
        skip_nf = {}
        for out_nf in ndf[cfg]:
            setattr(self, 'down_{}'.format(conv_num),
                    DownBlock(in_nf, out_nf, 3, 1, 1, activation=act, init_func=init_func, bias=True,
                              use_resnet=use_down_resblocks[cfg], use_norm=False))
            skip_nf['down_{}'.format(conv_num)] = out_nf
            in_nf = out_nf
            conv_num += 1
        conv_num -= 1
        if use_down_resblocks[cfg]:
            self.c1 = Conv(in_nf, 2 * in_nf, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                           use_resnet=False, use_norm=False)
            self.t = ((lambda x: x) if resnet_nblocks[cfg] == 0
                      else ResnetTransformer(2 * in_nf, resnet_nblocks[cfg], init_func))
            self.c2 = Conv(2 * in_nf, in_nf, 1, 1, 0, activation=act, init_func=init_func, bias=True,
                           use_resnet=False, use_norm=False)
        # ------------- Up-sampling path
        act = up_activation[cfg]
        for out_nf in nuf[cfg]:
            setattr(self, 'up_{}'.format(conv_num),
                    Conv(in_nf + skip_nf['down_{}'.format(conv_num)], out_nf, 3, 1, 1, bias=True, activation=act,
                         init_fun=init_func, use_norm=False, use_resnet=False))
            in_nf = out_nf
            conv_num -= 1
        if refine_output[cfg]:
            self.refine = nn.Sequential(ResnetTransformer(in_nf, 1, init_func),
                                        Conv(in_nf, in_nf, 1, 1, 0, use_resnet=False, init_func=init_func,
                                             activation=act,
                                             use_norm=False)
                                        )
        else:
            self.refine = lambda x: x
        self.output = Conv(in_nf, 2, 3, 1, 1, use_resnet=False, bias=True,
                           init_func=('zeros' if init_to_identity else init_func), activation=None,
                           use_norm=False)
    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], 1)
        skip_vals = {}
        conv_num = 1
        # Down
        while conv_num <= self.ndown_blocks:
            x, skip = getattr(self, 'down_{}'.format(conv_num))(x)
            skip_vals['down_{}'.format(conv_num)] = skip
            conv_num += 1
        if hasattr(self, 't'):
            x = self.c1(x)
            x = self.t(x)
            x = self.c2(x)
        # Up
        conv_num -= 1
        while conv_num > (self.ndown_blocks - self.nup_blocks):
            s = skip_vals['down_{}'.format(conv_num)]
            x = F.interpolate(x, (s.size(2), s.size(3)), mode='bilinear')
            x = torch.cat([x, s], 1)
            x = getattr(self, 'up_{}'.format(conv_num))(x)
            conv_num -= 1
        x = self.refine(x)
        x = self.output(x)
        return x

class Incremental_Modulation(nn.Module):
    '''
    only input learned feature distance
    use LR branch features mean and sigma as base, then add the
    '''
    def __init__(self, norm_nc=64, guide_nc=1, norm_type='instance'):
        super().__init__()
        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in Mod1'
                             % norm_type)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(guide_nc, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, guidance):
        b, c, h, w = x.size()
        x_ = x.view(b, c, h * w)
        mu_0 = torch.mean(x_, dim=-1, keepdim=True).unsqueeze(3)
        sigma_0 = torch.std(x_, dim=-1, keepdim=True).unsqueeze(3)

        normalized = self.param_free_norm(x)

        actv = self.mlp_shared(guidance)
        sigma = self.mlp_gamma(actv) + sigma_0
        mu = self.mlp_beta(actv) + mu_0
        out = normalized * sigma + mu

        return out
class DeformableConv2d_guide(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d_guide, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels*2,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels*2,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x, guide):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.
        fea = torch.cat([x, guide], dim=1)
        offset = self.offset_conv(fea)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(fea))
        # op = (n - (k * d - 1) + 2p / s)
        x = deform_conv2d(input=x,
                          offset=offset,
                          weight=self.regular_conv.weight,
                          bias=self.regular_conv.bias,
                          padding=self.padding,
                          mask=modulator,
                          stride=self.stride,
                          dilation=self.dilation)
        return x


def get_nonspade_norm_layer(mpdist=False, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        elif subnorm_type == 'sync_batch' and mpdist:
            norm_layer = nn.SyncBatchNorm(get_out_channel(layer), affine=True)
        else:
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
def make_layer(block, n_layers):
	layers = []
	for _ in range(n_layers):
		layers.append(block())
	return nn.Sequential(*layers)
def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt

class ResidualDenseBlock_5C(nn.Module):
	def __init__(self, nf=64, gc=32, bias=True):
		super(ResidualDenseBlock_5C, self).__init__()
		# gc: growth channel, i.e. intermediate channels
		self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
		self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
		self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
		self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
		self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
		self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x):
		x1 = self.lrelu(self.conv1(x))
		x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
		x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
		x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
		x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
		return x5 * 0.2 + x
class RRDB(nn.Module):
	'''Residual in Residual Dense Block'''
	def __init__(self, nf, gc=32):
		super(RRDB, self).__init__()
		self.RDB1 = ResidualDenseBlock_5C(nf, gc)
		self.RDB2 = ResidualDenseBlock_5C(nf, gc)
		self.RDB3 = ResidualDenseBlock_5C(nf, gc)

	def forward(self, x):
		out = self.RDB1(x)
		out = self.RDB2(out)
		out = self.RDB3(out)
		return out * 0.2 + x