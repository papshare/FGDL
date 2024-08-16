import os
import shutil
from PIL import Image
import numpy as np
import sys
import lpips
sys.path.append(".")
sys.path.append("..")
sys.path.append("models")

os.environ["BASICSR_JIT"] = 'True'
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision import transforms

from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

from utils import train_utils, common

from data.dataset import MCSRDataset
from utils.percep.percep import PerceptualNetwork
from utils.ranger import Ranger
import time
from pytorch_msssim import MS_SSIM
from archs.main_arch import CompositeModel, Discriminator, DiffProjDiscriminator2
# from mpmath import mp


class FGDL:
	def __init__(self, config):

		self.config = config

		# Initialize network
		self.model = CompositeModel(self.config).cuda()
		self.starttime = time.time()
		self.lpips_metric = lpips.LPIPS(net='alex').cuda()


	def _init_train(self):
		self.global_step = 0
		if self.config['use_pretrained_cms']:
			print('freezing...')
			for param in self.model.cms.parameters():
				param.requires_grad = False
			for param in self.model.resunet.parameters():
				param.requires_grad = False

		if self.config['use_proj_gan']:
			self.dis = DiffProjDiscriminator2().cuda()
		else:
			self.dis = Discriminator((self.config['input_nc'], self.config['output_size'], self.config['output_size'])).cuda()
		if self.config['train_cms']:
			self.dis2 = Discriminator((self.config['input_nc'], self.config['output_size'], self.config['output_size'])).cuda()


		if self.config['dis_path'] is not None:
			d_ckpt = torch.load(self.config['dis_path'], map_location='cpu')
			print('Loading discriminator from checkpoint: {}'.format(self.config['dis_path']))
			self.dis.load_state_dict(d_ckpt["state_dict"])
		else:
			print('No pretrained discriminator model. Training from begining!')


		self.d_optim = Ranger(self.dis.parameters(), self.config['learning_rate'])
		if self.config['train_cms']:
			self.d_optim2 = Ranger(self.dis2.parameters(), self.config['learning_rate'])
		self.GANLoss = nn.BCEWithLogitsLoss().cuda()
		self.msssim = MS_SSIM(data_range=1.0, size_average=True, channel=1)

		# Initialize loss todo
		self.mse_loss = nn.MSELoss().cuda().eval()
		if self.config['percep_lambda'] > 0:
			self.percep_loss = PerceptualNetwork(net_type='alex').cuda().eval()


		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.val_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.config['batch_size'],
										   shuffle=True,
										   num_workers=int(self.config['workers']),
										   drop_last=True)
		self.val_dataloader = DataLoader(self.val_dataset,
										  batch_size=self.config['val_batch_size'],
										  shuffle=False,
										  num_workers=int(self.config['val_workers']),
										  drop_last=True)

		# Initialize logger
		log_dir = os.path.join(self.config['exp_dir'], 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(self.config['exp_dir'], 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val = None
		if self.config['save_interval'] is None:
			self.config['save_interval'] = self.config['max_steps']
	def train(self):
		self._init_train()
		if self.config['train_cms']:
			self.train_cms()
		else:
			self.train_sr()
	def train_cms(self):
		d_loss = 0
		self.model.cms.train()
		d_iter = 6
		while self.global_step < self.config['max_steps']:
			for batch_idx, batch in enumerate(self.train_dataloader):
				self.optimizer.zero_grad()

				tar_na, tar_gt, ref = batch
				tar_na, tar_gt, ref = tar_na.cuda().float(), tar_gt.cuda().float(), ref.cuda().float()

				rs_cms, features_cms, tar_a, deformation = self.model.forward_cms(ref, tar_na)
				self.D_infty = tar_a - rs_cms

				if self.global_step % d_iter == 0 and self.global_step > 200000:
					# update discriminator
					self.requires_grad(self.dis, True)
					self.requires_grad(self.dis2, True)
					self.requires_grad(self.model, False)
					d_loss = 0.0
					if self.config['use_proj_gan']:
						_, fake_dis, _, real_dis = self.discriminate(self.D_infty, rs_cms, tar_a)
						for fd, rd in zip(fake_dis, real_dis):
							valid = torch.ones(fd.shape, requires_grad=False).cuda()
							fake = torch.zeros(fd.shape, requires_grad=False).cuda()
							loss_real = self.GANLoss(rd - fd.mean(0, keepdim=True), valid)
							loss_fake = self.GANLoss(fd - rd.mean(0, keepdim=True), fake)
							d_loss += (loss_real + loss_fake) / 2
						real_dis_coarse = self.dis2(tar_na)
						fake_dis_coarse = self.dis2(tar_a)
						valid = torch.ones(real_dis_coarse.shape, requires_grad=False).cuda()
						fake = torch.zeros(fake_dis_coarse.shape, requires_grad=False).cuda()
						loss_real_coarse = self.GANLoss(real_dis_coarse - fake_dis_coarse.mean(0, keepdim=True), valid)
						loss_fake_coarse = self.GANLoss(fake_dis_coarse - real_dis_coarse.mean(0, keepdim=True), fake)
						d_loss2 = (loss_real_coarse + loss_fake_coarse) / 2
					else:
						real_dis = self.dis(tar_a)
						fake_dis = self.dis(rs_cms)
						valid = torch.ones(real_dis.shape, requires_grad=False).cuda()
						fake = torch.zeros(fake_dis.shape, requires_grad=False).cuda()
						loss_real = self.GANLoss(real_dis - fake_dis.mean(0, keepdim=True), valid)
						loss_fake = self.GANLoss(fake_dis - real_dis.mean(0, keepdim=True), fake)
						d_loss = (loss_real + loss_fake) / 2
					self.d_optim.zero_grad()
					self.d_optim2.zero_grad()

					d_loss.backward(retain_graph=True)
					d_loss2.backward(retain_graph=True)
					self.d_optim.step()
					self.d_optim2.step()

					del real_dis, fake_dis, real_dis_coarse, fake_dis_coarse
					self.requires_grad(self.dis, False)
					self.requires_grad(self.dis2, False)
					self.requires_grad(self.model.cms, True)
					self.requires_grad(self.model.resunet, True)

				

				loss, loss_rec = self.calc_g_loss(tar_a, rs_cms, y_=tar_na)

				loss += self.smoothing_loss(deformation)

				loss.backward(retain_graph=True)
				self.optimizer.step()

				if self.global_step % self.config['board_interval'] == 0:
					loss_num  = loss_rec.item()
					print(f'Step:{self.global_step}, loss:{loss_num}')
					self.logger.add_scalar(f'loss', loss_num, self.global_step)


				# Validation related
				quan_dict = None
				if self.global_step != 0:
					if self.global_step % self.config['val_interval'] == 0 or self.global_step == self.config['max_steps']:
						quan_dict = self.validate_cms()
						if self.best_val is None or quan_dict['psnr'] > self.best_val:
							self.best_val = quan_dict['psnr']
							self.save_checkpoint(quan_dict, is_best=True)


				if self.global_step == self.config['max_steps']:
					self.save_checkpoint(quan_dict, is_best=False)
					print('Training completed. Exiting...')
					break

				self.global_step += 1
	def train_sr(self):
		d_loss = 0
		if self.config['use_pretrained_cms']:
			self.model.sr.train()
		else:
			self.model.train()
		d_iter = 6

		while self.global_step < self.config['max_steps'] :
			for batch_idx, batch in enumerate(self.train_dataloader):
				self.optimizer.zero_grad()

				lr, tar_na, tar_gt, ref = batch
				lr, tar_na, tar_gt, ref = lr.cuda().float(), tar_na.cuda().float(), tar_gt.cuda().float(), ref.cuda().float()

				rs_sr, rs_cms, features, features_cms, fea_cos_dist, tar_a = self.model.forward(lr, ref, tar_na)  # todo
				self.D_infty = tar_a - rs_cms

				if self.global_step % d_iter == 0 and self.global_step > 200000:
					# update discriminator
					self.requires_grad(self.dis, True)
					self.requires_grad(self.model, False)
					d_loss = 0.0
					if self.config['use_proj_gan']:
						_, fake_dis, _, real_dis = self.discriminate(self.D_infty, rs_sr, tar_a)
						for fd, rd in zip(fake_dis, real_dis):
							valid = torch.ones(fd.shape, requires_grad=False).cuda()
							fake = torch.zeros(fd.shape, requires_grad=False).cuda()
							loss_real = self.GANLoss(rd - fd.mean(0, keepdim=True), valid)
							loss_fake = self.GANLoss(fd - rd.mean(0, keepdim=True), fake)
							d_loss += (loss_real + loss_fake) / 2
					else:
						real_dis = self.dis(tar_a)
						fake_dis = self.dis(rs_sr)
						valid = torch.ones(real_dis.shape, requires_grad=False).cuda()
						fake = torch.zeros(fake_dis.shape, requires_grad=False).cuda()
						loss_real = self.GANLoss(real_dis - fake_dis.mean(0, keepdim=True), valid)
						loss_fake = self.GANLoss(fake_dis - real_dis.mean(0, keepdim=True), fake)
						d_loss = (loss_real + loss_fake) / 2
					self.d_optim.zero_grad()
					d_loss.backward(retain_graph=True)
					self.d_optim.step()
					del real_dis, fake_dis
					self.requires_grad(self.dis, False)
					self.requires_grad(self.model.sr, True)
					self.requires_grad(self.model.resunet, True)

				# update generator
				f_pairs = []
				for i in range(len(fea_cos_dist)):
					size = fea_cos_dist[i].shape[-1]
					d_i = F.interpolate(self.D_infty, (size, size))
					f_pairs.append((fea_cos_dist[i], d_i))

				loss, loss_rec = self.calc_g_loss(tar_a, rs_sr, f_pairs=f_pairs)
				loss.backward(retain_graph=True)
				self.optimizer.step()

				if self.global_step % self.config['board_interval'] == 0:
					loss_num = loss_rec.item()
					print(f'Step:{self.global_step}, loss:{loss_num}')
					self.logger.add_scalar(f'loss', loss_num, self.global_step)

				# Validation related
				quan_dict = None
				if self.global_step != 0:
					if self.global_step % self.config['val_interval'] == 0 or self.global_step == self.config['max_steps']:
						quan_dict = self.validate()
						if self.best_val is None or quan_dict['psnr'] > self.best_val:
							self.best_val = quan_dict['psnr']
							self.save_checkpoint(quan_dict, is_best=True)


				if self.global_step == self.config['max_steps']:
					self.save_checkpoint(quan_dict, is_best=False)
					print('Training completed. Exiting...')
					break

				self.global_step += 1

	def validate(self):
		self.model.eval()

		quan_results = {'psnr':[],'ssim':[], 'lpips':[]}
		for batch_idx, batch in enumerate(self.val_dataloader):
			lr, tar_na, tar_gt, ref = batch
			lr, tar_na, tar_gt, ref = lr.cuda().float(), tar_na.cuda().float(), tar_gt.cuda().float(), ref.cuda().float()

			with torch.no_grad():

				rs_sr, rs_cms, features, features_cms,fea_cos_dist, tar_a = self.model.forward(lr, ref, tar_na)  # todo
				D_infty = tar_a - rs_cms
				f_pairs = []
				for i in range(len(fea_cos_dist)):
					size = fea_cos_dist[i].shape[-1]
					d_i = F.interpolate(D_infty, (size, size))
					f_pairs.append((fea_cos_dist[i], d_i))

			quan_results['psnr'].append(psnr(rs_sr, tar_gt).detach().cpu().numpy())
			quan_results['ssim'].append(ssim(rs_sr, tar_gt).detach().cpu().numpy())
			quan_results['lpips'].append(self.lpips_metric(rs_sr.repeat(1,3,1,1), tar_gt.repeat(1,3,1,1)).detach().cpu().numpy())
		## quantitative result
		results = {}
		for key in quan_results.keys():
			print(f'{key}:\n', np.mean(quan_results[key]))
			results[key] = np.mean(quan_results[key])
		self.model.train()
		return results
	def validate_cms(self):
		self.model.eval()
		quan_results = {'psnr':[],'ssim':[], 'lpips':[]}

		for batch_idx, batch in enumerate(self.val_dataloader):
			tar_na, tar_gt, ref = batch
			tar_na, tar_gt, ref = tar_na.cuda().float(), tar_gt.cuda().float(), ref.cuda().float()
			with torch.no_grad():
				rs_cms, features_cms, tar_a, deformation = self.model.forward_cms(ref, tar_na)
			quan_results['psnr'].append(psnr(rs_cms, tar_gt).detach().cpu().numpy())
			quan_results['ssim'].append(ssim(rs_cms, tar_gt).detach().cpu().numpy())
			quan_results['lpips'].append(self.lpips_metric(rs_cms.repeat(1,3,1,1), tar_gt.repeat(1,3,1,1)).detach().cpu().numpy())
		## quantitative result
		results = {}
		for key in quan_results.keys():
			print(f'{key}:\n', np.mean(quan_results[key]))
			results[key] = np.mean(quan_results[key])

		self.model.train()
		return results

	def test(self, out_domains=None): # 'lr', 'rs_sr', 'rs_cms', 'tar_a', 'tar_na', 'tar_gt', 'ref','f_paris'

		if out_domains is None:
			if self.config['test']['cms_only']:
				print('WILL ONLY TEST THE CMS MODULE!!')
				out_domains = ['rs_cms', 'tar_a', 'tar_na', 'tar_gt', 'ref','D_infty']
			else:
				out_domains = ['lr', 'rs_sr', 'rs_cms', 'tar_a', 'tar_na', 'tar_gt', 'ref','f_pairs','D_infty']

		out_dirs = {}
		for outd in out_domains:
			out_path = os.path.join(self.config['exp_dir'], outd)
			out_dirs[outd] = out_path
			os.makedirs(out_path, exist_ok=True)
		quan_out_dir = os.path.join(self.config['exp_dir'], 'quantitative')
		os.makedirs(quan_out_dir, exist_ok=True)

		self.model.eval()
		test_dataset = self.configure_test_dataset()
		self.test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False, num_workers=1,drop_last=False)
		quan_results = {'psnr':[],'ssim':[], 'lpips':[]}
		for batch_idx, batch in enumerate(self.test_dataloader):
			with torch.no_grad():
				if self.config['test']['cms_only']:
					tar_na, tar_gt, ref = batch
					tar_na, tar_gt, ref = tar_na.cuda().float(), tar_gt.cuda().float(), ref.cuda().float()
					rs_cms, features_cms, tar_a, deformation = self.model.forward_cms(ref, tar_na)
					print('deformation:', torch.norm(deformation, p=2))
					pred = rs_cms
				else:
					lr, tar_na, tar_gt, ref = batch
					lr, tar_na, tar_gt, ref = lr.cuda().float(), tar_na.cuda().float(), tar_gt.cuda().float(), ref.cuda().float()
					rs_sr, rs_cms, features, features_cms, fea_cos_dist, tar_a = self.model.forward(lr, ref, tar_na)  # todo
					pred = rs_sr
				D_infty = tar_a - rs_cms
				if 'f_pairs' in out_domains:
					f_pairs = []
					for i in range(len(fea_cos_dist)):
						size = fea_cos_dist[i].shape[-1]
						d_i = F.interpolate(D_infty, (size, size))
						f_pairs.append((fea_cos_dist[i], d_i))

				pp = psnr(pred, tar_gt).detach().cpu().numpy()
				ss = ssim(pred, tar_gt).detach().cpu().numpy()
				lpp = self.lpips_metric(pred.repeat(1,3,1,1), tar_gt.repeat(1,3,1,1)).detach().cpu().numpy()[0][0][0][0]
				quan_results['psnr'].append(pp)
				quan_results['ssim'].append(ss)
				quan_results['lpips'].append(lpp)
				im_path = test_dataset.ref_paths[batch_idx]
				im_name = os.path.basename(im_path)
				print(f'psnr:{pp},ssim:{ss}, LPIPS:{lpp}\n')
				with open(os.path.join(quan_out_dir,'quan_single.txt'), 'a') as f:
					f.write(im_name + ': ')
					f.write(f'psnr:{pp},ssim:{ss}, LPIPS:{lpp}\n')
				for outd in out_domains:
					if outd == 'f_pairs':
						for j, (fdist, fgt) in enumerate(f_pairs):
							out_im = np.concatenate([np.array(common.tensor2im(fdist[0])), np.array(common.tensor2im(fgt[0]))], axis=1)
							Image.fromarray(out_im).save(os.path.join(out_dirs[outd], im_name.replace('.png', f'_{j}.png')))
					else:
						Image.fromarray(np.array(common.tensor2im(eval(outd)[0]))).save(os.path.join(out_dirs[outd], im_name))
		psnrm, ssimm, lppm = np.mean(quan_results['psnr']), np.mean(quan_results['ssim']), np.mean(quan_results['lpips'])
		with open(os.path.join(quan_out_dir, 'quan_total.txt'), 'a') as f:
			f.write(f'psnr:{psnrm},ssim:{ssimm}, LPIPS:{lppm}\n')
	def discriminate(self, map, fake_image, real_image):
		fake_and_real_img = torch.cat([fake_image, real_image], dim=0)
		d_out = self.dis(fake_and_real_img, segmap=torch.cat((map, map), dim=0))
		fake_feats, fake_preds, real_feats, real_preds = self.divide_pred(d_out)
		return fake_feats, fake_preds, real_feats, real_preds
	

	def divide_pred(self, pred):
		fake_feats = []
		fake_preds = []
		real_feats = []
		real_preds = []
		for p in pred[0]:
			fake_feats.append(p[:p.size(0) // 2])
			real_feats.append(p[p.size(0) // 2:])
		for p in pred[1]:
			fake_preds.append(p[:p.size(0) // 2])
			real_preds.append(p[p.size(0) // 2:])

		return fake_feats, fake_preds, real_feats, real_preds

	def smoothing_loss(self, y_pred):
		dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
		dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

		dx = dx * dx
		dy = dy * dy
		d = torch.mean(dx) + torch.mean(dy)
		return d
	def calc_g_loss(self, y, hr, y_=None, f_pairs=None):

		loss_dict = {}
		loss = 0.0
		if self.config['fea_lambda'] > 0 and f_pairs is not None:
			i = 0
			for pair in f_pairs:
				loss_f_l2 = F.mse_loss(pair[0],pair[1], reduction='mean')
				loss_dict[f'loss_f_l2_{i}'] = float(loss_f_l2)
				loss += loss_f_l2 * self.config['fea_lambda']
				i += 1

		if self.config['l2_lambda'] > 0:
			loss_hr_l2 = F.mse_loss(hr, y)
			loss += loss_hr_l2 * self.config['l2_lambda']
		if self.config['l1_lambda'] > 0:
			loss_hr_l1 = F.l1_loss(hr, y)
			loss += loss_hr_l1 * self.config['l1_lambda']
		if self.config['msssim_lambda'] > 0:
			loss_hr_msssim = 1-self.msssim((hr+1.0)/2.0, (y+1.0)/2.0)
			loss += loss_hr_msssim * self.config['msssim_lambda']
			loss_rec = loss.clone()
		if self.config['percep_lambda'] > 0:
			loss_percep = self.percep_loss(hr, y)
			loss += loss_percep * self.config['percep_lambda']
		if self.config['adv_lambda'] > 0 and self.global_step > 200000:
			adv_loss = 0.0
			if self.config['use_proj_gan']:
				feat_fake, fake_dis, feat_real, real_dis = self.discriminate(self.D_infty, hr, y)
				for fd,rd in zip(fake_dis,real_dis):
					valid = torch.ones(fd.shape, requires_grad=False).cuda()
					adv_loss += self.GANLoss(fd - rd.mean(0, keepdim=True), valid)
				GAN_Feat_loss = 0.0
				num_D = len(feat_fake)
				for i in range(num_D):
					GAN_Feat_loss += F.l1_loss(
						feat_fake[i], feat_real[i].detach()) * 10.0 / num_D
				loss += GAN_Feat_loss
				if y_ is not None:
					fake_dis_coarse = self.dis2(y)
					real_dis_coarse = self.dis2(y_)
					valid = torch.ones(real_dis_coarse.shape, requires_grad=False).cuda()
					adv_loss += self.GANLoss(fake_dis_coarse - real_dis_coarse.mean(0, keepdim=True), valid)
			else:
				fake_dis = self.dis(hr)
				real_dis = self.dis(y)
				valid = torch.ones(real_dis.shape, requires_grad=False).cuda()
				adv_loss = self.GANLoss(fake_dis - real_dis.mean(0, keepdim=True), valid)
			loss += adv_loss * self.config['adv_lambda']
		return loss, loss_rec

	def save_checkpoint(self, loss_dict, is_best):

		save_name = f'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_name_dis = 'best_model_dis.pt' if is_best else f'iteration_{self.global_step}_dis.pt'
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		checkpoint_path_dis = os.path.join(self.checkpoint_dir, save_name_dis)

		torch.save({'state_dict':self.model.state_dict()}, checkpoint_path)
		torch.save({'state_dict':self.dis.state_dict()}, checkpoint_path_dis)

		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val} \n{loss_dict}\n')
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def configure_optimizers(self):
		if self.config['train_cms']:
			params = list(self.model.cms.parameters()) + list(self.model.resunet.parameters())
		else:
			params = list(self.model.parameters())
		if self.config['optim_name'] == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.config['learning_rate'])
		else:
			optimizer = Ranger(params, lr=self.config['learning_rate'])
		return optimizer

	def configure_datasets(self):
		# co-training of multiple datasets
		tfs = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.5], [0.5])])
		k = int(self.config['output_size'] / self.config['input_size'])
		tm = self.config['target_modality']
		rm = self.config['ref_modality']
		st = self.config['strength']
		if st in ['low','mid','high']:
			target_dir = f'{tm}_elastic_{st}'
		else:
			target_dir = tm
		train_datasets = []
		val_datasets = []
		for dataset_root in self.config['dataset_roots']:

			train_datasets.append(MCSRDataset(
				lr_root=None if self.config['train_cms'] else os.path.join(dataset_root, self.config['train_sub_folder'], target_dir,f'LR{k}x'),
			    ref_root=os.path.join(dataset_root, self.config['train_sub_folder'], rm, 'HR'),
			    tar_gt_root=os.path.join(dataset_root, self.config['train_sub_folder'], tm, 'HR'),
			    tar_na_root=os.path.join(dataset_root, self.config['train_sub_folder'], target_dir, 'HR'),
			    source_transform=tfs,
			    target_transform=tfs
			))
		
			val_datasets.append(MCSRDataset(
				lr_root=None if self.config['train_cms'] else os.path.join(dataset_root, self.config['vali_sub_folder'], target_dir, f'LR{k}x'),
			    ref_root=os.path.join(dataset_root, self.config['vali_sub_folder'], rm, 'HR'),
			    tar_gt_root=os.path.join(dataset_root, self.config['vali_sub_folder'], tm, 'HR'),
			    tar_na_root=os.path.join(dataset_root, self.config['vali_sub_folder'], target_dir, 'HR'),
			    source_transform=tfs,
			    target_transform=tfs
			))

		train_dataset = ConcatDataset(train_datasets)
		val_dataset = ConcatDataset(val_datasets)
		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of validation samples: {len(val_dataset)}")
		return train_dataset, val_dataset

	def configure_test_dataset(self):
		tfs = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.5], [0.5])])
		k = int(self.config['output_size'] / self.config['input_size'])
		tm = self.config['target_modality']
		rm = self.config['ref_modality']
		st = self.config['strength']
		if st in ['low','mid','high']:
			target_dir = f'{tm}_elastic_{st}'
		else:
			target_dir = tm
		dataset_root =  self.config['test']['test_dataset_root']
		sub_folder = self.config['test']['sub_folder']
		test_dataset = MCSRDataset(
				lr_root=None if self.config['test']['cms_only'] else os.path.join(dataset_root, sub_folder, target_dir,f'LR{k}x'),
			    ref_root=os.path.join(dataset_root, sub_folder, rm, 'HR'),
			    tar_gt_root=os.path.join(dataset_root, sub_folder, tm, 'HR'),
			    tar_na_root=os.path.join(dataset_root, sub_folder, target_dir, 'HR'),
			    source_transform=tfs,
			    target_transform=tfs
			)


		print(f"Number of test samples: {len(test_dataset)}")
		return test_dataset
	#
	# def adjust(self, ws, global_step):
	# 	return float(mp.exp(-ws*(global_step**2)))
	def requires_grad(self, model, flag=True):
		for p in model.parameters():
			p.requires_grad = flag






