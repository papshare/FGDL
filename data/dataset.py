import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms


class MCSRDataset(Dataset):

	def __init__(self, lr_root=None, tar_gt_root=None, tar_na_root=None, ref_root=None, target_transform=None, source_transform=None):
		if lr_root is None:
			self.lr_paths = None
		else:
			self.lr_paths = sorted(data_utils.make_dataset(lr_root))
		self.tar_gt_paths = sorted(data_utils.make_dataset(tar_gt_root))
		self.tar_na_paths = sorted(data_utils.make_dataset(tar_na_root))
		self.ref_paths = sorted(data_utils.make_dataset(ref_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
	def __len__(self):
		return len(self.tar_gt_paths)

	def __getitem__(self, index):


		tar_gt_path = self.tar_gt_paths[index]
		tar_gt_im = Image.open(tar_gt_path)
		tar_na_path = self.tar_na_paths[index]
		tar_na_im = Image.open(tar_na_path)
		ref_path = self.ref_paths[index]
		ref_im = Image.open(ref_path)

		tar_gt_im = tar_gt_im.convert('L')
		tar_na_im = tar_na_im.convert('L')
		ref_im = ref_im.convert('L')

		if self.target_transform:
			tar_na_im = self.target_transform(tar_na_im)
			tar_gt_im = self.target_transform(tar_gt_im)

		if self.source_transform:
			ref_im = self.source_transform(ref_im)

		if self.lr_paths is not None:
			lr_path = self.lr_paths[index]
			lr_im = Image.open(lr_path)
			lr_im = lr_im.convert('L')
			if self.source_transform:
				lr_im = self.source_transform(lr_im)
			return lr_im, tar_na_im, tar_gt_im, ref_im
		else:
			return tar_na_im, tar_gt_im, ref_im



def add_grid(im):
	grid_color = 0
	d = 16
	im[:,::d] = grid_color
	im[::d,:] = grid_color
	return im
if __name__ == '__main__':
	ds = MCSRDataset(
		lr_root=os.path.join('/home/yd/data/BraTSReg_Training_Data_v3/brats22', 'train', 't2_elastic', f'LR4x'),
		ref_root=os.path.join('/home/yd/data/BraTSReg_Training_Data_v3/brats22', 'train', 't1', 'HR'),
		tar_gt_root=os.path.join('/home/yd/data/BraTSReg_Training_Data_v3/brats22', 'train', 't2', 'HR'),
		tar_na_root=os.path.join('/home/yd/data/BraTSReg_Training_Data_v3/brats22', 'train', 't2_elastic', 'HR'),
		source_transform=transforms.ToTensor(),
		target_transform=transforms.ToTensor()
	)
	for i in range(100):
		print(i)
		lr,gt_false, gt, ref = ds[i]

		im_gt = gt[0]
		im_gtf = gt_false[0]
		im_ref = ref[0]
		plt.subplot(2,2,1)
		plt.imshow(add_grid(im_gtf), cmap='gray')
		plt.subplot(2,2,2)
		plt.imshow(add_grid(im_gt), cmap='gray')
		plt.subplot(2,2,3)

		plt.imshow(add_grid(im_ref), cmap='gray')
		plt.subplot(2,2,4)

		plt.imshow(lr[0], cmap='gray')

		plt.show()
