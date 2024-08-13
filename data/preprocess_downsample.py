from torchvision import transforms
import cv2
import torch
import numpy as np
import glob
import os
from skimage.restoration import denoise_bilateral
from skimage import img_as_ubyte
from PIL import Image
import matplotlib.pyplot as plt

'''
no phases, for flexible test data
'''
def FFT2D(x):
    return torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)))


def IFFT2D(x):
    return torch.fft.ifft2(torch.fft.ifftshift(x), dim=(-2, -1))


def im2k(im):
    ## from 2*h*w im to 2*h*w kspace
    # im = im.permute(1, 2, 0).contiguous()
    k = torch.view_as_real(FFT2D(im)).contiguous()
    return k

def center_crop(data, shape):
    w_from = (data.shape[1] - shape[0]) // 2
    h_from = (data.shape[0] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[w_from:w_to, h_from:h_to, :]

def k2im(k):
    ## from 2*h*w im to 2*h*w kspace
    # k = k.permute(1, 2, 0).contiguous()
    im = torch.view_as_real(IFFT2D(torch.view_as_complex(k.contiguous()))).contiguous()
    return im

def do_downsample(ratios, modality='t2', image_size=256, data_dir='your path'):
    resize_func = transforms.Resize(image_size)
    pths = glob.glob(f'{data_dir}/{modality}/HR/*.png')
    for ratio in ratios:
        os.makedirs(f'{data_dir}/{modality}/LR{ratio}x', exist_ok=True)
    for pth in pths:
        name = pth.split('/')[-1]
        save_LR = f'{data_dir}/{modality}/LR{ratio}x/{name}'
        Tar_HR = np.expand_dims(cv2.imread(pth, 0) / 255.0, axis=2)
        Tar_HR = torch.from_numpy(Tar_HR).float().squeeze()
        Tar_HR = Tar_HR.unsqueeze(dim=0)
        Tar_HR = resize_func(Tar_HR)
        Tar_HR = Tar_HR.squeeze(dim=0)
        Tar_HR_k = im2k(Tar_HR)
        H, W = Tar_HR.size()
        for ratio in ratios:
            lh, lw = H // ratio, W // ratio
            Tar_LR_k = center_crop(Tar_HR_k, (lh, lw))
            Tar_LR = k2im(Tar_LR_k)
            Tar_LR = torch.norm(Tar_LR, dim=2)
            Tar_LR = (Tar_LR - Tar_LR.min()) / (Tar_LR.max() - Tar_LR.min())
            im_LR = (Tar_LR.numpy() * 255.0).astype('uint8')
            cv2.imwrite(save_LR, im_LR)

def fastmri_denoise(data_dir, modality):
    # rename the HR dir as HR_original
    os.rename(f'{data_dir}/{modality}/HR', f'{data_dir}/{modality}/HR_original')
    os.makedirs(f'{data_dir}/{modality}/HR')
    pths = glob.glob(f'{data_dir}/{modality}/HR/*.png')
    for pth in pths:
        img = cv2.imread(pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bilateral = denoise_bilateral(img, sigma_color=0.005, sigma_spatial=5, multichannel=True)
        bilateral = Image.fromarray(img_as_ubyte(bilateral))
        bilateral.save(pth.replace('HR_original', 'HR'))






