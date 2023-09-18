import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import pdb

def ssim_numpy(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """
    start_time = time.time()
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # 為了 Conv 退縮矩陣
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

	# SSIM 簡化公式計算
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    print("This is numpy computer time: " + "{:.3f}".format((time.time()-start_time)))
    return ssim_map.mean()


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std) 
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

def ssim_torch(img1, img2, kernel_size=11):
    start_time = time.time()
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    transf = transforms.ToTensor()
    guassian_filter = gkern(kernel_size, std=1.5).expand(1, 3, kernel_size, kernel_size)
    guassian_filter = nn.Parameter(data=guassian_filter, requires_grad=False)
    
    img1 = transf(img1).unsqueeze(0)
    img2 = transf(img2).unsqueeze(0)

    #img1 = F.conv2d(img1, weight=guassian_filter, padding=(int((kernel_size-1)/2), int((kernel_size-1)/2)), stride=1)
    #img2 = F.conv2d(img2, weight=guassian_filter, padding=(int((kernel_size-1)/2), int((kernel_size-1)/2)), stride=1)
    mu1 = F.conv2d(img1, weight=guassian_filter, stride=1)
    mu2 = F.conv2d(img2, weight=guassian_filter, stride=1)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1**2, weight=guassian_filter, stride=1) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, weight=guassian_filter, stride=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, weight=guassian_filter, stride=1) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    print("This is torch computer time: " + "{:.3f}".format((time.time()-start_time)))
    return ssim_map.mean().detach().numpy()

if __name__=='__main__':
    img1_path = ''
    img2_path = ''
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    print("This is SSIM(numpy): " + str("{:.3f}".format(ssim_numpy(img1, img2))))
    print("This is SSIM(torch): " + str("{:.3f}".format(ssim_torch(img1, img2))))
    
