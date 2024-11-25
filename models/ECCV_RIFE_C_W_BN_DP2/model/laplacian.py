import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch

from torchvision.transforms import Grayscale
gs = Grayscale()


def gauss_kernel(size=5, channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1]))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr


def detect_edges(img):
    gray_img = gs(img)

    sobel_x = torch.tensor([[2,2,4,2,2], [1,1,2,1,1], [0,0,0,0,0], [-1,-1,-2,-1,-1], [-2,-2,-4,-2,-2]], dtype=torch.float32).view(1, 1, 5, 5)
    sobel_y = torch.tensor([[2,1,0,-1,-2], [2,1,0,-1,-2], [4,2,0,-2,-4], [2,1,0,-1,-2], [2,1,0,-1,-2]], dtype=torch.float32).view(1, 1, 5, 5)
    
    edge_x = F.conv2d(gray_img, sobel_x, padding=2)
    edge_y = F.conv2d(gray_img, sobel_y, padding=2)
    
    edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2).squeeze(0)
    
    return edge_magnitude/edge_magnitude.max()


class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)
        
    def forward(self, imgs, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        loss_lap = sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
        
        # weighted
        img0, img1 = imgs[:, :3], imgs[:, -3:]
        mask = abs(img0-target) + abs(img1-target) 
        mask = mask/mask.max()
        loss_weigted = mask*abs(input-target)
        loss_weigted = loss_weigted.mean()
         
        return loss_lap + loss_weigted
