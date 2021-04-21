# -*- coding: utf-8 -*-
# @Time    : 2021/4/21 10:03
# @Author  : XinTong
# @FileName: densesift.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from time import perf_counter as p_time

pi = 3.1415926535897932384626433


class ImFilter:
    def __init__(self, device="cuda"):
        self.device = device
        kern_x = torch.zeros([1, 1, 5, 5])
        kern_y = torch.zeros([1, 1, 5, 5])
        kern_x[:, 0, 2] = torch.tensor([1, -8, 0, 8, -1]) / 12.
        kern_y[:, 0, :, 2] = torch.tensor([1, -8, 0, 8, -1]) / 12.
        self.kern_dx = kern_x.to(self.device)
        self.kern_dy = kern_y.to(self.device)
        kern_h = torch.zeros([1, 1, 7, 7])
        kern_v = torch.zeros([1, 1, 7, 7])
        kern_h[:, 0, 3] = torch.tensor([0, 0.25, 1, 1, 1, 0.25, 0])
        kern_v[:, 0, :, 3] = torch.tensor([0, 0.25, 1, 1, 1, 0.25, 0])
        self.kern_h = kern_h.to(self.device)
        self.kern_v = kern_v.to(self.device)

    def grad(self, img):
        dx = F.conv2d(img, self.kern_dx, stride=1, padding=2)
        dy = F.conv2d(img, self.kern_dy, stride=1, padding=2)
        dxy = torch.cat([dx, dy], dim=1)
        return dxy

    def window(self, img):
        img = img.permute([1, 0, 2, 3])
        h = F.conv2d(img, self.kern_h, stride=1, padding=3)
        v = F.conv2d(h, self.kern_v, stride=1, padding=3).permute([1, 0, 2, 3])
        return v


class CudaDSift:
    def __init__(self, shape=None, device="cuda"):
        self.d_mag = 0
        self.device = device
        self.cellSize = 3
        self.stepSize = 1
        self.Bins = 8
        self.filter = ImFilter(device)
        # init
        self.filter.grad(torch.tensor([[[[1.]]]]).to(device))
        self._sin = torch.sin(pi * 2 / self.Bins * torch.arange(8).type(torch.float)).to(self.device)
        self._cos = torch.cos(pi * 2 / self.Bins * torch.arange(8).type(torch.float)).to(self.device)
        self.theta = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0]]])
        self.idx = torch.tensor([-3, 0, 3, 6, -3, 0, 3, 6, -3, 0, 3, 6, -3, 0, 3, 6]).to(device)
        self.idy = torch.tensor([-3, -3, -3, -3, 0, 0, 0, 0, 3, 3, 3, 3, 6, 6, 6, 6]).to(device)
        self.shape = shape
        self.grid = None
        self.sift = None
        if shape is not None:
            grid = (F.affine_grid(self.theta, [1, 1, shape[0], shape[1]]).permute([0, 3, 1, 2]) + 1) / 2
            grid[:, 0] = grid[:, 0] * shape[0]
            grid[:, 1] = grid[:, 1] * shape[1]
            self.grid = grid.type(torch.long)
            self.sift = torch.zeros([1, 128, shape[0], shape[1]]).to(self.device)

    def bin(self, gradient):
        temp = gradient[:, 0] * self._cos.reshape([-1, 1, 1]) + gradient[:, 1] * self._sin.reshape([-1, 1, 1])
        temp = temp.clamp_min(0).unsqueeze(0)
        return temp

    def gather(self, imband_cell):
        for i, (x, y) in enumerate(zip(self.idx, self.idy)):
            grid_x = (self.grid[0, 0] + x).clamp(0, self.shape[0]-1)
            grid_y = (self.grid[0, 1] + y).clamp(0, self.shape[1]-1)
            self.sift[:, i * self.Bins:(i + 1) * self.Bins] = imband_cell[:, :, grid_x, grid_y]

    def normalization(self):
        mag = self.sift.norm(dim=1)
        return (self.sift / (mag + 1e-7) * 255).clamp_max(255).type(torch.uint8)

    def compute(self, img):
        if self.shape is not None:
            assert self.shape == img.shape, "error shape"
            img = torch.tensor(img).to(self.device).reshape([1, -1, self.shape[0], self.shape[1]])
        else:
            self.shape = img.shape
            img = torch.tensor(img).to(self.device).reshape([1, -1, self.shape[0], self.shape[1]])
            shape = img.shape
            grid = (F.affine_grid(self.theta, [1, 1, shape[0], shape[1]]).permute([0, 3, 1, 2]) + 1) / 2
            grid[:, 0] = grid[:, 0] * shape[0]
            grid[:, 1] = grid[:, 1] * shape[1]
            self.grid = grid.type(torch.long)
            self.sift = torch.zeros([1, 128, shape[0], shape[1]]).to(self.device)
        dxy = self.filter.grad(img)
        self.d_mag = dxy.norm(dim=1)
        grad = dxy / self.d_mag
        # 划分到八个方向
        imband = self.bin(grad)
        # 高斯窗
        imband_cell = self.filter.window(imband)
        # 拼接
        self.gather(imband_cell)
        # 正则化
        return self.normalization()



if __name__ == "__main__":
    im = cv2.imread("512.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    cu = CudaDSift(device="cuda", shape=im.shape)
    st = p_time()
    imgd = cu.compute(im)
    ed = p_time()
    print(ed - st)
