import torch
from torch import nn
from pytorch_msssim import SSIM as _SSIM


class PSNR(object):
    def __init__(self, data_range, reduction='none', eps=1e-8):
        self.data_range = data_range
        self.reduction = reduction
        self.eps = eps

    def __call__(self, outputs, targets):
        with torch.no_grad():
            mse = torch.mean((outputs - targets) ** 2., dim=(1, 2, 3, 4))
            psnr = 10. * torch.log10((self.data_range ** 2.) / (mse + self.eps))

            if self.reduction == 'mean':
                return psnr.mean()
            if self.reduction == 'sum':
                return psnr.sum()

            return psnr


class SSIM(object):
    def __init__(self, channels, data_range, size_average=True):
        self.data_range = data_range
        self.ssim_module = _SSIM(data_range=data_range, size_average=size_average, channel=channels, spatial_dims=3)

    def __call__(self, outputs, targets):
        with torch.no_grad():
            return self.ssim_module(outputs, targets)
