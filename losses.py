"""
Combined loss:  L1  +  SSIM (pytorch_msssim)  +  VGG-19 Perceptual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import SSIM


# ======================== SSIM Loss (library) ========================

class SSIMLoss(nn.Module):
    """``1 − SSIM`` using the ``pytorch_msssim`` library."""

    def __init__(self, data_range=1.0, channel=3):
        super().__init__()
        self.ssim_module = SSIM(
            data_range=data_range,
            size_average=True,
            channel=channel,
            nonnegative_ssim=True,
        )

    def forward(self, pred, target):
        return 1.0 - self.ssim_module(pred, target)


# ======================== VGG-19 Perceptual Loss (pretrained) ========================

class VGGFeatureExtractor(nn.Module):
    """Extract conv3_3 features from a **pretrained** VGG-19."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.layers = nn.Sequential(*list(vgg.children())[:16])
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.layers(x)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VGGFeatureExtractor()

    def forward(self, pred, target):
        return F.mse_loss(self.vgg(pred), self.vgg(target))


# ======================== Combined Loss ========================

class CombinedLoss(nn.Module):
    """``L_total = λ₁·L₁ + λ₂·(1−SSIM) + λ₃·L_perceptual``"""

    def __init__(self, lambda_l1=1.0, lambda_ssim=0.1, lambda_perceptual=0.04):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.perceptual = PerceptualLoss()
        self.w = (lambda_l1, lambda_ssim, lambda_perceptual)

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_perc = self.perceptual(pred, target)
        total = self.w[0] * loss_l1 + self.w[1] * loss_ssim + self.w[2] * loss_perc
        return total, loss_l1, loss_ssim, loss_perc
