"""
Combined loss:  L1  +  SSIM (pytorch_msssim)  +  Edge/Gradient (Sobel).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ======================== Edge / Gradient Loss (Sobel) ========================

class EdgeLoss(nn.Module):
    """L1 loss on Sobel-filtered gradients to preserve sharp edges and textures.

    Computational cost is negligible (~O(1) per pixel) compared to a
    VGG-based perceptual loss.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _gradient(self, img):
        C = img.shape[1]
        kx = self.sobel_x.expand(C, -1, -1, -1)
        ky = self.sobel_y.expand(C, -1, -1, -1)
        gx = F.conv2d(img, kx, padding=1, groups=C)
        gy = F.conv2d(img, ky, padding=1, groups=C)
        return gx, gy

    def forward(self, pred, target):
        pred_gx, pred_gy = self._gradient(pred)
        tgt_gx, tgt_gy = self._gradient(target)
        return F.l1_loss(pred_gx, tgt_gx) + F.l1_loss(pred_gy, tgt_gy)


# ======================== Combined Loss ========================

class CombinedLoss(nn.Module):
    """``L_total = λ₁·L₁ + λ₂·(1−SSIM) + λ₃·L_edge``"""

    def __init__(self, lambda_l1=1.0, lambda_ssim=0.1, lambda_edge=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        self.w = (lambda_l1, lambda_ssim, lambda_edge)

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_edge = self.edge(pred, target)
        total = self.w[0] * loss_l1 + self.w[1] * loss_ssim + self.w[2] * loss_edge
        return total, loss_l1, loss_ssim, loss_edge
