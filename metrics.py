"""PSNR and SSIM evaluation metrics.

Uses ``torchmetrics`` when available (Kaggle), falls back to pure PyTorch.
"""

import torch
import torch.nn.functional as F

try:
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )
    _HAS_TM = True
except ImportError:
    _HAS_TM = False


# ======================== Fallback implementations ========================

def _psnr_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


def _gaussian_kernel_1d(size: int, sigma: float, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def _ssim_torch(pred: torch.Tensor, target: torch.Tensor,
                window_size: int = 11, sigma: float = 1.5) -> float:
    C = pred.shape[1]
    k1 = _gaussian_kernel_1d(window_size, sigma, pred.device)
    kernel = (k1[:, None] @ k1[None, :]).expand(C, 1, -1, -1)

    mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=C)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, kernel, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, kernel, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=C) - mu12

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# ======================== Public API ========================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    if _HAS_TM:
        return peak_signal_noise_ratio(pred, target, data_range=1.0).item()
    return _psnr_torch(pred, target)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    if _HAS_TM:
        return structural_similarity_index_measure(pred, target, data_range=1.0).item()
    return _ssim_torch(pred, target)
