"""
Building blocks: Haar DWT/IDWT, Standard ViT Block, Patch-based Cross-Attention
Fusion, Gated Skip Connection, Patch Merging, Upsample.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ======================== Haar Wavelet Transform ========================

class HaarDWT2D(nn.Module):
    """2D Haar Discrete Wavelet Transform (lossless frequency decomposition)."""

    def forward(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        LL = (x00 + x01 + x10 + x11) * 0.5
        LH = (x00 + x01 - x10 - x11) * 0.5
        HL = (x00 - x01 + x10 - x11) * 0.5
        HH = (x00 - x01 - x10 + x11) * 0.5
        return LL, LH, HL, HH


class HaarIDWT2D(nn.Module):
    """2D Haar Inverse Discrete Wavelet Transform."""

    def forward(self, LL, LH, HL, HH):
        x00 = (LL + LH + HL + HH) * 0.5
        x01 = (LL + LH - HL - HH) * 0.5
        x10 = (LL - LH + HL - HH) * 0.5
        x11 = (LL - LH - HL + HH) * 0.5

        B, C, H2, W2 = x00.shape
        x = torch.empty(B, C, H2 * 2, W2 * 2, device=x00.device, dtype=x00.dtype)
        x[:, :, 0::2, 0::2] = x00
        x[:, :, 0::2, 1::2] = x01
        x[:, :, 1::2, 0::2] = x10
        x[:, :, 1::2, 1::2] = x11
        return x


# ======================== Utility ========================

def _pad_to(x, divisor):
    """Pad spatial dims of (B, C, H, W) so they are divisible by *divisor*."""
    _, _, H, W = x.shape
    pad_h = (divisor - H % divisor) % divisor
    pad_w = (divisor - W % divisor) % divisor
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, H, W


def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows.

    Args:
        x: (B, C, H, W)   H, W must be divisible by window_size.
    Returns:
        (B * nH * nW, C, ws, ws)
    """
    B, C, H, W = x.shape
    ws = window_size
    x = x.view(B, C, H // ws, ws, W // ws, ws)
    return x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, ws, ws)


def window_reverse(windows, window_size, H, W):
    """Reverse of window_partition.

    Args:
        windows: (B * nH * nW, C, ws, ws)
    Returns:
        (B, C, H, W)
    """
    ws = window_size
    nH, nW = H // ws, W // ws
    B = windows.shape[0] // (nH * nW)
    x = windows.view(B, nH, nW, -1, ws, ws)
    return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)


# ======================== Standard ViT Block ========================

class ViTBlock(nn.Module):
    """Standard Vision Transformer block: global Multi-head Self-Attention + FFN.

    Uses ``nn.MultiheadAttention`` from PyTorch for the attention computation.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)."""
        B, C, H, W = x.shape
        tokens = rearrange(x, 'b c h w -> b (h w) c')

        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out

        tokens = tokens + self.mlp(self.norm2(tokens))
        return rearrange(tokens, 'b (h w) c -> b c h w', h=H, w=W)


# ======================== Patch-based Cross-Attention Fusion ========================

class PatchCrossAttentionFusion(nn.Module):
    """Patch-based cross-attention: Q from LF context, K/V from HF detail.

    Both inputs share the same channel dimension (projected to C).
    Non-overlapping patches of size ``patch_size`` keep computation bounded.
    Uses ``nn.MultiheadAttention`` from PyTorch internally.
    """

    def __init__(self, dim, num_heads, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.norm_lf = nn.LayerNorm(dim)
        self.norm_hf = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, y_lf, y_hf):
        """y_lf, y_hf: (B, C, H, W) -> (B, C, H, W)."""
        B, C, H, W = y_lf.shape
        ps = self.patch_size

        if H < ps or W < ps:
            lf_tokens = rearrange(y_lf, 'b c h w -> b (h w) c')
            hf_tokens = rearrange(y_hf, 'b c h w -> b (h w) c')
            hf_residual = hf_tokens
            out, _ = self.cross_attn(
                self.norm_lf(lf_tokens),
                self.norm_hf(hf_tokens),
                hf_tokens,
            )
            out = out + hf_residual
            return rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)

        y_lf, orig_H, orig_W = _pad_to(y_lf, ps)
        y_hf, _, _ = _pad_to(y_hf, ps)
        _, _, Hp, Wp = y_lf.shape

        lf_patches = window_partition(y_lf, ps)
        hf_patches = window_partition(y_hf, ps)

        lf_tokens = rearrange(lf_patches, 'bw c h w -> bw (h w) c')
        hf_tokens = rearrange(hf_patches, 'bw c h w -> bw (h w) c')

        hf_residual = hf_tokens

        out, _ = self.cross_attn(
            self.norm_lf(lf_tokens),
            self.norm_hf(hf_tokens),
            hf_tokens,
        )
        out = out + hf_residual

        out = rearrange(out, 'bw (h w) c -> bw c h w', h=ps, w=ps)
        out = window_reverse(out, ps, Hp, Wp)
        return out[:, :, :orig_H, :orig_W]


# ======================== Gated Skip Connection ========================

class GatedSkipConnection(nn.Module):
    """Spatial gating to suppress rain streaks leaking from encoder."""

    def __init__(self, dim):
        super().__init__()
        self.conv_e = nn.Conv2d(dim, dim, 1)
        self.conv_d = nn.Conv2d(dim, dim, 1)

    def forward(self, x_enc, x_dec):
        gate = torch.sigmoid(self.conv_e(x_enc) + self.conv_d(x_dec))
        return x_enc * gate


# ======================== Down / Up sampling ========================

class PatchMerging(nn.Module):
    """Downsample by 2x: (B, C, H, W) -> (B, 2C, H/2, W/2)."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        x, _, _ = _pad_to(x, 2)
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 0::2, 1::2]
        x2 = x[:, :, 1::2, 0::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.reduction(self.norm(x))
        return rearrange(x, 'b h w c -> b c h w')


class Upsample(nn.Module):
    """Upsample by 2x using transposed convolution."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)
