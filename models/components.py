"""
Building blocks: Haar DWT/IDWT, Window Attention, Swin HF Block,
Window Cross-Attention Fusion, Gated Skip Connection, Patch Merging, Upsample.
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


# ======================== Window Utilities ========================

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


def _pad_to(x, divisor):
    """Pad spatial dims of (B, C, H, W) so they are divisible by *divisor*."""
    _, _, H, W = x.shape
    pad_h = (divisor - H % divisor) % divisor
    pad_w = (divisor - W % divisor) % divisor
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, H, W


# ======================== Relative Position Bias ========================

class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads

        self.table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'))
        coords_flat = coords.flatten(1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer('index', rel.sum(-1))

    def forward(self):
        """Returns (num_heads, ws*ws, ws*ws)."""
        N = self.window_size * self.window_size
        bias = self.table[self.index.view(-1)].view(N, N, -1)
        return bias.permute(2, 0, 1).contiguous()


# ======================== Window Self-Attention (for HF) ========================

class WindowSelfAttention(nn.Module):
    """W-MSA inside a single window."""

    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rpb = RelativePositionBias(window_size, num_heads)

    def forward(self, x):
        """x: (num_win, N, C) where N = ws*ws."""
        BW, N, C = x.shape
        qkv = self.qkv(x).reshape(BW, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rpb().unsqueeze(0)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(BW, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    """Swin Transformer block (W-MSA + FFN) for the HF branch."""

    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4.0):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowSelfAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)."""
        B, C, H, W = x.shape
        ws = self.window_size
        x, orig_H, orig_W = _pad_to(x, ws)
        _, _, Hp, Wp = x.shape

        wins = window_partition(x, ws)
        tokens = rearrange(wins, 'bw c h w -> bw (h w) c')

        tokens = self.attn(self.norm1(tokens)) + tokens
        tokens = self.mlp(self.norm2(tokens)) + tokens

        wins = rearrange(tokens, 'bw (h w) c -> bw c h w', h=ws, w=ws)
        x = window_reverse(wins, ws, Hp, Wp)
        return x[:, :, :orig_H, :orig_W]


# ======================== Window Cross-Attention Fusion ========================

class WindowCrossAttentionFusion(nn.Module):
    """Cross-Attention: Q from LF (C), K/V from HF (3C).

    Output has shape of HF (3C) with residual skip from HF.
    """

    def __init__(self, lf_dim, hf_dim, num_heads, window_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hf_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.q_proj = nn.Linear(lf_dim, hf_dim)
        self.k_proj = nn.Linear(hf_dim, hf_dim)
        self.v_proj = nn.Linear(hf_dim, hf_dim)
        self.out_proj = nn.Linear(hf_dim, hf_dim)

        self.norm_lf = nn.LayerNorm(lf_dim)
        self.norm_hf = nn.LayerNorm(hf_dim)
        self.rpb = RelativePositionBias(window_size, num_heads)

    def forward(self, y_lf, y_hf):
        """y_lf: (B, C, H, W), y_hf: (B, 3C, H, W) -> (B, 3C, H, W)."""
        B, C_lf, H, W = y_lf.shape
        C_hf = y_hf.shape[1]
        ws = self.window_size

        y_lf, orig_H, orig_W = _pad_to(y_lf, ws)
        y_hf, _, _ = _pad_to(y_hf, ws)
        _, _, Hp, Wp = y_lf.shape
        N = ws * ws

        lf_w = rearrange(window_partition(y_lf, ws), 'bw c h w -> bw (h w) c')
        hf_w = rearrange(window_partition(y_hf, ws), 'bw c h w -> bw (h w) c')

        hf_residual = hf_w
        lf_w = self.norm_lf(lf_w)
        hf_w = self.norm_hf(hf_w)

        Q = self.q_proj(lf_w).reshape(-1, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(hf_w).reshape(-1, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(hf_w).reshape(-1, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn + self.rpb().unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(-1, N, C_hf)
        out = self.out_proj(out) + hf_residual

        wins = rearrange(out, 'bw (h w) c -> bw c h w', h=ws, w=ws)
        out = window_reverse(wins, ws, Hp, Wp)
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
