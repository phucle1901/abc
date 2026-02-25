"""
MambaTransformerDerain – full U-Net with Hybrid Mamba-Transformer blocks.

Macro:  4-level U-Net (3 encoder stages + bottleneck + 3 decoder stages).
Micro:  Each HybridBlock = DWT → Mamba-LF ‖ Swin-HF → Cross-Attn Fusion → IDWT.
Skip:   Gated Skip-Connections between encoder and decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .ss2d import SS2D
from .components import (
    HaarDWT2D, HaarIDWT2D,
    SwinBlock, WindowCrossAttentionFusion,
    GatedSkipConnection, PatchMerging, Upsample,
)


# ======================== Vision State-Space Block ========================

class VSSBlock(nn.Module):
    """Mamba block: LayerNorm → SS2D → residual → LayerNorm → FFN → residual."""

    def __init__(self, dim, d_state=16, d_conv=3, expand=2, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state, d_conv, expand)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)."""
        B, C, H, W = x.shape

        # SS2D with pre-norm
        x_ln = rearrange(x, 'b c h w -> b (h w) c')
        x_ln = self.norm1(x_ln)
        x_ln = rearrange(x_ln, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.ss2d(x_ln) + x

        # FFN with pre-norm
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_flat = self.mlp(self.norm2(x_flat)) + x_flat
        return rearrange(x_flat, 'b (h w) c -> b c h w', h=H, w=W)


# ======================== Hybrid Building Block ========================

class HybridBlock(nn.Module):
    """DWT → Mamba-LF branch ‖ Swin-HF branch → Fusion → IDWT.

    Preserves spatial size: (B, C, H, W) → (B, C, H, W).
    """

    def __init__(self, dim, num_heads, window_size=8,
                 d_state=16, d_conv=3, ssm_expand=2, mlp_ratio=4.0):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        hf_dim = dim * 3

        self.mamba_branch = VSSBlock(dim, d_state, d_conv, ssm_expand, mlp_ratio)
        self.swin_branch = SwinBlock(hf_dim, num_heads, window_size, mlp_ratio)
        self.fusion = WindowCrossAttentionFusion(
            lf_dim=dim, hf_dim=hf_dim,
            num_heads=num_heads, window_size=window_size)

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape

        # pad to even for DWT
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        LL, LH, HL, HH = self.dwt(x)
        x_lf = LL
        x_hf = torch.cat([LH, HL, HH], dim=1)

        y_lf = self.mamba_branch(x_lf)
        y_hf = self.swin_branch(x_hf)

        y_fused = self.fusion(y_lf, y_hf)
        LH_o, HL_o, HH_o = y_fused.chunk(3, dim=1)

        out = self.idwt(y_lf, LH_o, HL_o, HH_o)
        out = out[:, :, :H, :W]
        return out + identity


# ======================== Stage (sequence of HybridBlocks) ========================

class Stage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=8, **kw):
        super().__init__()
        self.blocks = nn.ModuleList([
            HybridBlock(dim, num_heads, window_size, **kw)
            for _ in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ======================== Full Network ========================

class MambaTransformerDerain(nn.Module):
    """
    4-level U-Net with Mamba-Transformer hybrid blocks.

    Channel progression:  C → 2C → 4C → 8C (bottleneck) → 4C → 2C → C
    """

    def __init__(self, in_chans=3, base_dim=48,
                 num_blocks=(1, 1, 1, 1),
                 d_state=16, d_conv=3, ssm_expand=2,
                 window_size=8, mlp_ratio=4.0):
        super().__init__()
        dims = [base_dim * (2 ** i) for i in range(4)]  # [C, 2C, 4C, 8C]

        def _heads(d):
            return max(1, (d * 3) // 24)

        kw = dict(d_state=d_state, d_conv=d_conv,
                  ssm_expand=ssm_expand, mlp_ratio=mlp_ratio)

        # --- shallow feature extraction ---
        self.shallow = nn.Conv2d(in_chans, dims[0], 3, padding=1)

        # --- encoder ---
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(3):
            self.encoders.append(
                Stage(dims[i], num_blocks[i], _heads(dims[i]), window_size, **kw))
            self.downsamples.append(PatchMerging(dims[i]))

        # --- bottleneck ---
        self.bottleneck = Stage(
            dims[3], num_blocks[3], _heads(dims[3]), window_size, **kw)

        # --- decoder ---
        self.upsamples = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(2, -1, -1):
            self.upsamples.append(Upsample(dims[i + 1], dims[i]))
            self.gates.append(GatedSkipConnection(dims[i]))
            self.decoders.append(
                Stage(dims[i], num_blocks[i], _heads(dims[i]), window_size, **kw))

        # --- reconstruction ---
        self.output_conv = nn.Conv2d(dims[0], in_chans, 3, padding=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (B, 3, H, W) rainy image -> (B, 3, H, W) clean estimate."""
        inp = x

        x = self.shallow(x)

        # encoder
        skips = []
        for enc, down in zip(self.encoders, self.downsamples):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        # decoder
        for up, gate, dec, skip in zip(
                self.upsamples, self.gates, self.decoders, reversed(skips)):
            x = up(x)
            # handle size mismatch after upsample
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                                  align_corners=False)
            x = x + gate(skip, x)
            x = dec(x)

        x = self.output_conv(x)
        return x + inp
