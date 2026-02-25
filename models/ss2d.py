"""
SS2D – Selective Scan 2D (Vision Mamba core).

Uses the official ``mamba_ssm`` CUDA kernel when available (10-50× faster).
Falls back to a pure-PyTorch parallel prefix scan otherwise.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# --------------- try to import CUDA selective scan from mamba_ssm -----------
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_CUDA = True
except ImportError:
    HAS_MAMBA_CUDA = False
    selective_scan_fn = None


# --------------------------------------------------------------------------- #
#  Fallback: parallel prefix scan for  y_t = a_t · y_{t-1} + b_t
# --------------------------------------------------------------------------- #

def _parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    L = a.shape[-1]
    num_steps = int(math.ceil(math.log2(max(L, 2))))
    aa, bb = a.clone(), b.clone()
    for k in range(num_steps):
        s = 1 << k
        if s >= L:
            break
        aa_prev = F.pad(aa[..., :-s], (s, 0), value=1.0)
        bb_prev = F.pad(bb[..., :-s], (s, 0), value=0.0)
        bb = aa * bb_prev + bb
        aa = aa * aa_prev
    return bb


# --------------------------------------------------------------------------- #
#  SS2D Module
# --------------------------------------------------------------------------- #

class SS2D(nn.Module):
    """Selective Scan 2D with 4-directional Cross-Scan.

    Input / output shape: ``(B, d_model, H, W)``.
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 3, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand)
        self.dt_rank = max(1, d_model // 16)
        self.n_dirs = 4

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv // 2, groups=self.d_inner)
        self.act = nn.SiLU()

        proj_dim = self.dt_rank + 2 * self.d_state
        self.x_proj_weight = nn.Parameter(
            torch.randn(self.n_dirs, self.d_inner, proj_dim) * 0.02)

        self.dt_proj_weight = nn.Parameter(
            torch.randn(self.n_dirs, self.d_inner, self.dt_rank) * 0.02)
        self.dt_proj_bias = nn.Parameter(
            torch.rand(self.n_dirs, self.d_inner) * 0.1)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))
        self.D_skip = nn.Parameter(torch.ones(self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W = x.shape
        L = H * W

        x_flat = rearrange(x, 'b d h w -> b (h w) d')
        xz = self.in_proj(x_flat)
        x_branch, z = xz.chunk(2, dim=-1)

        x_branch = rearrange(x_branch, 'b (h w) e -> b e h w', h=H, w=W)
        x_branch = self.act(self.conv2d(x_branch))

        # --- 4 scan sequences ---
        x_rm = x_branch.reshape(B, self.d_inner, L)
        x_rm_r = x_rm.flip(-1)
        x_cm = x_branch.permute(0, 1, 3, 2).reshape(B, self.d_inner, L)
        x_cm_r = x_cm.flip(-1)
        xs = torch.stack([x_rm, x_rm_r, x_cm, x_cm_r], dim=0)  # (4,B,E,L)

        # --- per-direction parameter projection ---
        x_for_proj = rearrange(xs, 'k b e l -> k b l e')
        x_dbl = torch.einsum('kble, ker -> kblr',
                             x_for_proj, self.x_proj_weight)
        dts, Bs, Cs = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dts = torch.einsum('kblr, ker -> kble',
                           dts, self.dt_proj_weight)
        dts = dts + self.dt_proj_bias[:, None, None, :]
        # NOTE: softplus is deferred — CUDA kernel fuses it; fallback applies it

        A = -torch.exp(self.A_log)

        # --- batched selective scan (all 4 dirs at once) ---
        use_cuda = HAS_MAMBA_CUDA and x.is_cuda
        ys = self._scan_batched_cuda(xs, dts, A, Bs, Cs) if use_cuda \
            else self._scan_batched_pytorch(xs, F.softplus(dts), A, Bs, Cs)

        # --- reverse the reversed directions & merge ---
        ys[1] = ys[1].flip(-1)
        ys[3] = ys[3].flip(-1)

        y_row = (ys[0] + ys[1]).reshape(B, self.d_inner, H, W)
        y_col = (ys[2] + ys[3]).reshape(B, self.d_inner, W, H)
        y_col = y_col.permute(0, 1, 3, 2)
        y = y_row + y_col

        # --- output gating ---
        y = rearrange(y, 'b e h w -> b (h w) e')
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return rearrange(y, 'b (h w) d -> b d h w', h=H, w=W)

    # ======================== CUDA path (mamba_ssm) ======================== #
    def _scan_batched_cuda(self, xs, dts, A, Bs, Cs):
        """All 4 directions in one ``selective_scan_fn`` call."""
        K, B, E, L = xs.shape
        u = rearrange(xs, 'k b e l -> (k b) e l').contiguous()
        delta = rearrange(dts, 'k b l e -> (k b) e l').contiguous()
        B_ssm = rearrange(Bs, 'k b l n -> (k b) n l').contiguous()
        C_ssm = rearrange(Cs, 'k b l n -> (k b) n l').contiguous()

        y = selective_scan_fn(
            u, delta,
            A.contiguous(), B_ssm, C_ssm,
            D=self.D_skip.float().contiguous(),
            delta_softplus=True,
        )
        return list(rearrange(y, '(k b) e l -> k b e l', k=K).unbind(0))

    # =================== PyTorch fallback (parallel scan) ================== #
    def _scan_batched_pytorch(self, xs, dts, A, Bs, Cs):
        """Pure-PyTorch batched scan using parallel prefix doubling."""
        K, B, E, L = xs.shape
        N = A.shape[1]

        u = rearrange(xs, 'k b e l -> (k b) e l')
        dt = rearrange(dts, 'k b l e -> (k b) e l')
        B_t = rearrange(Bs, 'k b l n -> (k b) l n')
        C_t = rearrange(Cs, 'k b l n -> (k b) l n')

        deltaA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))
        deltaBx = (dt * u).unsqueeze(-1) * B_t.unsqueeze(1)

        a = rearrange(deltaA, 'b e l n -> (b e n) l')
        b = rearrange(deltaBx, 'b e l n -> (b e n) l')
        h = _parallel_scan(a, b)
        h = rearrange(h, '(b e n) l -> b e l n', b=K * B, e=E, n=N)

        y = torch.einsum('beln, bln -> bel', h, C_t)
        y = y + u * self.D_skip.unsqueeze(0).unsqueeze(-1)
        return list(rearrange(y, '(k b) e l -> k b e l', k=K).unbind(0))
