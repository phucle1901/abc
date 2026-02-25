# Mamba-Transformer Image Deraining

Hybrid Mamba-Transformer architecture with frequency-domain decomposition and selective gating for single-image rain removal.

## Architecture

```
Input (rainy) ──► Shallow Conv ──► Encoder (3 stages) ──► Bottleneck ──► Decoder (3 stages) ──► Conv ──► + Input ──► Output (clean)
                                        │                                       ▲
                                        └─── Gated Skip-Connections ────────────┘
```

**Hybrid Building Block** (at every stage):

1. **Haar DWT** decomposes features into low-frequency (LL) and high-frequency (LH, HL, HH) sub-bands
2. **Mamba-LF branch** (SS2D with 4-directional Cross-Scan) restores global context on the LL band
3. **Swin-HF branch** (Window Multi-Head Self-Attention) removes rain streaks from HF bands
4. **Window Cross-Attention Fusion** merges LF context into HF detail
5. **Haar IDWT** reconstructs the full-resolution feature map

## Libraries Used

| Component | Library | Pretrained? |
|---|---|---|
| Selective Scan (Mamba) | `mamba_ssm` CUDA kernel (fallback: pure PyTorch) | — |
| SSIM Loss | `pytorch_msssim` | — |
| PSNR / SSIM Metrics | `torchmetrics` | — |
| Perceptual Loss | `torchvision` VGG-19 | **Yes** (ImageNet) |
| Mixed Precision | `torch.autocast` + `GradScaler` | — |

## Project Structure

```
├── models/
│   ├── components.py   # DWT/IDWT, Swin attention, fusion, gated skip, down/up
│   ├── ss2d.py         # SS2D with mamba_ssm CUDA + PyTorch fallback
│   └── network.py      # Full U-Net: VSSBlock, HybridBlock, MambaTransformerDerain
├── dataset.py          # Rain100L data loader with augmentation
├── losses.py           # L1 + SSIM (pytorch_msssim) + VGG-19 perceptual
├── metrics.py          # PSNR / SSIM (torchmetrics)
├── config.py           # CLI argument parser
├── train.py            # Training loop with AMP
├── test.py             # Evaluation & result saving
├── setup_kaggle.sh     # One-click Kaggle setup
├── requirements.txt
└── Rain100L/           # Dataset (input/ and target/ sub-folders)
```

## Local Setup (Mac / CPU)

```bash
pip install -r requirements.txt
```

> `mamba_ssm` requires CUDA and will NOT install on Mac — the code automatically
> falls back to a pure-PyTorch parallel scan.

## Kaggle Setup (GPU)

In a Kaggle notebook, run these cells:

```python
# Cell 1: Clone repo
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
```

```python
# Cell 2: Install dependencies + CUDA mamba
!bash setup_kaggle.sh
```

```python
# Cell 3: Upload Rain100L dataset to Kaggle as a Dataset, then symlink:
!ln -s /kaggle/input/rain100l ./Rain100L
```

```python
# Cell 4: Train
!python train.py --data_dir Rain100L --epochs 300 --batch_size 4 --patch_size 128
```

## Training

```bash
python train.py --data_dir Rain100L --epochs 300 --batch_size 4 --patch_size 128
```

| Flag | Default | Description |
|---|---|---|
| `--base_channels` | 48 | Base channel width C |
| `--num_blocks` | 1 1 1 1 | Hybrid blocks per stage |
| `--d_state` | 16 | SSM state dimension |
| `--window_size` | 8 | Swin / fusion window size |
| `--lambda_l1` | 1.0 | L1 loss weight |
| `--lambda_ssim` | 0.1 | SSIM loss weight |
| `--lambda_perceptual` | 0.04 | VGG perceptual loss weight |
| `--lr` | 2e-4 | Initial learning rate |
| `--resume` | — | Path to checkpoint to resume |

Training logs are written to `logs/` (viewable with TensorBoard).

## Evaluation

```bash
python test.py --checkpoint checkpoints/best.pth --save_images
```

## Loss Function

```
L_total = λ₁·L₁ + λ₂·(1 − SSIM) + λ₃·L_perceptual(VGG-19)
```

## References

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
- VMamba: Visual State Space Model (Liu et al., 2024)
- Swin Transformer (Liu et al., 2021)
- U-Net (Ronneberger et al., 2015)
# abc
