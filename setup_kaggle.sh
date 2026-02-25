#!/bin/bash
# ============================================================
#  Setup script for Kaggle environment (fast — no compilation)
#
#    !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
#    %cd YOUR_REPO
#    !bash setup_kaggle.sh
# ============================================================

echo "===== Installing dependencies ====="
pip install -q pytorch-msssim torchmetrics einops

echo ""
echo "===== Setup complete! ====="
echo ""
echo "NOTE: mamba-ssm is NOT installed (takes 20-30 min to compile)."
echo "The code uses a PyTorch sequential scan fallback automatically."
echo ""
echo "To optionally install mamba-ssm for faster training, run in a SEPARATE cell:"
echo "  !pip install ninja causal-conv1d && TORCH_CUDA_ARCH_LIST='7.5' pip install mamba-ssm"
echo ""
echo "Train:  !python train.py --data_dir Rain100L --epochs 300"
