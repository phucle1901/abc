#!/bin/bash
# ============================================================
#  Setup script for Kaggle environment
#  Run this ONCE at the start of a Kaggle notebook:
#
#    !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
#    %cd YOUR_REPO
#    !bash setup_kaggle.sh
# ============================================================

set -e

echo "===== Installing base dependencies ====="
pip install -q pytorch-msssim torchmetrics einops

echo ""
echo "===== Installing mamba-ssm (CUDA selective scan) ====="
echo "  This may take 5-10 minutes to compile..."
pip install -q causal-conv1d 2>/dev/null || true
pip install -q mamba-ssm 2>/dev/null || true

echo ""
if python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" 2>/dev/null; then
    echo "✅ mamba-ssm installed — CUDA acceleration ENABLED"
else
    echo "⚠️  mamba-ssm not available — will use PyTorch fallback (slower but works)"
fi

echo ""
echo "===== Setup complete! ====="
echo "Run training with:  python train.py --data_dir Rain100L --epochs 300"
