#!/bin/bash
# ============================================================
#  Setup script for Kaggle environment
#  Run this ONCE at the start of a Kaggle notebook:
#
#    !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
#    %cd YOUR_REPO
#    !bash setup_kaggle.sh
# ============================================================

echo "===== Installing base dependencies ====="
pip install -q pytorch-msssim torchmetrics einops

echo ""
echo "===== Installing mamba-ssm (CUDA selective scan) ====="
echo "  Kaggle T4 = compute capability 7.5"
echo "  This may take 5-10 minutes to compile..."
pip install -q ninja 2>/dev/null || true
export TORCH_CUDA_ARCH_LIST="7.5"
pip install -q causal-conv1d 2>/dev/null || true
pip install -q mamba-ssm 2>/dev/null || true

echo ""
if python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('OK')" 2>/dev/null; then
    echo "====================================================="
    echo "  mamba-ssm: INSTALLED (CUDA acceleration ENABLED)"
    echo "====================================================="
else
    echo "====================================================="
    echo "  mamba-ssm: NOT AVAILABLE"
    echo "  Using PyTorch sequential scan fallback (still works)"
    echo "====================================================="
fi

echo ""
echo "===== Setup complete! ====="
echo ""
echo "Train:  python train.py --data_dir Rain100L --epochs 300"
echo "Test:   python test.py --checkpoint checkpoints/best.pth --save_images"
