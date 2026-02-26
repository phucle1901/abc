"""Training script for MambaTransformerDerain (multi-GPU + AMP for Kaggle)."""

import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config
from models import MambaTransformerDerain
from dataset import build_datasets
from losses import CombinedLoss
from metrics import compute_psnr, compute_ssim


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _unwrap(model):
    """Get the underlying model from DataParallel wrapper."""
    return model.module if isinstance(model, nn.DataParallel) else model


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    model.eval()
    total_psnr, total_ssim, n = 0.0, 0.0, 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        _, _, H, W = inp.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        inp_pad = torch.nn.functional.pad(inp, (0, pad_w, 0, pad_h), mode='reflect') \
            if (pad_h or pad_w) else inp

        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(inp_pad)

        pred = pred[:, :, :H, :W].clamp(0, 1).float()
        total_psnr += compute_psnr(pred, tgt) * inp.shape[0]
        total_ssim += compute_ssim(pred, tgt) * inp.shape[0]
        n += inp.shape[0]
    return total_psnr / max(n, 1), total_ssim / max(n, 1)


def main():
    args = get_config()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    n_gpus = torch.cuda.device_count()
    print(f'Device: {device} | GPUs: {n_gpus} | AMP: {use_amp}')

    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
        print('mamba_ssm CUDA kernel: ENABLED')
    except ImportError:
        print('mamba_ssm CUDA kernel: not found (using PyTorch fallback)')

    # ---- data ----
    train_set, val_set = build_datasets(
        args.data_dir, args.patch_size, args.train_split)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    print(f'Train: {len(train_set)} | Val: {len(val_set)}')

    # ---- model ----
    model = MambaTransformerDerain(
        base_dim=args.base_channels,
        num_blocks=args.num_blocks,
        d_state=args.d_state,
        ssm_expand=args.ssm_expand,
        patch_size=args.fusion_patch_size,
    )

    if not args.no_grad_checkpoint:
        model.enable_gradient_checkpointing()
        print('Gradient checkpointing: ENABLED')

    model = model.to(device)

    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f'DataParallel: using {n_gpus} GPUs')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params / 1e6:.2f} M')

    # ---- loss / optim / scheduler ----
    criterion = CombinedLoss(
        args.lambda_l1, args.lambda_ssim, args.lambda_edge).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup = args.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.GradScaler(device=device.type, enabled=use_amp)
    print(f'LR: {args.lr} | Warmup: {warmup} epochs | then cosine decay')

    start_epoch = 0
    best_psnr = 0.0

    # ---- resume ----
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        _unwrap(model).load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('best_psnr', 0.0)
        print(f'Resumed from epoch {start_epoch}')

    writer = SummaryWriter(args.log_dir)

    if args.eval_only:
        psnr, ssim = evaluate(model, val_loader, device, use_amp)
        print(f'Eval  PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}')
        return

    # ---- training loop ----
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for inp, tgt in pbar:
            inp, tgt = inp.to(device), tgt.to(device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(inp)
                loss, l1, ssim_l, edge_l = criterion(pred.clamp(0, 1), tgt)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}',
                             l1=f'{l1.item():.4f}',
                             ssim=f'{ssim_l.item():.4f}',
                             edge=f'{edge_l.item():.4f}')

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

        # ---- evaluate on train + val (every eval_interval epochs) ----
        do_eval = ((epoch + 1) % args.eval_interval == 0) or (epoch == args.epochs - 1)
        if do_eval:
            train_psnr, train_ssim = evaluate(model, train_loader, device, use_amp)
            val_psnr, val_ssim = evaluate(model, val_loader, device, use_amp)

            writer.add_scalar('train/psnr', train_psnr, epoch)
            writer.add_scalar('train/ssim', train_ssim, epoch)
            writer.add_scalar('val/psnr', val_psnr, epoch)
            writer.add_scalar('val/ssim', val_ssim, epoch)

            print(f'  Train PSNR: {train_psnr:.2f} dB | SSIM: {train_ssim:.4f}')
            print(f'  Val   PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}')

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    'epoch': epoch,
                    'model': _unwrap(model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_psnr': best_psnr,
                }, os.path.join(args.save_dir, 'best.pth'))

        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model': _unwrap(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_psnr,
        }, os.path.join(args.save_dir, f'epoch_{epoch+1}.pth'))

    writer.close()
    print(f'Training complete.  Best PSNR: {best_psnr:.2f} dB')


if __name__ == '__main__':
    main()
