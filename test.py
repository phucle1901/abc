"""Evaluate a trained MambaTransformerDerain model and save results."""

import os
import argparse
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import MambaTransformerDerain
from dataset import DerainDataset
from metrics import compute_psnr, compute_ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input_dir', type=str, default='Rain100L/input')
    parser.add_argument('--target_dir', type=str, default='Rain100L/target')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--base_channels', type=int, default=48)
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[2, 2, 2, 2])
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--ssm_expand', type=int, default=2)
    parser.add_argument('--fusion_patch_size', type=int, default=8)
    parser.add_argument('--save_images', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    model = MambaTransformerDerain(
        base_dim=args.base_channels,
        num_blocks=args.num_blocks,
        d_state=args.d_state,
        ssm_expand=args.ssm_expand,
        patch_size=args.fusion_patch_size,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded checkpoint from epoch {ckpt["epoch"] + 1}')

    dataset = DerainDataset(args.input_dir, args.target_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    if args.save_images:
        os.makedirs(args.output_dir, exist_ok=True)

    total_psnr, total_ssim = 0.0, 0.0

    with torch.no_grad():
        for i, (inp, tgt) in enumerate(tqdm(loader, desc='Testing')):
            if i>5:
                break
            inp, tgt = inp.to(device), tgt.to(device)
            _, _, H, W = inp.shape

            pad_h = (16 - H % 16) % 16
            pad_w = (16 - W % 16) % 16
            inp_pad = torch.nn.functional.pad(
                inp, (0, pad_w, 0, pad_h), mode='reflect') \
                if (pad_h or pad_w) else inp

            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(inp_pad)

            pred = pred[:, :, :H, :W].clamp(0, 1).float()

            total_psnr += compute_psnr(pred, tgt)
            total_ssim += compute_ssim(pred, tgt)

            if args.save_images:
                img = TF.to_pil_image(pred.squeeze(0).cpu())
                img.save(os.path.join(args.output_dir, f'{i + 1}.png'))

    n = len(dataset)
    print(f'Average PSNR: {total_psnr / n:.2f} dB')
    print(f'Average SSIM: {total_ssim / n:.4f}')


if __name__ == '__main__':
    main()
