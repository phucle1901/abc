import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description='Mamba-Transformer Image Deraining')

    parser.add_argument('--data_dir', type=str, default='Rain100L',
                        help='Root directory of dataset (with input/ and target/ subfolders)')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data used for training')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Training patch size (must be divisible by 16)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--base_channels', type=int, default=48)
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[2, 2, 2, 2],
                        help='Number of HybridBlocks per stage (enc1, enc2, enc3, bottleneck)')
    parser.add_argument('--d_state', type=int, default=16,
                        help='SSM state dimension for Mamba')
    parser.add_argument('--ssm_expand', type=int, default=2,
                        help='Expansion factor in SS2D')
    parser.add_argument('--fusion_patch_size', type=int, default=8,
                        help='Patch size for cross-attention fusion')

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=15,
                        help='Linear LR warmup epochs before cosine decay')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lambda_l1', type=float, default=1.0)
    parser.add_argument('--lambda_ssim', type=float, default=0.1)
    parser.add_argument('--lambda_edge', type=float, default=0.1)

    parser.add_argument('--no_grad_checkpoint', action='store_true',
                        help='Disable gradient checkpointing (uses more GPU memory)')

    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_only', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    return args
