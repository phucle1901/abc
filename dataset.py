"""Dataset loader for paired image-deraining datasets (e.g. Rain100L)."""

import os
import random
from glob import glob

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class DerainDataset(Dataset):
    """Loads (rainy, clean) image pairs from *input_dir* and *target_dir*.

    Training mode  : random crop + random flip augmentation.
    Evaluation mode: full-resolution images (padded to be divisible by *pad_to*).
    """

    def __init__(self, input_dir, target_dir, patch_size=None,
                 augment=False, pad_to=16):
        super().__init__()
        self.input_paths = sorted(glob(os.path.join(input_dir, '*.*')))
        self.target_paths = sorted(glob(os.path.join(target_dir, '*.*')))
        assert len(self.input_paths) == len(self.target_paths), \
            f'Mismatch: {len(self.input_paths)} inputs vs {len(self.target_paths)} targets'
        self.patch_size = patch_size
        self.augment = augment
        self.pad_to = pad_to

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        inp = Image.open(self.input_paths[idx]).convert('RGB')
        tgt = Image.open(self.target_paths[idx]).convert('RGB')

        inp = TF.to_tensor(inp)
        tgt = TF.to_tensor(tgt)

        if self.patch_size is not None:
            inp, tgt = self._random_crop(inp, tgt, self.patch_size)

        if self.augment:
            inp, tgt = self._augment(inp, tgt)

        return inp, tgt

    @staticmethod
    def _random_crop(inp, tgt, size):
        _, h, w = inp.shape
        if h < size or w < size:
            inp = TF.resize(inp, [max(h, size), max(w, size)])
            tgt = TF.resize(tgt, [max(h, size), max(w, size)])
            _, h, w = inp.shape
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        inp = inp[:, top:top + size, left:left + size]
        tgt = tgt[:, top:top + size, left:left + size]
        return inp, tgt

    @staticmethod
    def _augment(inp, tgt):
        if random.random() > 0.5:
            inp = TF.hflip(inp)
            tgt = TF.hflip(tgt)
        if random.random() > 0.5:
            inp = TF.vflip(inp)
            tgt = TF.vflip(tgt)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            inp = torch.rot90(inp, k, [1, 2])
            tgt = torch.rot90(tgt, k, [1, 2])
        return inp, tgt


def build_datasets(data_dir, patch_size=128, train_split=0.8):
    """Build train / val datasets from a single directory.

    Expected layout::

        data_dir/
            input/   (rainy images)
            target/  (clean images)
    """
    input_dir = os.path.join(data_dir, 'input')
    target_dir = os.path.join(data_dir, 'target')

    full = DerainDataset(input_dir, target_dir)
    n = len(full)
    n_train = int(n * train_split)

    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_set = _SubsetDataset(full, train_idx, patch_size=patch_size, augment=True)
    val_set = _SubsetDataset(full, val_idx, patch_size=None, augment=False)
    return train_set, val_set


class _SubsetDataset(Dataset):
    """Wraps a DerainDataset with index subset & overridden augmentation."""

    def __init__(self, base, indices, patch_size=None, augment=False):
        self.base = base
        self.indices = indices
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        inp = Image.open(self.base.input_paths[real_idx]).convert('RGB')
        tgt = Image.open(self.base.target_paths[real_idx]).convert('RGB')

        inp = TF.to_tensor(inp)
        tgt = TF.to_tensor(tgt)

        if self.patch_size is not None:
            inp, tgt = DerainDataset._random_crop(inp, tgt, self.patch_size)
        if self.augment:
            inp, tgt = DerainDataset._augment(inp, tgt)

        return inp, tgt
