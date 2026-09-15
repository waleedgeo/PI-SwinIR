"""
=============================================================================
dataset.py — PyTorch Dataset for DEM Patches (City-Based Splits)
=============================================================================
Loads pre-processed (N, C, 128, 128) .npy patch arrays with lazy memmap
and optional augmentation.

Supports two split modes:
  1. City-based holdout: 3 cities for train/val, 1 city for inference test
  2. Random split (fallback): split combined train_X.npy / train_Y.npy

Usage:
  from src.dataset import build_dataloaders
  train_loader, val_loader = build_dataloaders(cfg)
=============================================================================
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    Dataset, DataLoader, Subset, ConcatDataset, WeightedRandomSampler,
)

from src.config import PROCESSED_DIR, IN_CHANNELS


class DEMPatchDataset(Dataset):
    """
    Memory-mapped dataset for (X, Y) patch pairs stored as .npy files.

    Lazy loading (memmap opened on first __getitem__) is critical for:
      • Avoiding multiprocessing serialisation issues on Windows
      • Keeping memory footprint low with large datasets
    """

    def __init__(self, x_path, y_path, augment=False):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.augment = augment

        # Read header to get length without loading data
        self._X = None
        self._Y = None
        self._len = self._read_npy_length(self.x_path)

    @staticmethod
    def _read_npy_length(path):
        """Read the array length from the .npy header without loading data."""
        arr = np.load(path, mmap_mode="r")
        length = arr.shape[0]
        del arr
        return length

    def _ensure_loaded(self):
        """Lazy-load: open memory-mapped files on first access."""
        if self._X is None:
            self._X = np.load(self.x_path, mmap_mode="r")
            self._Y = np.load(self.y_path, mmap_mode="r")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._ensure_loaded()

        x = torch.from_numpy(self._X[idx].astype(np.float32))  # (C, H, W)
        y = torch.from_numpy(self._Y[idx].astype(np.float32))  # (1, H, W)

        if self.augment:
            x, y = self._random_augment(x, y)

        return x, y

    @staticmethod
    def _random_augment(x, y):
        """
        Apply random augmentations using PyTorch ops (views, not copies):
          • Horizontal flip (50%)
          • Vertical flip (50%)
          • Random 90° rotation (0/90/180/270°)

        Both x and y receive the SAME transformation.
        torch.flip and torch.rot90 return views — zero-copy.
        """
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])

        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])

        # Random 90° rotation (k ∈ {0, 1, 2, 3})
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            x = torch.rot90(x, k, dims=[1, 2])
            y = torch.rot90(y, k, dims=[1, 2])

        return x.contiguous(), y.contiguous()


# ═══════════════════════════════════════════════════════════════════════════
#  DataLoader Factory
# ═══════════════════════════════════════════════════════════════════════════

def _build_city_balanced_sampler(city_sizes, train_indices, seed=42):
    """
    Build a WeightedRandomSampler so each city contributes equally per epoch.

    Each patch gets weight = 1 / (num_cities * city_patch_count), ensuring
    that a city with 5,903 patches is sampled as often as one with 3,017.

    Parameters
    ----------
    city_sizes : list[int]
        Number of patches per city, in order of concatenation.
    train_indices : list[int]
        Indices into the ConcatDataset used for training.
    seed : int
        For reproducibility.

    Returns
    -------
    WeightedRandomSampler
    """
    total = sum(city_sizes)
    num_cities = len(city_sizes)

    # Build per-sample weight: maps global index → city weight
    sample_weights_full = np.zeros(total, dtype=np.float64)
    offset = 0
    for city_n in city_sizes:
        # Each city gets equal total weight = 1/num_cities,
        # distributed across its patches
        w = 1.0 / (num_cities * city_n) if city_n > 0 else 0.0
        sample_weights_full[offset: offset + city_n] = w
        offset += city_n

    # Only keep weights for training indices
    train_weights = [sample_weights_full[i] for i in train_indices]

    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_indices),
        replacement=True,
        generator=generator,
    )


def build_dataloaders(cfg, processed_dir=None):
    """
    Build train and val DataLoaders using city-based splits.

    Split logic:
      • Available development cities: per-city .npy files concatenated
      • Validation: random 15% of training patches
      • Holdout city: separate (loaded only at inference/eval time)
      • City-balanced sampling: WeightedRandomSampler ensures equal
        representation from each city per epoch.

    Parameters
    ----------
    cfg : TrainingProfile
    processed_dir : Path or None (defaults to config.PROCESSED_DIR)

    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    if processed_dir is None:
        processed_dir = PROCESSED_DIR
    processed_dir = Path(processed_dir)

    # On Windows, multiprocessing with mmap is fragile
    num_workers = cfg.num_workers
    if os.name == "nt" and num_workers > 0:
        print(f"  ⚠ Windows detected: setting num_workers=0 "
              f"(was {num_workers}) for pickle safety")
        num_workers = 0

    # ── Try city-based split first ──
    train_cities = cfg.train_cities
    city_datasets_aug = []       # augmented for training
    city_datasets_noaug = []     # no augmentation for validation
    city_sizes = []              # patch count per city (for balanced sampling)
    total_patches = 0

    for city in train_cities:
        x_path = processed_dir / f"{city}_X_patches.npy"
        y_path = processed_dir / f"{city}_Y_patches.npy"
        if x_path.exists() and y_path.exists():
            ds_aug   = DEMPatchDataset(x_path, y_path, augment=True)
            ds_noaug = DEMPatchDataset(x_path, y_path, augment=False)
            city_datasets_aug.append(ds_aug)
            city_datasets_noaug.append(ds_noaug)
            city_sizes.append(len(ds_aug))
            total_patches += len(ds_aug)
            print(f"    {city}: {len(ds_aug):,} patches")

    # Fallback: use combined train_X/Y.npy if no per-city files
    if len(city_datasets_aug) == 0:
        x_path = processed_dir / "train_X.npy"
        y_path = processed_dir / "train_Y.npy"
        if not x_path.exists():
            raise FileNotFoundError(
                f"No patch files found in {processed_dir}. "
                f"Run `python -m src.preprocess` first."
            )
        ds_aug   = DEMPatchDataset(x_path, y_path, augment=True)
        ds_noaug = DEMPatchDataset(x_path, y_path, augment=False)
        city_datasets_aug.append(ds_aug)
        city_datasets_noaug.append(ds_noaug)
        city_sizes.append(len(ds_aug))
        total_patches = len(ds_aug)
        print(f"    Combined: {total_patches:,} patches (fallback mode)")

    # ── Build train/val split using indices ──
    n_val   = int(total_patches * cfg.val_split)
    n_train = total_patches - n_val

    gen = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(total_patches, generator=gen).tolist()
    train_indices = indices[:n_train]
    val_indices   = indices[n_train:]

    # Concatenate all city datasets
    full_aug   = ConcatDataset(city_datasets_aug)
    full_noaug = ConcatDataset(city_datasets_noaug)

    # Augmented subset for training, non-augmented for validation
    train_ds = Subset(full_aug, train_indices)
    val_ds   = Subset(full_noaug, val_indices)

    print(f"  Dataset: {total_patches:,} patches → "
          f"train={n_train:,}, val={n_val:,}")
    print(f"  Holdout city: {cfg.holdout_city} (not used in training)")

    # ── City-balanced sampler (only for training) ──
    use_balanced = len(city_sizes) > 1
    if use_balanced:
        sampler = _build_city_balanced_sampler(
            city_sizes, train_indices, seed=cfg.seed
        )
        print(f"  City-balanced sampling: ON ({len(city_sizes)} cities)")
    else:
        sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),   # shuffle only if no sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
