"""
Dataset utilities for PI-SwinIR super-resolution training and evaluation.

Supports:
- Paired HR/LR image datasets (LR images loaded from disk).
- Unpaired HR-only datasets (LR images generated on-the-fly via bicubic downsampling).

Expected directory structure::

    root/
    ├── train/
    │   ├── HR/
    │   └── LR/      (optional)
    └── val/
        ├── HR/
        └── LR/      (optional)
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class SRDataset(Dataset):
    """Super-resolution dataset.

    Loads HR images from ``hr_dir`` and paired LR images from
    ``lr_dir`` (if provided and exists).  When ``lr_dir`` is ``None``
    or the directory is absent, LR images are synthesised on-the-fly
    via bicubic downsampling.

    Args:
        hr_dir (str): path to the HR image directory.
        lr_dir (str | None): path to the LR image directory.
            Pass ``None`` to generate LR images on-the-fly.
        scale (int): super-resolution scale factor.  Default: 4.
        patch_size (int): LR patch size for training crops.  Set to
            ``0`` to disable random cropping (used for validation).
            Default: 64.
        augment (bool): apply random horizontal/vertical flips and
            90-degree rotations during training.  Default: ``True``.
    """

    def __init__(
        self,
        hr_dir: str,
        lr_dir: Optional[str] = None,
        scale: int = 4,
        patch_size: int = 64,
        augment: bool = True,
    ):
        super().__init__()
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir) if (lr_dir and os.path.isdir(lr_dir)) else None
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment

        self.hr_paths = sorted(p for p in self.hr_dir.iterdir() if _is_image(p))
        if not self.hr_paths:
            raise FileNotFoundError(f"No images found in HR directory: {hr_dir}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.hr_paths)

    def _random_crop(
        self, hr: torch.Tensor, lr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random paired crop of LR and HR tensors."""
        _, lr_h, lr_w = lr.shape
        ps = self.patch_size
        if lr_h < ps or lr_w < ps:
            # Pad if image is smaller than patch size
            pad_h = max(0, ps - lr_h)
            pad_w = max(0, ps - lr_w)
            lr = TF.pad(lr, (0, 0, pad_w, pad_h), padding_mode="reflect")
            hr = TF.pad(
                hr,
                (0, 0, pad_w * self.scale, pad_h * self.scale),
                padding_mode="reflect",
            )
            _, lr_h, lr_w = lr.shape

        top = random.randint(0, lr_h - ps)
        left = random.randint(0, lr_w - ps)
        lr = lr[:, top : top + ps, left : left + ps]
        hr = hr[
            :,
            top * self.scale : (top + ps) * self.scale,
            left * self.scale : (left + ps) * self.scale,
        ]
        return hr, lr

    def _augment(
        self, hr: torch.Tensor, lr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply consistent random flips and rotations to HR and LR."""
        if random.random() > 0.5:
            hr = TF.hflip(hr)
            lr = TF.hflip(lr)
        if random.random() > 0.5:
            hr = TF.vflip(hr)
            lr = TF.vflip(lr)
        k = random.randint(0, 3)
        if k > 0:
            hr = torch.rot90(hr, k, dims=[1, 2])
            lr = torch.rot90(lr, k, dims=[1, 2])
        return hr, lr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (lr, hr) tensor pair.

        Returns:
            Tuple[Tensor, Tensor]: LR image (C, H, W) and HR image
            (C, H*scale, W*scale), both in [0, 1].
        """
        hr_path = self.hr_paths[idx]
        hr_img = _load_image(str(hr_path))
        hr = self.to_tensor(hr_img)

        if self.lr_dir is not None:
            lr_path = self.lr_dir / hr_path.name
            if lr_path.exists():
                lr_img = _load_image(str(lr_path))
                lr = self.to_tensor(lr_img)
            else:
                lr = self._bicubic_downsample(hr)
        else:
            lr = self._bicubic_downsample(hr)

        if self.patch_size > 0:
            hr, lr = self._random_crop(hr, lr)

        if self.augment:
            hr, lr = self._augment(hr, lr)

        return lr, hr

    def _bicubic_downsample(self, hr: torch.Tensor) -> torch.Tensor:
        """Downsample HR tensor to LR using bicubic interpolation."""
        _, h, w = hr.shape
        lr_h = h // self.scale
        lr_w = w // self.scale
        lr = TF.resize(
            hr,
            (lr_h, lr_w),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        return lr.clamp(0.0, 1.0)
