"""
Utility functions for PI-SwinIR.

Provides image quality metrics (PSNR, SSIM), an AverageMeter for
tracking running statistics during training, and checkpoint
save/load helpers.
"""

import math
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Image Quality Metrics
# ---------------------------------------------------------------------------


def calculate_psnr(
    sr: torch.Tensor,
    hr: torch.Tensor,
    max_val: float = 1.0,
    crop_border: int = 0,
) -> float:
    """Compute PSNR between SR and HR tensors (in dB).

    Args:
        sr: super-resolved image tensor (B, C, H, W) or (C, H, W), [0, max_val].
        hr: ground-truth HR tensor of the same shape.
        max_val: maximum pixel value.  Default: 1.0.
        crop_border: number of border pixels to exclude on each side.

    Returns:
        PSNR value in dB (float).
    """
    assert sr.shape == hr.shape, "SR and HR must have the same shape."
    if crop_border > 0:
        sr = sr[..., crop_border:-crop_border, crop_border:-crop_border]
        hr = hr[..., crop_border:-crop_border, crop_border:-crop_border]
    mse = torch.mean((sr.float() - hr.float()) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def calculate_ssim(
    sr: torch.Tensor,
    hr: torch.Tensor,
    max_val: float = 1.0,
    crop_border: int = 0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Compute mean SSIM between SR and HR tensors.

    Args:
        sr: super-resolved image, (B, C, H, W) or (C, H, W), [0, max_val].
        hr: ground-truth HR of the same shape.
        max_val: data range.  Default: 1.0.
        crop_border: number of border pixels to exclude.
        window_size: Gaussian window size.  Default: 11.
        sigma: Gaussian sigma.  Default: 1.5.
        k1: SSIM stability constant.  Default: 0.01.
        k2: SSIM stability constant.  Default: 0.03.

    Returns:
        Mean SSIM value (float) in [0, 1].
    """
    if crop_border > 0:
        sr = sr[..., crop_border:-crop_border, crop_border:-crop_border]
        hr = hr[..., crop_border:-crop_border, crop_border:-crop_border]

    if sr.dim() == 3:
        sr = sr.unsqueeze(0)
        hr = hr.unsqueeze(0)

    sr = sr.float()
    hr = hr.float()

    # Build Gaussian kernel
    kernel = _gaussian_kernel(window_size, sigma, sr.device)
    C, H, W = sr.shape[1:]

    ssim_vals = []
    for c in range(C):
        sr_c = sr[:, c : c + 1]
        hr_c = hr[:, c : c + 1]
        ssim_vals.append(
            _ssim_channel(sr_c, hr_c, kernel, max_val, k1, k2)
        )
    return float(np.mean(ssim_vals))


def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    kernel /= kernel.sum()
    return kernel.view(1, 1, size, size)


def _ssim_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    max_val: float,
    k1: float,
    k2: float,
) -> float:
    import torch.nn.functional as F

    C1 = (k1 * max_val) ** 2
    C2 = (k2 * max_val) ** 2
    pad = kernel.shape[-1] // 2

    mu_x = F.conv2d(x, kernel, padding=pad)
    mu_y = F.conv2d(y, kernel, padding=pad)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, padding=pad) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=pad) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / denominator
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------


class AverageMeter:
    """Computes and stores running average and current value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: avg={self.avg:.4f} (n={self.count})"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    state: Dict,
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None,
) -> None:
    """Save a training checkpoint.

    Args:
        state: dict containing model/optimiser state and metadata.
        filepath: path to save the checkpoint file.
        is_best: if ``True``, also copy to ``best_filepath``.
        best_filepath: path for the best-model checkpoint.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(state, filepath)
    if is_best and best_filepath:
        import shutil
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """Load a checkpoint into model (and optionally optimizer).

    Args:
        filepath: path to the checkpoint file.
        model: model to load weights into.
        optimizer: optimizer to restore state (optional).
        device: target device.  Defaults to CPU.

    Returns:
        The full checkpoint dict (contains epoch, metrics, etc.).
    """
    if device is None:
        device = torch.device("cpu")
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
