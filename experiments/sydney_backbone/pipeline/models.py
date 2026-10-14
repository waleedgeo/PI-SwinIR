"""R1C1 backbone swap: original Swin, or compact U-Net in the same gated head.

No training, weight search, pretrained encoder, or final elevation clamp occurs
here. The original Swin source is imported rather than reimplemented.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


PAPER_SWIN_KWARGS = dict(
    in_channels=11, embed_dim=96, num_rstb=6, num_stl=4, num_heads=4,
    window_size=8, mlp_ratio=2.0, fabdem_channel_idx=6,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
)


def source_identity(source_root=None):
    """Record the exact original model source used for both head initialization paths."""
    root = Path(source_root or os.environ.get("PI_SWINIR_CODE_ROOT", str(Path(__file__).resolve().parents[1])))
    path = (root / "src" / "model.py").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Original model source not found: {path}")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def original_swin_class(source_root=None):
    """Import original Swin class without mutating the active project or src namespace."""
    identity = source_identity(source_root)
    key = "_r1c1_original_model_" + identity["sha256"][:16]
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, identity["path"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(key, None)
            raise
    return sys.modules[key].SwinIRDEM


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class UNetFeatures(nn.Module):
    """Three-pool U-Net: 96 input features -> 96 full-resolution features."""
    def __init__(self):
        super().__init__()
        self.enc0 = DoubleConv(96, 32)
        self.enc1 = DoubleConv(32, 64)
        self.enc2 = DoubleConv(64, 128)
        self.bottleneck = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)
        self.dec0 = DoubleConv(64 + 32, 32)
        self.projection = nn.Conv2d(32, 96, 1)

    @staticmethod
    def join(deep, skip):
        deep = F.interpolate(deep, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((deep, skip), dim=1)

    def forward(self, x):
        x0 = self.enc0(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(self.pool(x1))
        deep = self.bottleneck(self.pool(x2))
        deep = self.dec2(self.join(deep, x2))
        deep = self.dec1(self.join(deep, x1))
        deep = self.dec0(self.join(deep, x0))
        return self.projection(deep)


class UNetDEM(nn.Module):
    """Only deep features change; retain paper stem, fusion, skip, and head."""
    def __init__(self, *, img_size=128, source_root=None):
        super().__init__()
        self.in_channels = 11
        self.embed_dim = 96
        self.fabdem_channel_idx = 6
        self.img_size = img_size
        self.shallow_feature = nn.Conv2d(11, 96, 3, padding=1)
        self.backbone = UNetFeatures()
        self.deep_feature_fusion = nn.Conv2d(96, 96, 3, padding=1)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
        )
        self.gate_conv = nn.Sequential(nn.Conv2d(96, 1, 1), nn.Sigmoid())
        # Use the source's exact shared-module initialization policy.
        self.apply(original_swin_class(source_root)._init_weights)
        nn.init.zeros_(self.reconstruction[-1].weight)
        nn.init.zeros_(self.reconstruction[-1].bias)
        nn.init.constant_(self.gate_conv[0].bias, 2.0)

    def features(self, x):
        shallow = self.shallow_feature(x)
        return self.deep_feature_fusion(self.backbone(shallow)) + shallow

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 11:
            raise ValueError("Expected input shape B x 11 x H x W")
        if min(x.shape[-2:]) < 8:
            raise ValueError("U-Net's three pooling stages need dimensions >= 8")
        deep = self.features(x)
        fabdem = x[:, 6:7]
        return self.gate_conv(deep) * fabdem + self.reconstruction(deep)


def build_model(backbone, *, img_size=128, source_root=None):
    """Return declared compact U-Net or the untouched paper-profile Swin class."""
    if backbone == "swin":
        return original_swin_class(source_root)(img_size=img_size, **PAPER_SWIN_KWARGS)
    if backbone == "unet":
        return UNetDEM(img_size=img_size, source_root=source_root)
    raise ValueError(f"Unknown backbone {backbone!r}; choose 'swin' or 'unet'")


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
