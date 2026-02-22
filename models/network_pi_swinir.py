"""
Physics-Informed SwinIR (PI-SwinIR).

Wraps the SwinIR backbone with a physics-informed training interface.
The physics constraints are applied through the loss function during
training (see losses/physics_loss.py).
"""

import torch
import torch.nn as nn
from models.network_swinir import SwinIR


class PISwinIR(nn.Module):
    """Physics-Informed SwinIR for image super-resolution.

    Extends the SwinIR backbone with optional physics-aware residual
    guidance.  During forward pass the network behaves identically to
    standard SwinIR; the physics-informed constraints are enforced via
    :class:`losses.physics_loss.PhysicsInformedLoss` during training.

    Args:
        img_size (int): LR input patch size.  Default: 64.
        in_chans (int): number of input image channels.  Default: 3.
        embed_dim (int): patch embedding dimension.  Default: 60.
        depths (tuple[int]): per-RSTB depth.  Default: (6, 6, 6, 6).
        num_heads (tuple[int]): per-RSTB attention heads.
        window_size (int): Swin window size.  Default: 8.
        mlp_ratio (float): MLP expansion ratio.  Default: 4.0.
        upscale (int): super-resolution scale factor (2, 3, or 4).  Default: 4.
        num_feat (int): intermediate feature channels.  Default: 64.
        drop_rate (float): dropout rate.  Default: 0.0.
        attn_drop_rate (float): attention dropout rate.  Default: 0.0.
        drop_path_rate (float): stochastic depth rate.  Default: 0.1.
    """

    def __init__(
        self,
        img_size=64,
        in_chans=3,
        embed_dim=60,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=8,
        mlp_ratio=4.0,
        upscale=4,
        num_feat=64,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.upscale = upscale

        self.backbone = SwinIR(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            upscale=upscale,
            num_feat=num_feat,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): LR input of shape (B, C, H, W).

        Returns:
            Tensor: SR output of shape (B, C, H*scale, W*scale).
        """
        return self.backbone(x)
