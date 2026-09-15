"""
=============================================================================
Stage 2 — SwinIR Vision Transformer for DEM Refinement
=============================================================================
Pipeline  : DEM Super-Resolution (PI-SwinIR)
Purpose   : 10 m → 10 m DEM refinement (denoising / correction) using a
            custom Swin Transformer V2 architecture.  NO spatial upsampling.

Architecture overview:
  Input  (B, C, 128, 128)   C = 11 by default
      │
      ├── Shallow Feature Extractor: Conv2d(C, embed_dim, 3, 1, 1)
      │
      ├── Deep Feature Extraction: num_rstb × RSTB
      │       Each RSTB = num_stl × SwinTransformerLayer (V2)
      │                 + Conv2d(embed_dim, embed_dim, 3, 1, 1)
      │                 + Skip connection
      │       window_size = 8, num_heads = 4 (released l4_full profile)
      │
      ├── Conv2d(embed_dim, embed_dim, 3, 1, 1)  — deep feature fusion
      │
      ├── Reconstruction Tail  (NO PixelShuffle):
      │       Conv2d(embed_dim, 64, 3, 1, 1) + LeakyReLU
      │       Conv2d(64, 1, 3, 1, 1)
      │
      └── Gated refinement: output = gate * FABDEM_channel_from_input + residual
  Output (B, 1, 128, 128)

Usage:
    model = SwinIRDEM(in_channels=11, embed_dim=180)
    x = torch.randn(2, 11, 128, 128)
    y = model(x)                    # (2, 1, 128, 128)
    print_model_stats(model)
=============================================================================
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
#  Utility — window partition / reverse
# ═══════════════════════════════════════════════════════════════════════════

def window_partition(x, window_size):
    """
    Partition a (B, H, W, C) tensor into non-overlapping windows.

    Returns
    -------
    windows : (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
                   W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Merge windows back into a (B, H, W, C) tensor.
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


# ═══════════════════════════════════════════════════════════════════════════
#  Swin Transformer V2 — Window Attention
# ═══════════════════════════════════════════════════════════════════════════

class WindowAttentionV2(nn.Module):
    """
    Window-based Multi-Head Self-Attention (Swin V2 style).

    Key V2 improvements over V1:
      • Log-spaced Continuous Position Bias (log-CPB) instead of a
        parametric table — better generalisation to unseen window sizes.
      • Cosine attention (post-norm on Q, K) with a learnable temperature τ
        instead of scaled dot-product — improved training stability.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size   # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # ── Learnable temperature (V2) ──
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        # ── Log-spaced Continuous Position Bias (V2) ──
        # Small MLP: 2 → 512 → num_heads
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # Build relative coordinate table
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Wh, Ww)
        coords_flat = torch.flatten(coords, 1)                                    # (2, Wh*Ww)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]        # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous().float()    # (N, N, 2)

        # Log-space transform (V2 style)
        relative_coords_log = torch.sign(relative_coords) * torch.log2(
            torch.abs(relative_coords) + 1.0
        ) / math.log2(8)                                     # normalise by log2(8)

        self.register_buffer("relative_coords_table", relative_coords_log, persistent=False)

        # ── Q, K, V projection ──
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        x    : (num_windows * B, N, C)   where N = Wh * Ww
        mask : (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)    # (3, B_, nH, N, head_dim)
        q, k, v = qkv.unbind(0)

        # ── Cosine attention (V2) ──
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        logit_scale = torch.clamp(
            self.logit_scale, max=math.log(1.0 / 0.01)
        ).exp()
        attn = (q @ k.transpose(-2, -1)) * logit_scale       # (B_, nH, N, N)

        # ── Continuous position bias ──
        relative_position_bias = self.cpb_mlp(
            self.relative_coords_table                        # (N, N, 2)
        ).permute(2, 0, 1).contiguous()                       # (nH, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)

        # ── Attention mask for shifted windows ──
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
#  Swin Transformer Layer (V2)
# ═══════════════════════════════════════════════════════════════════════════

class SwinTransformerLayerV2(nn.Module):
    """
    A single Swin Transformer V2 layer with (shifted) window attention.

    Applies:
      1. Layer Norm → Window Attention → Skip
      2. Layer Norm → MLP → Skip
    """

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttentionV2(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )

        self.drop_path = nn.Identity() if drop_path <= 0 else DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, attn_mask):
        """
        x : (B, H * W, C)
        """
        H = W = int(math.sqrt(x.shape[1]))
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # ── Cyclic shift ──
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x

        # ── Window partition ──
        x_windows = window_partition(shifted_x, self.window_size)   # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size ** 2, C)   # (nW*B, ws*ws, C)

        # ── Window attention ──
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # ── Merge windows ──
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # ── Reverse cyclic shift ──
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # ── MLP ──
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ═══════════════════════════════════════════════════════════════════════════
#  Drop Path (Stochastic Depth)
# ═══════════════════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# ═══════════════════════════════════════════════════════════════════════════
#  Residual Swin Transformer Block (RSTB)
# ═══════════════════════════════════════════════════════════════════════════

class RSTB(nn.Module):
    """
    Residual Swin Transformer Block.

    Contains `num_layers` SwinTransformerLayerV2 modules with alternating
    shift patterns, followed by a Conv2d to re-inject translational
    equivariance, plus a residual skip connection.
    """

    def __init__(self, dim, num_heads, window_size=8, num_layers=6,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, img_size=128):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.window_size = window_size

        # Build alternating (W-MSA, SW-MSA) layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            shift = 0 if (i % 2 == 0) else window_size // 2
            dp = drop_path if isinstance(drop_path, float) else drop_path[i]
            self.layers.append(
                SwinTransformerLayerV2(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dp,
                )
            )

        # Conv tail — re-inject locality into transformer features
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        # Pre-compute the attention mask for shifted windows
        self.register_buffer("attn_mask", self._compute_attn_mask(), persistent=False)

    def _compute_attn_mask(self):
        """
        Build attention mask for SW-MSA (shift_size = window_size // 2).
        Returns None if no shift is used, else (nW, ws*ws, ws*ws).
        """
        H = W = self.img_size
        ws = self.window_size
        shift = ws // 2

        if shift == 0:
            return None

        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)     # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, ws * ws)     # (nW, ws*ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask                                  # (nW, ws*ws, ws*ws)

    def forward(self, x):
        """
        x : (B, C, H, W)

        Internally reshaped to (B, H*W, C) for the transformer layers,
        then back to (B, C, H, W) for the convolution + residual.
        """
        B, C, H, W = x.shape
        residual = x

        # Reshape: (B, C, H, W) → (B, H*W, C)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        # Apply transformer layers
        for layer in self.layers:
            mask = self.attn_mask if layer.shift_size > 0 else None
            x = layer(x, attn_mask=mask)

        # Back to spatial: (B, H*W, C) → (B, C, H, W)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Conv + residual
        x = self.conv(x) + residual
        return x


# ═══════════════════════════════════════════════════════════════════════════
#  SwinIR — DEM Refinement Model (No Upsampling)
# ═══════════════════════════════════════════════════════════════════════════

class SwinIRDEM(nn.Module):
    """
    Custom SwinIR for 10 m → 10 m DEM refinement.

    Key differences from standard SwinIR:
      • No PixelShuffle / sub-pixel upsampling — input and output are
        both (B, *, 128, 128).
      • Global residual skip — the model learns the correction Δ and
        adds it to the FABDEM input channel.
      • Swin Transformer V2 blocks for improved stability.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 11: 9 raw + 2 indices).
    embed_dim : int
        Feature dimension throughout the transformer backbone (default 180).
    num_rstb : int
        Number of Residual Swin Transformer Blocks (default 4).
    num_stl : int
        Number of Swin Transformer Layers per RSTB (default 6).
    num_heads : int
        Attention heads per layer (default 6).
    window_size : int
        Local attention window size (default 8; 128 must be divisible by 8).
    mlp_ratio : float
        Hidden dimension ratio in the MLP (default 4.0).
    fabdem_channel_idx : int
        Index of the FABDEM channel in the input tensor (default 6).
        Used for the global residual skip connection.
    img_size : int
        Spatial size of input patches (default 128).
    drop_rate : float
        Dropout rate (default 0.0).
    attn_drop_rate : float
        Attention dropout rate (default 0.0).
    drop_path_rate : float
        Stochastic depth rate (default 0.1).
    """

    def __init__(
        self,
        in_channels=11,
        embed_dim=180,
        num_rstb=4,
        num_stl=6,
        num_heads=6,
        window_size=8,
        mlp_ratio=4.0,
        fabdem_channel_idx=6,
        img_size=128,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.fabdem_channel_idx = fabdem_channel_idx
        self.img_size = img_size

        assert img_size % window_size == 0, (
            f"img_size ({img_size}) must be divisible by window_size ({window_size})"
        )

        # ── 1. Shallow Feature Extractor ──
        self.shallow_feature = nn.Conv2d(
            in_channels, embed_dim, kernel_size=3, stride=1, padding=1
        )

        # ── 2. Deep Feature Extraction — RSTB stack ──
        # Linearly increasing drop-path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_rstb * num_stl)]

        self.rstb_blocks = nn.ModuleList()
        for i in range(num_rstb):
            block_dpr = dpr[i * num_stl: (i + 1) * num_stl]
            self.rstb_blocks.append(
                RSTB(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    num_layers=num_stl,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=block_dpr,
                    img_size=img_size,
                )
            )

        # ── 3. Deep feature fusion conv ──
        self.deep_feature_fusion = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # ── 4. Reconstruction Tail (NO PixelShuffle) ──
        self.reconstruction = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1),
        )

        # ── 5. Gated Residual Skip ──
        # Learnable gate: sigmoid output controls how much FABDEM is retained
        # vs. replaced by the learned correction.
        self.gate_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid(),
        )

        # ── Initialise weights ──
        self.apply(self._init_weights)

        # ── Critical: override init for output heads ──
        # Zero-init reconstruction tail's final conv so residual ≈ 0 at start.
        # Without this, Kaiming init on Conv2d(64→1) produces output std ≈ 80+
        # because fan_in=576 amplifies variance through the 1-channel bottleneck.
        nn.init.zeros_(self.reconstruction[-1].weight)
        nn.init.zeros_(self.reconstruction[-1].bias)

        # Bias gate toward 1.0 so output starts as ≈ FABDEM (strong anchor).
        # sigmoid(2.0) ≈ 0.88 → output ≈ 0.88 × FABDEM + 0 ≈ FABDEM.
        nn.init.constant_(self.gate_conv[0].bias, 2.0)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, C, H, W) — normalised multi-modal input tensor

        Returns
        -------
        out : (B, 1, H, W) — refined DEM prediction
        """
        B, C, H, W = x.shape

        # ── Extract FABDEM channel for global residual ──
        fabdem = x[:, self.fabdem_channel_idx: self.fabdem_channel_idx + 1, :, :]
        # shape: (B, 1, H, W)

        # ── Shallow features ──
        shallow = self.shallow_feature(x)          # (B, embed_dim, H, W)

        # ── Deep features — transformer backbone ──
        deep = shallow
        for rstb in self.rstb_blocks:
            deep = rstb(deep)

        # ── Deep feature fusion + shallow skip ──
        deep = self.deep_feature_fusion(deep) + shallow    # (B, embed_dim, H, W)

        # ── Reconstruction tail ──
        residual = self.reconstruction(deep)       # (B, 1, H, W)

        # ── Gated residual: gate dynamically blends FABDEM with correction ──
        # gate ≈ 1 → trust FABDEM; gate ≈ 0 → override with correction
        gate = self.gate_conv(deep)                # (B, 1, H, W)
        out = gate * fabdem + residual

        return out


# ═══════════════════════════════════════════════════════════════════════════
#  Model Statistics
# ═══════════════════════════════════════════════════════════════════════════

def print_model_stats(model):
    """
    Print a summary of model parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    print("=" * 60)
    print("  SwinIR-DEM  Model Statistics")
    print("=" * 60)
    print(f"  Total parameters       : {total:>12,d}")
    print(f"  Trainable parameters   : {trainable:>12,d}")
    print(f"  Non-trainable params   : {non_trainable:>12,d}")
    print(f"  Approx. size (MB)      : {total * 4 / 1024**2:>12.1f}")
    print("=" * 60)

    # Per-block breakdown
    print("\n  Block breakdown:")
    print(f"    {'Block':<30s}  {'Params':>12s}")
    print(f"    {'─'*30}  {'─'*12}")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"    {name:<30s}  {n:>12,d}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  Quick self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Instantiating SwinIR-DEM model ...")
    model = SwinIRDEM(in_channels=11, embed_dim=180)
    print_model_stats(model)

    print("Running forward pass with dummy tensor ...")
    x = torch.randn(2, 11, 128, 128)
    with torch.no_grad():
        y = model(x)
    print(f"  Input shape  : {x.shape}")
    print(f"  Output shape : {y.shape}")
    assert y.shape == (2, 1, 128, 128), f"Shape mismatch! Got {y.shape}"
    print("  ✓ Forward pass OK — shape verified.")
