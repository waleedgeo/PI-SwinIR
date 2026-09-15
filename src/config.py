"""
=============================================================================
config.py — Centralised Configuration with Multi-Profile Support
=============================================================================
All paths, hyperparameters, and constants for the PI-SwinIR DEM pipeline.

Profiles:
  • quick      — Rapid sanity check (~10-20 min, 5 epochs)
  • dev        — Development iteration (~1-2 hr, 15 epochs)
  • local      — Overnight training (~8-12 hr on RTX 3070)
  • full_local — Full local convergence (~2-4 days, RTX 3070 8 GB)
  • gcp        — Cloud training (Vertex AI, T4/A100)

Usage:
  from src.config import get_profile
  cfg = get_profile("quick")
=============================================================================
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import os

# ═══════════════════════════════════════════════════════════════════════════
#  GCS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
GCS_BUCKET = os.environ.get("DEM_GCS_BUCKET", "")


# ═══════════════════════════════════════════════════════════════════════════
#  PATHS  (shared across all profiles)
# ═══════════════════════════════════════════════════════════════════════════
# DEM_DATA_ROOT overrides the local data/output root; it must be a filesystem path.

_env_root = os.environ.get("DEM_DATA_ROOT")
PROJECT_ROOT   = Path(_env_root) if _env_root else Path(__file__).resolve().parent.parent
RAW_DIR        = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR        = PROJECT_ROOT / "logs"
RESULTS_DIR    = PROJECT_ROOT / "results"
VIS_DIR        = PROJECT_ROOT / "results" / "figures"


# ═══════════════════════════════════════════════════════════════════════════
#  CITY REGISTRY  (shared)
# ═══════════════════════════════════════════════════════════════════════════

CITIES = {
    # ── Original 4 cities ──
    "New_Orleans":   "EPSG:32615",
    "London":        "EPSG:27700",
    "Seattle":       "EPSG:32610",
    "San_Francisco": "EPSG:32610",
    # ── New cities (Feb 2026) ──
    "Houston":       "EPSG:32615",
    "Rotterdam":     "EPSG:28992",
    "Sydney":        "EPSG:32756",
    "Miami":         "EPSG:32617",
}

# Default split: seven development cities, one unseen evaluation city
DEFAULT_HOLDOUT_CITY = "San_Francisco"


# ═══════════════════════════════════════════════════════════════════════════
#  PREPROCESSING  (shared — data is already processed)
# ═══════════════════════════════════════════════════════════════════════════

PATCH_SIZE     = 128
STRIDE         = 64        # 50 % overlap
NODATA_THRESH  = 0.20      # >20 % NaN → discard patch
TARGET_RES     = 10.0      # metres

# Band indices in the 9-band Features_10m.tif (GEE export)
IDX_VV     = 0
IDX_VH     = 1
IDX_R      = 2
IDX_G      = 3
IDX_B      = 4
IDX_NIR    = 5
IDX_FABDEM = 6
IDX_HAND   = 7
IDX_ROADS  = 8

# Historical GEE export scales, for provenance only; loaders expect physical units.
SCALE_SAR    = 100.0    # SAR dB × 100 → Int16
SCALE_S2     = 10000.0  # S2 reflectance × 10000 → Int16
SCALE_ELEV   = 100.0    # FABDEM / GT / HAND metres × 100 → Int16

# Fixed normalization bounds used with the released checkpoint.
SAR_MIN, SAR_MAX   = -32.0,   44.0    # dB  (actual: -27.84 … +39.97)
ELEV_MIN, ELEV_MAX = -150.0, 500.0    # metres (widened for Rotterdam <0m, higher terrain)
HAND_MIN, HAND_MAX =    0.0, 500.0    # metres (widened for new city range)

# Final channel count after spectral indices (NDVI, NDWI — no NDBI)
# Channel order:
#   0  VV        (SAR, min-max → [0, 1])
#   1  VH        (SAR, min-max → [0, 1])
#   2  Red       (S2, already 0-1)
#   3  Green     (S2, already 0-1)
#   4  Blue      (S2, already 0-1)
#   5  NIR       (S2, already 0-1)
#   6  FABDEM    (elevation, min-max → [0, 1])
#   7  HAND      (min-max → [0, 1])
#   8  Roads     (binary 0/1)
#   9  NDVI      (mapped from [-1, 1] to [0, 1])
#  10  NDWI      (mapped from [-1, 1] to [0, 1])
IN_CHANNELS = 11

# Index of FABDEM in the final 11-channel normalised tensor
FABDEM_CHANNEL_IDX = 6

# Channel display names (for visualisation)
CHANNEL_NAMES = [
    "SAR VV", "SAR VH", "Red", "Green", "Blue", "NIR",
    "FABDEM", "HAND", "Roads", "NDVI", "NDWI",
]


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING PROFILE  dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingProfile:
    """All tuneable parameters for model architecture, training, and loss."""

    name: str

    # ── Model architecture ──
    embed_dim: int          = 96
    num_rstb: int           = 4
    num_stl: int            = 4       # Swin layers per RSTB
    num_heads: int          = 4       # must divide embed_dim
    window_size: int        = 8
    mlp_ratio: float        = 2.0
    drop_path: float        = 0.1

    # ── Training loop ──
    batch_size: int         = 2
    grad_accum_steps: int   = 4       # effective batch = batch_size × this
    epochs: int             = 200
    warmup_epochs: int      = 2       # linear LR warmup before cosine decay
    lr: float               = 2e-4    # AdamW
    lr_min: float           = 1e-6    # cosine annealing floor
    weight_decay: float     = 1e-4
    grad_clip: float        = 1.0     # critical for physics loss
    use_amp: bool           = True    # mixed-precision
    num_workers: int        = 4       # dataloader workers (safe on Linux)
    val_split: float        = 0.15    # fraction of train cities used for val
    early_stop_pat: int     = 30      # patience (epochs)
    save_every: int         = 10      # checkpoint interval
    seed: int               = 42      # reproducibility

    # ── Loss weights ──
    # L1 should dominate early training for elevation accuracy.
    # Physics losses are regularisers — keep small to avoid gradient conflicts.
    lambda_l1: float        = 1.0
    lambda_slope: float     = 0.1
    lambda_curvature: float = 0.05
    lambda_flow: float      = 0.01
    flow_iters: int         = 2       # flow accumulation diffusion passes (2 is sufficient)

    # ── City-based holdout ──
    holdout_city: str       = DEFAULT_HOLDOUT_CITY

    # ── Batch limits (None = unlimited) ──
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int]   = None

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum_steps

    @property
    def train_cities(self) -> List[str]:
        """Cities used for training/validation (all except holdout)."""
        return [c for c in CITIES if c != self.holdout_city]

    def summary(self) -> str:
        """One-liner for logging."""
        return (
            f"[{self.name}] embed={self.embed_dim}, "
            f"RSTB={self.num_rstb}×{self.num_stl}STL, "
            f"heads={self.num_heads}, mlp={self.mlp_ratio}, "
            f"batch={self.batch_size}×{self.grad_accum_steps}={self.effective_batch}, "
            f"epochs={self.epochs}, lr={self.lr}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  PROFILE PRESETS
# ═══════════════════════════════════════════════════════════════════════════
#
#  RTX 3070 Laptop (8 GB VRAM, 40 GB RAM) — compute budget guide:
#    embed=64,  RSTB=2×2 → ~300K params → batch=8 fits easily
#    embed=96,  RSTB=4×4 → ~1.7M params → batch=4 fits with AMP
#    embed=120, RSTB=6×6 → ~5M params   → batch=2 with AMP
#    embed=180, RSTB=6×8 → ~25M params  → needs T4/A100

PROFILES = {
    # ── quick: rapid sanity check (~10-20 min) ──
    # Smallest model, limited batches. Tests: data pipeline,
    # loss computation, checkpointing, visualisation hooks.
    "quick": TrainingProfile(
        name="quick",
        embed_dim=64,
        num_rstb=2,
        num_stl=2,
        num_heads=4,
        mlp_ratio=2.0,
        drop_path=0.0,
        batch_size=4,
        grad_accum_steps=1,
        epochs=5,
        warmup_epochs=1,             # ≥1 to avoid SequentialLR T_max=0
        lr=2e-4,
        num_workers=4,
        val_split=0.15,
        early_stop_pat=10,
        save_every=1,
        lambda_flow=0.05,            # keep low for fast runs
        max_train_batches=50,        # cap for speed
        max_val_batches=20,
    ),

    # ── dev: development iteration (~1-2 hr) ──
    # Medium model, full dataset, enough epochs to see convergence trends.
    # Use for: testing architecture changes, loss tuning, debugging.
    "dev": TrainingProfile(
        name="dev",
        embed_dim=64,
        num_rstb=3,
        num_stl=3,
        num_heads=4,
        mlp_ratio=2.0,
        drop_path=0.05,
        batch_size=4,
        grad_accum_steps=2,          # eff batch = 8
        epochs=15,
        warmup_epochs=2,
        lr=2e-4,
        lr_min=1e-6,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=True,
        num_workers=4,
        val_split=0.15,
        early_stop_pat=10,
        save_every=3,
    ),

    # ── test: 10-epoch deployment test (~2-3 hr on T4 GPU) ──
    # Medium model, full dataset, 10 epochs. For validating the full
    # pipeline before committing to overnight training.
    "test": TrainingProfile(
        name="test",
        embed_dim=64,
        num_rstb=3,
        num_stl=3,
        num_heads=4,
        mlp_ratio=2.0,
        drop_path=0.05,
        batch_size=4,
        grad_accum_steps=2,          # eff batch = 8
        epochs=10,
        warmup_epochs=1,
        lr=2e-4,
        lr_min=1e-6,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=True,
        num_workers=4,
        val_split=0.15,
        early_stop_pat=10,
        save_every=5,
    ),

    # ── local: overnight training (~8-12 hr on RTX 3070) ──
    # Solid model, full dataset. The everyday workhorse profile.
    # ~1.7M params, batch=4 with AMP fits easily in 8 GB.
    # ~15 min/epoch × 50 epochs ≈ 12 hours
    "local": TrainingProfile(
        name="local",
        embed_dim=96,
        num_rstb=4,
        num_stl=4,
        num_heads=4,                 # 96/4 = 24 per head
        window_size=8,
        mlp_ratio=2.0,
        drop_path=0.1,
        batch_size=4,
        grad_accum_steps=2,          # eff batch = 8
        epochs=50,
        warmup_epochs=3,
        lr=2e-4,
        lr_min=1e-6,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=True,
        num_workers=4,
        val_split=0.15,
        early_stop_pat=20,
        save_every=5,
    ),

    # ── full_local: complete convergence on RTX 3070 (~2-4 days) ──
    # Deeper model, longer training. "No cloud needed" option.
    # ~5M params, batch=2 with AMP fits in 8 GB.
    # ~25 min/epoch × 200 epochs ≈ 3.5 days (early stop likely ~100-150)
    "full_local": TrainingProfile(
        name="full_local",
        embed_dim=120,
        num_rstb=6,
        num_stl=6,
        num_heads=6,                 # 120/6 = 20 per head
        window_size=8,
        mlp_ratio=2.0,
        drop_path=0.1,
        batch_size=2,
        grad_accum_steps=4,          # eff batch = 8
        epochs=200,
        warmup_epochs=5,
        lr=1e-4,                     # lower LR for deeper model
        lr_min=1e-6,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=True,
        num_workers=4,
        val_split=0.15,
        early_stop_pat=40,
        save_every=10,
    ),

    # ── gcp: Vertex AI single A100 40 GB ──
    # Full SwinIR-scale model: ~16.4M params, batch=8×2=16 eff.
    # embed_dim=180, 6 RSTB × 6 STL — balanced for single-GPU training.
    # ~20-30 min/epoch on A100 × 200 epochs ≈ 3-4 days.
    # Perfect fit for 40GB VRAM (uses ~25-30GB).
    "gcp": TrainingProfile(
        name="gcp",
        embed_dim=180,
        num_rstb=6,
        num_stl=6,
        num_heads=6,                 # 180/6 = 30 per head
        window_size=8,
        mlp_ratio=4.0,
        drop_path=0.1,
        batch_size=8,
        grad_accum_steps=2,          # eff batch = 16
        epochs=200,
        warmup_epochs=10,
        lr=1e-4,
        lr_min=1e-6,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=True,
        num_workers=8,
        val_split=0.15,
        early_stop_pat=50,
        save_every=10,
    ),

    # ═══════════════════════════════════════════════════════════════════
    # L4-OPTIMIZED PROFILES (benchmarked on NVIDIA L4 24 GB)
    # ═══════════════════════════════════════════════════════════════════

    # ── sanity: ultra-fast pipeline validation (~2-5 min) ──
    # Tiny model (303K params). Confirms no errors, loss decreases,
    # outputs are sane. Run before any real training.
    "sanity": TrainingProfile(
        name="sanity",
        embed_dim=64,
        num_rstb=2,
        num_stl=2,
        num_heads=4,                 # 64/4 = 16 per head
        window_size=8,
        mlp_ratio=2.0,
        drop_path=0.05,
        batch_size=4,
        grad_accum_steps=1,
        epochs=2,
        warmup_epochs=0,
        lr=2e-4,
        lr_min=1e-6,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=True,
        num_workers=2,               # L4: 2 workers to save RAM
        val_split=0.15,
        early_stop_pat=2,
        save_every=1,
        flow_iters=2,
        max_train_batches=30,        # 30 train + 10 val batches
        max_val_batches=10,
    ),

    # ── l4_probe: convergence validation on L4 (~2.5 hr) ──
    # 2.5M params (6 RSTB × 4 STL). 3 full epochs to validate that
    # the model learns and show meaningful MAE reduction.
    # Same architecture as l4_full — checkpoint can seed the long run.
    # Optimised: flow_iters=2, num_workers=2.
    # ~47 min/epoch × 3 = ~2.5 hours.
    "l4_probe": TrainingProfile(
        name="l4_probe",
        embed_dim=96,
        num_rstb=6,
        num_stl=4,
        num_heads=4,                 # 96/4 = 24 per head
        window_size=8,
        mlp_ratio=2.0,
        drop_path=0.1,
        batch_size=8,                # 17.5 GB peak (benchmarked)
        grad_accum_steps=1,          # no accumulation → fastest
        epochs=3,
        warmup_epochs=1,
        lr=2e-4,
        lr_min=1e-6,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=True,
        num_workers=2,               # L4: 2 workers to save RAM
        val_split=0.15,
        early_stop_pat=3,
        save_every=1,                # save every epoch for safety
        flow_iters=2,                # 2 iterations (was 6 — saves ~30%)
    ),

    # ── l4_full: paper-quality production run on L4 (~14 hr) ──
    # IDENTICAL architecture to l4_probe (2.5M params, 6 RSTB × 4 STL).
    # 15 epochs with early stopping. Resume from l4_probe best checkpoint
    # or start fresh. Optimised: flow_iters=2, num_workers=3.
    # ~58 min/epoch × 15 = ~14.5 hours (early stop may cut to ~10 hr).
    "l4_full": TrainingProfile(
        name="l4_full",
        embed_dim=96,
        num_rstb=6,
        num_stl=4,
        num_heads=4,                 # 96/4 = 24 per head
        window_size=8,
        mlp_ratio=2.0,
        drop_path=0.1,
        batch_size=8,                # 17.5 GB peak (benchmarked)
        grad_accum_steps=1,          # no accumulation needed
        epochs=15,
        warmup_epochs=2,
        lr=2e-4,
        lr_min=1e-6,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=True,
        num_workers=3,               # L4: 3 workers (~3.5 GB RAM, reduces GPU stalls)
        val_split=0.15,
        early_stop_pat=10,
        save_every=1,                # save every epoch for safety
        flow_iters=2,                # 2 iterations (was 6 — saves ~30%)
    ),

    # ── rtx3070_full: l4_full architecture, resized to fit 8 GB VRAM ──
    # IDENTICAL architecture/hyperparameters to l4_full (2.53M params, 6 RSTB
    # x 4 STL, matches the released checkpoint) so a Phase 4 retrain stays a
    # faithful reproduction of the paper's recipe. l4_full's batch_size=8
    # with no grad accumulation benchmarked at ~17.5 GB peak on the L4 - too
    # large for an 8 GB card. Same effective batch (8) via batch_size=2 x
    # grad_accum_steps=4 instead, expected to cut peak activation memory to
    # roughly a quarter - NOT yet empirically benchmarked on this GPU. Run a
    # short smoke test (a handful of batches) to confirm peak VRAM and
    # sec/step before committing to a full run, and state the resulting cost
    # estimate before committing to a full training run.
    "rtx3070_full": TrainingProfile(
        name="rtx3070_full",
        embed_dim=96,
        num_rstb=6,
        num_stl=4,
        num_heads=4,                 # 96/4 = 24 per head
        window_size=8,
        mlp_ratio=2.0,
        drop_path=0.1,
        batch_size=2,
        grad_accum_steps=4,          # eff batch = 8, matches l4_full
        epochs=15,
        warmup_epochs=2,
        lr=2e-4,
        lr_min=1e-6,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=True,
        num_workers=2,               # laptop GPU: fewer workers to save RAM
        val_split=0.15,
        early_stop_pat=10,
        save_every=1,                # save every epoch for safety
        flow_iters=2,
    ),

    # ═══════════════════════════════════════════════════════════════════
    # ABLATION STUDY PROFILES (for paper — same arch as l4_full)
    # ═══════════════════════════════════════════════════════════════════

    # ── M0: L1-only baseline (no physics losses) ──
    # Tests: how much do physics-informed losses contribute?
    # Compare measured elevation and terrain metrics against the full objective.
    "ablation_l1_only": TrainingProfile(
        name="ablation_l1_only",
        embed_dim=96,
        num_rstb=6,
        num_stl=4,
        num_heads=4,
        window_size=8,
        mlp_ratio=2.0,
        drop_path=0.1,
        batch_size=8,
        grad_accum_steps=1,
        epochs=15,
        warmup_epochs=2,
        lr=2e-4,
        lr_min=1e-6,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=True,
        num_workers=3,
        val_split=0.15,
        early_stop_pat=10,
        save_every=1,
        flow_iters=2,
        # ── All physics losses DISABLED ──
        lambda_l1=1.0,
        lambda_slope=0.0,
        lambda_curvature=0.0,
        lambda_flow=0.0,
    ),

    # ── M4-A: No flow accumulation loss ──
    # Tests: does flow routing loss improve hydrological consistency?
    # Evaluate the contribution of flow regularization without assuming its effect.
    "ablation_no_flow": TrainingProfile(
        name="ablation_no_flow",
        embed_dim=96,
        num_rstb=6,
        num_stl=4,
        num_heads=4,
        window_size=8,
        mlp_ratio=2.0,
        drop_path=0.1,
        batch_size=8,
        grad_accum_steps=1,
        epochs=15,
        warmup_epochs=2,
        lr=2e-4,
        lr_min=1e-6,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=True,
        num_workers=3,
        val_split=0.15,
        early_stop_pat=10,
        save_every=1,
        flow_iters=2,
        # ── Flow loss DISABLED, slope + curvature kept ──
        lambda_l1=1.0,
        lambda_slope=0.1,
        lambda_curvature=0.05,
        lambda_flow=0.0,
    ),

    # ── M4-B: No curvature loss ──
    # Tests: does curvature loss improve surface smoothness & terrain shape?
    # Evaluate the contribution of curvature regularization.
    "ablation_no_curv": TrainingProfile(
        name="ablation_no_curv",
        embed_dim=96,
        num_rstb=6,
        num_stl=4,
        num_heads=4,
        window_size=8,
        mlp_ratio=2.0,
        drop_path=0.1,
        batch_size=8,
        grad_accum_steps=1,
        epochs=15,
        warmup_epochs=2,
        lr=2e-4,
        lr_min=1e-6,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_amp=True,
        num_workers=3,
        val_split=0.15,
        early_stop_pat=10,
        save_every=1,
        flow_iters=2,
        # ── Curvature loss DISABLED, slope + flow kept ──
        lambda_l1=1.0,
        lambda_slope=0.1,
        lambda_curvature=0.0,
        lambda_flow=0.01,
    ),
}


def get_profile(name: str) -> TrainingProfile:
    """Return a training profile by name. Raises KeyError if not found."""
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise KeyError(
            f"Unknown profile '{name}'. Available: {available}"
        )
    return PROFILES[name]


# ═══════════════════════════════════════════════════════════════════════════
#  BACKWARD-COMPATIBLE MODULE-LEVEL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
#  These point to the "local" profile so existing imports still work.
#  New code should prefer: cfg = get_profile("local")

_default = PROFILES["local"]

EMBED_DIM    = _default.embed_dim
NUM_RSTB     = _default.num_rstb
NUM_STL      = _default.num_stl
NUM_HEADS    = _default.num_heads
WINDOW_SIZE  = _default.window_size
MLP_RATIO    = _default.mlp_ratio
DROP_PATH    = _default.drop_path

BATCH_SIZE       = _default.batch_size
GRAD_ACCUM_STEPS = _default.grad_accum_steps
USE_AMP          = _default.use_amp
EPOCHS           = _default.epochs
LR               = _default.lr
WEIGHT_DECAY     = _default.weight_decay
LR_MIN           = _default.lr_min
GRAD_CLIP        = _default.grad_clip
VAL_SPLIT        = _default.val_split
NUM_WORKERS      = _default.num_workers
EARLY_STOP_PAT   = _default.early_stop_pat
SAVE_EVERY       = _default.save_every

LAMBDA_L1        = _default.lambda_l1
LAMBDA_SLOPE     = _default.lambda_slope
LAMBDA_CURVATURE = _default.lambda_curvature
LAMBDA_FLOW      = _default.lambda_flow
