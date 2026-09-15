"""
=============================================================================
visualize.py — Publication-Quality Visualisation Suite
=============================================================================
Nature Communications-grade figures for DEM downscaling pipeline.

Functions auto-called by train.py and inference.py, or run standalone:
    python -m src.visualize --mode training --csv logs/train_test_*.csv
    python -m src.visualize --mode patches
    python -m src.visualize --mode inference --pred results/city_Predicted_DEM_10m.tif
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np

from src.config import (
    PROCESSED_DIR, VIS_DIR, LOG_DIR,
    IN_CHANNELS, FABDEM_CHANNEL_IDX,
)

# ═══════════════════════════════════════════════════════════════════════════
#  STYLE — Nature Communications / Publication Quality
# ═══════════════════════════════════════════════════════════════════════════

# Colour palette — muted, print-friendly
PALETTE = {
    "blue":      "#2166ac",
    "red":       "#b2182b",
    "green":     "#4dac26",
    "orange":    "#e08214",
    "purple":    "#7b3294",
    "grey":      "#636363",
    "teal":      "#01665e",
    "gold":      "#d4a017",
    "light_bg":  "#fafafa",
    "grid":      "#e0e0e0",
}

COMPONENT_COLOURS = {
    "l1":        PALETTE["blue"],
    "slope":     PALETTE["green"],
    "curvature": PALETTE["orange"],
    "flow":      PALETTE["purple"],
}

CHANNEL_NAMES = [
    "SAR VV", "SAR VH", "Red", "Green", "Blue", "NIR",
    "FABDEM", "HAND", "Roads", "NDVI", "NDWI",
]


def _apply_style():
    """Apply a clean, publication-quality matplotlib style."""
    plt.rcParams.update({
        # Figure
        "figure.facecolor":      "white",
        "figure.dpi":            150,
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.15,

        # Font — sans-serif, Nature Comms style
        "font.family":           "sans-serif",
        "font.sans-serif":       ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size":             9,
        "axes.titlesize":        10,
        "axes.labelsize":        9,
        "xtick.labelsize":       8,
        "ytick.labelsize":       8,
        "legend.fontsize":       8,

        # Axes
        "axes.facecolor":        PALETTE["light_bg"],
        "axes.edgecolor":        "#cccccc",
        "axes.grid":             True,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.linewidth":        0.6,

        # Grid
        "grid.color":            PALETTE["grid"],
        "grid.linewidth":        0.4,
        "grid.alpha":            0.7,

        # Lines
        "lines.linewidth":       1.5,
        "lines.markersize":      4,
    })


def _save_fig(fig, path: Path, title: str = ""):
    """Save figure and print confirmation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"    ✓ {title}: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  1. TRAINING DASHBOARD  (auto-called after training)
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_dashboard(csv_path: str, output_dir: str | None = None):
    """
    4-panel training dashboard from CSV log.
    Panels: Total Loss | Component Losses | LR Schedule | Train-Val Gap
    """
    _apply_style()

    csv_path = Path(csv_path)
    out_dir = Path(output_dir) if output_dir else VIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Read CSV ──
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("    ⚠ Empty CSV, skipping dashboard.")
        return

    epochs     = [int(r["epoch"]) for r in rows]
    lr         = [float(r["lr"]) for r in rows]
    train_tot  = [float(r["train_total"]) for r in rows]
    val_tot    = [float(r["val_total"]) for r in rows]

    components = {}
    for key in ["l1", "slope", "curvature", "flow"]:
        components[key] = {
            "train": [float(r.get(f"train_{key}", 0)) for r in rows],
            "val":   [float(r.get(f"val_{key}", 0)) for r in rows],
        }

    # ── Figure ──
    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel (a): Total Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_tot, color=PALETTE["blue"], label="Train", marker="o", markersize=3)
    ax1.plot(epochs, val_tot, color=PALETTE["red"], label="Val", marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.set_title("(a) Total Loss", fontweight="bold", loc="left")
    ax1.legend(frameon=True, fancybox=False, edgecolor="#ccc")
    ax1.set_yscale("log")

    # Panel (b): Component Losses (validation)
    ax2 = fig.add_subplot(gs[0, 1])
    for key, colour in COMPONENT_COLOURS.items():
        vals = components[key]["val"]
        ax2.plot(epochs, vals, color=colour, label=key.capitalize(), marker="o", markersize=2.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Weighted Loss (Val)")
    ax2.set_title("(b) Loss Components", fontweight="bold", loc="left")
    ax2.legend(frameon=True, fancybox=False, edgecolor="#ccc", ncol=2)
    ax2.set_yscale("log")

    # Panel (c): Learning Rate Schedule
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, lr, color=PALETTE["teal"], linewidth=2)
    ax3.fill_between(epochs, lr, alpha=0.15, color=PALETTE["teal"])
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("(c) LR Schedule", fontweight="bold", loc="left")
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))

    # Panel (d): Generalisation Gap
    ax4 = fig.add_subplot(gs[1, 1])
    gap = [v - t for v, t in zip(val_tot, train_tot)]
    colours = [PALETTE["green"] if g >= 0 else PALETTE["red"] for g in gap]
    ax4.bar(epochs, gap, color=colours, width=0.7, alpha=0.8)
    ax4.axhline(0, color="#999", linewidth=0.5, linestyle="--")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Val − Train Loss")
    ax4.set_title("(d) Generalisation Gap", fontweight="bold", loc="left")

    fig.suptitle("Training Dashboard", fontsize=13, fontweight="bold", y=1.01)

    _save_fig(fig, out_dir / "training_dashboard.png", "Training dashboard")
    return out_dir / "training_dashboard.png"


# ═══════════════════════════════════════════════════════════════════════════
#  2. MODEL ARCHITECTURE DIAGRAM  (auto-called after training)
# ═══════════════════════════════════════════════════════════════════════════

def plot_model_diagram(cfg, output_dir: str | None = None):
    """
    Block diagram of the SwinIR-DEM architecture with param counts per block.
    Professional figure without requiring external packages like torchviz.
    """
    _apply_style()

    out_dir = Path(output_dir) if output_dir else VIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Block definitions: (label, x_center, width, colour, sublabel)
    blocks = [
        (f"Input\n{IN_CHANNELS}-ch Features", 0.8, 1.2, "#e3f2fd", f"128×128×{IN_CHANNELS}"),
        ("Shallow\nFeature\nExtraction", 2.3, 1.1, PALETTE["blue"], f"Conv 3×3\n→ {cfg.embed_dim}d"),
    ]

    # RSTB blocks
    rstb_x_start = 3.7
    rstb_width = 0.55
    rstb_gap = 0.12
    for i in range(cfg.num_rstb):
        x = rstb_x_start + i * (rstb_width + rstb_gap)
        blocks.append(
            (f"RSTB\n{i+1}", x, rstb_width, PALETTE["teal"],
             f"{cfg.num_stl} STL\n{cfg.num_heads}H")
        )

    last_rstb_x = rstb_x_start + (cfg.num_rstb - 1) * (rstb_width + rstb_gap)

    blocks.extend([
        ("Deep\nFeature\nFusion", last_rstb_x + 1.2, 1.0, PALETTE["green"], f"Conv 3×3\n{cfg.embed_dim}d"),
        ("Gated\nResidual (+)", last_rstb_x + 2.5, 0.9, PALETTE["gold"], "σ(G)·FABDEM\n+ correction"),
        ("Recon\nHead", last_rstb_x + 3.6, 0.8, PALETTE["red"], "Conv→64→1"),
        ("Output\nDEM 10m", last_rstb_x + 4.8, 1.0, "#fce4ec", "128×128×1"),
    ])

    y_center = 2.5
    box_h = 1.8

    for i, (label, xc, w, colour, sublabel) in enumerate(blocks):
        rect = mpatches.FancyBboxPatch(
            (xc - w/2, y_center - box_h/2), w, box_h,
            boxstyle="round,pad=0.08",
            facecolor=colour, edgecolor="#555", linewidth=0.8, alpha=0.85,
        )
        ax.add_patch(rect)

        # Choose text colour for readability
        text_col = "white" if colour in [PALETTE["blue"], PALETTE["teal"],
                                          PALETTE["green"], PALETTE["red"],
                                          PALETTE["purple"]] else "#222"

        ax.text(xc, y_center + 0.2, label, ha="center", va="center",
                fontsize=7, fontweight="bold", color=text_col)
        ax.text(xc, y_center - 0.55, sublabel, ha="center", va="center",
                fontsize=5.5, color=text_col, alpha=0.9, style="italic")

        # Arrow to next block
        if i < len(blocks) - 1:
            next_xc = blocks[i + 1][1]
            next_w = blocks[i + 1][2]
            ax.annotate("", xy=(next_xc - next_w/2 - 0.02, y_center),
                        xytext=(xc + w/2 + 0.02, y_center),
                        arrowprops=dict(arrowstyle="->", color="#666",
                                        lw=1.2, connectionstyle="arc3,rad=0"))

    # Title and stats
    ax.text(8, 4.6, f"PI-SwinIR DEM Architecture — {cfg.name} profile",
            ha="center", va="center", fontsize=12, fontweight="bold")

    stats_text = (
        f"embed_dim={cfg.embed_dim}  |  "
        f"{cfg.num_rstb} RSTB × {cfg.num_stl} STL  |  "
        f"{cfg.num_heads} heads  |  "
        f"window={cfg.window_size}  |  "
        f"mlp_ratio={cfg.mlp_ratio}"
    )
    ax.text(8, 0.3, stats_text, ha="center", va="center",
            fontsize=7.5, color="#555", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                      edgecolor="#ddd", linewidth=0.5))

    _save_fig(fig, out_dir / "model_architecture.png", "Model architecture")
    return out_dir / "model_architecture.png"


# ═══════════════════════════════════════════════════════════════════════════
#  3. SAMPLE PATCHES  (standalone)
# ═══════════════════════════════════════════════════════════════════════════

def plot_sample_patches(x_path=None, y_path=None, n_samples=6,
                        output_dir=None, seed=42):
    """
    Grid of sample training patches: key input channels + target DEM.
    """
    _apply_style()

    x_path = Path(x_path) if x_path else PROCESSED_DIR / "train_X.npy"
    y_path = Path(y_path) if y_path else PROCESSED_DIR / "train_Y.npy"
    out_dir = Path(output_dir) if output_dir else VIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(x_path, mmap_mode="r")
    Y = np.load(y_path, mmap_mode="r")

    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, len(X))
    indices = rng.choice(len(X), n_samples, replace=False)

    display = [
        ("RGB",      None, "viridis"),
        ("FABDEM",   6,    "terrain"),
        ("HAND",     7,    "YlOrBr"),
        ("NDVI",     9,    "RdYlGn"),
        ("SAR VV",   0,    "gray"),
        ("Target",   None, "terrain"),
    ]
    n_cols = len(display)

    fig, axes = plt.subplots(n_samples, n_cols,
                             figsize=(n_cols * 2.0, n_samples * 2.0))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        xp = X[idx]
        yp = Y[idx]

        for col, (label, ch, cmap) in enumerate(display):
            ax = axes[row, col]

            if label == "RGB":
                rgb = np.stack([xp[2], xp[3], xp[4]], axis=-1)
                rgb = np.clip(rgb * 2.5, 0, 1)
                ax.imshow(rgb, interpolation="nearest")
            elif label == "Target":
                im = ax.imshow(yp[0], cmap=cmap, interpolation="nearest")
            else:
                im = ax.imshow(xp[ch], cmap=cmap, interpolation="nearest")

            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=9, fontweight="bold",
                             color=PALETTE["grey"])
            if col == 0:
                ax.set_ylabel(f"#{idx}", fontsize=8, color=PALETTE["grey"])

    fig.suptitle("Sample Training Patches", fontsize=12, fontweight="bold", y=1.01)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    _save_fig(fig, out_dir / "sample_patches.png", "Sample patches")
    return out_dir / "sample_patches.png"


# ═══════════════════════════════════════════════════════════════════════════
#  4. ALL 11 INPUT CHANNELS  (standalone)
# ═══════════════════════════════════════════════════════════════════════════

def plot_all_channels(x_path=None, patch_idx=0, output_dir=None):
    """Visualise all 11 input channels for a single patch."""
    _apply_style()

    x_path = Path(x_path) if x_path else PROCESSED_DIR / "train_X.npy"
    out_dir = Path(output_dir) if output_dir else VIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(x_path, mmap_mode="r")
    patch = X[patch_idx]

    cmaps = ["gray", "gray", "Reds", "Greens", "Blues", "RdYlGn",
             "terrain", "YlOrBr", "binary", "RdYlGn", "BrBG"]

    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    for i, ax in enumerate(axes.flat):
        if i < IN_CHANNELS:
            im = ax.imshow(patch[i], cmap=cmaps[i], interpolation="nearest")
            ax.set_title(CHANNEL_NAMES[i], fontsize=8, fontweight="bold",
                         color=PALETTE["grey"])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"All {IN_CHANNELS} Input Channels (Patch #{patch_idx})",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    _save_fig(fig, out_dir / f"all_channels_patch{patch_idx}.png", "All channels")
    return out_dir / f"all_channels_patch{patch_idx}.png"


# ═══════════════════════════════════════════════════════════════════════════
#  5. DATA DISTRIBUTION  (standalone)
# ═══════════════════════════════════════════════════════════════════════════

def plot_data_distribution(x_path=None, y_path=None, output_dir=None,
                           max_patches=2000):
    """Channel-wise histograms and target elevation distribution."""
    _apply_style()

    x_path = Path(x_path) if x_path else PROCESSED_DIR / "train_X.npy"
    y_path = Path(y_path) if y_path else PROCESSED_DIR / "train_Y.npy"
    out_dir = Path(output_dir) if output_dir else VIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(x_path, mmap_mode="r")
    Y = np.load(y_path, mmap_mode="r")

    n = min(max_patches, len(X))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), n, replace=False)

    fig, axes = plt.subplots(3, 5, figsize=(13, 7))
    axes = axes.flat

    for i in range(IN_CHANNELS):
        ax = axes[i]
        samples = X[idx, i].flatten()
        ax.hist(samples, bins=80, color=PALETTE["blue"], alpha=0.7,
                edgecolor="none", density=True)
        ax.set_title(CHANNEL_NAMES[i], fontsize=8, fontweight="bold")
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6)

    # Target DEM distribution
    ax_t = axes[12]
    target_samples = Y[idx, 0].flatten()
    ax_t.hist(target_samples, bins=80, color=PALETTE["red"], alpha=0.7,
              edgecolor="none", density=True)
    ax_t.set_title("Target DEM (norm)", fontsize=8, fontweight="bold")
    ax_t.set_yticks([])

    # Stats panel
    ax_s = axes[13]
    ax_s.axis("off")
    stats = (
        f"Total patches: {len(X):,}\n"
        f"Sampled: {n:,}\n"
        f"Shape: {X.shape}\n"
        f"Target range: [{target_samples.min():.3f}, {target_samples.max():.3f}]"
    )
    ax_s.text(0.1, 0.5, stats, transform=ax_s.transAxes, fontsize=8,
              va="center", family="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["light_bg"],
                        edgecolor="#ddd"))

    # Hide extra
    for j in range(14, 15):
        axes[j].axis("off")

    fig.suptitle("Input Channel Distributions", fontsize=12,
                 fontweight="bold", y=1.01)
    plt.tight_layout()

    _save_fig(fig, out_dir / "data_distribution.png", "Data distribution")
    return out_dir / "data_distribution.png"


# ═══════════════════════════════════════════════════════════════════════════
#  6. INFERENCE RESULT  (auto-called after inference)
# ═══════════════════════════════════════════════════════════════════════════

def _hillshade(dem, azimuth=315, altitude=45):
    """Compute hillshade from a 2D DEM array."""
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    dy, dx = np.gradient(dem)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    hs = (np.sin(alt_rad) * np.cos(slope) +
          np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    return np.clip(hs, 0, 1)


def plot_inference_result(pred_path: str, features_path: str | None = None,
                          gt_path: str | None = None,
                          city: str = "Unknown", output_dir: str | None = None):
    """
    6-panel inference result with proper 3-way comparison:
    Row 1: (a) Predicted DEM  | (b) Hillshade          | (c) Summary Stats
    Row 2: (d) FABDEM Input   | (e) Ground Truth (LiDAR)| (f) Elevation Distributions

    All elevations in metres.  NoData corners masked to NaN.
    """
    _apply_style()

    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ImportError:
        print("    ⚠ rasterio not available, skipping inference visualisation.")
        return None

    pred_path = Path(pred_path)
    out_dir = Path(output_dir) if output_dir else VIS_DIR / f"inference_{city}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Read prediction ──
    with rasterio.open(pred_path) as src:
        pred = src.read(1).astype(np.float32)
        pred_profile = src.profile.copy()

    # ── Mask NoData corners (CRS tilt artefacts) ──
    # Treat exactly 0.0 at the borders as NoData (projection fill value)
    nodata_mask = (pred == 0) | ~np.isfinite(pred)
    pred[nodata_mask] = np.nan

    # ── Load & convert FABDEM (Int16 ×100 → metres) ──
    fabdem = None
    if features_path and Path(features_path).exists():
        with rasterio.open(features_path) as src:
            fabdem_raw = src.read(FABDEM_CHANNEL_IDX + 1).astype(np.float32)
        fabdem = fabdem_raw / 100.0   # Int16 ×100 → metres
        fabdem[nodata_mask] = np.nan  # apply same mask

    # ── Load Ground Truth (if available) ──
    gt = None
    if gt_path and Path(gt_path).exists():
        gt = _load_gt_for_viz(gt_path, pred_profile)
        gt[nodata_mask] = np.nan

    # ── Shared elevation range (from valid pixels) ──
    all_valid = [pred[np.isfinite(pred)]]
    if fabdem is not None:
        all_valid.append(fabdem[np.isfinite(fabdem)])
    if gt is not None:
        all_valid.append(gt[np.isfinite(gt)])
    combined = np.concatenate(all_valid)
    vmin, vmax = np.percentile(combined, [2, 98])

    # ═══════════════════ Figure ═══════════════════
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.28, wspace=0.22,
                          width_ratios=[1, 1, 0.85])

    # Panel (a): Predicted DEM
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(pred, cmap="terrain", vmin=vmin, vmax=vmax,
                     interpolation="bilinear")
    ax1.set_title("(a) Predicted DEM (10 m)", fontweight="bold", loc="left")
    ax1.set_xticks([]); ax1.set_yticks([])
    plt.colorbar(im1, ax=ax1, label="Elevation (m)", fraction=0.046)

    # Panel (b): Hillshade
    ax2 = fig.add_subplot(gs[0, 1])
    hs = _hillshade(np.nan_to_num(pred, nan=0.0))
    hs[nodata_mask] = np.nan
    ax2.imshow(hs, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    ax2.set_title("(b) Hillshade", fontweight="bold", loc="left")
    ax2.set_xticks([]); ax2.set_yticks([])

    # Panel (c): Summary Statistics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    pv = pred[np.isfinite(pred)]
    stats_text = (
        f"City: {city}\n\n"
        f"Shape: {pred.shape[1]}×{pred.shape[0]} px\n"
        f"Min:  {pv.min():8.2f} m\n"
        f"Max:  {pv.max():8.2f} m\n"
        f"Mean: {pv.mean():8.2f} m\n"
        f"Std:  {pv.std():8.2f} m\n"
        f"NaN:  {nodata_mask.sum():,} px"
    )
    ax3.text(0.08, 0.6, stats_text, transform=ax3.transAxes, fontsize=9.5,
             va="center", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["light_bg"],
                       edgecolor="#ddd", linewidth=0.5))

    # Panel (d): FABDEM Input
    if fabdem is not None:
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(fabdem, cmap="terrain", vmin=vmin, vmax=vmax,
                         interpolation="bilinear")
        ax4.set_title("(d) FABDEM Input (30 m)", fontweight="bold", loc="left")
        ax4.set_xticks([]); ax4.set_yticks([])
        plt.colorbar(im4, ax=ax4, label="Elevation (m)", fraction=0.046)

    # Panel (e): Ground Truth
    if gt is not None:
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(gt, cmap="terrain", vmin=vmin, vmax=vmax,
                         interpolation="bilinear")
        ax5.set_title("(e) Ground Truth (LiDAR)", fontweight="bold", loc="left")
        ax5.set_xticks([]); ax5.set_yticks([])
        plt.colorbar(im5, ax=ax5, label="Elevation (m)", fraction=0.046)
    elif fabdem is not None:
        # Fallback: show residual if no GT available
        ax5 = fig.add_subplot(gs[1, 1])
        from scipy.ndimage import zoom
        if fabdem.shape != pred.shape:
            sy = pred.shape[0] / fabdem.shape[0]
            sx = pred.shape[1] / fabdem.shape[1]
            fabdem_rs = zoom(fabdem, (sy, sx), order=1)
        else:
            fabdem_rs = fabdem
        diff = pred - fabdem_rs
        dmax = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 98)
        im5 = ax5.imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax,
                         interpolation="bilinear")
        ax5.set_title("(e) Learned Correction (Pred − FABDEM)",
                       fontweight="bold", loc="left")
        ax5.set_xticks([]); ax5.set_yticks([])
        plt.colorbar(im5, ax=ax5, label="Δ Elevation (m)", fraction=0.046)

    # Panel (f): Combined Elevation Histogram
    ax6 = fig.add_subplot(gs[1, 2])
    pred_v = pred[np.isfinite(pred)].flatten()
    ax6.hist(pred_v, bins=120, color=PALETTE["red"], alpha=0.55,
             edgecolor="none", density=True, label="Predicted")
    if fabdem is not None:
        fab_v = fabdem[np.isfinite(fabdem)].flatten()
        ax6.hist(fab_v, bins=120, color=PALETTE["blue"], alpha=0.45,
                 edgecolor="none", density=True, label="FABDEM")
    if gt is not None:
        gt_v = gt[np.isfinite(gt)].flatten()
        ax6.hist(gt_v, bins=120, color=PALETTE["green"], alpha=0.45,
                 edgecolor="none", density=True, label="Ground Truth")
    ax6.set_xlabel("Elevation (m)")
    ax6.set_ylabel("Density")
    ax6.set_title("(f) Elevation Distribution", fontweight="bold", loc="left")
    ax6.legend(frameon=True, fancybox=False, edgecolor="#ccc", fontsize=7.5)

    fig.suptitle(f"DEM Inference Result — {city}", fontsize=13,
                 fontweight="bold", y=1.01)

    _save_fig(fig, out_dir / f"inference_{city}.png", f"Inference {city}")
    return out_dir / f"inference_{city}.png"


def _load_gt_for_viz(gt_path, ref_profile):
    """Load 1m ground truth, aggregate to 10m, reproject to match reference."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(gt_path) as src:
        raw = src.read(1).astype(np.float32)
        gt_crs = src.crs
        gt_transform = src.transform
        native_res = abs(src.res[0])
        nodata = src.nodata

    if nodata is not None:
        raw[raw == nodata] = np.nan
    raw = raw / 100.0   # Int16 ×100 → metres

    # Average pool to 10m
    factor = max(1, int(round(10.0 / native_res)))
    if factor > 1:
        h_trim = (raw.shape[0] // factor) * factor
        w_trim = (raw.shape[1] // factor) * factor
        raw = raw[:h_trim, :w_trim]
        blocks = raw.reshape(h_trim // factor, factor,
                             w_trim // factor, factor)
        with np.errstate(all="ignore"):
            pooled = np.nanmean(blocks, axis=(1, 3))
    else:
        pooled = raw

    pooled_h, pooled_w = pooled.shape
    pooled_transform = rasterio.transform.from_bounds(
        *rasterio.transform.array_bounds(
            raw.shape[0] if factor <= 1 else h_trim,
            raw.shape[1] if factor <= 1 else w_trim,
            gt_transform,
        ), pooled_w, pooled_h,
    )

    dst_crs       = ref_profile["crs"]
    dst_transform = ref_profile["transform"]
    dst_w, dst_h  = ref_profile["width"], ref_profile["height"]

    gt_aligned = np.empty((dst_h, dst_w), dtype=np.float32)
    gt_aligned[:] = np.nan

    reproject(
        source=pooled,
        destination=gt_aligned,
        src_transform=pooled_transform,
        src_crs=gt_crs,
        dst_transform=dst_transform,
        dst_crs=ref_profile["crs"],
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )

    return gt_aligned


# ═══════════════════════════════════════════════════════════════════════════
#  7. PROFILE COMPARISON TABLE  (standalone)
# ═══════════════════════════════════════════════════════════════════════════

def plot_profile_comparison(output_dir: str | None = None):
    """Generate a professional comparison table of all training profiles."""
    _apply_style()

    from src.config import PROFILES

    out_dir = Path(output_dir) if output_dir else VIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")

    headers = ["Profile", "embed_dim", "RSTB×STL", "Heads", "MLP", "Batch",
               "Epochs", "LR", "Warmup", "λ_flow", "λ_curv"]

    rows = []
    for name, p in PROFILES.items():
        rows.append([
            name,
            str(p.embed_dim),
            f"{p.num_rstb}×{p.num_stl}",
            str(p.num_heads),
            f"{p.mlp_ratio:.1f}",
            f"{p.batch_size}×{p.grad_accum_steps}={p.effective_batch}",
            str(p.epochs),
            f"{p.lr:.0e}",
            str(p.warmup_epochs),
            f"{p.lambda_flow:.1f}",
            f"{p.lambda_curvature:.1f}",
        ])

    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)

    # Style header
    for j, key in enumerate(headers):
        cell = table[0, j]
        cell.set_facecolor(PALETTE["blue"])
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colours
    for i in range(len(rows)):
        bg = PALETTE["light_bg"] if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(bg)
            table[i + 1, j].set_edgecolor("#ddd")

    ax.set_title("Training Profile Comparison", fontsize=12,
                 fontweight="bold", pad=20)

    _save_fig(fig, out_dir / "profile_comparison.png", "Profile comparison")
    return out_dir / "profile_comparison.png"


# ═══════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    from src.cli import configure_console
    configure_console()
    parser = argparse.ArgumentParser(
        description="PI-SwinIR DEM Visualisation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.visualize --mode training --csv logs/train_test_*.csv
  python -m src.visualize --mode patches
  python -m src.visualize --mode channels
  python -m src.visualize --mode distribution
  python -m src.visualize --mode inference --pred results/New_Orleans_Predicted_DEM_10m.tif --city New_Orleans
  python -m src.visualize --mode profiles
  python -m src.visualize --mode all
        """,
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["training", "patches", "channels",
                                 "distribution", "inference", "profiles", "all"],
                        help="Visualisation mode")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to training CSV log")
    parser.add_argument("--pred", type=str, default=None,
                        help="Path to predicted DEM GeoTIFF")
    parser.add_argument("--features", type=str, default=None,
                        help="Path to features GeoTIFF (for FABDEM baseline)")
    parser.add_argument("--gt", type=str, default=None,
                        help="Path to ground truth GeoTIFF (1m, Int16 ×100)")
    parser.add_argument("--city", type=str, default="Unknown")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--profile", type=str, default="test",
                        help="Profile name (for model diagram)")

    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  PI-SwinIR Visualisation → {args.mode}")
    print(f"{'='*50}\n")

    out = args.output_dir

    if args.mode in ("training", "all"):
        csv_path = args.csv
        if not csv_path:
            # Auto-find latest CSV
            csvs = sorted(LOG_DIR.glob("train_*.csv"))
            if csvs:
                csv_path = str(csvs[-1])
                print(f"  Auto-detected CSV: {csv_path}")
            else:
                print("  ⚠ No training CSV found. Skipping training dashboard.")
                csv_path = None
        if csv_path:
            plot_training_dashboard(csv_path, output_dir=out)

        # Model diagram for specified profile
        from src.config import get_profile
        cfg = get_profile(args.profile)
        plot_model_diagram(cfg, output_dir=out)

    if args.mode in ("patches", "all"):
        plot_sample_patches(output_dir=out)

    if args.mode in ("channels", "all"):
        plot_all_channels(output_dir=out)

    if args.mode in ("distribution", "all"):
        plot_data_distribution(output_dir=out)

    if args.mode in ("inference", "all"):
        if args.pred:
            plot_inference_result(args.pred, args.features,
                                 gt_path=args.gt,
                                 city=args.city, output_dir=out)
        else:
            # Auto-find predictions
            from src.config import RESULTS_DIR, RAW_DIR
            tifs = list(RESULTS_DIR.glob("*_Predicted_DEM_10m.tif"))
            for tif in tifs:
                city = tif.stem.replace("_Predicted_DEM_10m", "")
                feats = RAW_DIR / f"{city}_Features_10m.tif"
                gt_f  = RAW_DIR / f"{city}_GroundTruth_1m.tif"
                plot_inference_result(str(tif),
                                     str(feats) if feats.exists() else None,
                                     gt_path=str(gt_f) if gt_f.exists() else None,
                                     city=city, output_dir=out)

    if args.mode in ("profiles", "all"):
        plot_profile_comparison(output_dir=out)

    print(f"\n  ✓ All figures saved to: {out or VIS_DIR}\n")


if __name__ == "__main__":
    main()
