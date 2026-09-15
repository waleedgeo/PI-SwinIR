"""
=============================================================================
evaluate.py — Stage 6: Evaluation Metrics & Visualization
=============================================================================
Quantitative and visual assessment of PI-SwinIR DEM predictions against
the ground-truth LiDAR DEM.

Metrics:
  • MAE  (Mean Absolute Error, metres)
  • RMSE (Root Mean Square Error, metres)
  • R²   (Coefficient of Determination)
  • SSIM (Structural Similarity Index, 11×11 Gaussian window)
  • Slope RMSE (terrain gradient error, degrees)

Outputs:
  results/{city}/
    ├── metrics.json
    ├── error_map.tif        (spatial error raster)
    ├── comparison.png       (4-panel: FABDEM | Predicted | GT | Error)
    ├── scatter.png           (pred vs GT scatter)
    └── histogram.png         (error histogram)

Usage:
  python -m src.evaluate --city New_Orleans \
      --pred results/New_Orleans_Predicted_DEM_10m.tif
  python -m src.evaluate --pred pred.tif --gt gt.tif --fabdem fabdem.tif
=============================================================================
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.config import (
    RAW_DIR, ELEV_MIN, ELEV_MAX,
    IDX_FABDEM,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Metric Functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(pred, gt, mask=None):
    """
    Compute MAE, RMSE, R², and SSIM between predicted and ground-truth DEMs.

    Parameters
    ----------
    pred, gt : 2D arrays (H, W) — elevation in metres
    mask     : optional bool mask (True = valid pixels)

    Returns
    -------
    dict : metric name → value
    """
    if mask is None:
        mask = np.isfinite(pred) & np.isfinite(gt)

    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)

    if len(p) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan,
                "ssim": np.nan, "n_pixels": 0}

    error = p - g
    mae  = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((g - np.mean(g)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    ssim = _compute_ssim_windowed(pred, gt, mask)

    # ── Slope RMSE (degrees) ──
    slope_rmse = _compute_slope_rmse(pred, gt, mask)

    return {
        "mae":        float(mae),
        "rmse":       float(rmse),
        "r2":         float(r2),
        "ssim":       float(ssim),
        "slope_rmse": float(slope_rmse),
        "bias":       float(np.mean(error)),
        "n_pixels":   int(np.sum(mask)),
        "pred_min":   float(np.min(p)),
        "pred_max":   float(np.max(p)),
        "gt_min":     float(np.min(g)),
        "gt_max":     float(np.max(g)),
    }


def _gaussian_kernel(size=11, sigma=1.5):
    """Create a 2D Gaussian kernel for windowed SSIM."""
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2
    g = np.exp(-0.5 * (coords / sigma) ** 2)
    kernel = np.outer(g, g)
    return kernel / kernel.sum()


def _compute_ssim_windowed(pred, gt, mask, win_size=11, sigma=1.5):
    """
    Compute windowed SSIM using an 11×11 Gaussian kernel.
    This is closer to the standard Wang et al. 2004 formulation
    than a single global SSIM value.
    """
    from scipy.ndimage import uniform_filter, convolve

    p = pred.copy().astype(np.float64)
    g = gt.copy().astype(np.float64)

    # Normalise to [0, 1] using combined range
    vmin = min(np.nanmin(p[mask]), np.nanmin(g[mask]))
    vmax = max(np.nanmax(p[mask]), np.nanmax(g[mask]))
    rng = vmax - vmin if vmax != vmin else 1.0
    p = (p - vmin) / rng
    g = (g - vmin) / rng

    # Replace invalid pixels with local mean (to avoid edge effects)
    p[~mask] = 0.0
    g[~mask] = 0.0
    w = mask.astype(np.float64)

    kernel = _gaussian_kernel(win_size, sigma)

    # Weighted local statistics
    w_sum   = convolve(w, kernel, mode='constant', cval=0.0)
    w_sum   = np.clip(w_sum, 1e-10, None)

    mu_p    = convolve(p * w, kernel, mode='constant', cval=0.0) / w_sum
    mu_g    = convolve(g * w, kernel, mode='constant', cval=0.0) / w_sum
    mu_pp   = convolve(p * p * w, kernel, mode='constant', cval=0.0) / w_sum
    mu_gg   = convolve(g * g * w, kernel, mode='constant', cval=0.0) / w_sum
    mu_pg   = convolve(p * g * w, kernel, mode='constant', cval=0.0) / w_sum

    sig_pp  = mu_pp - mu_p ** 2
    sig_gg  = mu_gg - mu_g ** 2
    sig_pg  = mu_pg - mu_p * mu_g

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_p * mu_g + C1) * (2 * sig_pg + C2)) / \
               ((mu_p**2 + mu_g**2 + C1) * (sig_pp + sig_gg + C2))

    # Average SSIM only over valid pixels
    return float(np.nanmean(ssim_map[mask]))


def _compute_slope_rmse(pred, gt, mask, cell_size=10.0):
    """
    Compute RMSE of slope (degrees) between predicted and GT DEMs.
    Uses numpy gradient with cell_size in metres.
    """
    def _slope_deg(dem, cs):
        dy, dx = np.gradient(dem, cs)
        return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    slope_p = _slope_deg(np.nan_to_num(pred, nan=0.0), cell_size)
    slope_g = _slope_deg(np.nan_to_num(gt, nan=0.0), cell_size)

    diff = slope_p[mask] - slope_g[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


# ═══════════════════════════════════════════════════════════════════════════
#  Data Loading Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_dem(path):
    """Load a single-band DEM GeoTIFF. Returns (data, profile)."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nd = src.nodata
    if nd is not None:
        data[data == nd] = np.nan
    return data, profile


def load_gt_and_align(gt_path, ref_profile):
    """
    Load 1m ground truth, average pool to 10m, reproject to match
    the reference grid. Returns elevation in metres.
    """
    with rasterio.open(gt_path) as src:
        raw = src.read(1).astype(np.float32)
        gt_crs = src.crs
        gt_transform = src.transform
        native_res = abs(src.res[0])
        nodata = src.nodata

    if nodata is not None:
        raw[raw == nodata] = np.nan
    # NOTE: we do NOT treat raw == 0 as nodata — zero is a valid
    # elevation (sea level). Nodata comes from file metadata only.
    # Data is already in metres (float32) — no /100 conversion needed.

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

    # Build pooled transform
    pooled_h, pooled_w = pooled.shape
    pooled_transform = rasterio.transform.from_bounds(
        *rasterio.transform.array_bounds(
            raw.shape[0] if factor <= 1 else h_trim,
            raw.shape[1] if factor <= 1 else w_trim,
            gt_transform,
        ), pooled_w, pooled_h,
    )

    # Reproject to match reference grid
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
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )

    return gt_aligned


def load_fabdem_from_features(features_path):
    """Extract the FABDEM band from the Features_10m.tif (already in metres)."""
    with rasterio.open(features_path) as src:
        fabdem = src.read(IDX_FABDEM + 1).astype(np.float32)  # 1-indexed bands
        nd = src.nodata
    if nd is not None:
        fabdem[fabdem == nd] = np.nan
    # Data is already in metres (float32) — no /100 conversion needed.
    return fabdem


# ═══════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════

def _setup_style():
    """Dark theme for plots."""
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor":   "#16213e",
        "text.color":       "#e0e0e0",
        "axes.labelcolor":  "#e0e0e0",
        "xtick.color":      "#b0b0b0",
        "ytick.color":      "#b0b0b0",
        "axes.edgecolor":   "#3a3a5a",
        "grid.color":       "#2a2a4a",
        "font.family":      "sans-serif",
        "font.size":        10,
    })


def plot_comparison(fabdem, pred, gt, error, metrics, output_path, city=""):
    """
    4-panel comparison figure with stats legend:
    [FABDEM] [Predicted] [GT] [Error Map]
    Each panel shows min/max/mean/std stats.
    """
    _setup_style()

    mask = np.isfinite(gt) & np.isfinite(pred)
    vmin = np.nanpercentile(gt[mask], 1) if mask.any() else 0
    vmax = np.nanpercentile(gt[mask], 99) if mask.any() else 100

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f"PI-SwinIR DEM Refinement — {city}", fontsize=14,
                 fontweight="bold", color="white", y=1.02)

    titles = ["FABDEM (Input)", "Predicted DEM", "Ground Truth (LiDAR)", "Error (Pred − GT)"]
    data   = [fabdem, pred, gt, error]
    cmaps  = ["terrain", "terrain", "terrain", "RdBu_r"]
    cbar_labels = ["Elevation (m)", "Elevation (m)", "Elevation (m)", "Error (m)"]

    for i, (ax, d, t, cm, cbl) in enumerate(zip(axes, data, titles, cmaps, cbar_labels)):
        if i == 3:  # error map
            err_abs = np.nanmax(np.abs(error[mask])) if mask.any() else 5
            err_lim = min(err_abs, 10.0)
            im = ax.imshow(d, cmap=cm, vmin=-err_lim, vmax=err_lim)
        else:
            im = ax.imshow(d, cmap=cm, vmin=vmin, vmax=vmax)

        ax.set_title(t, fontsize=11, color="white", pad=8)
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(cbl, fontsize=8, color="#b0b0b0")

        # ── Per-panel stats annotation ──
        valid = d[np.isfinite(d)] if d is not None else np.array([])
        if len(valid) > 0:
            stats_text = (
                f"min={valid.min():.1f}\n"
                f"max={valid.max():.1f}\n"
                f"μ={valid.mean():.1f}\n"
                f"σ={valid.std():.1f}"
            )
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                    fontsize=7, color="white", va="bottom", ha="left",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                              alpha=0.6, edgecolor="none"))

    # Metrics text
    metrics_text = (
        f"MAE={metrics['mae']:.2f}m  "
        f"RMSE={metrics['rmse']:.2f}m  "
        f"R²={metrics['r2']:.4f}  "
        f"SSIM={metrics['ssim']:.4f}  "
        f"Slope RMSE={metrics.get('slope_rmse', 0):.2f}°"
    )
    fig.text(0.5, -0.02, metrics_text, ha="center", fontsize=11,
             color="#66ccff", fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {output_path}")


def plot_scatter(pred, gt, mask, output_path, city=""):
    """Scatter plot of predicted vs GT elevations."""
    _setup_style()

    p = pred[mask]
    g = gt[mask]

    # Subsample if too many points
    n = len(p)
    if n > 50000:
        idx = np.random.choice(n, 50000, replace=False)
        p, g = p[idx], g[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(g, p, s=0.5, alpha=0.3, c="#4ecdc4", edgecolors="none")

    lims = [min(g.min(), p.min()), max(g.max(), p.max())]
    ax.plot(lims, lims, "--", color="#ff6b6b", lw=1.5, label="1:1 line")

    ax.set_xlabel("Ground Truth (m)", fontsize=11)
    ax.set_ylabel("Predicted (m)", fontsize=11)
    ax.set_title(f"Predicted vs GT — {city}", fontsize=12,
                 fontweight="bold", color="white")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {output_path}")


def plot_error_histogram(error, mask, output_path, city=""):
    """Histogram of pixel-wise errors."""
    _setup_style()

    e = error[mask]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(e, bins=200, color="#4ecdc4", alpha=0.8, edgecolor="none",
            density=True)
    ax.axvline(0, color="#ff6b6b", lw=1.5, ls="--", label="Zero")
    ax.axvline(np.mean(e), color="#ffd93d", lw=1.5, ls="-",
               label=f"Mean={np.mean(e):.2f}m")

    ax.set_xlabel("Error (Predicted − GT) [m]", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Error Distribution — {city}", fontsize=12,
                 fontweight="bold", color="white")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {output_path}")


def plot_elevation_histogram(gt, fabdem, pred, mask, output_path, city=""):
    """
    Combined elevation histogram: GT vs FABDEM vs Predicted.
    Publication-quality overlay to show model improvement over FABDEM.
    """
    _setup_style()

    g = gt[mask]
    f = fabdem[mask]
    p = pred[mask]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bins = np.linspace(
        min(g.min(), f.min(), p.min()),
        max(g.max(), f.max(), p.max()),
        150,
    )

    ax.hist(g, bins=bins, color="#2ecc71", alpha=0.5, edgecolor="none",
            density=True, label=f"Ground Truth (μ={g.mean():.1f}m)")
    ax.hist(f, bins=bins, color="#3498db", alpha=0.4, edgecolor="none",
            density=True, label=f"FABDEM (μ={f.mean():.1f}m)")
    ax.hist(p, bins=bins, color="#e74c3c", alpha=0.45, edgecolor="none",
            density=True, label=f"Predicted (μ={p.mean():.1f}m)")

    ax.set_xlabel("Elevation (m)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Elevation Distribution — {city}", fontsize=12,
                 fontweight="bold", color="white")
    ax.legend(fontsize=9, frameon=True, fancybox=False,
              edgecolor="#444", facecolor="#1a1a2e")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Error Map GeoTIFF
# ═══════════════════════════════════════════════════════════════════════════

def save_error_map(error, ref_profile, output_path):
    """Save spatial error as a GeoTIFF."""
    profile = ref_profile.copy()
    profile.update(dtype="float32", count=1, compress="deflate",
                   nodata=np.nan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(error.astype(np.float32), 1)
    print(f"  ✓ Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_city(pred_path, gt_path, features_path, out_dir, city=""):
    """
    Full evaluation pipeline for one city.

    Parameters
    ----------
    pred_path     : predicted DEM GeoTIFF (metres)
    gt_path       : ground-truth 1m DEM (Int16 ×100)
    features_path : Features_10m.tif (for FABDEM baseline)
    out_dir       : output directory for results
    city          : name string for plot titles
    """
    print(f"\n{'='*65}")
    print(f"  Evaluating: {city or pred_path}")
    print(f"{'='*65}")

    out_dir = Path(out_dir)

    # ── Load data ──
    print("  Loading predicted DEM ...")
    pred, pred_profile = load_dem(pred_path)

    print("  Loading & aligning GT ...")
    gt = load_gt_and_align(gt_path, pred_profile)

    print("  Loading FABDEM baseline ...")
    fabdem = load_fabdem_from_features(features_path)

    # ── Valid mask (exclude CRS tilt NoData corners) ──
    nodata_corners = (pred == 0) | ~np.isfinite(pred)
    pred[nodata_corners] = np.nan
    fabdem[nodata_corners] = np.nan

    mask = np.isfinite(pred) & np.isfinite(gt) & np.isfinite(fabdem)
    print(f"  Valid pixels: {mask.sum():,} / {mask.size:,} "
          f"({100*mask.sum()/mask.size:.1f}%)")
    print(f"  NoData corners excluded: {nodata_corners.sum():,} px")

    # ── Compute metrics: Model vs GT ──
    print("\n  Metrics (PI-SwinIR):")
    m_model = compute_metrics(pred, gt, mask)
    for k, v in m_model.items():
        if isinstance(v, float):
            print(f"    {k:12s}: {v:>10.4f}")
        else:
            print(f"    {k:12s}: {v}")

    # Baseline: FABDEM vs GT
    print("\n  Metrics (FABDEM baseline):")
    m_fabdem = compute_metrics(fabdem, gt, mask)
    for k, v in m_fabdem.items():
        if isinstance(v, float):
            print(f"    {k:12s}: {v:>10.4f}")
        else:
            print(f"    {k:12s}: {v}")

    # ── Improvement ──
    if m_fabdem["rmse"] > 0:
        improve = (1 - m_model["rmse"] / m_fabdem["rmse"]) * 100
        print(f"\n  ➤ RMSE improvement over FABDEM: {improve:+.1f}%")

    # ── Error map ──
    error = np.full_like(pred, np.nan)
    error[mask] = pred[mask] - gt[mask]

    # ── Save outputs ──
    save_error_map(error, pred_profile, out_dir / "error_map.tif")

    plot_comparison(fabdem, pred, gt, error, m_model,
                    out_dir / "comparison.png", city)
    plot_scatter(pred, gt, mask, out_dir / "scatter.png", city)
    plot_error_histogram(error, mask, out_dir / "histogram.png", city)
    plot_elevation_histogram(gt, fabdem, pred, mask,
                             out_dir / "elevation_histogram.png", city)

    # Save metrics as JSON
    results = {
        "city": city,
        "model_metrics": m_model,
        "fabdem_metrics": m_fabdem,
        "rmse_improvement_pct": improve if m_fabdem["rmse"] > 0 else None,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved: {metrics_path}")

    print(f"\n{'='*65}")
    print(f"  Evaluation complete → {out_dir}")
    print(f"{'='*65}\n")

    return results


def main():
    from src.cli import configure_console
    configure_console()
    parser = argparse.ArgumentParser(description="Evaluate PI-SwinIR predictions")
    parser.add_argument("--city", type=str, default=None,
                        help="City name (uses default paths)")
    parser.add_argument("--pred", type=str, required=True,
                        help="Predicted DEM GeoTIFF (metres)")
    parser.add_argument("--gt", type=str, default=None,
                        help="Ground truth GeoTIFF in metres (native 1m, 5m or 10m)")
    parser.add_argument("--features", type=str, default=None,
                        help="Features_10m.tif (for FABDEM baseline)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    if args.city:
        city = args.city
        gt_candidates = sorted(RAW_DIR.glob(f"{city}_GroundTruth_*.tif"))
        gt = args.gt or (str(gt_candidates[0]) if gt_candidates else None)
        features = args.features or str(RAW_DIR / f"{city}_Features_10m.tif")
        out_dir = Path(args.output_dir) if args.output_dir else Path("results") / city
    else:
        city = ""
        gt = args.gt
        features = args.features
        out_dir = Path(args.output_dir or "results")

    if not gt:
        parser.error("Provide --gt or a city with an available ground-truth raster")
    evaluate_city(args.pred, gt, features, out_dir, city)


if __name__ == "__main__":
    main()
