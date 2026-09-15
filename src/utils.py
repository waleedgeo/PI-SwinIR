"""
=============================================================================
utils.py — Inference & GeoTIFF Inspection Utilities
=============================================================================
Quick tools for:
  1. inspect_geotiff()    — Print stats, range, CRS, resolution of any GeoTIFF
  2. compare_geotiffs()   — Side-by-side comparison of two GeoTIFFs
  3. quick_inference()    — Run inference on a city with auto-resolved paths
  4. compare_with_gt()    — Compare inference output against ground truth

Usage (CLI):
  python -m src.utils inspect data/raw/New_Orleans_Features_10m.tif
  python -m src.utils inspect results/New_Orleans_Predicted_DEM_10m.tif
  python -m src.utils compare results/New_Orleans_Predicted_DEM_10m.tif data/raw/New_Orleans_GroundTruth_1m.tif
  python -m src.utils infer --city New_Orleans --profile l4 --checkpoint checkpoints/best_l4.pt

Usage (Python):
  from src.utils import inspect_geotiff, quick_inference
  inspect_geotiff("results/New_Orleans_Predicted_DEM_10m.tif")
=============================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
#  1. GeoTIFF Inspector
# ═══════════════════════════════════════════════════════════════════════════

def inspect_geotiff(path, band=None, percentiles=(1, 5, 25, 50, 75, 95, 99)):
    """
    Print comprehensive statistics for a GeoTIFF file.

    Parameters
    ----------
    path : str or Path
        Path to the GeoTIFF file.
    band : int or None
        Specific band to inspect (1-indexed). If None, all bands.
    percentiles : tuple
        Percentiles to compute for each band.
    """
    import rasterio

    path = Path(path)
    if not path.exists():
        print(f"  ✗ File not found: {path}")
        return

    with rasterio.open(path) as src:
        print(f"\n{'='*65}")
        print(f"  GeoTIFF Inspector: {path.name}")
        print(f"{'='*65}")
        print(f"  Path       : {path}")
        print(f"  Dimensions : {src.width} × {src.height} px")
        print(f"  Bands      : {src.count}")
        print(f"  Dtype      : {src.dtypes[0]}")
        print(f"  CRS        : {src.crs}")
        print(f"  Resolution : {abs(src.res[0]):.2f} × {abs(src.res[1]):.2f} m")
        print(f"  Bounds     : {src.bounds}")
        print(f"  Nodata     : {src.nodata}")
        print(f"  Compress   : {src.profile.get('compress', 'none')}")

        file_size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  File size  : {file_size_mb:.1f} MB")

        # Determine which bands to read
        bands = [band] if band else list(range(1, src.count + 1))

        print(f"\n  {'Band':>4s}  {'Min':>12s}  {'Max':>12s}  {'Mean':>12s}  "
              f"{'Std':>10s}  {'NaN%':>6s}  {'Zero%':>6s}")
        print(f"  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*12}  "
              f"{'─'*10}  {'─'*6}  {'─'*6}")

        all_stats = []
        for b in bands:
            data = src.read(b).astype(np.float32)

            # Handle nodata
            if src.nodata is not None:
                data[data == src.nodata] = np.nan

            total_px = data.size
            nan_count = int(np.isnan(data).sum())
            nan_pct = nan_count / total_px * 100
            valid = data[~np.isnan(data)]

            if len(valid) == 0:
                print(f"  {b:4d}  {'all NaN':>12s}")
                continue

            zero_pct = (valid == 0).sum() / total_px * 100
            stats = {
                "band": b,
                "min": float(valid.min()),
                "max": float(valid.max()),
                "mean": float(valid.mean()),
                "std": float(valid.std()),
                "nan_pct": nan_pct,
                "zero_pct": zero_pct,
            }

            print(f"  {b:4d}  {stats['min']:12.4f}  {stats['max']:12.4f}  "
                  f"{stats['mean']:12.4f}  {stats['std']:10.4f}  "
                  f"{nan_pct:5.1f}%  {zero_pct:5.1f}%")

            all_stats.append(stats)

        # Percentile table for single-band or specified band
        if len(bands) <= 3:
            print(f"\n  Percentiles:")
            print(f"  {'Band':>4s}", end="")
            for p in percentiles:
                print(f"  {'P'+str(p):>8s}", end="")
            print()
            print(f"  {'─'*4}", end="")
            for _ in percentiles:
                print(f"  {'─'*8}", end="")
            print()

            for b in bands:
                data = src.read(b).astype(np.float32)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                valid = data[~np.isnan(data)]
                if len(valid) == 0:
                    continue
                pcts = np.percentile(valid, percentiles)
                print(f"  {b:4d}", end="")
                for v in pcts:
                    print(f"  {v:8.3f}", end="")
                print()

        print(f"{'='*65}\n")
        return all_stats


def compare_geotiffs(path_a, path_b, label_a="A", label_b="B"):
    """
    Side-by-side comparison of two single-band GeoTIFFs.
    Computes difference statistics (A - B).
    """
    import rasterio

    path_a, path_b = Path(path_a), Path(path_b)

    with rasterio.open(path_a) as src_a, rasterio.open(path_b) as src_b:
        print(f"\n{'='*65}")
        print(f"  GeoTIFF Comparison")
        print(f"{'='*65}")
        print(f"  {label_a}: {path_a.name}  ({src_a.width}×{src_a.height}, "
              f"CRS={src_a.crs})")
        print(f"  {label_b}: {path_b.name}  ({src_b.width}×{src_b.height}, "
              f"CRS={src_b.crs})")

        a = src_a.read(1).astype(np.float32)
        b = src_b.read(1).astype(np.float32)

        if src_a.nodata is not None:
            a[a == src_a.nodata] = np.nan
        if src_b.nodata is not None:
            b[b == src_b.nodata] = np.nan

    # Check compatible shapes
    min_h = min(a.shape[0], b.shape[0])
    min_w = min(a.shape[1], b.shape[1])
    if a.shape != b.shape:
        print(f"\n  ⚠ Shape mismatch: {a.shape} vs {b.shape}")
        print(f"    Comparing overlapping region: {min_h}×{min_w}")
        a = a[:min_h, :min_w]
        b = b[:min_h, :min_w]

    # Stats
    valid = ~(np.isnan(a) | np.isnan(b))
    av, bv = a[valid], b[valid]

    print(f"\n  {'Metric':<20s}  {label_a:>12s}  {label_b:>12s}")
    print(f"  {'─'*20}  {'─'*12}  {'─'*12}")
    print(f"  {'Min':<20s}  {av.min():12.4f}  {bv.min():12.4f}")
    print(f"  {'Max':<20s}  {av.max():12.4f}  {bv.max():12.4f}")
    print(f"  {'Mean':<20s}  {av.mean():12.4f}  {bv.mean():12.4f}")
    print(f"  {'Std':<20s}  {av.std():12.4f}  {bv.std():12.4f}")
    print(f"  {'Valid pixels':<20s}  {len(av):12,d}  {len(bv):12,d}")

    # Difference stats
    diff = av - bv
    print(f"\n  Difference ({label_a} − {label_b}):")
    print(f"    Min       : {diff.min():.4f}")
    print(f"    Max       : {diff.max():.4f}")
    print(f"    Mean      : {diff.mean():.4f}")
    print(f"    Std       : {diff.std():.4f}")
    print(f"    MAE       : {np.abs(diff).mean():.4f}")
    print(f"    RMSE      : {np.sqrt((diff**2).mean()):.4f}")

    # Percentiles of absolute difference
    abs_diff = np.abs(diff)
    pcts = [50, 90, 95, 99]
    pct_vals = np.percentile(abs_diff, pcts)
    print(f"\n  |Diff| percentiles:")
    for p, v in zip(pcts, pct_vals):
        print(f"    P{p:<3d}     : {v:.4f}")

    print(f"{'='*65}\n")

    return {"mae": float(np.abs(diff).mean()),
            "rmse": float(np.sqrt((diff**2).mean())),
            "mean_diff": float(diff.mean())}


# ═══════════════════════════════════════════════════════════════════════════
#  2. Quick Inference
# ═══════════════════════════════════════════════════════════════════════════

def quick_inference(city, profile="l4", checkpoint=None, device=None):
    """
    One-liner to run inference on a city.
    Auto-resolves feature path, checkpoint, and output path.

    Parameters
    ----------
    city : str
        City name, e.g. 'New_Orleans'
    profile : str
        Training profile name (l4, l4_overnight, gcp, etc.)
    checkpoint : str or None
        Path to checkpoint file. If None, auto-discovers best_{profile}.pt
    device : str or None
        'cuda', 'cpu', or None (auto-detect)

    Returns
    -------
    pred_metres : (H, W) float32 — predicted DEM in metres
    """
    import torch
    from src.config import RAW_DIR, CHECKPOINT_DIR, RESULTS_DIR
    from src.inference import run_inference

    features = RAW_DIR / f"{city}_Features_10m.tif"
    output = RESULTS_DIR / f"{city}_Predicted_DEM_10m.tif"

    if checkpoint is None:
        # Try profile-specific, then generic
        candidates = [
            CHECKPOINT_DIR / f"best_{profile}.pt",
            CHECKPOINT_DIR / "best.pt",
        ]
        for c in candidates:
            if c.exists():
                checkpoint = str(c)
                break
        if checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoint found. Looked for: {[str(c) for c in candidates]}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Quick inference: {city}")
    print(f"  Profile   : {profile}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Output    : {output}")

    pred = run_inference(str(features), checkpoint, str(output),
                         profile_name=profile, device=device)
    return pred


def compare_with_gt(city, pred_tif=None):
    """
    Compare a prediction GeoTIFF against the ground truth for a city.

    Parameters
    ----------
    city : str
        City name, e.g. 'New_Orleans'
    pred_tif : str or None
        Path to prediction. If None, uses results/{city}_Predicted_DEM_10m.tif
    """
    from src.config import RAW_DIR, RESULTS_DIR

    if pred_tif is None:
        pred_tif = RESULTS_DIR / f"{city}_Predicted_DEM_10m.tif"

    # Find GT file
    gt_candidates = list(RAW_DIR.glob(f"{city}_GroundTruth_*.tif"))
    if not gt_candidates:
        print(f"  ✗ No ground truth found for {city}")
        return

    gt_tif = gt_candidates[0]

    print(f"\n  Comparing prediction vs ground truth for {city}")
    print(f"  Prediction: {pred_tif}")
    print(f"  GT:         {gt_tif}")

    inspect_geotiff(pred_tif)
    result = compare_geotiffs(pred_tif, gt_tif,
                              label_a="Predicted", label_b="GT")
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  3. Data Upload to GCS
# ═══════════════════════════════════════════════════════════════════════════

def upload_processed_to_gcs(bucket=None):
    """Upload all processed .npy files to GCS bucket."""
    import subprocess
    from src.config import PROCESSED_DIR, GCS_BUCKET
    bucket = bucket or GCS_BUCKET
    if not bucket:
        raise ValueError("Set DEM_GCS_BUCKET to enable cloud uploads")

    if not PROCESSED_DIR.exists():
        print("  ✗ No processed data directory found")
        return

    npy_files = list(PROCESSED_DIR.glob("*.npy"))
    joblib_files = list(PROCESSED_DIR.glob("*.joblib"))
    all_files = npy_files + joblib_files

    print(f"\n  Uploading {len(all_files)} files to {bucket}/processed/ ...")

    for f in all_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        dst = f"{bucket}/processed/{f.name}"
        print(f"    ↑ {f.name} ({size_mb:.1f} MB) → {dst}")
        try:
            subprocess.run(
                ["gsutil", "-m", "cp", str(f), dst],
                capture_output=True, timeout=600, check=True
            )
            print(f"      ✓ Done")
        except Exception as e:
            print(f"      ✗ Failed: {e}")

    print(f"  ✓ Upload complete")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    from src.cli import configure_console
    configure_console()
    parser = argparse.ArgumentParser(
        description="DEM Super-Resolution Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.utils inspect data/raw/New_Orleans_Features_10m.tif
  python -m src.utils inspect results/New_Orleans_Predicted_DEM_10m.tif --band 1
  python -m src.utils compare results/pred.tif data/raw/gt.tif
  python -m src.utils infer --city New_Orleans --profile l4
  python -m src.utils upload-processed
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # ── inspect ──
    sp_inspect = subparsers.add_parser("inspect", help="Inspect a GeoTIFF")
    sp_inspect.add_argument("path", type=str, help="Path to GeoTIFF")
    sp_inspect.add_argument("--band", type=int, default=None,
                            help="Specific band (1-indexed)")

    # ── compare ──
    sp_compare = subparsers.add_parser("compare", help="Compare two GeoTIFFs")
    sp_compare.add_argument("path_a", type=str, help="First GeoTIFF")
    sp_compare.add_argument("path_b", type=str, help="Second GeoTIFF")
    sp_compare.add_argument("--label-a", type=str, default="A")
    sp_compare.add_argument("--label-b", type=str, default="B")

    # ── infer ──
    sp_infer = subparsers.add_parser("infer", help="Quick inference on a city")
    sp_infer.add_argument("--city", type=str, required=True)
    sp_infer.add_argument("--profile", type=str, default="l4")
    sp_infer.add_argument("--checkpoint", type=str, default=None)

    # ── compare-gt ──
    sp_gt = subparsers.add_parser("compare-gt",
                                   help="Compare prediction with ground truth")
    sp_gt.add_argument("--city", type=str, required=True)
    sp_gt.add_argument("--pred", type=str, default=None,
                       help="Prediction GeoTIFF path (auto-detected if omitted)")

    # ── upload-processed ──
    subparsers.add_parser("upload-processed",
                          help="Upload processed data to GCS")

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_geotiff(args.path, band=args.band)
    elif args.command == "compare":
        compare_geotiffs(args.path_a, args.path_b,
                         label_a=args.label_a, label_b=args.label_b)
    elif args.command == "infer":
        quick_inference(args.city, profile=args.profile,
                        checkpoint=args.checkpoint)
    elif args.command == "compare-gt":
        compare_with_gt(args.city, pred_tif=args.pred)
    elif args.command == "upload-processed":
        upload_processed_to_gcs()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
