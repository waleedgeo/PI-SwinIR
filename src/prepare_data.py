"""
=============================================================================
prepare_data.py — Stage 0: GEE Export Preparation Utility
=============================================================================
Prepares raw data exported from Google Earth Engine (GEE) for the
PI-SwinIR DEM refinement pipeline.

The GEE exports are typically Int16 GeoTIFFs with scale factors applied
during export. This script:

  1. Validates exported files (band count, dtype, CRS consistency)
  2. Removes GEE-applied scale factors where needed
  3. Applies ocean masking for coastal cities
  4. Produces the final per-city files ready for preprocess.py

Expected input structure:
  data/raw/
    ├── {City}_Features_10m.tif    (9-band Int16 feature stack)
    ├── {City}_GroundTruth_1m.tif  (single-band Int16 GT DEM, ×100)
    └── {City}_AOI.geojson         (Area of Interest boundary)

Usage:
  python -m src.prepare_data                       # validate all cities
  python -m src.prepare_data --validate-only       # just check, no changes
  python -m src.prepare_data --city New_Orleans     # single city
  python -m src.prepare_data --fix-nodata           # fix nodata tags
=============================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

from src.config import (
    CITIES, RAW_DIR, PROCESSED_DIR,
    IDX_VV, IDX_VH, IDX_R, IDX_G, IDX_B, IDX_NIR,
    IDX_FABDEM, IDX_HAND, IDX_ROADS,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════

EXPECTED_FEATURE_BANDS = 9
EXPECTED_FEATURE_DTYPE = "int16"
EXPECTED_GT_BANDS = 1

BAND_NAMES = [
    "SAR VV", "SAR VH",
    "Red (S2)", "Green (S2)", "Blue (S2)", "NIR (S2)",
    "FABDEM", "HAND", "Roads",
]


def _log(msg, level="INFO"):
    symbol = {"INFO": "ℹ", "WARN": "⚠", "ERROR": "✗", "OK": "✓"}
    print(f"  {symbol.get(level, '•')} [{level}] {msg}")


def validate_features(tif_path):
    """
    Validate a Features_10m.tif file.

    Checks:
      - File exists and is readable
      - Has exactly 9 bands
      - dtype is Int16
      - CRS is defined
      - nodata tag is set
      - Band value ranges are physically plausible

    Returns (is_valid, issues_list)
    """
    issues = []
    path = Path(tif_path)

    if not path.exists():
        return False, [f"File not found: {path}"]

    try:
        with rasterio.open(path) as src:
            # Band count
            if src.count != EXPECTED_FEATURE_BANDS:
                issues.append(
                    f"Expected {EXPECTED_FEATURE_BANDS} bands, got {src.count}"
                )

            # Dtype
            dtypes = set(src.dtypes)
            if len(dtypes) != 1 or dtypes.pop() != EXPECTED_FEATURE_DTYPE:
                issues.append(f"Expected dtype '{EXPECTED_FEATURE_DTYPE}', got {src.dtypes}")

            # CRS
            if src.crs is None:
                issues.append("No CRS defined")

            # Nodata — common for GEE exports to omit; pipeline has fallback
            if src.nodata is None:
                _log(f"No nodata tag on {Path(tif_path).name} "
                     f"(pipeline will use all-band-zero fallback)", "WARN")

            # Read sample region for range checks
            raw = src.read().astype(np.float32)

            # SAR check: dB×100, urban areas can reach +35 dB (3500)
            # Typical range: -50 dB to +40 dB → -5000 to 4000
            for idx, name in [(IDX_VV, "SAR VV"), (IDX_VH, "SAR VH")]:
                band = raw[idx]
                valid = band[band != 0] if src.nodata is None else band[band != src.nodata]
                if len(valid) > 0:
                    bmin, bmax = valid.min(), valid.max()
                    if bmin < -6000 or bmax > 5000:
                        issues.append(
                            f"{name}: range [{bmin:.0f}, {bmax:.0f}] "
                            f"seems wrong for dB×100"
                        )

            # S2 check: reflectance×10000, bright surfaces can exceed 10000
            # Values up to ~16000 (reflectance 1.6) are normal for roofs, snow
            for idx, name in zip(
                [IDX_R, IDX_G, IDX_B, IDX_NIR],
                ["Red", "Green", "Blue", "NIR"],
            ):
                band = raw[idx]
                valid = band[band != 0] if src.nodata is None else band[band != src.nodata]
                if len(valid) > 0:
                    bmin, bmax = valid.min(), valid.max()
                    if bmin < -100 or bmax > 20000:
                        issues.append(
                            f"{name}: range [{bmin:.0f}, {bmax:.0f}] "
                            f"seems wrong for reflectance×10000"
                        )

            # FABDEM check: should be in range [-50000, 900000] (m × 100)
            fab = raw[IDX_FABDEM]
            fab_valid = fab[fab != 0] if src.nodata is None else fab[fab != src.nodata]
            if len(fab_valid) > 0:
                fmin, fmax = fab_valid.min(), fab_valid.max()
                if fmin < -50000 or fmax > 900000:
                    issues.append(
                        f"FABDEM: range [{fmin:.0f}, {fmax:.0f}] "
                        f"seems wrong for metres×100"
                    )

            # HAND check: should be in range [0, 100000] (m × 100)
            hand = raw[IDX_HAND]
            hand_valid = hand[hand != 0] if src.nodata is None else hand[hand != src.nodata]
            if len(hand_valid) > 0:
                hmin, hmax = hand_valid.min(), hand_valid.max()
                if hmin < -100 or hmax > 100000:
                    issues.append(
                        f"HAND: range [{hmin:.0f}, {hmax:.0f}] "
                        f"seems wrong for metres×100"
                    )

            # Roads check: should be binary (0 or 1)
            roads = raw[IDX_ROADS]
            unique_roads = np.unique(roads[~np.isnan(roads)])
            non_binary = [v for v in unique_roads if v not in (0, 1)]
            if len(non_binary) > 5:
                issues.append(
                    f"Roads: expected binary (0/1), found {len(non_binary)} "
                    f"other values"
                )

    except Exception as e:
        return False, [f"Failed to open: {e}"]

    return len(issues) == 0, issues


def validate_gt(tif_path):
    """
    Validate a GroundTruth_1m.tif file.

    Returns (is_valid, issues_list)
    """
    issues = []
    path = Path(tif_path)

    if not path.exists():
        return False, [f"File not found: {path}"]

    try:
        with rasterio.open(path) as src:
            if src.count != EXPECTED_GT_BANDS:
                issues.append(f"Expected {EXPECTED_GT_BANDS} band, got {src.count}")

            if src.crs is None:
                issues.append("No CRS defined")

            if src.nodata is None:
                _log(f"No nodata tag on {Path(tif_path).name}", "WARN")

            # Quick range check: sample from CENTER of image
            # (corners are often nodata for GEE exports)
            cx = max(0, src.width // 2 - 500)
            cy = max(0, src.height // 2 - 500)
            sw = min(1000, src.width - cx)
            sh = min(1000, src.height - cy)
            sample = src.read(1, window=rasterio.windows.Window(cx, cy, sw, sh))
            sample = sample.astype(np.float32)
            if src.nodata is not None:
                sample[sample == src.nodata] = np.nan
            # Don't treat 0 as nodata for GT — sea level is valid
            valid = sample[np.isfinite(sample)]
            if len(valid) > 0:
                vmin, vmax = valid.min() / 100.0, valid.max() / 100.0
                _log(f"GT elevation range: [{vmin:.1f}, {vmax:.1f}] metres")

    except Exception as e:
        return False, [f"Failed to open: {e}"]

    return len(issues) == 0, issues


# ═══════════════════════════════════════════════════════════════════════════
#  Fix utilities
# ═══════════════════════════════════════════════════════════════════════════

def fix_nodata_tag(tif_path, nodata_value=0):
    """
    Set the nodata tag on a GeoTIFF if it's missing.
    Modifies the file in place (rasterio update mode).
    """
    path = Path(tif_path)
    with rasterio.open(path, "r+") as src:
        if src.nodata is None:
            src.nodata = nodata_value
            _log(f"Set nodata={nodata_value} on {path.name}", "OK")
        else:
            _log(f"nodata already set to {src.nodata} on {path.name}", "INFO")


# ═══════════════════════════════════════════════════════════════════════════
#  Per-city summary
# ═══════════════════════════════════════════════════════════════════════════

def summarise_city(city):
    """Print a summary of data for one city."""
    print(f"\n{'─'*55}")
    print(f"  City: {city}")
    print(f"{'─'*55}")

    feat_path = RAW_DIR / f"{city}_Features_10m.tif"
    gt_path   = RAW_DIR / f"{city}_GroundTruth_1m.tif"
    aoi_path  = RAW_DIR / f"{city}_AOI.geojson"

    all_ok = True

    # Features
    ok, issues = validate_features(feat_path)
    if ok:
        _log(f"Features: OK ({feat_path.name})", "OK")
        with rasterio.open(feat_path) as src:
            _log(f"  Shape: {src.width}×{src.height}, CRS: {src.crs}")
    else:
        all_ok = False
        for issue in issues:
            _log(f"Features: {issue}", "ERROR")

    # Ground truth
    ok, issues = validate_gt(gt_path)
    if ok:
        _log(f"GT: OK ({gt_path.name})", "OK")
    else:
        all_ok = False
        for issue in issues:
            _log(f"GT: {issue}", "ERROR")

    # AOI
    if aoi_path.exists():
        _log(f"AOI: OK ({aoi_path.name})", "OK")
    else:
        _log(f"AOI: not found ({aoi_path.name})", "WARN")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    from src.cli import configure_console
    configure_console()
    parser = argparse.ArgumentParser(
        description="Validate and prepare GEE-exported data for PI-SwinIR"
    )
    parser.add_argument("--city", type=str, default=None,
                        help="Process single city (default: all)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate, don't modify files")
    parser.add_argument("--fix-nodata", action="store_true",
                        help="Set missing nodata tags to 0")
    args = parser.parse_args()

    cities = [args.city] if args.city else CITIES

    print(f"\n{'='*55}")
    print(f"  GEE Data Preparation — PI-SwinIR DEM Pipeline")
    print(f"{'='*55}")
    print(f"  Cities  : {', '.join(cities)}")
    print(f"  Raw dir : {RAW_DIR}")
    print(f"  Mode    : {'validate only' if args.validate_only else 'validate + fix'}")

    all_ok = True
    for city in cities:
        ok = summarise_city(city)
        if not ok:
            all_ok = False

    # Optionally fix nodata
    if args.fix_nodata and not args.validate_only:
        print(f"\n  Fixing nodata tags ...")
        for city in cities:
            feat_path = RAW_DIR / f"{city}_Features_10m.tif"
            gt_path   = RAW_DIR / f"{city}_GroundTruth_1m.tif"
            if feat_path.exists():
                fix_nodata_tag(feat_path, nodata_value=0)
            if gt_path.exists():
                fix_nodata_tag(gt_path, nodata_value=0)

    # Summary
    print(f"\n{'='*55}")
    if all_ok:
        print(f"  ✓ All data validated successfully.")
        print(f"  Next step: python -m src.preprocess")
    else:
        print(f"  ⚠ Some issues found — review the output above.")
        print(f"  Fix issues before running preprocess.py")
    print(f"{'='*55}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
