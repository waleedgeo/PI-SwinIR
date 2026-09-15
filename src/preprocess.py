"""
=============================================================================
Stage 1 — Data Preprocessing & Tensor Engineering
=============================================================================
Pipeline  : DEM Super-Resolution (PI-SwinIR)
Purpose   : Convert prepared float32 GeoTIFFs into normalised, patched
            NumPy arrays ready for PyTorch training.

Expected data/raw/ layout:
  data/raw/
    ├── {City}_Features_10m.tif   — 9-band float32 feature stack at 10 m
    ├── {City}_GroundTruth_*.tif  — 1-band float32 DEM in metres at 1m/5m
    └── {City}_roi.geojson        — study-area boundary

Output (11-channel input + 1-channel target):
  data/processed/
    ├── {City}_X_patches.npy     (N, 11, 128, 128)
    ├── {City}_Y_patches.npy     (N,  1, 128, 128)
    ├── train_X.npy              combined (train cities only)
    ├── train_Y.npy              combined
    ├── scaler_X.joblib          per-channel normalisation params
    └── scaler_Y.joblib          elevation scaler params
=============================================================================
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.config import (
    CITIES, CHANNEL_NAMES, DEFAULT_HOLDOUT_CITY,
    PATCH_SIZE, STRIDE, NODATA_THRESH, TARGET_RES,
    IDX_VV, IDX_VH, IDX_R, IDX_G, IDX_B, IDX_NIR,
    IDX_FABDEM, IDX_HAND, IDX_ROADS,
    SAR_MIN, SAR_MAX, ELEV_MIN, ELEV_MAX, HAND_MIN, HAND_MAX,
    IN_CHANNELS, FABDEM_CHANNEL_IDX,
    RAW_DIR, PROCESSED_DIR,
)

SAR_BANDS = [IDX_VV, IDX_VH]
S2_BANDS  = [IDX_R, IDX_G, IDX_B, IDX_NIR]


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER — Min-Max scaler (channel-wise, fixed bounds)
# ═══════════════════════════════════════════════════════════════════════════

class FixedMinMaxScaler:
    """
    Channel-wise min-max scaler that uses fixed (pre-defined) bounds.
    Stores them so they can be serialised and reversed at inference time.
    """
    def __init__(self):
        self.mins = None     # (C,)
        self.maxs = None     # (C,)

    def set_bounds(self, mins, maxs):
        self.mins = np.asarray(mins, dtype=np.float32)
        self.maxs = np.asarray(maxs, dtype=np.float32)
        # safety: avoid zero-range
        self.maxs = np.where(self.maxs == self.mins, self.mins + 1.0, self.maxs)

    def transform(self, arr):
        """Scale (C, H, W) array channel-wise to [0, 1]."""
        C = arr.shape[0]
        mins = self.mins[:C].reshape(C, 1, 1)
        maxs = self.maxs[:C].reshape(C, 1, 1)
        scaled = (arr - mins) / (maxs - mins)
        scaled = np.clip(scaled, 0.0, 1.0)
        return np.nan_to_num(scaled, nan=0.0).astype(np.float32)

    def inverse_transform(self, arr):
        """Reverse the scaling."""
        C = arr.shape[0]
        mins = self.mins[:C].reshape(C, 1, 1)
        maxs = self.maxs[:C].reshape(C, 1, 1)
        return arr * (maxs - mins) + mins


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load & denormalise the 9-band feature stack
# ═══════════════════════════════════════════════════════════════════════════

def load_features(tif_path):
    """
    Load the 9-band feature stack (float32 in physical units) and build
    a nodata mask.

    Returns
    -------
    features : (9, H, W) float32  — in physical units
    profile  : rasterio profile
    nodata_mask : (H, W) bool — True where pixel is nodata
    """
    with rasterio.open(tif_path) as src:
        raw = src.read().astype(np.float32)   # (9, H, W)
        profile = src.profile.copy()
        file_nodata = src.nodata

    # Build nodata mask from file metadata or fallback to all-band-zero
    if file_nodata is not None:
        nodata_mask = (raw == file_nodata).all(axis=0)
    else:
        nodata_mask = (raw == 0).all(axis=0)   # all 9 bands == 0

    raw[:, nodata_mask] = np.nan

    # ── Data is already in physical units (float32) ──
    # SAR: dB,  S2: reflectance [0,1],  FABDEM/HAND: metres,  Roads: binary
    features = raw.copy()

    # Clip S2 reflectance to [0, 1] for safety
    for b in S2_BANDS:
        features[b] = np.clip(features[b], 0.0, 1.0)

    return features, profile, nodata_mask


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — Compute spectral indices from S2 bands
# ═══════════════════════════════════════════════════════════════════════════

def compute_spectral_indices(features):
    """
    Compute NDVI and NDWI from the denormalised S2 bands (already 0-1).

    Parameters
    ----------
    features : (9, H, W) float32

    Returns
    -------
    ndvi, ndwi : each (H, W) float32 in [-1, 1]
    """
    red   = features[IDX_R]
    green = features[IDX_G]
    nir   = features[IDX_NIR]

    eps = 1e-10

    ndvi = (nir - red)   / (nir + red   + eps)   # vegetation
    ndwi = (green - nir) / (green + nir + eps)   # water

    return ndvi, ndwi


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3 — Normalise & build the 11-channel feature tensor
# ═══════════════════════════════════════════════════════════════════════════

def normalise_and_stack(features, ndvi, ndwi):
    """
    Apply fixed min-max normalisation to each sensor group and stack
    into the final 11-channel tensor.

    Channel order:
      0  VV        (SAR, min-max → [0, 1])
      1  VH        (SAR, min-max → [0, 1])
      2  R         (S2, already 0-1)
      3  G         (S2, already 0-1)
      4  B         (S2, already 0-1)
      5  NIR       (S2, already 0-1)
      6  FABDEM    (elevation, min-max → [0, 1])
      7  HAND      (min-max → [0, 1])
      8  ROADS     (binary 0/1)
      9  NDVI      (index, ~ [-1, 1])
     10  NDWI      (index, ~ [-1, 1])

    Returns
    -------
    X : (11, H, W) float32
    scaler : FixedMinMaxScaler (fitted for the 11 channels)
    """
    _, H, W = features.shape
    X = np.empty((IN_CHANNELS, H, W), dtype=np.float32)

    # SAR → min-max
    for i, b in enumerate(SAR_BANDS):
        X[i] = np.clip((features[b] - SAR_MIN) / (SAR_MAX - SAR_MIN), 0, 1)

    # S2 → already 0-1
    for i, b in enumerate(S2_BANDS):
        X[2 + i] = features[b]

    # FABDEM → global elevation min-max
    X[6] = np.clip(
        (features[IDX_FABDEM] - ELEV_MIN) / (ELEV_MAX - ELEV_MIN), 0, 1
    )

    # HAND → min-max
    X[7] = np.clip(
        (features[IDX_HAND] - HAND_MIN) / (HAND_MAX - HAND_MIN), 0, 1
    )

    # ROADS → binary
    X[8] = features[IDX_ROADS]

    # Spectral indices → map [-1, 1] to [0, 1]
    X[9]  = np.clip((ndvi + 1.0) / 2.0, 0, 1)
    X[10] = np.clip((ndwi + 1.0) / 2.0, 0, 1)

    # Replace residual NaN with 0
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)

    # Build scaler object (stores bounds for inverse at inference)
    scaler = FixedMinMaxScaler()
    scaler.set_bounds(
        mins=np.array([SAR_MIN, SAR_MIN,  0, 0, 0, 0, ELEV_MIN, HAND_MIN, 0, -1, -1]),
        maxs=np.array([SAR_MAX, SAR_MAX,  1, 1, 1, 1, ELEV_MAX, HAND_MAX, 1,  1,  1]),
    )

    return X, scaler


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3b — Ocean mask (multi-source)
# ═══════════════════════════════════════════════════════════════════════════

def create_ocean_mask(features, ndwi, gt_aligned=None):
    """
    Create an ocean/water-body mask from multiple sources.

    Ocean pixels are:
      1. NDWI > 0.3  (spectral water detection)
      2. HAND < 0.5 m AND FABDEM < -5 m  (at drainage level, below sea level)
      3. GT DEM is all-NaN (ocean areas have no LiDAR coverage)

    Parameters
    ----------
    features  : (9, H, W) float32 — denormalised to physical units
    ndwi      : (H, W) float32    — water index
    gt_aligned: (1, H, W) or None — ground truth in metres (NaN = no data)

    Returns
    -------
    ocean_mask : (H, W) bool — True = ocean/water pixel
    """
    fabdem_m = features[IDX_FABDEM]
    hand_m   = features[IDX_HAND]

    # Spectral water detection
    water_spectral = ndwi > 0.3

    # Hydrological: low HAND + below sea level
    water_hydro = (hand_m < 0.5) & (fabdem_m < -5.0)

    # Combine
    ocean = water_spectral & water_hydro

    # If GT available: areas with no GT coverage in water regions
    if gt_aligned is not None:
        gt_nodata = np.isnan(gt_aligned[0])
        ocean = ocean | (water_spectral & gt_nodata)

    return ocean


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4 — Load & aggregate Ground-Truth DEM (1 m → 10 m)
# ═══════════════════════════════════════════════════════════════════════════

def load_ground_truth(gt_path, ref_profile, strip_rows=2048):
    """
    Load the 1 m ground-truth DEM, average-pool to 10 m, then reproject /
    align to the reference feature grid.

    **Memory-optimised**: reads the raster in horizontal strips instead of
    loading the entire array (~10 GB for 51 k × 51 k float32) at once.
    Peak GT memory ≈ strip_rows × width × 4 bytes (~0.4 GB).

    Parameters
    ----------
    gt_path     : path to the GT GeoTIFF (float32, metres)
    ref_profile : rasterio profile of the 10 m feature stack (target grid)
    strip_rows  : number of 1 m rows to read per iteration (default 2048)

    Returns
    -------
    gt_norm    : (1, H, W) float32 — normalised to [0, 1] via global min-max
    gt_aligned : (1, H, W) float32 — actual metres (for diagnostics)
    scaler     : FixedMinMaxScaler for elevation
    """
    with rasterio.open(gt_path) as src:
        gt_nodata    = src.nodata
        gt_crs       = src.crs
        gt_transform = src.transform
        native_res   = abs(src.res[0])
        full_h       = src.height
        full_w       = src.width

    factor = max(1, int(round(TARGET_RES / native_res)))

    if factor > 1:
        # Trim dimensions to exact multiples of the pooling factor
        h_trim = (full_h // factor) * factor
        w_trim = (full_w // factor) * factor

        pooled_h = h_trim // factor
        pooled_w = w_trim // factor

        # Pre-allocate the pooled output (10 m resolution) — small array
        pooled = np.empty((pooled_h, pooled_w), dtype=np.float32)

        # Make strip_rows a multiple of factor so each strip pools cleanly
        strip_rows = max(factor, (strip_rows // factor) * factor)
        pooled_strip_rows = strip_rows // factor

        with rasterio.open(gt_path) as src:
            for row_start in range(0, h_trim, strip_rows):
                row_end = min(row_start + strip_rows, h_trim)
                actual_rows = row_end - row_start

                # Read only this horizontal strip (stays as int16 on disk)
                window = rasterio.windows.Window(0, row_start, w_trim, actual_rows)
                strip = src.read(1, window=window).astype(np.float32)

                # Replace nodata
                if gt_nodata is not None:
                    strip[strip == gt_nodata] = np.nan

                # Data is already in metres (float32)
                # No division needed

                # Average-pool this strip
                n_pool_rows = actual_rows // factor
                blocks = strip[:n_pool_rows * factor, :].reshape(
                    n_pool_rows, factor, pooled_w, factor
                )
                with np.errstate(all="ignore"):
                    pool_out = np.nanmean(blocks, axis=(1, 3))

                # Write into the pre-allocated output
                out_row = row_start // factor
                pooled[out_row:out_row + n_pool_rows] = pool_out

                del strip, blocks, pool_out  # free immediately

        print(f"    Pooled {full_h}×{full_w} → {pooled_h}×{pooled_w}  "
              f"(factor={factor}, strip={strip_rows} rows)")
    else:
        # No pooling needed — still read in strips to limit memory
        pooled = np.empty((full_h, full_w), dtype=np.float32)
        h_trim, w_trim = full_h, full_w

        with rasterio.open(gt_path) as src:
            for row_start in range(0, full_h, strip_rows):
                row_end = min(row_start + strip_rows, full_h)
                window = rasterio.windows.Window(0, row_start, full_w, row_end - row_start)
                strip = src.read(1, window=window).astype(np.float32)
                if gt_nodata is not None:
                    strip[strip == gt_nodata] = np.nan
                # Data is already in metres (float32)
                pooled[row_start:row_end] = strip
                del strip

    # Build an intermediate profile for the pooled raster
    pooled_h, pooled_w = pooled.shape
    pooled_transform = rasterio.transform.from_bounds(
        *rasterio.transform.array_bounds(h_trim, w_trim, gt_transform),
        pooled_w, pooled_h,
    )

    # ── Reproject to match the feature grid exactly ──
    dst_crs       = ref_profile["crs"]
    dst_transform = ref_profile["transform"]
    dst_w         = ref_profile["width"]
    dst_h         = ref_profile["height"]

    gt_aligned = np.empty((1, dst_h, dst_w), dtype=np.float32)
    gt_aligned[:] = np.nan

    reproject(
        source=pooled,
        destination=gt_aligned[0],
        src_transform=pooled_transform,
        src_crs=gt_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )

    del pooled  # free the pooled array after reprojection

    # ── Global min-max normalisation (same range as FABDEM) ──
    gt_norm = np.clip(
        (gt_aligned - ELEV_MIN) / (ELEV_MAX - ELEV_MIN), 0, 1
    ).astype(np.float32)
    gt_norm = np.nan_to_num(gt_norm, nan=0.0)

    scaler = FixedMinMaxScaler()
    scaler.set_bounds(mins=np.array([ELEV_MIN]), maxs=np.array([ELEV_MAX]))

    return gt_norm, gt_aligned, scaler


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5 — Patch extraction with 50 % overlap + ocean filtering
# ═══════════════════════════════════════════════════════════════════════════

def extract_patches(X, Y, ocean_mask=None, patch_size=PATCH_SIZE,
                    stride=STRIDE, nodata_thresh=NODATA_THRESH):
    """
    Sliding-window extraction of co-registered patches.

    Parameters
    ----------
    X : (C, H, W) float32  — normalised input stack
    Y : (1, H, W) float32  — normalised ground-truth
    ocean_mask : (H, W) bool or None — True = ocean pixel to exclude
    patch_size : int
    stride     : int        — step size (64 = 50 % overlap)
    nodata_thresh : float   — max NaN fraction allowed

    Returns
    -------
    X_patches, Y_patches : lists of (C, ps, ps) / (1, ps, ps) arrays
    """
    _, H, W = X.shape
    X_patches, Y_patches = [], []

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            x_p = X[:, r:r + patch_size, c:c + patch_size]
            y_p = Y[:, r:r + patch_size, c:c + patch_size]

            total_px = patch_size * patch_size

            # ── Filter: >20 % NaN in any band of X or in Y ──
            nan_frac_x = np.isnan(x_p).any(axis=0).sum() / total_px
            nan_frac_y = np.isnan(y_p).sum() / total_px
            if nan_frac_x > nodata_thresh or nan_frac_y > nodata_thresh:
                continue

            # ── Filter: >20 % zero-only pixels (likely nodata masked) ──
            zero_frac = (x_p == 0).all(axis=0).sum() / total_px
            if zero_frac > nodata_thresh:
                continue

            # ── Filter: ocean pixels ──
            if ocean_mask is not None:
                ocean_patch = ocean_mask[r:r + patch_size, c:c + patch_size]
                ocean_frac = ocean_patch.sum() / total_px
                if ocean_frac > nodata_thresh:
                    continue

            # ── Filter: deep ocean via denormalized Y ──
            elev_orig = y_p[0] * (ELEV_MAX - ELEV_MIN) + ELEV_MIN
            deep_ocean_frac = np.nansum(elev_orig < -50) / total_px
            if deep_ocean_frac > nodata_thresh:
                continue

            X_patches.append(x_p.copy())
            Y_patches.append(y_p.copy())

    return X_patches, Y_patches


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def process_city(city_name, raw_dir, out_dir, dry_run=False):
    """
    Full pipeline for a single city: load → denorm → indices → normalise →
    aggregate GT → ocean mask → patch → save.

    **Memory-optimised**: frees intermediate arrays as soon as they are no
    longer needed, and returns only (n_patches, scaler_X, scaler_Y) to
    avoid keeping patch lists alive across cities.

    Returns (n_patches, scaler_X, scaler_Y).
    """
    import gc

    feat_path = raw_dir / f"{city_name}_Features_10m.tif"
    # Auto-discover GT file (supports 1m, 5m, etc.)
    gt_candidates = list(raw_dir.glob(f"{city_name}_GroundTruth_*.tif"))
    gt_path = gt_candidates[0] if gt_candidates else raw_dir / f"{city_name}_GroundTruth_1m.tif"

    # ── Skip if already processed ──
    x_out = out_dir / f"{city_name}_X_patches.npy"
    y_out = out_dir / f"{city_name}_Y_patches.npy"
    if not dry_run and x_out.exists() and y_out.exists():
        existing = np.load(x_out, mmap_mode='r')
        n = existing.shape[0]
        del existing
        print(f"  ✓ Already processed ({n} patches) — skipping.")
        return n, None, None

    # ── Validate files exist ──
    if not feat_path.exists():
        print(f"  ⚠ Features file not found: {feat_path} — skipping {city_name}")
        return 0, None, None
    if not gt_path.exists():
        print(f"  ⚠ GT file not found: {gt_path} — skipping {city_name}")
        return 0, None, None

    t0 = time.time()

    # 1. Load features
    print(f"  → Loading features: {feat_path.name}")
    features, feat_profile, nodata_mask = load_features(feat_path)
    print(f"    Raw shape: {features.shape}  |  CRS: {feat_profile['crs']}")

    # 2. Compute spectral indices (NDVI + NDWI only, no NDBI)
    print("  → Computing NDVI, NDWI ...")
    ndvi, ndwi = compute_spectral_indices(features)

    # 3. Normalise & stack to 11-channel tensor
    print(f"  → Normalising & stacking {IN_CHANNELS}-channel input ...")
    X, scaler_X = normalise_and_stack(features, ndvi, ndwi)
    print(f"    X shape: {X.shape}  |  dtype: {X.dtype}")

    # Print per-channel stats
    ch_names = [
        "VV", "VH", "R", "G", "B", "NIR",
        "FABDEM", "HAND", "ROADS", "NDVI", "NDWI",
    ]
    print("    Channel stats (min / mean / max):")
    for i, name in enumerate(ch_names):
        ch = X[i]
        valid = ch[~np.isnan(ch)] if np.isnan(ch).any() else ch
        if len(valid.flat) > 0:
            print(f"      {i:2d} {name:8s}: {valid.min():+8.4f} / "
                  f"{valid.mean():+8.4f} / {valid.max():+8.4f}")

    # 4. Load & aggregate GT
    print(f"  → Loading & pooling GT: {gt_path.name}")
    Y, Y_raw, scaler_Y = load_ground_truth(gt_path, feat_profile)
    print(f"    Y shape: {Y.shape}  |  dtype: {Y.dtype}")

    # 5. Create ocean mask
    print("  → Creating ocean mask ...")
    ocean_mask = create_ocean_mask(features, ndwi, gt_aligned=Y_raw)
    ocean_pct = ocean_mask.sum() / ocean_mask.size * 100
    print(f"    Ocean pixels: {ocean_mask.sum():,} / {ocean_mask.size:,} "
          f"({ocean_pct:.1f}%)")

    # Free large intermediates no longer needed
    del features, ndvi, ndwi, nodata_mask, Y_raw
    gc.collect()

    # 6. Patch extraction
    print(f"  → Extracting {PATCH_SIZE}×{PATCH_SIZE} patches "
          f"(stride={STRIDE}, drop >{NODATA_THRESH*100:.0f}% nodata) ...")
    X_patches, Y_patches = extract_patches(X, Y, ocean_mask=ocean_mask)
    n_patches = len(X_patches)
    print(f"    ✓ Valid patches: {n_patches}")

    # Free the full rasters now that patches are extracted
    del X, Y, ocean_mask
    gc.collect()

    elapsed = time.time() - t0
    print(f"    ⏱ Processing time: {elapsed:.1f}s")

    if dry_run:
        print("  [DRY RUN] — patches not saved.")
        del X_patches, Y_patches
        gc.collect()
        return n_patches, scaler_X, scaler_Y

    # 7. Save per-city patches
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_patches > 0:
        X_arr = np.stack(X_patches, axis=0)
        del X_patches
        np.save(out_dir / f"{city_name}_X_patches.npy", X_arr)
        print(f"    ✓ Saved → {city_name}_X_patches.npy  {X_arr.shape}")
        del X_arr

        Y_arr = np.stack(Y_patches, axis=0)
        del Y_patches
        np.save(out_dir / f"{city_name}_Y_patches.npy", Y_arr)
        print(f"    ✓ Saved → {city_name}_Y_patches.npy  {Y_arr.shape}")
        del Y_arr
    else:
        del X_patches, Y_patches

    gc.collect()
    return n_patches, scaler_X, scaler_Y


def main():
    from src.cli import configure_console
    configure_console()
    parser = argparse.ArgumentParser(
        description="Stage 1 — Preprocess raw GeoTIFFs into patched training arrays"
    )
    parser.add_argument(
        "--city", type=str, default=None,
        help="Process a single city (e.g. New_Orleans). "
             "If omitted, all cities are processed.",
    )
    parser.add_argument(
        "--raw-dir", type=str, default=None,
        help="Directory containing raw GeoTIFFs (default: data/raw)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for patches & scalers (default: data/processed)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the full pipeline but skip saving files",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else RAW_DIR
    out_dir = Path(args.out_dir) if args.out_dir else PROCESSED_DIR

    cities = {args.city: CITIES[args.city]} if args.city else CITIES

    total_patches = 0
    city_patch_files = []   # list of (X_path, Y_path) for concatenation
    final_scaler_X, final_scaler_Y = None, None

    for city_name, crs in cities.items():
        print(f"\n{'='*65}")
        print(f"  City: {city_name}  |  CRS: {crs}")
        print(f"{'='*65}")

        n_patches, scX, scY = process_city(
            city_name, raw_dir, out_dir, dry_run=args.dry_run
        )

        if scX is not None:
            final_scaler_X = scX
            final_scaler_Y = scY

        total_patches += n_patches

        # Track per-city files for later concatenation
        if n_patches > 0 and not args.dry_run:
            city_patch_files.append((
                out_dir / f"{city_name}_X_patches.npy",
                out_dir / f"{city_name}_Y_patches.npy",
            ))

    # Fallback: build scalers from config constants if all cities were skipped
    if final_scaler_X is None:
        final_scaler_X = FixedMinMaxScaler()
        final_scaler_X.set_bounds(
            mins=np.array([SAR_MIN, SAR_MIN, 0, 0, 0, 0, ELEV_MIN, HAND_MIN, 0, -1, -1]),
            maxs=np.array([SAR_MAX, SAR_MAX, 1, 1, 1, 1, ELEV_MAX, HAND_MAX, 1,  1,  1]),
        )
        final_scaler_Y = FixedMinMaxScaler()
        final_scaler_Y.set_bounds(mins=np.array([ELEV_MIN]), maxs=np.array([ELEV_MAX]))

    # ── Combine all cities (memory-efficient) ──
    print(f"\n{'='*65}")
    print(f"  Total patches across all cities: {total_patches}")
    print(f"{'='*65}")

    if args.dry_run or total_patches == 0:
        print("  Done (dry-run or no patches).")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate per-city .npy files using memory-mapped reads
    # so we never hold more than 1 city's patches in RAM at once.
    print("\n  → Concatenating per-city patches (memory-mapped) ...")
    x_path_out = out_dir / "train_X.npy"
    y_path_out = out_dir / "train_Y.npy"

    # First pass: determine total N and shapes
    sample_x = np.load(city_patch_files[0][0], mmap_mode='r')
    sample_y = np.load(city_patch_files[0][1], mmap_mode='r')
    x_shape_tail = sample_x.shape[1:]   # (C, H, W)
    y_shape_tail = sample_y.shape[1:]   # (1, H, W)
    del sample_x, sample_y

    # Pre-allocate output arrays on disk via np.lib.format
    X_all = np.empty((total_patches, *x_shape_tail), dtype=np.float32)
    offset = 0
    for x_file, _ in city_patch_files:
        chunk = np.load(x_file, mmap_mode='r')
        n = chunk.shape[0]
        X_all[offset:offset + n] = chunk
        offset += n
        del chunk
    np.save(x_path_out, X_all)
    print(f"    train_X : {X_all.shape}  (N, C={IN_CHANNELS}, 128, 128)")
    del X_all

    Y_all = np.empty((total_patches, *y_shape_tail), dtype=np.float32)
    offset = 0
    for _, y_file in city_patch_files:
        chunk = np.load(y_file, mmap_mode='r')
        n = chunk.shape[0]
        Y_all[offset:offset + n] = chunk
        offset += n
        del chunk
    np.save(y_path_out, Y_all)
    print(f"    train_Y : {Y_all.shape}  (N, C=1,  128, 128)")
    del Y_all

    print(f"  ✓ Saved → {x_path_out}")
    print(f"  ✓ Saved → {y_path_out}")

    # Save scalers
    if final_scaler_X is not None:
        joblib.dump(final_scaler_X, out_dir / "scaler_X.joblib")
        joblib.dump(final_scaler_Y, out_dir / "scaler_Y.joblib")
        print(f"  ✓ Scalers saved → {out_dir}")

    print("\n  ✓ Preprocessing complete.")


if __name__ == "__main__":
    main()
