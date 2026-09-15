"""
=============================================================================
inference.py — Stage 5: Tiled Inference & GeoTIFF Export
=============================================================================
Run a trained PI-SwinIR checkpoint on full-resolution GeoTIFFs:

  1. Load the 9-band Features_10m.tif
  2. Apply the same normalisation as Stage 1 (11 channels, no NDBI)
  3. Tile into overlapping 128×128 patches
  4. Forward each tile through the model (AMP, GPU)
  5. Blend overlapping predictions with cosine weighting
  6. Invert normalisation back to metres (with clamping + warnings)
  7. Write output GeoTIFF with matching CRS & transform

Usage:
  python -m src.inference --city New_Orleans --checkpoint checkpoints/best_local.pt
  python -m src.inference --city New_Orleans --profile local
  python -m src.inference --features data/raw/New_Orleans_Features_10m.tif \
                          --checkpoint checkpoints/best.pt \
                          --output results/New_Orleans_pred.tif
=============================================================================
"""

import argparse
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

from src.config import (
    RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    IN_CHANNELS, FABDEM_CHANNEL_IDX,
    PATCH_SIZE, USE_AMP,
    SAR_MIN, SAR_MAX, ELEV_MIN, ELEV_MAX, HAND_MIN, HAND_MAX,
    IDX_VV, IDX_VH, IDX_R, IDX_G, IDX_B, IDX_NIR,
    IDX_FABDEM, IDX_HAND, IDX_ROADS,
    get_profile,
)
from src.model import SwinIRDEM


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1 — Load & normalise features (mirrors preprocess.py logic)
# ═══════════════════════════════════════════════════════════════════════════

SAR_BANDS = [IDX_VV, IDX_VH]
S2_BANDS  = [IDX_R, IDX_G, IDX_B, IDX_NIR]


def load_and_normalise(tif_path):
    """
    Load the 9-band feature GeoTIFF (float32, physical units),
    compute spectral indices (NDVI + NDWI, no NDBI),
    and normalise to the 11-channel tensor expected by the model.

    Returns
    -------
    X          : (11, H, W) float32 — normalised input tensor
    profile    : rasterio profile (for writing the output)
    nodata_mask: (H, W) bool — True where all input bands are nodata
    """
    with rasterio.open(tif_path) as src:
        raw = src.read().astype(np.float32)   # (9, H, W)
        profile = src.profile.copy()
        file_nodata = src.nodata

    # Nodata mask: use file metadata or fallback to all-band-zero
    if file_nodata is not None:
        nodata_mask = (raw == file_nodata).all(axis=0)
    else:
        nodata_mask = (raw == 0).all(axis=0)
    raw[:, nodata_mask] = np.nan

    # ── Data is already in physical units (float32) ──
    # SAR: dB,  S2: reflectance [0,1],  FABDEM/HAND: metres
    features = raw.copy()
    for b in S2_BANDS:
        features[b] = np.clip(features[b], 0.0, 1.0)

    # ── Spectral indices (NDVI + NDWI only, no NDBI) ──
    eps = 1e-10
    red, green, nir = features[IDX_R], features[IDX_G], features[IDX_NIR]
    ndvi = (nir - red)   / (nir + red   + eps)
    ndwi = (green - nir) / (green + nir + eps)

    # ── Normalise & stack 11 channels ──
    _, H, W = features.shape
    X = np.empty((IN_CHANNELS, H, W), dtype=np.float32)

    for i, b in enumerate(SAR_BANDS):
        X[i] = np.clip((features[b] - SAR_MIN) / (SAR_MAX - SAR_MIN), 0, 1)
    for i, b in enumerate(S2_BANDS):
        X[2 + i] = features[b]
    X[6] = np.clip((features[IDX_FABDEM] - ELEV_MIN) / (ELEV_MAX - ELEV_MIN), 0, 1)
    X[7] = np.clip((features[IDX_HAND] - HAND_MIN) / (HAND_MAX - HAND_MIN), 0, 1)
    X[8] = features[IDX_ROADS]
    X[9]  = np.clip((ndvi + 1.0) / 2.0, 0, 1)
    X[10] = np.clip((ndwi + 1.0) / 2.0, 0, 1)

    X = np.nan_to_num(X, nan=0.0).astype(np.float32)

    return X, profile, nodata_mask


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2 — Cosine-blended tiled inference
# ═══════════════════════════════════════════════════════════════════════════

def _make_blend_weight(tile_size):
    """
    Build a 2D Hann window for seamless blending of overlapping tiles.

    Hann window: w(n) = sin²(π·n/N) — with 50% overlap, two adjacent
    windows sum to exactly 1.0, guaranteeing perfect reconstruction.
    """
    n = np.arange(tile_size, dtype=np.float32)
    w1d = np.sin(np.pi * (n + 0.5) / tile_size) ** 2
    w2d = np.outer(w1d, w1d)
    w2d = np.clip(w2d, 1e-6, 1.0)
    return w2d


def tiled_inference(model, X, device, tile_size=PATCH_SIZE, stride=None,
                    batch_size=8, use_amp=USE_AMP):
    """
    Run tiled inference with overlap and cosine blending.

    Parameters
    ----------
    model    : SwinIRDEM, CNNLite, or UNetLite (on device) — any nn.Module mapping
               (B, IN_CHANNELS, tile_size, tile_size) -> (B, 1, tile_size, tile_size)
    X        : (C, H, W) float32 — full normalised feature tensor
    device   : torch.device
    tile_size : int
    stride   : int (default tile_size // 2)
    batch_size : int — tiles per GPU batch
    use_amp  : bool

    Returns
    -------
    pred : (H, W) float32 — predicted DEM (normalised 0-1 space)
    """
    if stride is None:
        stride = tile_size // 2

    C, H, W = X.shape
    blend_w = _make_blend_weight(tile_size)

    # Pad X so every position gets a full tile
    # Use 'edge' (repeat boundary pixel) instead of 'reflect' to avoid
    # creating artificial terrain features at boundaries.
    pad_h = max(tile_size, tile_size + int(np.ceil(max(0, H - tile_size) / stride)) * stride) - H
    pad_w = max(tile_size, tile_size + int(np.ceil(max(0, W - tile_size) / stride)) * stride) - W
    X_pad = np.pad(X, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    _, Hp, Wp = X_pad.shape

    # Output accumulators
    pred_sum   = np.zeros((Hp, Wp), dtype=np.float32)
    weight_sum = np.zeros((Hp, Wp), dtype=np.float32)

    # Collect tile coordinates
    coords = []
    for r in range(0, Hp - tile_size + 1, stride):
        for c in range(0, Wp - tile_size + 1, stride):
            coords.append((r, c))

    n_tiles = len(coords)
    print(f"  → {n_tiles} tiles ({Hp}×{Wp} padded, stride={stride})")

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, n_tiles, batch_size):
            batch_coords = coords[batch_start:batch_start + batch_size]
            tiles = []
            for r, c in batch_coords:
                tile = X_pad[:, r:r + tile_size, c:c + tile_size]
                tiles.append(tile)

            batch_t = torch.from_numpy(np.stack(tiles, axis=0)).to(device)

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                out = model(batch_t)                # (B, 1, ts, ts)

            out_np = out.cpu().numpy()[:, 0, :, :]  # (B, ts, ts)

            for idx, (r, c) in enumerate(batch_coords):
                pred_sum[r:r + tile_size, c:c + tile_size]   += out_np[idx] * blend_w
                weight_sum[r:r + tile_size, c:c + tile_size] += blend_w

    # Normalise
    pred = pred_sum / weight_sum

    # Crop back to original size
    pred = pred[:H, :W]
    return pred


# ═══════════════════════════════════════════════════════════════════════════
#  Step 3 — Inverse normalisation (with clamping + warnings)
# ═══════════════════════════════════════════════════════════════════════════

def denorm_elevation(pred_norm):
    """
    Convert normalised prediction back to metres.

    Applies safety clamping with margin and logs a warning if the model
    produced values significantly outside the [0, 1] normalised range.
    """
    vmin, vmax = float(pred_norm.min()), float(pred_norm.max())

    # ── Diagnostic: normalised space ──
    print(f"\n    ── Denormalisation diagnostics ──")
    print(f"    Normalised range : [{vmin:.4f}, {vmax:.4f}]")
    if vmin < -0.5 or vmax > 1.5:
        print(f"    ⚠ WARNING: Values far outside [0,1] — model may be undertrained.")

    # ── Pre-clamp in normalised space ──
    # Physically: valid elevations lie in [ELEV_MIN, ELEV_MAX] = [-150, 500]
    # so normalised values should remain roughly in [-0.1, 1.1].
    # Anything beyond that is numerical noise from an early/undertrained model.
    n_extreme = int(np.sum((pred_norm < -0.15) | (pred_norm > 1.15)))
    if n_extreme > 0:
        pct = n_extreme / pred_norm.size * 100
        print(f"    ⚠ Pre-clamping {n_extreme:,} extreme normalised pixels ({pct:.2f}%)")
    pred_norm = np.clip(pred_norm, -0.15, 1.15)

    # Denormalise: x_metres = x_norm × (ELEV_MAX − ELEV_MIN) + ELEV_MIN
    pred_metres = pred_norm * (ELEV_MAX - ELEV_MIN) + ELEV_MIN

    m_min, m_max = float(pred_metres.min()), float(pred_metres.max())
    print(f"    Denormed (metres): [{m_min:.2f}, {m_max:.2f}]")

    # ── Clamp to physically plausible range with 50 m margin ──
    clamp_lo = ELEV_MIN - 50.0
    clamp_hi = ELEV_MAX + 50.0
    n_clamped = int(np.sum((pred_metres < clamp_lo) | (pred_metres > clamp_hi)))
    if n_clamped > 0:
        pct = n_clamped / pred_metres.size * 100
        print(f"    ⚠ Clamping {n_clamped:,} pixels ({pct:.2f}%) "
              f"to [{clamp_lo:.0f}, {clamp_hi:.0f}] m")
    pred_metres = np.clip(pred_metres, clamp_lo, clamp_hi)

    return pred_metres


# ═══════════════════════════════════════════════════════════════════════════
#  Step 4 — GeoTIFF export
# ═══════════════════════════════════════════════════════════════════════════

def save_geotiff(data, ref_profile, output_path):
    """Write a single-band COG (Cloud Optimized GeoTIFF) with ZSTD
    compression and built-in overviews for fast rendering."""
    from rasterio.enums import Resampling
    from rasterio.shutil import copy as rio_copy

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = ref_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.nan,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="zstd",
        zstd_level=3,
        predictor=2,
        BIGTIFF="IF_SAFER",
    )

    # Step 1: Write tiled GeoTIFF with ZSTD
    tmp_path = output_path.with_suffix(".tmp.tif")
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)

    # Step 2: Build overviews (pyramids)
    overview_levels = [2, 4, 8, 16]
    with rasterio.open(tmp_path, "r+") as ds:
        ds.build_overviews(overview_levels, Resampling.average)
        ds.update_tags(ns="rio_overview", resampling="average")

    # Step 3: Copy to COG
    rio_copy(
        tmp_path, str(output_path),
        driver="COG",
        compress="zstd",
        level=3,
        overview_resampling="average",
        blocksize=512,
    )

    # Cleanup
    if tmp_path.exists():
        tmp_path.unlink()

    size_mb = output_path.stat().st_size / 1024**2
    print(f"  ✓ Saved COG: {output_path}  ({data.shape[0]}×{data.shape[1]}, {size_mb:.0f} MB)")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(features_path, checkpoint_path, output_path,
                  profile_name="local", device=None, batch_size=8, make_figures=True):
    """
    End-to-end inference on a single GeoTIFF.

    Returns the predicted DEM (metres) as a 2D numpy array.
    """
    cfg = get_profile(profile_name)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"  PI-SwinIR DEM Inference")
    print(f"{'='*65}")
    print(f"  Profile    : {cfg.name}")
    print(f"  Features   : {features_path}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Output     : {output_path}")
    print(f"  Device     : {device}")

    # ── Load & normalise ──
    print(f"\n  Loading & normalising features ...")
    X, profile, nodata_mask = load_and_normalise(features_path)
    n_nodata = int(nodata_mask.sum())
    pct_nodata = 100 * n_nodata / nodata_mask.size
    print(f"    Shape: {X.shape}  ({X.shape[1]}×{X.shape[2]} px)")
    print(f"    Nodata pixels: {n_nodata:,} ({pct_nodata:.1f}%)")

    # ── Load checkpoint ──
    print(f"  Loading model checkpoint ...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Auto-detect profile from checkpoint
    saved_profile = ckpt.get("profile", None)
    if saved_profile and saved_profile != cfg.name:
        print(f"    ⚠ Checkpoint was trained with '{saved_profile}' profile, "
              f"not '{cfg.name}'. Auto-switching to '{saved_profile}'.")
        cfg = get_profile(saved_profile)

    # ── Build model ──
    # Baseline checkpoints (baselines/scripts/train_learned_baseline.py) save a "model" key
    # ("cnn_lite" / "unet_lite"); SwinIR checkpoints (src/train.py) never set that key but always
    # set "profile" instead, so this is an unambiguous way to tell them apart.
    baseline_model_name = ckpt.get("model")
    if baseline_model_name in ("cnn_lite", "unet_lite"):
        from baselines.models.cnn_lite import CNNLite
        from baselines.models.unet_lite import UNetLite
        model_cls = {"cnn_lite": CNNLite, "unet_lite": UNetLite}[baseline_model_name]
        model = model_cls(in_channels=IN_CHANNELS, fabdem_channel_idx=FABDEM_CHANNEL_IDX).to(device)
    else:
        model = SwinIRDEM(
            in_channels=IN_CHANNELS,
            embed_dim=cfg.embed_dim,
            num_rstb=cfg.num_rstb,
            num_stl=cfg.num_stl,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            mlp_ratio=cfg.mlp_ratio,
            fabdem_channel_idx=FABDEM_CHANNEL_IDX,
            img_size=PATCH_SIZE,
            drop_path_rate=0.0,   # no stochastic depth at inference
        ).to(device)

    model.load_state_dict(ckpt["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model loaded ({n_params:,} params, epoch {ckpt.get('epoch', '?')})")

    # ── Tiled inference ──
    print(f"\n  Running tiled inference ...")
    pred_norm = tiled_inference(model, X, device, batch_size=batch_size)
    print(f"    Prediction shape: {pred_norm.shape}")

    # ── FABDEM baseline stats ──
    fabdem_norm = X[FABDEM_CHANNEL_IDX]
    fabdem_m = fabdem_norm * (ELEV_MAX - ELEV_MIN) + ELEV_MIN
    print(f"\n    FABDEM input (metres): [{fabdem_m.min():.2f}, {fabdem_m.max():.2f}]")

    # ── Denormalise ──
    pred_metres = denorm_elevation(pred_norm)

    # ── Apply nodata mask ──
    # Set predictions to NaN where the input features were nodata.
    # Without this, the model produces small non-zero values for
    # nodata/ocean areas, which inflates whole-raster statistics.
    pred_metres[nodata_mask] = np.nan
    print(f"    Applied nodata mask: {int(nodata_mask.sum()):,} pixels set to NaN")

    # ── Residual stats (valid pixels only) ──
    valid = ~nodata_mask
    residual_m = pred_metres[valid] - fabdem_m[valid]
    print(f"    Learned residual (m): [{residual_m.min():.2f}, {residual_m.max():.2f}]")
    print(f"    Residual std (m)    : {residual_m.std():.3f}")

    # ── Save ──
    save_geotiff(pred_metres, profile, Path(output_path))

    elapsed = time.time() - t0
    print(f"\n  ⏱ Total inference time: {elapsed:.1f}s")
    print(f"{'='*65}\n")

    # ── Auto-generate inference visualisation ──
    if not make_figures:
        return pred_metres
    try:
        from src.visualize import plot_inference_result
        from src.config import RAW_DIR
        city_name = Path(features_path).stem.replace("_Features_10m", "")
        gt_candidates = sorted(Path(features_path).parent.glob(f"{city_name}_GroundTruth_*.tif"))
        if not gt_candidates:
            gt_candidates = sorted(RAW_DIR.glob(f"{city_name}_GroundTruth_*.tif"))
        gt_f = gt_candidates[0] if gt_candidates else None
        print(f"  Generating inference figures ...")
        plot_inference_result(
            str(Path(output_path)),
            str(features_path),
            gt_path=str(gt_f) if gt_f is not None else None,
            city=city_name,
            output_dir=str(Path(output_path).parent / "figures" / Path(output_path).stem),
        )
    except Exception as e:
        print(f"  ⚠ Visualisation skipped: {e}")

    return pred_metres


def main():
    from src.cli import configure_console
    configure_console()
    parser = argparse.ArgumentParser(description="Run PI-SwinIR inference")
    parser.add_argument("--profile", type=str, default="local",
                        help="Training profile the model was built with (default: local). "
                             "Auto-detected from checkpoint if saved.")
    parser.add_argument("--city", type=str, default=None,
                        help="City name (uses default paths)")
    parser.add_argument("--features", type=str, default=None,
                        help="Path to Features_10m.tif")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (auto-detected from profile)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output GeoTIFF path")
    parser.add_argument("--batch-size", type=int, default=8, help="Tiles per inference batch")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--no-figures", action="store_true", help="Skip diagnostic plots")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    # Resolve paths
    if args.city:
        features = Path(args.features) if args.features else RAW_DIR / f"{args.city}_Features_10m.tif"
        output = Path(args.output) if args.output else RESULTS_DIR / f"{args.city}_Predicted_DEM_10m.tif"
    elif args.features:
        features = Path(args.features)
        output = Path(args.output or "results/predicted_dem.tif")
    else:
        parser.error("Specify --city or --features")

    # Resolve checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        profile_ckpt = CHECKPOINT_DIR / f"best_{args.profile}.pt"
        generic_ckpt = CHECKPOINT_DIR / "best.pt"
        if profile_ckpt.exists():
            checkpoint = str(profile_ckpt)
        elif generic_ckpt.exists():
            checkpoint = str(generic_ckpt)
        else:
            parser.error(f"No checkpoint found. Looked for {profile_ckpt} and {generic_ckpt}")
        print(f"  Auto-detected checkpoint: {checkpoint}")

    run_inference(features, checkpoint, output, profile_name=args.profile,
                  device=torch.device(args.device) if args.device else None,
                  batch_size=args.batch_size, make_figures=not args.no_figures)


if __name__ == "__main__":
    main()
