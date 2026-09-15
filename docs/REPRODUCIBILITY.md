# Reviewer guide

## Checks

1. Install runtime and run `python scripts/verify_release.py`: checksums, strict state loading for all six models, finite forwards, composite-loss gradients, matching normalization, and tile coverage on small/irregular grids.
2. Run the README synthetic example for GeoTIFF inference/evaluation without external data.
3. Inspect architecture/loss/config source, checkpoint manifest, saved metrics and training logs.
4. With original prepared rasters, rerun inference/evaluation using identical extent/grid/masks/software. Prepared arrays enable fresh training with supplied profiles.

## Limits

This provides the base computational materials. Full study rasters, complete original acquisition/export scripts and a permanent data archive are absent; all paper results cannot yet be reconstructed from one public download. Synthetic data only check software. Private manuscript/review notes, paper figure generators and wider hydrology/revision workflows are outside this core release.

Production weights are unchanged; compact exports preserve tensor values. No full training or full-city campaign was rerun during assembly. The default split excludes San Francisco and includes Houston in development. Random overlapping-patch validation is spatially dependent; the production file does not embed the full historical resolved split manifest.

The actual prediction is `gate * FABDEM + residual`. Terrain losses use normalized surfaces, and flow propagation is a local two-iteration regularizer. Invalid targets can contribute normalized-zero loss because a separate pixel mask is not used. Normalization clips elevations. These details limit interpretation and transfer to new terrain.

## Public portability adjustments

The repository copy keeps architecture, weights and objectives, while configuring local data roots, disabling author-bucket uploads by default, honoring explicit city output paths, resolving Sydney reference filenames, exposing device/batch size, separating diagnostic outputs and fixing tile coverage below one tile. Fixed-path manuscript plotting and placeholder baseline-evaluation scripts were omitted. These changes do not alter the active research workspace or retroactively recompute historical scores.

Actual checks are recorded in [RELEASE_VALIDATION.md](RELEASE_VALIDATION.md). For future runs record Git commit, checkpoint checksum, environment, input extent/grid/mask and data versions.

## Remaining submission materials

Provide an accessible archive or an explicit access arrangement for exact study data. Complete source versions/dates, export parameters and elevation-reference metadata. Align manuscript city roles and performance claims with code/benchmarks. Add accepted-publication citation metadata when available.
