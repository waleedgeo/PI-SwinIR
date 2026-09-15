# Data contract

## Prepared features

Preprocessing and inference expect a **nine-band float32 GeoTIFF at 10 m**, with a projected CRS in metres and all bands aligned. Registered filenames: `{City}_Features_10m.tif`.

| Band (1-based) | Feature | Units |
|---|---|---|
| 1 | Sentinel-1 VV | dB |
| 2 | Sentinel-1 VH | dB |
| 3 | Sentinel-2 Red | Reflectance 0–1 |
| 4 | Sentinel-2 Green | Reflectance 0–1 |
| 5 | Sentinel-2 Blue | Reflectance 0–1 |
| 6 | Sentinel-2 NIR | Reflectance 0–1 |
| 7 | FABDEM | Metres, resampled to 10 m |
| 8 | HAND | Metres |
| 9 | Roads | Binary 0/1 |

Loaders **do not decode scaled Int16 exports**. Historical exports used SAR `/100`, optical `/10000`, and elevation/HAND `/100`; decode those before use while preserving NoData, after verifying the actual source scales. Export conversion is outside this release. Use prepared physical-unit float32 rasters with `src.preprocess` and `src.inference`.

## Normalization

Tensors are `(batch, 11, height, width)`. SAR is clipped/mapped from −32…44 dB, FABDEM and targets from −150…500 m, HAND from 0…500 m, and optical reflectance to 0…1. Roads stay binary. FABDEM is channel 6 (zero-based).

`NDVI = (NIR - Red) / (NIR + Red + epsilon)` and `NDWI = (Green - NIR) / (Green + NIR + epsilon)`. Both are mapped to 0…1 using `(index + 1) / 2`, clipped, and appended at channels 9 and 10. Preserve these bounds and channel order for released weights. Out-of-range elevations are clipped during input/target normalization.

## Reference and missing data

Reference rasters are single-band elevation **in metres**, with CRS, transform and NoData metadata: `{City}_GroundTruth_1m.tif`, `_5m.tif` or `_10m.tif`. Native pixels are averaged to approximately 10 m and then aligned with bilinear reprojection. Zero elevation is valid ground truth.

Feature NoData uses all-nine-bands matching declared NoData, or all-band zero without metadata. Residual invalid input values are filled with normalized zero. Preprocessing rejects patches with excessive missing/ocean content; remaining invalid target pixels are filled with normalized zero **without a separate per-pixel loss mask**. Inference restores the all-band NoData footprint; evaluation masks invalid prediction/reference pixels. Check partial-band missingness and coastal masks for new inputs.

## Study areas

Boundaries are under [data/regions](../data/regions/); city keys/CRS are registered in [src/config.py](../src/config.py).

| City key | CRS | Reference source recorded locally | Native resolution | Default role |
|---|---|---|---|---|
| Houston | EPSG:32615 | USGS 3DEP | 1 m | Development |
| New_Orleans | EPSG:32615 | USGS 3DEP | 1 m | Development |
| Seattle | EPSG:32610 | USGS 3DEP | 1 m | Development |
| San_Francisco | EPSG:32610 | USGS 3DEP | 1 m | Unseen |
| Miami | EPSG:32617 | USGS 3DEP | 1 m | Development |
| London | EPSG:27700 | Environment Agency | 1 m | Development |
| Rotterdam | EPSG:28992 | AHN | 1 m | Development |
| Sydney | EPSG:32756 | NSW Government | 5 m | Development |

Exact source versions/dates, acquisition/compositing/export parameters, and vertical-datum harmonization are not fully specified here. Confirm them against original exports and manuscript before claiming exact data reconstruction.

## Availability

Included: code, weights, saved metrics/logs, study boundaries and a synthetic-data generator. Multi-gigabyte study rasters and processed arrays are outside Git. **No verified public benchmark download or permanent data archive is supplied.** A private project bucket is not a reviewer download source.

Contact [Mirza Waleed](mailto:waleedgeo@outlook.com) for exact prepared rasters. Source products retain their respective terms. A public archive with a persistent identifier and complete acquisition/export metadata is still needed for independent reproduction of all manuscript results.

For your own inputs, align products to a common projected 10 m grid, match bands/units, confirm elevation references, and provide reference terrain for training/evaluation. Add new city/CRS entries to `CITIES` for preprocessing/training. Arbitrary areas can use inference `--features` directly.
