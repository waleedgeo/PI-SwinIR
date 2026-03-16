<div align="center">
  <img src="img/PI-SwinIR-Logo.png" alt="PI-SwinIR Logo" width="75%">

  <p align="center">
    <strong>PI-SwinIR: A Physics-Informed Swin Transformer for Hydrology-Aware Vertical Refinement of Global DEMs from Multimodal Earth Observation Data</strong>
    <br /><br />
    <a href="#overview">Overview</a>
    ·
    <a href="#graphical-abstract">Graphical Abstract</a>
    ·
    <a href="#model-summary">Model Summary</a>
    ·
    <a href="#key-results">Key Results</a>
    ·
    <a href="#repository-status">Repository Status</a>
    ·
    <a href="#citation">Citation</a>
    ·
    <a href="#contact">Contact</a>
  </p>

  <p align="center">
    <img src="https://img.shields.io/badge/Status-Under%20Submission-orange?style=for-the-badge&logo=gitbook" alt="Status">
    <img src="https://img.shields.io/badge/Method-Physics--Informed%20Deep%20Learning-blue?style=for-the-badge&logo=pytorch" alt="Method">
    <img src="https://img.shields.io/badge/Data-Sentinel--1%20%7C%20Sentinel--2%20%7C%20FABDEM-brightgreen?style=for-the-badge" alt="Data">
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License">
    </a>
  </p>
</div>

<br />

## Overview

PI-SwinIR is a physics-informed deep learning framework for same-resolution vertical refinement of globally available digital elevation models (DEMs) using multimodal Earth observation data. Designed for hydrology-aware terrain modeling, the framework integrates Sentinel-1 SAR, Sentinel-2 multispectral imagery, FABDEM elevation, Height Above Nearest Drainage (HAND), road masks, and spectral indices to improve DEM vertical accuracy while preserving terrain structure relevant to drainage and surface form.

This is **not** a generic image super-resolution method. PI-SwinIR performs **same-resolution vertical refinement** — it refines the elevation surface of existing DEMs at 10 m resolution, reducing vertical error and recovering hydrologically meaningful terrain structure that FABDEM and similar globally available products do not fully capture.

---

## Why This Matters

High-resolution, vertically accurate DEMs are essential for:

- Flood mapping and inundation modeling
- Hydrological analysis and drainage extraction
- Geomorphological and terrain-dependent environmental assessment

Globally available DEM products such as FABDEM contain residual vertical error and often fail to resolve fine-scale terrain features required for reliable 10 m applications, particularly in regions without airborne LiDAR coverage. PI-SwinIR addresses this gap by learning a hydrology-aware vertical refinement function from freely available, globally accessible Earth observation inputs.

---

## Graphical Abstract

<div align="center">
  <img src="img/PI-SwinIR Graphical Abstract - 1920rescale.png" alt="PI-SwinIR Graphical Abstract" width="100%" style="border-radius: 10px; border: 1px solid #ddd; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
</div>

> PI-SwinIR processes an 11-channel multimodal Earth observation input through a physics-informed Swin Transformer and produces a geometrically accurate, hydrologically consistent 10 m DEM refinement.

---

## Model Summary

PI-SwinIR combines a **Swin Transformer V2 backbone** with a **gated residual refinement head** to predict elevation corrections from multimodal Earth observation context. The architecture is designed to refine the source DEM surface while controlling the magnitude and spatial structure of residual updates.

### Input Data

The model ingests an **11-channel input stack** composed of:

| Source | Channels |
|--------|----------|
| Sentinel-1 SAR | VV, VH |
| Sentinel-2 multispectral | RGB, NIR |
| FABDEM | Elevation |
| HAND | Height Above Nearest Drainage |
| OSM road network | Road mask |
| Spectral indices | NDVI, NDWI |

All inputs are freely available and globally reproducible, without reliance on proprietary datasets.

### Physics-Informed Loss

Training uses a **composite loss** that simultaneously optimizes:

- **Elevation reconstruction** — reduces pixel-wise vertical error relative to the reference
- **Slope consistency** — preserves realistic gradient magnitudes across the terrain surface
- **Curvature consistency** — suppresses artificial smoothing and maintains ridge and valley morphology
- **Differentiable D8 flow-routing constraints** — aligns predicted drainage paths with reference stream networks

This formulation ensures that the refined DEM is not only vertically accurate, but also physically consistent with terrain behavior relevant to hydrology.

---

## Key Results

Evaluated against LiDAR-derived terrain models across **eight cities on three continents**, PI-SwinIR demonstrates robust and consistent improvements over FABDEM:

- **25.3% average RMSE reduction** relative to FABDEM across all evaluation sites
- **Sub-meter MAE in five cities**
- **|bias| < 0.1 m in seven cities**
- Measurable stream-network recovery at the 10 m grid
- Strongest gains in **moderate- and high-relief terrain**

The evaluation covered approximately **138 million valid 10 m pixels**, providing a large-scale, geographically diverse assessment of the model's generalization capacity.

---

## Repository Status

> The manuscript describing this methodology is currently **under submission**. The full source code, trained model weights, and inference pipeline will be made publicly available following peer review and manuscript acceptance.

> For early access to the codebase for validation or research collaboration purposes, please [contact the corresponding author](#contact).

### Planned Release

| Component | Status | Timeline |
|-----------|--------|----------|
| Graphical Abstract & Logo | Available | Now |
| Manuscript Submission | Under Review | Submitted |
| Model Architecture Code | Restricted | Upon acceptance |
| Pretrained Weights | Pending | Upon acceptance |
| Training & Inference Pipeline | Pending | Upon acceptance |
| Benchmark Dataset | Pending | Upon acceptance |

---

## Citation

If you use PI-SwinIR or its methodology in your research, please cite the following manuscript once published:

```bibtex
@article{waleed2026piswinir,
  title={PI-SwinIR: A Physics-Informed Swin Transformer for Hydrology-Aware Vertical Refinement of Global DEMs from Multimodal Earth Observation Data},
  author={Waleed, Mirza},
  journal={Under Submission},
  year={2026}
}
```

---

## Contact

**Mirza Waleed** (First Author, Developer, & Corresponding Author)  
Department of Geography, Hong Kong Baptist University  
Hong Kong Special Administrative Region of China

- Website: [waleedgeo.com](https://waleedgeo.com)
- Email: [waleedgeo@outlook.com](mailto:waleedgeo@outlook.com)
- GitHub: [@waleedgeo](https://github.com/waleedgeo)


---

## Acknowledgments

The author gratefully acknowledges the **European Space Agency (ESA)** for providing freely available Sentinel-1 and Sentinel-2 imagery through the Copernicus programme, and the **FABDEM** and **HAND** development teams for their open-access terrain datasets. I also thank the **OpenStreetMap** community for the road network data used in this study.

---

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
