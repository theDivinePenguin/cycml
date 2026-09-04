# TCIR Dataset Audit Report: Cyclone Identification & Center Localization
**SIH Problem Statement 26070 — Identification Component**  
**Date**: 2026-09-04  
**Audit Author**: DeepCycloNet Engineering & Scientific Team  

---

## 1. Objective of the Audit
Before training any model, verify what the TCIR dataset genuinely contains regarding:
1. Spatial dimensions and channel layout
2. Center representation and coordinate mapping
3. Centering status (pre-centered vs. raw/off-center)
4. Presence of negative / non-cyclone imagery
5. Feasibility of a scientifically rigorous identification and center localization benchmark

---

## 2. Empirical Findings

### A. Raw Storage & Dimensions
* **Available HDF5 Archives**:
  * `data/raw/TCIR-CPAC_IO_SH.h5`: **23,118 images**, shape `(23118, 201, 201, 4)`, float32
  * `data/raw/TCIR-ATLN_EPAC_WPAC.h5`: **47,381 images**, shape `(47381, 201, 201, 4)`, float32
  * **Total Observations**: **70,499 satellite images** across 1,285 unique tropical cyclones.
* **Spatial Dimensions**: Exactly **$201 \times 201$ pixels**.
* **Channel Layout**:
  * **Channel 0 (`IR1`)**: 10.8 µm Clean Infrared Brightness Temperature (physical Kelvin, $\mu \approx 267.8\text{ K}$)
  * **Channel 1 (`WV`)**: 6.7 µm Upper-Tropospheric Water Vapor (physical Kelvin, $\mu \approx 236.1\text{ K}$)
  * **Channel 2 (`VIS`)**: 0.65 µm High-Resolution Visible Albedo ($0.0 \to 1.0$)
  * **Channel 3 (`PMW`)**: 89 GHz Passive Microwave Polarized Brightness Temperature (sparse)

---

### B. Spatial Geometry & Coordinate Mapping
* **Grid Spacing**: Interpolated to a uniform spatial resolution of **$0.07^\circ$ latitude/longitude per pixel**.
* **Field of View (FOV)**:
  $$\text{FOV} = 201 \times 0.07^\circ = 14.07^\circ \times 14.07^\circ \approx 1,560\text{ km} \times 1,560\text{ km}$$
* **Nominal Ground-Truth Center**:
  Every image is centered on the official JTWC/NHC Best Track location $(\text{lat}_{\text{center}}, \text{lon}_{\text{center}})$.
  In the raw $201 \times 201$ array, the best-track eye is fixed at pixel coordinate:
  $$(u_{\text{center}}, v_{\text{center}}) = (100, 100)$$
* **Pixel-to-Geographic Mapping**:
  $$\text{lat}(u, v) = \text{lat}_{\text{center}} - (u - 100) \times 0.07^\circ$$
  $$\text{lon}(u, v) = \text{lon}_{\text{center}} + (v - 100) \times 0.07^\circ$$
  Physical distance per pixel: $1^\circ \text{ latitude} \approx 111.12\text{ km} \implies 0.07^\circ \approx 7.78\text{ km/pixel}$ (in Y).

---

### C. Dataset Limitations & Constraints

| Question | Verification Result | Scientific Implication |
| :--- | :---: | :--- |
| **Are existing frames already centered?** | **YES (100%)** | All 70,499 images were pre-centered at $(100, 100)$ by the dataset creators. |
| **Are raw / uncentered full-disk images available?** | **NO** | TCIR contains only pre-cut $14^\circ \times 14^\circ$ cyclone-centric crops. No raw full-disk geostationary sweeps exist in the repository. |
| **Do natural negative (non-cyclone) examples exist?** | **NO** | All 70,499 records belong to officially named/numbered cyclones ($V_{\max} \ge 10\text{ kt}$). There are 0 open-ocean background images in the H5 files. |
| **Can multiple cyclones appear in one image?** | **NO** | The $14^\circ$ bounding box isolates a single tropical cyclone vortex. |

---

## 3. Scientific Formulation: How to Defensibly Construct the Task

Because TCIR does not provide full-disk sweeps or natural negative patches, **we must not fake a benchmark by claiming raw global detection on centered data**. 

Instead, there are two scientifically defensible formulations:

### Formulation 1: Controlled Off-Center Crop Localization (Recommended for Phase 2–5)
* **Rationale**: In real operations, an initial rough advisory box or previous 3-hour extrapolation places the cyclone somewhere inside a sensor window, but with navigation or center-fixing errors ($\pm 50 \text{ to } 250\text{ km}$).
* **Mechanism**:
  * Take sub-windows of size $128 \times 128$ (or $160 \times 160$) randomly or systematically cropped from the $201 \times 201$ canvas.
  * For a $128 \times 128$ window, the true center $(100, 100)$ can appear anywhere between pixel $(27, 27)$ and $(100, 100)$ in the crop, representing realistic center displacements up to:
    $$\Delta x, \Delta y \in [-36.5, +36.5]\text{ pixels} \approx \pm 280\text{ km}$$
  * The model is tasked with predicting the 2D Gaussian heatmap $\mathcal{H}(x, y)$ centered at $(x_{\text{center}}, y_{\text{center}})$ within this sub-window.

### Formulation 2: Negative (Cyclone-Absent) Synthesis via Outer Peripheral Ocean Crops
* **Rationale**: For negative detection samples (`cyclone_present = 0`), we can extract peripheral corner crops ($80 \times 80$) from the outer edges of compact or weak systems ($V_{\max} < 30\text{ kt}$, $R_{35} = 0$).
* At a radial distance $> 650\text{ km}$ from the center, these peripheral sub-crops contain ambient tropical ocean and trade-wind cumulus with zero vortex signature, providing physically valid negative training samples without fabricating artificial noise.

---

## 4. Next Step Recommendation
Proceed with **Formulation 1 (Controlled Off-Center Sub-Window Cropping + 2D Gaussian Heatmap Localization)** and **Formulation 2 (Peripheral Ocean Negative Sampling)**:
1. **Detection Head**: Binary cross-entropy $\hat{y} \in [0, 1]$ (Cyclone Present vs. Absent).
2. **Center Localization Head**: Spatial decoder predicting Gaussian target heatmap $\mathcal{H}^*(x, y) = \exp\left(-\frac{(x - x_c)^2 + (y - y_c)^2}{2\sigma^2}\right)$ with $\sigma \approx 2.5\text{ pixels}$ ($\approx 20\text{ km}$).
3. **Evaluation**: Physical kilometer error ($E_{\text{center}} = \sqrt{\Delta x^2 + \Delta y^2} \times 7.78\text{ km}$) reporting Mean, Median, 90th percentile, and $\% < 25\text{ km}$.
