# TCIR Multi-Channel Satellite Dataset Audit Report

**Problem Statement 26070**: Tropical Cyclone Identification, Pattern Classification, and Intensity Estimation using Multi-Source Satellite Imagery.

**Audit Date**: September 2026 | **Total Archived Frames**: 70,499 across 1,285 cyclones.

## 1. Executive Summary & Channel Matrix Overview

The Tropical Cyclone Image and Information Repository (TCIR) consists of 4 coregistered, centered $201 \times 201$ pixel satellite imagery channels per cyclone observation fix. Each channel represents a distinct physical wavelength with distinct physical units, valid numerical ranges, and missing-data characteristics.

### Master Channel Specification Table

| Channel | Name | Physical Semantics | Units | Matrix Dtype | Valid Min | Valid Max | Valid Mean ± Std | Missing % | Missing-Value Convention |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **0** | **IR1** | Infrared Window (10.7 µm) | `Kelvin (K)` | `float32` | 76.13 | 347.82 | 267.30 ± 27.23 | **0.66%** | NaN (0.2493%) |
| **1** | **WV** | Water Vapor (6.7 µm) | `Kelvin (K)` | `float32` | 118.68 | 301.62 | 235.94 ± 12.09 | **0.54%** | NaN (0.1456%) |
| **2** | **VIS** | Visible (0.65 µm) | `Dimensionless / Normalized Reflectance` | `float32` | 0.00 | 2.20 | 0.50 ± 0.70 | **35.81%** | NaN (25.9594%) |
| **3** | **PMW** | Passive Microwave / Rain Rate | `mm/hr or Proxy Value` | `float32` | -0.00 | 49.16 | 0.49 ± 1.47 | **0.48%** | NaN (0.1008%) + NetCDF/HDF NC_FILL_FLOAT ~9.96921e+36 (0.0144%) |

---

## 2. In-Depth Channel Analysis

### Channel 0: IR1 — Infrared Window (10.7 µm)
- **Physical Meaning**: Geostationary infrared window brightness temperature measuring cloud-top temperatures and sea-surface thermal emissions.
- **Meteorological Role**: Core backbone for the Advanced Dvorak Technique (ADT). Deep convective clouds in the eyewall/CDO register between 180 K – 215 K (colder = deeper updrafts), while warm tropical ocean backgrounds register ~295 K – 305 K. A warm eye (subsidence warming) registers 240 K – 275 K.
- **Data Completeness**: Highly continuous day and night. Only **0.2493%** (CPAC/IO/SH) and **1.0732%** (ATLN/EPAC/WPAC) isolated missing pixels.
- **Observed Value Range**: ~128.3 K to ~340.6 K.

### Channel 1: WV — Water Vapor (6.7 µm)
- **Physical Meaning**: Upper-to-mid tropospheric water vapor absorption band (~300–600 hPa).
- **Meteorological Role**: Captures upper-level moisture channels, dry air intrusion into cyclone cores, and radial upper-tropospheric outflow channels.
- **Data Completeness**: Highly continuous day and night. Less than **0.01%** missing pixels across all ocean basins.
- **Observed Value Range**: ~118.7 K to ~301.5 K (typically ~30–40 K cooler than IR1 due to atmospheric vapor absorption).

### Channel 2: VIS — Visible (0.65 µm)
- **Physical Meaning**: Solar reflectance / planetary albedo normalized to unit solar zenith angle.
- **Meteorological Role**: Ultra-high detail imagery of shallow low-level cumulus banding, eyewall pinhole structures, and convective cloud texture.
- **Data Completeness Limitation**: Subject to Earth's diurnal cycle. Exactly **3,823 frames (16.54%)** in CPAC/IO/SH and **16,528 frames (34.88%)** in ATLN/EPAC/WPAC are nighttime observations containing all-NaN values.
- **Observed Value Range**: 0.00 to 2.20 normalized albedo. Zero during twilight/night.

### Channel 3: PMW — Passive Microwave / Rain Rate Proxy
- **Physical Meaning**: Low-Earth Orbit (LEO) passive microwave radiometer measurements (85–91 GHz scattering / precipitation rate proxy).
- **Meteorological Role**: Directly penetrates non-precipitating cirrus clouds to reveal inner-core eyewall concentric rings and rainband organization.
- **Missing Value Representation Alert**: In the raw HDF5 dataset, missing microwave observations are represented by two distinct mechanisms:
  1. `IEEE NaN` (accounting for 0.1008% in CPAC/IO/SH and 0.8393% in ATLN/EPAC/WPAC)
  2. `NC_FILL_FLOAT = 9.969209968386869e+36` (values $> 10^20$, accounting for 134,670 pixels in CPAC/IO/SH)
- **Observed Valid Range**: -0.00 to ~32.6 mm/hr (or normalized precipitation proxy units). Mean of valid pixels: 0.40.


---

## 3. Scientific Preprocessing & Missing Value Protocol

Based on the rigorous physical and numerical audit, the multi-channel preprocessing pipeline must adhere to the following deterministic rules:

1. **Fill-Value Cleaning (Channel 3 PMW)**: Pixels with values $> 10^{20}$ or $< -100$ must be converted to `NaN` before downstream operations to eliminate NetCDF fill-value corruption.
2. **Nighttime Visible Handling (Channel 2 VIS)**: Nighttime NaNs (solar albedo = 0) must be imputed with `0.0` (physical absence of solar photons) rather than the global day-time mean.
3. **Microwave Imputation (Channel 3 PMW)**: Missing LEO microwave pixels/frames must be imputed with `0.0` (zero rain-rate / background ocean baseline).
4. **Thermal Window Imputation (Channels 0 & 1: IR1 & WV)**: Rare missing pixels (<0.1%) must be filled using channel-wise spatial medians or valid training means.
5. **Per-Channel Standardization**: Normalization mean and standard deviation must be computed independently per channel strictly over the **training set** (`splits_all_basins.json`), guaranteeing zero test or validation leakage.
