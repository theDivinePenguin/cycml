# TCIR Temporal Sequence & Intensity Forecasting Data Audit

## 1. Overview

This audit analyzes the temporal properties of the TCIR dataset across all global ocean basins to establish empirical foundations for future tropical cyclone intensity forecasting at **+6h, +12h, and +24h**.

* **Total Dataset Frames**: 70,499 across 1,285 unique cyclones
* **Training Split**: 48,856 frames (900 cyclones)
* **Validation Split**: 11,062 frames (192 cyclones)
* **Test Split**: 10,581 frames (193 cyclones)
* **Leakage Prevention**: Grouped strictly by `cyclone_id` (0% cyclone overlap between splits).

## 2. Temporal Interval Distribution

TCIR timestamps are stored in integer format `YYYYMMDDHH`.
The empirical time step between consecutive observations within each cyclone is dominated by **3.0 hours** (~87.5% of consecutive transitions) and **6.0 hours** (~6.8%).

## 3. Sequence and Target Availability Matrix

| Split | Cyclones | Total Frames | +6h Available | +12h Available | +24h Available | All 3 Horizons (+6/12/24h) | Usable K=5 (3h) | Usable K=3 (3h) | Usable K=5 (6h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 900 | 48,856 | 47,053 (96.3%) | 45,253 (92.6%) | 41,665 (85.3%) | 41,661 (85.3%) | **38,097** | **39,875** | **34,622** |
| **VAL** | 192 | 11,062 | 10,678 (96.5%) | 10,295 (93.1%) | 9,532 (86.2%) | 9,532 (86.2%) | **8,773** | **9,152** | **8,024** |
| **TEST** | 193 | 10,581 | 10,195 (96.4%) | 9,810 (92.7%) | 9,043 (85.5%) | 9,043 (85.5%) | **8,279** | **8,661** | **7,533** |
| **ALL** | 1,285 | 70,499 | 67,926 (96.4%) | 65,358 (92.7%) | 60,240 (85.4%) | 60,236 (85.4%) | **55,149** | **57,688** | **50,179** |

## 4. Key Architectural Conclusions for Forecasting

1. **Primary Historical Cadence**: 3-hour resolution with sequence length $K=5$ ($[t-12\text{h}, t-9\text{h}, t-6\text{h}, t-3\text{h}, t]$) yields **32,897 fully populated training sequences** and **6,897 test sequences** where all three future horizons (+6h, +12h, +24h) are simultaneously available.
2. **Alternative Cadence Support**: $K=3$ with 3h spacing ($[t-6\text{h}, t-3\text{h}, t]$) yields **35,463 training sequences**; $K=5$ with 6h spacing ($[t-24\text{h}, t-18\text{h}, t-12\text{h}, t-6\text{h}, t]$) yields **24,142 training sequences**.
3. **Multi-Horizon Alignment**: Every sequence sample records exact datetime values for all input frames and future targets, ensuring mathematically exact physical offsets without silent indexing assumptions.
