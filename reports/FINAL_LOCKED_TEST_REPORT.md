# Final Scientific Report: Locked Test Evaluation of DeepCycloNet Suite

**Execution Date**: 2026-09-05 20:38:34 UTC
**Locked Test Manifest**: `data/metadata/forecast_test_sequences_k5_aligned.csv` (N=6,825 sequences, 171 unique cyclones)
**Manifest SHA256**: `3698c48082a9b16f705776fa30a9bbe319a3d36814ce8202d3641ba503690b60`

## 1. Executive Performance Benchmark

| Model Architecture | Mean MAE | +6h MAE | +12h MAE | +24h MAE | +24h RMSE | +24h R² | False Dips |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline (V_t)** | 8.86 kt | 4.12 kt | 7.95 kt | 14.52 kt | 20.12 kt | 0.598 | 0 |
| **Stage 1: Frozen Residual Baseline** | **6.9509 kt** | 3.51 kt | 6.39 kt | 10.95 kt | 15.52 kt | 0.761 | **0** |
| **Stage 1+2+3: Final Hybrid Suite** | **6.6350 kt** | **3.46 kt** | **6.09 kt** | **10.36 kt** | **14.48 kt** | **0.792** | **0** |

## 2. Granular Subgroup Breakdown

* **True RI Events (N=431, 6.3%)**:
  * Residual Baseline: `32.43 kt`
  * Final Hybrid Model: `26.37 kt` (**+18.7% error reduction** / -6.05 kt)
* **Non-RI Events (N=6,394, 93.7%)**:
  * Residual Baseline: `9.51 kt`
  * Final Hybrid Model: `9.28 kt` (-0.23 kt)
* **Extreme Major Cyclones (>=95 kt, N=1,665, 24.4%)**:
  * Residual Baseline: `18.28 kt`
  * Final Hybrid Model: `16.66 kt` (-1.62 kt)

## 3. Statistical Testing & Bootstrap Analysis

* **Paired t-test**: $t = -10.291$, $p = 1.1734e-24$
* **Wilcoxon Signed-Rank**: $W = 10,121,222$, $p = 7.1042e-21$
* **Bootstrap 95% CI (Overall ΔMAE)**: `-0.3157 kt` [-0.3764 kt, -0.2539 kt] (Win Rate: 100.0%)
* **Bootstrap 95% CI (RI +24h ΔMAE)**: `-6.0475 kt` [-6.5447 kt, -5.5370 kt] (Win Rate: 100.0%)

## 4. Final Scientific Verdict

```text
FINAL MODEL:
Residual ΔV CNN + Temporal Transformer K=5
+
Dedicated Focal Loss RI Classifier
+
Ridge Fusion Gate (alpha=10.0)

FINAL LOCKED TEST:
Residual Baseline: 6.9509 kt
Hybrid:            6.6350 kt
Improvement:       +4.54% vs Residual (+25.14% vs Persistence)
RI Error Gain:     +18.7% error reduction on explosive deepening
Passed All Integrity Checks: YES (Zero Leakage, Zero Overlap, Zero False Dips)
```
