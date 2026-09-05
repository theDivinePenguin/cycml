# Scientific Sensitivity Report: RI Classifier Weighting Multiplier (λ)

**Execution Date**: 2026-09-05 20:48:13 UTC
**Validation Cohort**: `data/metadata/forecast_val_sequences_k5_aligned.csv` (N=7,295 sequences, 181 unique cyclones)
**Locked Test Set**: Strictly Untouched (Zero Test Data Evaluated or Inspected)

## 1. Executive Sensitivity Table

| λ | Overall MAE | +6h MAE | +12h MAE | +24h MAE | RI +24h MAE | Non-RI +24h MAE | False Dips |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | 8.4654 kt | 4.11 kt | 7.57 kt | 13.72 kt | 20.09 kt | 13.34 kt | 0 |
| **0.75** | 7.0455 kt | 3.52 kt | 6.37 kt | 11.24 kt | 21.80 kt | 10.62 kt | 0 |
| **1.00** *(Current)* | 6.4340 kt | 3.31 kt | 5.89 kt | 10.10 kt | 23.51 kt | 9.30 kt | 0 |
| **1.25** | 7.0462 kt | 3.61 kt | 6.44 kt | 11.08 kt | 25.22 kt | 10.24 kt | 0 |
| **1.50** | 8.4800 kt | 4.17 kt | 7.66 kt | 13.61 kt | 26.95 kt | 12.82 kt | 0 |
| **1.75** | 10.4042 kt | 4.92 kt | 9.30 kt | 16.99 kt | 28.68 kt | 16.30 kt | 0 |
| **2.00** | 12.5997 kt | 5.83 kt | 11.19 kt | 20.78 kt | 30.41 kt | 20.21 kt | 0 |

## 2. Granular Subgroup Deltas vs. Current Baseline (λ = 1.00)

| λ | Overall ΔMAE | RI +24h ΔMAE (% change) | Non-RI +24h ΔMAE | Extreme (>=95 kt) ΔMAE | Storms (+/-) | 95% CI (Overall ΔMAE) | p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.50 | +2.0315 kt | -3.42 kt (-14.5%) | +4.04 kt | +1.13 kt | 33/148 | [+1.946, +2.124] | 0.000e+00 |
| 0.75 | +0.6116 kt | -1.71 kt (-7.3%) | +1.31 kt | +0.12 kt | 46/135 | [+0.557, +0.664] | 5.100e-116 |
| 1.00 | +0.0000 kt | +0.00 kt (+0.0%) | +0.00 kt | +0.00 kt | 0/0 | — | — |
| 1.25 | +0.6122 kt | +1.72 kt (+7.3%) | +0.94 kt | +0.83 kt | 67/114 | [+0.560, +0.667] | 2.134e-111 |
| 1.50 | +2.0461 kt | +3.44 kt (+14.6%) | +3.52 kt | +2.69 kt | 46/135 | [+1.951, +2.147] | 0.000e+00 |
| 1.75 | +3.9702 kt | +5.17 kt (+22.0%) | +7.00 kt | +5.32 kt | 29/152 | [+3.851, +4.096] | 0.000e+00 |
| 2.00 | +6.1657 kt | +6.90 kt (+29.4%) | +10.91 kt | +8.43 kt | 19/162 | [+6.007, +6.328] | 0.000e+00 |

## 3. Scientific Analysis

### A. The RI vs. Non-RI Sensitivity Trade-Off
- At **λ = 1.00**, the Ridge model strikes an empirically optimized compromise: Overall MAE = **6.4340 kt**.
- At **λ = 1.25**: RI +24h error changes by **+1.72 kt**, while bulk non-RI error changes by **+0.94 kt**.
- At **λ = 1.50**: RI +24h error changes by **+3.44 kt**, while bulk non-RI error changes by **+3.52 kt**.
- At **λ = 1.75**: RI +24h error changes by **+5.17 kt**, while bulk non-RI error changes by **+7.00 kt**.
- At **λ = 2.00**: RI +24h error changes by **+6.90 kt**, while bulk non-RI error changes by **+10.91 kt**.

### B. Trajectory Monotonicity
- False dips remain **0** across all evaluated values of λ ∈ [0.50, 2.00].

## 4. Final Scientific Verdict

```text
VERDICT: "λ=1.0 remains optimal"
RATIONALE: λ = 1.00 achieves the optimal balance; scaling RI weight further produces no statistically significant overall gain or degrades bulk non-RI accuracy.
```
