# Scientific Validation Report: Learned RI-Aware Correction Model

**Execution Date**: 2026-09-05 21:04:41 UTC
**Training Split**: `data/metadata/forecast_train_sequences_k5_aligned.csv` (N = 31,280)
**Validation Cohort**: `data/metadata/forecast_val_sequences_k5_aligned.csv` (N = 7,295, 181 cyclones)
**Locked Test Set**: Strictly Untouched (Zero Test Predictions Generated or Inspected)
**Base Checkpoints**: 100% Frozen

## 1. Executive Scientific Verdict

```text
VERDICT: PROMISING (Candidate only. Canonical locked test remains unchanged.)
RATIONALE: Learned correction MLP_AllHorizons_scale_15kt achieves a genuine -2.38 kt reduction in RI error while maintaining overall MAE and keeping non-RI degradation within +-0.48 kt.
```

## 2. Full Model Comparison Table

| Model Configuration | Overall MAE (Δ) | +24h MAE | RI +24h MAE (Δ) | Non-RI +24h MAE (Δ) | RI Underpred % | False Dips | Storms (+/-) | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Canonical Baseline (No Correction)** *(Baseline)* | 6.4340 kt (+0.0000) | 10.10 kt | 23.51 kt (+0.00) | 9.30 kt (+0.00) | 98.5% | 0 | 0/0 | — |
| **Ridge_24h_alpha_10** | 6.3447 kt (-0.0892) | 9.83 kt | 21.99 kt (-1.52) | 9.11 kt (-0.19) | 90.7% | 0 | 103/78 | [-0.114, -0.066] |
| **Ridge_24h_alpha_50** | 6.3433 kt (-0.0907) | 9.83 kt | 22.01 kt (-1.50) | 9.11 kt (-0.20) | 90.7% | 0 | 105/76 | [-0.116, -0.067] |
| **Ridge_24h_alpha_100** | 6.3427 kt (-0.0913) | 9.83 kt | 22.01 kt (-1.50) | 9.10 kt (-0.20) | 90.7% | 0 | 105/76 | [-0.117, -0.068] |
| **Ridge_24h_alpha_500** | 6.3394 kt (-0.0945) | 9.82 kt | 21.94 kt (-1.56) | 9.10 kt (-0.21) | 90.7% | 0 | 104/77 | [-0.119, -0.072] |
| **Ridge_24h_alpha_1000** | 6.3375 kt (-0.0965) | 9.81 kt | 21.87 kt (-1.63) | 9.09 kt (-0.21) | 90.5% | 0 | 106/75 | [-0.120, -0.074] |
| **Ridge_24h_alpha_5000** | 6.3376 kt (-0.0963) | 9.81 kt | 21.50 kt (-2.01) | 9.12 kt (-0.19) | 90.5% | 0 | 112/69 | [-0.115, -0.077] |
| **Ridge_AllHorizons_alpha_100** | 6.0554 kt (-0.3786) | 9.83 kt | 22.01 kt (-1.50) | 9.10 kt (-0.20) | 90.7% | 0 | 131/50 | [-0.422, -0.333] |
| **Ridge_AllHorizons_alpha_500** | 6.0549 kt (-0.3791) | 9.82 kt | 21.94 kt (-1.56) | 9.10 kt (-0.21) | 90.7% | 0 | 133/48 | [-0.422, -0.336] |
| **Ridge_AllHorizons_alpha_1000** | 6.0561 kt (-0.3779) | 9.81 kt | 21.87 kt (-1.63) | 9.09 kt (-0.21) | 90.5% | 0 | 133/48 | [-0.420, -0.337] |
| **Ridge_AllHorizons_alpha_5000** | 6.0817 kt (-0.3523) | 9.81 kt | 21.50 kt (-2.01) | 9.12 kt (-0.19) | 90.5% | 0 | 137/44 | [-0.388, -0.318] |
| **MLP_24h_scale_5kt** | 6.2786 kt (-0.1553) | 9.63 kt | 21.56 kt (-1.94) | 8.93 kt (-0.38) | 95.6% | 0 | 127/54 | [-0.179, -0.134] |
| **MLP_AllHorizons_scale_5kt** | 5.9447 kt (-0.4892) | 9.63 kt | 21.95 kt (-1.55) | 8.90 kt (-0.40) | 95.6% | 0 | 141/40 | [-0.535, -0.445] |
| **MLP_24h_scale_10kt** | 6.2507 kt (-0.1833) | 9.55 kt | 21.31 kt (-2.20) | 8.85 kt (-0.45) | 94.9% | 0 | 121/60 | [-0.214, -0.155] |
| **MLP_AllHorizons_scale_10kt** | 5.8886 kt (-0.5453) | 9.53 kt | 21.37 kt (-2.14) | 8.82 kt (-0.48) | 95.1% | 0 | 138/43 | [-0.599, -0.495] |
| **MLP_24h_scale_15kt** | 6.2533 kt (-0.1807) | 9.56 kt | 20.92 kt (-2.59) | 8.88 kt (-0.42) | 93.4% | 0 | 120/61 | [-0.211, -0.150] |
| **MLP_AllHorizons_scale_15kt** | 5.8830 kt (-0.5509) | 9.51 kt | 21.12 kt (-2.38) | 8.82 kt (-0.48) | 94.4% | 0 | 137/44 | [-0.605, -0.500] |
| **MLP_24h_scale_20kt** | 6.2464 kt (-0.1875) | 9.54 kt | 20.77 kt (-2.74) | 8.87 kt (-0.43) | 92.9% | 0 | 119/62 | [-0.223, -0.154] |
| **MLP_AllHorizons_scale_20kt** | 5.8848 kt (-0.5492) | 9.52 kt | 21.10 kt (-2.41) | 8.83 kt (-0.48) | 94.4% | 0 | 136/45 | [-0.602, -0.497] |

## 3. Strengthening Regimes Audit (Signed Error Progression)

| Regime | Baseline MAE (Underpred %) | Best Candidate MAE (Underpred %) | Baseline Signed Error | Best Candidate Signed Error |
| :--- | :---: | :---: | :---: | :---: |
| **All Sequences (N=7,295)** | 10.10 kt (49.9%) | 9.51 kt (50.0%) | +0.00 kt | -0.14 kt |
| **True ΔV24 > 0 kt (N=3,522)** | 12.40 kt (83.8%) | 11.83 kt (81.9%) | -10.49 kt | -9.42 kt |
| **True ΔV24 >= 10 kt (N=1,887)** | 14.97 kt (88.6%) | 14.08 kt (85.7%) | -13.59 kt | -12.11 kt |
| **True ΔV24 >= 20 kt (N=989)** | 18.78 kt (93.1%) | 17.23 kt (89.9%) | -18.22 kt | -16.06 kt |
| **True RI: ΔV24 >= 30 kt (N=409)** | 23.51 kt (98.5%) | 21.12 kt (94.4%) | -23.42 kt | -20.72 kt |

## 4. Visual Diagnostics Generated

1. `plot1_pred_vs_true_delta24.png`: Scatter comparison of predicted vs true ΔV24.
2. `plot2_error_binned_by_pri.png`: +24h MAE stratified by predicted RI probability.
3. `plot3_signed_error_binned_by_true_delta24.png`: Signed bias curve across ground truth ΔV24.
4. `plot4_ri_event_pred_vs_true.png`: Close-up on the 409 true RI sequences.
5. `plot5_example_trajectories.png`: Real validation trajectories comparing Baseline vs Learned Correction.
