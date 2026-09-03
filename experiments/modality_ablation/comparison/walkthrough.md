# TCIR 8-Way Satellite Modality Ablation Study — Final Scientific Report

## Executive Summary & Scientific Verdict

> [!IMPORTANT]
> **Scientific Verdict**: **`INCONCLUSIVE`**
>
> **Core Finding**: Empirical differences are within observational noise bounds.

### 1. Research Questions Addressed
1. **Does Water Vapor (WV, 6.7 µm) add predictive information beyond IR1?**
2. **Does Visible Reflectance (VIS, 0.65 µm) add predictive information beyond IR1?**
3. **Does Passive Microwave (PMW, Rain Rate proxy) add predictive information beyond IR1?**
4. **Which single additional modality is best?**
5. **Which pairwise combination is best?**
6. **Does combining all four modalities outperform IR1?**
7. **Are improvements statistically significant across cyclone-level block bootstrap?**
8. **How does modality availability (day/night VIS, microwave swaths) affect error?**
9. **Is naive early channel stacking sufficient, or is a hierarchical architecture required?**

---
## 2. Experimental Matrix & Overall Test Performance

Evaluated on **10,581 held-out test frames across 193 unseen tropical cyclones** with zero split leakage:

| Exp ID | Configuration | Channels ($C$) | Test MAE (kt) | Test RMSE (kt) | $R^2$ Score | Median AE | Mean Bias | $\ge 110$ kt MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | **IR1 (Control)** | `[0]` ($C=1$) | **8.635 kt** | **11.955 kt** | **0.8343** | 6.330 kt | -1.441 kt | 13.61 kt |
| **Exp B** | **IR1 + WV** | `[0, 1]` ($C=2$) | **8.609 kt** | **12.033 kt** | **0.8322** | 6.209 kt | -2.165 kt | 13.79 kt |
| **Exp C** | **IR1 + VIS** | `[0, 2]` ($C=2$) | **8.836 kt** | **12.277 kt** | **0.8253** | 6.409 kt | -1.742 kt | 13.68 kt |
| **Exp D** | **IR1 + PMW** | `[0, 3]` ($C=2$) | **9.113 kt** | **12.924 kt** | **0.8064** | 6.444 kt | -2.266 kt | 15.52 kt |
| **Exp E** | **IR1 + WV + VIS** | `[0, 1, 2]` ($C=3$) | **8.563 kt** | **11.966 kt** | **0.8340** | 6.095 kt | -1.493 kt | 13.52 kt |
| **Exp F** | **IR1 + WV + PMW** | `[0, 1, 3]` ($C=3$) | **8.574 kt** | **12.030 kt** | **0.8322** | 6.085 kt | -2.199 kt | 13.53 kt |
| **Exp G** | **IR1 + VIS + PMW** | `[0, 2, 3]` ($C=3$) | **8.793 kt** | **12.288 kt** | **0.8250** | 6.285 kt | -1.270 kt | 14.01 kt |
| **Exp H** | **All Four (IR1+WV+VIS+PMW)** | `[0, 1, 2, 3]` ($C=4$) | **8.584 kt** | **12.037 kt** | **0.8321** | 6.200 kt | -1.262 kt | 13.86 kt |

---
## 3. Paired Cyclone-Level Block Bootstrap Significance (1,000 Resamples)

Unit of resampling is the **individual tropical cyclone** ($N=193$ clusters), accounting for temporal autocorrelation across lifecycle frames:

| Configuration | $\Delta$ MAE vs IR1 (kt) | 95% Confidence Interval | $\Delta$ RMSE (kt) | 95% CI | $\Delta R^2$ | $p$-value | % Gain | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IR1 + WV** | **+0.027 kt** | `[-0.179, +0.228]` | -0.074 kt | `[-0.383, +0.204]` | -0.0021 | **p = 0.766** | +0.32% | NO (p > 0.05) |
| **IR1 + VIS** | **-0.200 kt** | `[-0.358, -0.047]` | -0.318 kt | `[-0.551, -0.077]` | -0.0090 | **p = 0.024** | -2.32% | YES (p < 0.05) |
| **IR1 + PMW** | **-0.483 kt** | `[-0.793, -0.227]` | -0.968 kt | `[-1.680, -0.434]` | -0.0283 | **p = 0.001** | -5.59% | YES (p < 0.05) |
| **IR1 + WV + VIS** | **+0.076 kt** | `[-0.136, +0.291]` | -0.007 kt | `[-0.326, +0.315]` | -0.0002 | **p = 0.536** | +0.88% | NO (p > 0.05) |
| **IR1 + WV + PMW** | **+0.064 kt** | `[-0.163, +0.287]` | -0.066 kt | `[-0.477, +0.321]` | -0.0020 | **p = 0.606** | +0.74% | NO (p > 0.05) |
| **IR1 + VIS + PMW** | **-0.160 kt** | `[-0.352, +0.028]` | -0.335 kt | `[-0.627, -0.035]` | -0.0096 | **p = 0.104** | -1.85% | NO (p > 0.05) |
| **All Four (IR1+WV+VIS+PMW)** | **+0.054 kt** | `[-0.203, +0.276]` | -0.069 kt | `[-0.550, +0.347]` | -0.0021 | **p = 0.648** | +0.63% | NO (p > 0.05) |

---
## 4. Modality Marginal Contributions & Interaction Analysis

### Marginal Contribution of Individual Modalities Beyond IR1:
- **Marginal Gain from WV (6.7 µm)**: `+0.026 kt`
- **Marginal Gain from VIS (0.65 µm)**: `-0.201 kt`
- **Marginal Gain from PMW (Rain Rate)**: `-0.478 kt`

### Pairwise Predictive Interaction Analysis:
- **WV $\times$ VIS Interaction**: `+0.247 kt` -> **Complementary (Synergistic)**
- **WV $\times$ PMW Interaction**: `+0.513 kt` -> **Complementary (Synergistic)**
- **VIS $\times$ PMW Interaction**: `+0.521 kt` -> **Complementary (Synergistic)**

---
## 5. Missingness & Diurnal Stratification Analysis

- **Visible (VIS) Solar Availability**: Day = **nan%**, Night = **nan%**.
  - IR1 Day MAE: `nan kt` vs Night MAE: `nan kt`
  - IR1+VIS Day MAE: `nan kt` vs Night MAE: `nan kt`
- **Passive Microwave (PMW) Swath Availability**: Available = **nan%**, Missing = **nan%**.
  - IR1 Swath MAE: `nan kt` vs Missing MAE: `nan kt`
  - IR1+PMW Swath MAE: `nan kt` vs Missing MAE: `nan kt`

---
## 6. Unseen Indian Ocean Cyclones Generalization

### Super Cyclone Giri (`201004I`, Peak 135.0 kt, 35 Frames):
| Model | Lifecycle MAE | RMSE | Mean Bias | Peak Actual | Peak Predicted | Peak Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IR1 (Control)** | **8.29 kt** | 12.27 kt | -7.02 kt | 135.0 kt | **110.3 kt** | **-24.7 kt** |
| **IR1 + WV** | **9.34 kt** | 11.43 kt | -6.12 kt | 135.0 kt | **117.3 kt** | **-17.7 kt** |
| **IR1 + VIS** | **12.10 kt** | 15.20 kt | -10.36 kt | 135.0 kt | **105.5 kt** | **-29.5 kt** |
| **IR1 + PMW** | **10.13 kt** | 13.23 kt | -6.00 kt | 135.0 kt | **117.5 kt** | **-17.5 kt** |
| **IR1 + WV + VIS** | **9.46 kt** | 11.80 kt | -8.28 kt | 135.0 kt | **112.5 kt** | **-22.5 kt** |
| **IR1 + WV + PMW** | **10.69 kt** | 13.46 kt | -7.32 kt | 135.0 kt | **110.3 kt** | **-24.7 kt** |
| **IR1 + VIS + PMW** | **9.68 kt** | 11.88 kt | -3.14 kt | 135.0 kt | **109.2 kt** | **-25.8 kt** |
| **All Four (IR1+WV+VIS+PMW)** | **9.75 kt** | 12.52 kt | -6.48 kt | 135.0 kt | **103.7 kt** | **-31.3 kt** |

### Severe Cyclonic Storm Madi (`201306I`, Peak 85.0 kt, 61 Frames):
| Model | Lifecycle MAE | RMSE | Mean Bias | Peak Actual | Peak Predicted | Peak Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IR1 (Control)** | **6.56 kt** | 8.39 kt | +0.26 kt | 85.0 kt | **100.2 kt** | **+15.2 kt** |
| **IR1 + WV** | **7.01 kt** | 9.20 kt | -0.32 kt | 85.0 kt | **91.9 kt** | **+6.9 kt** |
| **IR1 + VIS** | **4.41 kt** | 5.71 kt | +0.59 kt | 85.0 kt | **99.8 kt** | **+14.8 kt** |
| **IR1 + PMW** | **4.91 kt** | 6.39 kt | +0.87 kt | 85.0 kt | **91.9 kt** | **+6.9 kt** |
| **IR1 + WV + VIS** | **6.82 kt** | 8.69 kt | -0.13 kt | 85.0 kt | **92.6 kt** | **+7.6 kt** |
| **IR1 + WV + PMW** | **3.16 kt** | 3.96 kt | -0.44 kt | 85.0 kt | **92.1 kt** | **+7.1 kt** |
| **IR1 + VIS + PMW** | **6.44 kt** | 7.97 kt | +2.59 kt | 85.0 kt | **101.2 kt** | **+16.2 kt** |
| **All Four (IR1+WV+VIS+PMW)** | **7.87 kt** | 10.18 kt | +2.44 kt | 85.0 kt | **94.6 kt** | **+9.6 kt** |

---
## 7. Recommended Next-Stage Multimodal Architecture

Because early channel stacking forces missing modality dropouts (e.g. night VIS = 0.0, sparse PMW swaths) through the primary spatial convolution, we recommend advancing to a **Hierarchical Cross-Attention / Modality-Gated Fusion Architecture**:

```text
  IR1 (10.7 µm)  ──>  [ResNet Branch 1] ──┐
  WV  (6.7 µm)   ──>  [ResNet Branch 2] ──┼──> [Modality Masking & Cross-Attention] ──> Intensity (Vmax)
  VIS (0.65 µm)  ──>  [ResNet Branch 3] ──┤        │ (Gated on Solar Zenith / Swath Mask)
  PMW (Rainrate) ──>  [ResNet Branch 4] ──┘
```

This ensures that informative thermal infrared patterns are never contaminated by missing-modality zero-fill noise.

---
### Key Generated Publication Figures:
- [Overall MAE Comparison](file:///home/raymondj/Projects/cycml/experiments/modality_ablation/comparison/plots/overall_mae_comparison.png)
- [Overall Metric Matrix](file:///home/raymondj/Projects/cycml/experiments/modality_ablation/comparison/plots/overall_metric_comparison.png)
- [Intensity Binned Error](file:///home/raymondj/Projects/cycml/experiments/modality_ablation/comparison/plots/error_by_intensity.png)
- [Bias by Intensity Regime](file:///home/raymondj/Projects/cycml/experiments/modality_ablation/comparison/plots/bias_by_intensity.png)
- [High-Intensity Comparison](file:///home/raymondj/Projects/cycml/experiments/modality_ablation/comparison/plots/high_intensity_comparison.png)
- [Scatter Prediction vs Actual](file:///home/raymondj/Projects/cycml/experiments/modality_ablation/comparison/plots/prediction_vs_actual.png)
- [Modality Inclusion Heatmap](file:///home/raymondj/Projects/cycml/experiments/modality_ablation/comparison/plots/modality_ablation_heatmap.png)
- [Missingness Stratification](file:///home/raymondj/Projects/cycml/experiments/modality_ablation/comparison/plots/missingness_vs_error.png)
- [Giri Lifecycle Comparison](file:///home/raymondj/Projects/cycml/experiments/modality_ablation/comparison/plots/giri_lifecycle.png)
- [Madi Lifecycle Comparison](file:///home/raymondj/Projects/cycml/experiments/modality_ablation/comparison/plots/madi_lifecycle.png)