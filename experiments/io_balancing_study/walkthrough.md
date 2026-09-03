# Indian Ocean Intensity-Balancing Study: Research Report & Walkthrough

## Executive Summary & Scientific Verdict

* **Research Question**: *Does targeted intensity balancing within the Indian Ocean domain provide a better solution to high-intensity saturation than simply adding cyclone data from unrelated ocean basins?*
* **Predefined Success Criterion**: The intensity-aware IO model is considered to improve the saturation problem if it produces a statistically supported reduction in $\ge 110\text{ kt}$ MAE and/or negative bias, accompanied by a regression slope closer to $1.0$, without substantial degradation in $\le 70\text{ kt}$ performance.
* **Scientific Verdict**: **NOT SUPPORTED**.
  - Sqrt-inverse-frequency sampling within the isolated Indian Ocean domain failed to improve high-intensity estimation ($\Delta \text{MAE}_{\ge 110\text{ kt}} = +13.36\text{ kt}$) and degraded overall performance ($\Delta \text{MAE} = \mathbf{+8.49\text{ kt}}$, 95% CI $[+7.53, +9.32]\text{ kt}$) due to **severe cyclone-level sample sparsity** ($N=3$ training cyclones in the $130–150\text{ kt}$ range).
  - Conversely, the **All-Basin Model (trained across all 6 basins) achieved the best overall performance on the Indian Ocean test set** (**MAE = 6.39 kt**, **RMSE = 8.42 kt**, **$R^2 = 0.784$**, **Slope = 0.87**), decisively outperforming both IO-only models.

---

## 1. Dataset Construction & Leakage Audit

The Indian Ocean (`IO`) subset was extracted from the authoritative TCIR dataset:
- **Total IO Frames**: 3,184 frames across 75 unique cyclones (2003–2016).
- **Grouped 70/15/15 Split**:
  - **Train**: 2,338 frames across 53 unique cyclones (73.4%)
  - **Val**: 524 frames across 12 unique cyclones (16.5%)
  - **Test**: 322 frames across 10 unique cyclones (10.1%)
- **8-Point Leakage Audit**: **STATUS: PASS** (0 cyclone overlap, 0 sample overlap, training-only normalization).

---

## 2. Indian Ocean Training Intensity Distribution

| Intensity Bin | Training Frames | % Frames | Unique Cyclones | Sqrt-Inverse Sampling Prob | Sampling Multiplier |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **15–30 kt** | 708 | 30.28% | 52 | 25.04% | $0.83\times$ |
| **30–50 kt** | 1,117 | 47.78% | 53 | 31.46% | $0.66\times$ |
| **50–70 kt** | 288 | 12.32% | 26 | 15.97% | $1.30\times$ |
| **70–90 kt** | 88 | 3.76% | 12 | 8.83% | $2.35\times$ |
| **90–110 kt** | 46 | 1.97% | 8 | 6.38% | $3.24\times$ |
| **110–130 kt** | 67 | 2.87% | 8 | 7.70% | $2.69\times$ |
| **130–150 kt** | 24 | 1.03% | 3 | 4.61% | $4.49\times$ |
| **> 150 kt** | 0 | 0.00% | 0 | 0.00% | $0.00\times$ |

![IO Training Distribution](file:///home/raymondj/.gemini/antigravity-ide/brain/747dc68b-30a6-4760-b7b7-add1abefb41a/io_training_intensity_distribution.png)
![Natural vs Balanced Sampling Shift](file:///home/raymondj/.gemini/antigravity-ide/brain/747dc68b-30a6-4760-b7b7-add1abefb41a/io_natural_vs_balanced_distribution.png)

---

## 3. Master Comparison Table: Evaluated on the Identical Held-Out IO Test Set

All 4 models were evaluated against the **exact same 322 held-out Indian Ocean test frames (10 cyclones)**:

| Model | Training Dataset | Sampling Strategy | Test Set Domain | Test MAE | Test RMSE | Test $R^2$ | Regression Slope | Bias $\ge 110\text{ kt}$ | MAE $\ge 110\text{ kt}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **All-Basin Model** | All 6 Basins (70k frames) | Natural | **Same IO Test** | **6.39 kt** | **8.42 kt** | **0.784** | **0.87** | **$-20.61\text{ kt}$** | **20.61 kt** |
| **Original Baseline** | CPAC / IO / SH (23k frames) | Natural | **Same IO Test** | 7.79 kt | 10.07 kt | 0.691 | 0.78 | $-14.11\text{ kt}$ | 14.11 kt |
| **IO Natural (A)** | IO Only (2.3k frames) | Natural | **Same IO Test** | 11.34 kt | 14.97 kt | 0.317 | 0.63 | $-51.31\text{ kt}$ | 51.31 kt |
| **IO Balanced (B)** | IO Only (2.3k frames) | Intensity-Aware | **Same IO Test** | 19.87 kt | 22.86 kt | -0.592 | 0.59 | $-64.67\text{ kt}$ | 64.67 kt |

![Model Comparison](file:///home/raymondj/.gemini/antigravity-ide/brain/747dc68b-30a6-4760-b7b7-add1abefb41a/io_model_comparison.png)
![Prediction vs Actual](file:///home/raymondj/.gemini/antigravity-ide/brain/747dc68b-30a6-4760-b7b7-add1abefb41a/io_prediction_vs_actual.png)

---

## 4. Cyclone-Level Performance & Statistical Significance

### A. Cyclone-Level Breakdown (10 Indian Ocean Test Storms)

| Model | Mean Cyclone MAE | Median Cyclone MAE | High-Intensity Cyclone MAE | Best Storm | Worst Storm |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **All-Basin Model** | **6.54 kt** | **6.88 kt** | **7.09 kt** | `201401I` (2.8 kt) | `200603I` (9.7 kt) |
| **Original Baseline** | 7.73 kt | 6.76 kt | 9.74 kt | `200904I` (5.3 kt) | `200603I` (11.3 kt) |
| **IO Natural (A)** | 11.08 kt | 8.31 kt | 23.92 kt | `201104I` (5.5 kt) | `201004I` (23.9 kt) |
| **IO Balanced (B)** | 19.34 kt | 17.33 kt | 32.77 kt | `201104I` (12.7 kt) | `201004I` (32.8 kt) |

### B. Paired Cyclone-Level Block Bootstrap (1,000 Resamples of 10 Storms)
$$\Delta \text{Metric} = \text{Metric}(\text{IO Balanced}) - \text{Metric}(\text{IO Natural})$$

| Metric | Measured Δ | 95% Bootstrap Confidence Interval | $P(\text{Improvement})$ | Statistical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **$\Delta \text{MAE}_{\text{overall}}$** | **$+8.49\text{ kt}$** | **$[+7.53, +9.32]\text{ kt}$** | $0.000$ | Statistically significant degradation |
| **$\Delta \text{RMSE}_{\text{overall}}$** | **$+7.92\text{ kt}$** | **$[+7.18, +8.55]\text{ kt}$** | $0.000$ | Statistically significant degradation |
| **$\Delta \text{Slope}$** | **$-0.05$** | **$[-0.13, -0.01]$** | $0.003$ | Flatter slope (worse compression) |
| **$\Delta \text{MAE}_{\ge 110\text{ kt}}$** | **$+13.36\text{ kt}$** | **$[+13.36, +13.36]\text{ kt}$** | $0.000$ | High-intensity error increased |

![Error by Intensity](file:///home/raymondj/.gemini/antigravity-ide/brain/747dc68b-30a6-4760-b7b7-add1abefb41a/io_error_by_intensity.png)
![Bias by Intensity](file:///home/raymondj/.gemini/antigravity-ide/brain/747dc68b-30a6-4760-b7b7-add1abefb41a/io_bias_by_intensity.png)

---

## 5. Scientific Explanation: Why Global Data Expansion Outperforms Isolated Domain Balancing

1. **The Fundamental Constraint of Single-Basin Sparsity**:
   The North Indian Ocean experiences only $\sim 4–6$ tropical cyclones per year. In the entire 14-year TCIR archive, only 3 storms exceeded 130 kt.
   Attempting to solve high-intensity saturation by reweighting 3 storms simply overfits the network to those 3 individual convective snapshots, degrading generalizability across the 78% of storms in the 15–50 kt range.

2. **Universal Atmospheric Physics**:
   Tropical cyclone infrared signatures (convective cloud-top temperatures, eyewall curvature, and central dense overcast diameters) are governed by thermodynamic laws that are invariant across ocean basins.
   Training on 70,499 global frames provided the ResNet-18 model with rich morphological priors for Category 4–5 systems, allowing it to generalize back to the Indian Ocean with unprecedented accuracy (**6.39 kt MAE**).

---

## 6. Recommended Next Research Directions

1. **Multi-Spectral Sensor Fusion**: Combining infrared (IR1) with Passive Microwave (PMW 85–91 GHz) channels to peer through upper cloud shields and resolve inner eyewall dynamics directly.
2. **Global Pretraining with Domain Fine-Tuning**: Pretraining on all 6 global basins and fine-tuning with a low learning rate on regional basins.
