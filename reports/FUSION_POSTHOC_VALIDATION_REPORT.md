# Scientific Validation Experiment: Post-Hoc Fusion of Residual ΔV & Dedicated RI Classifier

**Date**: 2026-09-06 01:51:07
**Validation Cohort**: Exactly 7,295 sequences (`data/metadata/forecast_val_sequences_k5_aligned.csv`)
**Gate Training Cohort**: 6,000 sequences from `data/metadata/forecast_train_sequences_k5_aligned.csv` (Zero validation leakage)
**Locked Test Set**: Strictly untouched.

## 1. Global Horizon Performance Comparison

| Configuration | Mean MAE | +6h MAE | +12h MAE | +24h MAE | +24h RMSE | +24h R² | False Dips |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Residual Forecaster Alone (Baseline)** | 6.68 kt | 3.33 kt | 6.10 kt | 10.62 kt | 15.19 kt | 0.748 | 0 |
| **2. Residual + RI Probability (Heuristic)** | 6.62 kt | 3.31 kt | 6.05 kt | 10.49 kt | 15.03 kt | 0.753 | 0 |
| **3. Residual + RI Prob + V_max (Stage-Conditioned)** | 6.62 kt | 3.31 kt | 6.05 kt | 10.50 kt | 15.03 kt | 0.753 | 0 |
| **4. Learned Ridge Gating Model** | 6.55 kt | 3.33 kt | 5.97 kt | 10.35 kt | 14.86 kt | 0.758 | 0 |

## 2. Sub-Cohort Granular Breakdown

| Configuration | RI Events (+24h MAE) | Non-RI (+24h MAE) | Extreme Intensity (+24h MAE) | RI Events Overall MAE |
| :--- | :---: | :---: | :---: | :---: |
| **1. Residual Forecaster Alone (Baseline)** | 29.81 kt | 9.48 kt | 19.03 kt | 16.90 kt |
| **2. Residual + RI Probability (Heuristic)** | 27.25 kt | 9.50 kt | 18.33 kt | 15.60 kt |
| **3. Residual + RI Prob + V_max (Stage-Conditioned)** | 27.46 kt | 9.49 kt | 18.42 kt | 15.69 kt |
| **4. Learned Ridge Gating Model** | 18.13 kt | 9.89 kt | 16.18 kt | 11.15 kt |

## 3. Statistical Significance vs. Pure Residual Baseline

| Configuration | ΔMAE vs. Residual | Paired t-statistic | p-value | Scientific Conclusion |
| :--- | :---: | :---: | :---: | :--- |
| **2. Residual + RI Probability (Heuristic)** | -0.0665 kt | -8.387 | 5.9387e-17 | **STATISTICALLY SIGNIFICANT IMPROVEMENT** |
| **3. Residual + RI Prob + V_max (Stage-Conditioned)** | -0.0626 kt | -8.911 | 6.3261e-19 | **STATISTICALLY SIGNIFICANT IMPROVEMENT** |
| **4. Learned Ridge Gating Model** | -0.1336 kt | -3.279 | 1.0478e-03 | **STATISTICALLY SIGNIFICANT IMPROVEMENT** |

## 4. Scientific Findings & Discussion

### A. Substantial Error Reduction on Rapid Intensification Events
* **Pure Residual Baseline Failure Mode**: While the baseline residual forecaster achieves high overall accuracy (6.68 kt MAE), it under-predicts extreme rapid intensification surges due to variance compression under Huber loss, exhibiting an MAE of **29.81 kt on true RI events** at +24h.
* **Impact of RI Fusion**:
  - The **Learned Ridge Gating Model** slashes +24h error on true RI events from **29.81 kt down to 18.13 kt** (a massive **11.68 kt / 39.2% error reduction**).
  - Extreme intensity errors ($V \ge 95\text{ kt}$) drop from **19.03 kt to 16.18 kt** (-2.85 kt).
  - Heuristic probability thresholding alone achieves a noticeable **2.56 kt reduction** on RI events (29.81 kt $\to$ 27.25 kt).

### B. The Non-RI False-Alarm Penalty & Statistical Significance
* **The Inherent Trade-Off**: Boosting intensity when $P(\text{RI})$ triggers causes a slight increase in error on non-RI events:
  - Non-RI +24h MAE increases modestly from **9.48 kt to 9.50 kt (+0.02 kt)** under heuristic gating, and to **9.89 kt (+0.41 kt)** under the learned Ridge gate.
* **Net Overall Validation Result**:
  - Because non-RI events comprise 94.4% of all sequences, the net overall gain is moderate but statistically decisive:
    - Heuristic Fusion: **6.68 kt $\to$ 6.62 kt** ($\Delta = -0.0665\text{ kt}, p = 5.94 \times 10^{-17}$).
    - Learned Ridge Gate: **6.68 kt $\to$ 6.55 kt** ($\Delta = -0.1336\text{ kt}, p = 1.05 \times 10^{-3}$).
  - All post-hoc improvements are statistically significant under both paired t-tests and Wilcoxon signed-rank tests.

### C. Physical Trajectory Coherence
* Across all 7,295 sequences, **every fusion configuration preserved 0 false dips** (0% rate of non-physical trajectory oscillations). The multi-horizon projection remains strictly monotonic and smooth.

### D. Operational Conclusion
Dedicated binary RI classification probability **does** contain valuable, non-redundant predictive signal that continuous regression alone partially compresses. A post-hoc learned gating model effectively acts as an adaptive variance expander during rare explosive deepening events, reducing RI under-forecasting by nearly 40% while preserving trajectory monotonicity.

