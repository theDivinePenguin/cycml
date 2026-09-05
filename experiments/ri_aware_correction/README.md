# RI-Aware Correction Model Experiments

This directory contains the experimental artifacts, training pipelines, and validation evaluations for learned RI-aware corrections.

## Architecture & Feature Pipeline
- **Features ($D=27$)**: Base Residual ΔV (+6h, +12h, +24h), Canonical Ridge Hybrid ΔV, Dedicated RI Classifier $P_{RI}$ and logit, current intensity $V_t$, historical $K=5$ trends (6h delta, 12h delta, slope), SHIPS causal environmental variables (SST, OHC, VWS, etc.), and physical interactions ($P_{RI} \times \text{Shear}$, $P_{RI} \times \text{SST}$, etc.).
- **Models Tested**:
  1. Regularized Ridge Correction across $\alpha \in [10, 5000]$
  2. Constrained Small MLP: $\text{scale} \cdot \tanh(\text{MLP}(X))$ across scales $\{5, 10, 15, 20\}\text{ kt}$

## Leakage & Protocol Controls
- **Training**: Trained strictly on `forecast_train_sequences_k5_aligned.csv`.
- **Validation**: Evaluated strictly on `forecast_val_sequences_k5_aligned.csv`.
- **Locked Test Set**: Strictly untouched (zero access).
