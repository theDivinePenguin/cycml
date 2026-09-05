# Final Locked-Test Scientific Report: Learned RI-Aware Correction Model

**Execution Date**: 2026-09-05 21:13:11 UTC
**Locked Test Manifest**: `data/metadata/forecast_test_sequences_k5_aligned.csv` (N = 6,825 sequences across 171 unique cyclones)
**Status**: Audited Experimental Candidate Evaluation (Frozen Model v1)
**Canonical Test Artifacts**: 100% Frozen & Untouched

## 1. Executive Scientific Verdict

```text
VERDICT: WIN

WIN: Learned RI-Aware Correction significantly outperforms Canonical Hybrid by 0.6509 kt on locked test (95% CI: [-0.7070, -0.5999] kt, p = 1.28e-107).
```

## 2. Locked-Test Benchmark Table

| Model Architecture | Overall MAE | +6h MAE | +12h MAE | +24h MAE | RI +24h MAE | Non-RI +24h MAE | Extreme (>=95kt) | False Dips |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Persistence Baseline** | **8.8628 kt** | 4.12 kt | 7.95 kt | 14.52 kt | 42.35 kt | 12.64 kt | 21.40 kt | 0 |
| **2. Canonical Residual ΔV Forecaster Alone** | **6.9509 kt** | 3.51 kt | 6.39 kt | 10.95 kt | 32.43 kt | 9.51 kt | 14.44 kt | 0 |
| **3. Canonical Final Hybrid** | **6.6350 kt** | 3.46 kt | 6.09 kt | 10.36 kt | 26.37 kt | 9.28 kt | 14.07 kt | 0 |
| **4. Learned RI-Aware Correction v1 (Audited MLP)** | **5.9841 kt** | 2.94 kt | 5.40 kt | 9.61 kt | 23.06 kt | 8.70 kt | 13.29 kt | 0 |

## 3. Statistical Significance vs. Canonical Champion (6.6350 kt)

- **Overall Test MAE**: `5.9841 kt` vs `6.6350 kt` (Δ = `-0.6509 kt`)
- **RI +24h MAE**: `23.06 kt` vs `26.37 kt` (Δ = `-3.31 kt`)
- **Non-RI +24h MAE**: `8.70 kt` vs `9.28 kt` (Δ = `-0.58 kt`)
- **95% Bootstrap Confidence Interval**: `[-0.7070, -0.5999] kt`
- **Bootstrap Win Rate**: `100.0%`
- **Paired t-test p-value**: `1.284e-107`
- **Cyclone Win Ratio**: `138 improved / 32 worsened` across 171 test cyclones

## 4. Methodological Safeguards & Audit Confirmation

- Single evaluation execution on the locked test partition.
- Model weights, scalers, and hyperparameters were 100% frozen prior to test inference.
- Zero future lookahead: all 27 input features were strictly computed from information available at forecast origin $t$.
- Canonical test reports and checkpoints in `experiments/final_locked_test/` remain completely unchanged.
