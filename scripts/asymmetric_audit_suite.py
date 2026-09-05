"""Asymmetric Strengthening vs Weakening Forensic Audit Suite."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

repo_root = Path(__file__).resolve().parents[1]
RESULTS_DIR = repo_root / "experiments" / "ri_stress_test" / "results"
PLOTS_DIR = repo_root / "experiments" / "ri_stress_test" / "plots"

# Load test sequences and predictions
test_seq = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")
ultra_pred = pd.read_csv("experiments/ri_target_loss/results/exp2_delta_1_6_12/test_predictions.csv")

# Ensure alignment
assert len(test_seq) == len(ultra_pred), "Length mismatch!"
merged_test = pd.concat([test_seq, ultra_pred[['pred_delta_6h', 'pred_delta_12h', 'pred_delta_24h', 'recon_plus_6h', 'recon_plus_12h', 'recon_plus_24h', 'pred_trend', 'pred_ri_prob']]], axis=1)

# Calculate ground truth deltas
merged_test['act_dv6'] = merged_test['vmax_plus_6h'] - merged_test['vmax_curr']
merged_test['act_dv12'] = merged_test['vmax_plus_12h'] - merged_test['vmax_curr']
merged_test['act_dv24'] = merged_test['vmax_plus_24h'] - merged_test['vmax_curr']

print("="*80)
print("PART 1: DIRECTIONAL STATISTICS (STRENGTHENING VS WEAKENING)")
print("="*80)

directional_results = []
for h, act_col, pred_col in [(6, 'act_dv6', 'pred_delta_6h'), (12, 'act_dv12', 'pred_delta_12h'), (24, 'act_dv24', 'pred_delta_24h')]:
    # Strengthening: act > 0
    str_mask = merged_test[act_col] > 0
    weak_mask = merged_test[act_col] < 0
    flat_mask = merged_test[act_col] == 0

    # Strengthening stats
    act_str = merged_test.loc[str_mask, act_col].values
    pred_str = merged_test.loc[str_mask, pred_col].values
    slope_str, int_str, r_str, p_str, _ = stats.linregress(act_str, pred_str)
    mae_str = np.mean(np.abs(pred_str - act_str))
    bias_str = np.mean(pred_str - act_str)

    # Weakening stats
    act_weak = merged_test.loc[weak_mask, act_col].values
    pred_weak = merged_test.loc[weak_mask, pred_col].values
    slope_weak, int_weak, r_weak, p_weak, _ = stats.linregress(act_weak, pred_weak)
    mae_weak = np.mean(np.abs(pred_weak - act_weak))
    bias_weak = np.mean(pred_weak - act_weak)

    directional_results.extend([
        {
            "Horizon": f"+{h}h",
            "Regime": "Strengthening (ΔV > 0)",
            "N": len(act_str),
            "Mean_Actual_ΔV": float(np.mean(act_str)),
            "Mean_Pred_ΔV": float(np.mean(pred_str)),
            "Median_Pred_ΔV": float(np.median(pred_str)),
            "MAE": float(mae_str),
            "Bias": float(bias_str),
            "Slope": float(slope_str),
            "Intercept": float(int_str),
            "Correlation_r": float(r_str),
        },
        {
            "Horizon": f"+{h}h",
            "Regime": "Weakening (ΔV < 0)",
            "N": len(act_weak),
            "Mean_Actual_ΔV": float(np.mean(act_weak)),
            "Mean_Pred_ΔV": float(np.mean(pred_weak)),
            "Median_Pred_ΔV": float(np.median(pred_weak)),
            "MAE": float(mae_weak),
            "Bias": float(bias_weak),
            "Slope": float(slope_weak),
            "Intercept": float(int_weak),
            "Correlation_r": float(r_weak),
        }
    ])

dir_df = pd.DataFrame(directional_results)
dir_df.to_csv(RESULTS_DIR / "audit_directional_statistics.csv", index=False)
print(dir_df.to_string(index=False))

print("\n" + "="*80)
print("PART 2: MAGNITUDE CALIBRATION (FINE-GRAINED BINS)")
print("="*80)

mag_bins = [
    ("< -30", lambda x: x < -30),
    ("-30 to -15", lambda x: (x >= -30) & (x < -15)),
    ("-15 to 0", lambda x: (x >= -15) & (x < 0)),
    ("0", lambda x: x == 0),
    ("0 to +15", lambda x: (x > 0) & (x <= 15)),
    ("+15 to +30", lambda x: (x > 15) & (x <= 30)),
    ("+30 to +45", lambda x: (x > 30) & (x <= 45)),
    ("+45 to +60", lambda x: (x > 45) & (x <= 60)),
    ("+60 to +75", lambda x: (x > 60) & (x <= 75)),
    ("> +75", lambda x: x > 75),
]

mag_rows = []
for b_name, b_fn in mag_bins:
    sub = merged_test[b_fn(merged_test['act_dv24'])]
    act = sub['act_dv24'].values
    pred = sub['pred_delta_24h'].values
    # Persistence ΔV = 0
    mae_pers = np.mean(np.abs(act))
    mae_pred = np.mean(np.abs(pred - act))
    bias_pred = np.mean(pred - act)

    mag_rows.append({
        "Bin": b_name,
        "N": len(sub),
        "Mean_Actual_ΔV24": float(np.mean(act)),
        "Mean_Pred_ΔV24": float(np.mean(pred)),
        "Median_Pred_ΔV24": float(np.median(pred)),
        "Std_Pred_ΔV24": float(np.std(pred)),
        "MAE_Model": float(mae_pred),
        "MAE_Persistence": float(mae_pers),
        "Bias_Signed": float(bias_pred),
        "Compression_Ratio": float(np.mean(pred) / np.mean(act)) if abs(np.mean(act)) > 1e-3 else np.nan,
    })

mag_df = pd.DataFrame(mag_rows)
mag_df.to_csv(RESULTS_DIR / "audit_magnitude_calibration.csv", index=False)
print(mag_df.to_string(index=False))

print("\n" + "="*80)
print("PART 3: REGRESSION ASYMMETRY (ACTUAL > 0 VS ACTUAL < 0)")
print("="*80)

reg_rows = []
for regime_name, mask in [("Actual ΔV24 > 0 (Intensification)", merged_test['act_dv24'] > 0),
                           ("Actual ΔV24 < 0 (Weakening)", merged_test['act_dv24'] < 0)]:
    sub = merged_test[mask]
    act = sub['act_dv24'].values
    pred = sub['pred_delta_24h'].values

    p_r, p_p = stats.pearsonr(act, pred)
    s_r, s_p = stats.spearmanr(act, pred)
    slope, intercept, _, _, stderr = stats.linregress(act, pred)
    mae = np.mean(np.abs(pred - act))
    rmse = np.sqrt(np.mean((pred - act)**2))

    reg_rows.append({
        "Regime": regime_name,
        "N": len(sub),
        "Pearson_r": float(p_r),
        "Spearman_rho": float(s_r),
        "Regression_Slope": float(slope),
        "Regression_Intercept": float(intercept),
        "Slope_StdErr": float(stderr),
        "MAE": float(mae),
        "RMSE": float(rmse),
    })

reg_df = pd.DataFrame(reg_rows)
reg_df.to_csv(RESULTS_DIR / "audit_regression_asymmetry.csv", index=False)
print(reg_df.to_string(index=False))

print("\n" + "="*80)
print("PART 4: TRAJECTORY SHAPE & ACCELERATION AUDIT")
print("="*80)

# Forecast trajectory across +6h, +12h, +24h
# Compute slope over 24h: (ΔV24 - 0) / 24
pred_slopes = merged_test['pred_delta_24h'].values / 24.0
act_slopes = merged_test['act_dv24'].values / 24.0

# Acceleration / non-linearity:
# (ΔV24 - ΔV12) / 12h - (ΔV12 - ΔV6) / 6h
pred_acc = ((merged_test['pred_delta_24h'] - merged_test['pred_delta_12h']) / 12.0) - ((merged_test['pred_delta_12h'] - merged_test['pred_delta_6h']) / 6.0)
act_acc = ((merged_test['act_dv24'] - merged_test['act_dv12']) / 12.0) - ((merged_test['act_dv12'] - merged_test['act_dv6']) / 6.0)

# Trajectory range: max(0, ΔV6, ΔV12, ΔV24) - min(0, ΔV6, ΔV12, ΔV24)
pred_trajs = np.stack([np.zeros(len(merged_test)), merged_test['pred_delta_6h'].values, merged_test['pred_delta_12h'].values, merged_test['pred_delta_24h'].values], axis=1)
act_trajs = np.stack([np.zeros(len(merged_test)), merged_test['act_dv6'].values, merged_test['act_dv12'].values, merged_test['act_dv24'].values], axis=1)

pred_ranges = np.ptp(pred_trajs, axis=1)
act_ranges = np.ptp(act_trajs, axis=1)
pred_vars = np.var(pred_trajs, axis=1)
act_vars = np.var(act_trajs, axis=1)

traj_summary = {
    "Actual_Mean_Slope (kt/h)": float(np.mean(act_slopes)),
    "Pred_Mean_Slope (kt/h)": float(np.mean(pred_slopes)),
    "Actual_Mean_Acceleration (kt/h^2)": float(np.mean(act_acc)),
    "Pred_Mean_Acceleration (kt/h^2)": float(np.mean(pred_acc)),
    "Actual_Std_Acceleration": float(np.std(act_acc)),
    "Pred_Std_Acceleration": float(np.std(pred_acc)),
    "Actual_Mean_Trajectory_Range (kt)": float(np.mean(act_ranges)),
    "Pred_Mean_Trajectory_Range (kt)": float(np.mean(pred_ranges)),
    "Actual_Mean_Trajectory_Variance": float(np.mean(act_vars)),
    "Pred_Mean_Trajectory_Variance": float(np.mean(pred_vars)),
    "Variance_Suppression_Factor": float(np.mean(pred_vars) / np.mean(act_vars)),
}

for k, v in traj_summary.items():
    print(f"  {k:38s}: {v:.4f}")

pd.DataFrame([traj_summary]).to_csv(RESULTS_DIR / "audit_trajectory_shape.csv", index=False)

print("\n" + "="*80)
print("PART 5: EXTREME INTENSIFICATION VS EXTREME WEAKENING TAIL")
print("="*80)

tail_rows = []
for thresh in [30, 45, 60, 75]:
    # Positive tail: act >= thresh
    p_sub = merged_test[merged_test['act_dv24'] >= thresh]
    act_p = p_sub['act_dv24'].values
    pred_p = p_sub['pred_delta_24h'].values

    # Negative tail: act <= -thresh
    n_sub = merged_test[merged_test['act_dv24'] <= -thresh]
    act_n = n_sub['act_dv24'].values
    pred_n = n_sub['pred_delta_24h'].values

    tail_rows.append({
        "Threshold": f"±{thresh} kt",
        "Pos_N": len(p_sub),
        "Pos_Mean_Actual": float(np.mean(act_p)),
        "Pos_Mean_Pred": float(np.mean(pred_p)),
        "Pos_Median_Pred": float(np.median(pred_p)),
        "Pos_Max_Pred": float(np.max(pred_p)),
        "Pos_Bias": float(np.mean(pred_p - act_p)),
        "Pos_Pct_Pred_ge_30": float((pred_p >= 30).sum() / len(p_sub) * 100),
        "Pos_Pct_Pred_ge_45": float((pred_p >= 45).sum() / len(p_sub) * 100),
        "Pos_Pct_Pred_ge_60": float((pred_p >= 60).sum() / len(p_sub) * 100),
        "Neg_N": len(n_sub),
        "Neg_Mean_Actual": float(np.mean(act_n)),
        "Neg_Mean_Pred": float(np.mean(pred_n)),
        "Neg_Median_Pred": float(np.median(pred_n)),
        "Neg_Min_Pred": float(np.min(pred_n)),
        "Neg_Bias": float(np.mean(pred_n - act_n)),
        "Neg_Pct_Pred_le_minus30": float((pred_n <= -30).sum() / len(n_sub) * 100),
        "Neg_Pct_Pred_le_minus45": float((pred_n <= -45).sum() / len(n_sub) * 100),
        "Neg_Pct_Pred_le_minus60": float((pred_n <= -60).sum() / len(n_sub) * 100),
    })

tail_df = pd.DataFrame(tail_rows)
tail_df.to_csv(RESULTS_DIR / "audit_extreme_tail_comparison.csv", index=False)
print(tail_df[["Threshold", "Pos_N", "Pos_Mean_Actual", "Pos_Mean_Pred", "Pos_Bias", "Pos_Pct_Pred_ge_30", "Neg_N", "Neg_Mean_Actual", "Neg_Mean_Pred", "Neg_Bias", "Neg_Pct_Pred_le_minus30"]].to_string(index=False))

print("\n" + "="*80)
print("PART 7: TRAINING VS VALIDATION VS TEST ASYMMETRY COMPARISON")
print("="*80)

# Compare positive vs negative slope across splits
# Load train extreme fits from Phase 10
train_ext = pd.read_csv(RESULTS_DIR / "phase10_train_extreme_fits.csv")

split_comp = [
    {
        "Split": "Test (N=7,901)",
        "Strengthening_Slope (ΔV>0)": dir_df.loc[(dir_df['Horizon'] == '+24h') & (dir_df['Regime'].str.contains('Strengthening')), 'Slope'].values[0],
        "Weakening_Slope (ΔV<0)": dir_df.loc[(dir_df['Horizon'] == '+24h') & (dir_df['Regime'].str.contains('Weakening')), 'Slope'].values[0],
        "Extreme_Pos_Mean_Pred (act>=45)": tail_df.loc[tail_df['Threshold'] == '±45 kt', 'Pos_Mean_Pred'].values[0],
        "Extreme_Pos_Bias (act>=45)": tail_df.loc[tail_df['Threshold'] == '±45 kt', 'Pos_Bias'].values[0],
        "Extreme_Neg_Mean_Pred (act<=-45)": tail_df.loc[tail_df['Threshold'] == '±45 kt', 'Neg_Mean_Pred'].values[0],
        "Extreme_Neg_Bias (act<=-45)": tail_df.loc[tail_df['Threshold'] == '±45 kt', 'Neg_Bias'].values[0],
    },
    {
        "Split": "Train (N=738 extremes evaluated)",
        "Strengthening_Slope (ΔV>0)": np.polyfit(train_ext['actual_dv24'], train_ext['pred_dv24'], 1)[0],
        "Weakening_Slope (ΔV<0)": np.nan, # evaluated on pos
        "Extreme_Pos_Mean_Pred (act>=45)": float(train_ext['pred_dv24'].mean()),
        "Extreme_Pos_Bias (act>=45)": float((train_ext['pred_dv24'] - train_ext['actual_dv24']).mean()),
        "Extreme_Neg_Mean_Pred (act<=-45)": np.nan,
        "Extreme_Neg_Bias (act<=-45)": np.nan,
    }
]
print(pd.DataFrame(split_comp).to_string(index=False))

# -------------------------------------------------------------------------
# PLOT: CONDITIONAL BIAS & ASYMMETRIC SATURATION CURVE
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Fine-grained conditional expectation curve
fine_bins = np.linspace(-60, 80, 29)
bin_centers = 0.5 * (fine_bins[:-1] + fine_bins[1:])
cond_means = []
cond_medians = []
counts = []

for i in range(len(fine_bins)-1):
    sub = merged_test[(merged_test['act_dv24'] >= fine_bins[i]) & (merged_test['act_dv24'] < fine_bins[i+1])]
    counts.append(len(sub))
    if len(sub) > 5:
        cond_means.append(np.mean(sub['pred_delta_24h']))
        cond_medians.append(np.median(sub['pred_delta_24h']))
    else:
        cond_means.append(np.nan)
        cond_medians.append(np.nan)

axes[0].plot(bin_centers, cond_means, 'o-', color='#10b981', lw=2.5, label='E[Pred ΔV24 | Actual ΔV24]')
axes[0].plot(bin_centers, cond_medians, 's--', color='#047857', lw=1.8, label='Median Pred ΔV24')
axes[0].plot(bin_centers, bin_centers, 'k--', alpha=0.6, label='Ideal Unbiased (y = x)')
axes[0].axvline(0, color='gray', linestyle=':', alpha=0.5)
axes[0].axhline(0, color='gray', linestyle=':', alpha=0.5)
axes[0].axhline(45.94, color='red', linestyle=':', lw=1.8, label='Max Positive Pred (+45.9 kt)')
axes[0].axhline(-42.06, color='blue', linestyle=':', lw=1.8, label='Min Negative Pred (-42.1 kt)')

axes[0].set_xlabel("Actual Ground Truth ΔV24 (kt)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Predicted ΔV24 (kt)", fontsize=12, fontweight="bold")
axes[0].set_title("Calibration Curve: Severe Positive Tail Flattening", fontsize=14, fontweight="bold")
axes[0].legend(fontsize=10, loc="upper left")
axes[0].grid(True, linestyle="--", alpha=0.4)

# Subplot 2: Trajectory Acceleration Comparison (Actual vs Predicted)
sns.kdeplot(act_acc, ax=axes[1], color='black', lw=2, label=f'Actual Acceleration (Std={np.std(act_acc):.3f})')
sns.kdeplot(pred_acc, ax=axes[1], color='#10b981', lw=2.5, fill=True, alpha=0.3, label=f'Predicted Acceleration (Std={np.std(pred_acc):.3f})')
axes[1].set_xlim(-2.5, 2.5)
axes[1].set_xlabel("Intensity Acceleration d²V/dt² (kt / h²)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Probability Density", fontsize=12, fontweight="bold")
axes[1].set_title("Trajectory Dynamics: Predicted vs Actual Acceleration", fontsize=14, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "asymmetric_bias_trajectory_curve.png", dpi=300)
plt.close()

print(f"\n[DONE] Saved plot to {PLOTS_DIR / 'asymmetric_bias_trajectory_curve.png'}")
