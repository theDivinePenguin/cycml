"""Generate additional forensic audit plots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
PLOTS_DIR = repo_root / "experiments" / "ri_stress_test" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

train_seq = pd.read_csv("data/metadata/forecast_train_sequences_k7.csv")
val_seq = pd.read_csv("data/metadata/forecast_val_sequences_k7.csv")
test_seq = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")
train_ext = pd.read_csv("experiments/ri_stress_test/results/phase10_train_extreme_fits.csv")
ultra_test = pd.read_csv("experiments/ri_target_loss/results/exp2_delta_1_6_12/test_predictions.csv")

for df in [train_seq, val_seq, test_seq]:
    df["delta_v24"] = df["vmax_plus_24h"] - df["vmax_curr"]
ultra_test["act_dv24"] = ultra_test["vmax_plus_24h"] - ultra_test["vmax_curr"]

# -------------------------------------------------------------------------
# PLOT 1: TARGET TAIL DISTRIBUTION (LOG SCALE & CDF)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

bins = np.linspace(-80, 105, 75)
axes[0].hist(train_seq["delta_v24"], bins=bins, density=True, alpha=0.5, label=f"Train (N={len(train_seq):,})", color="#2563eb")
axes[0].hist(test_seq["delta_v24"], bins=bins, density=True, alpha=0.5, label=f"Test (N={len(test_seq):,})", color="#10b981")
axes[0].set_yscale("log")
axes[0].axvline(30, color="orange", linestyle="--", lw=1.8, label="RI Threshold (+30 kt)")
axes[0].axvline(45, color="purple", linestyle="--", lw=1.8, label="Severe RI (+45 kt)")
axes[0].axvline(60, color="red", linestyle="--", lw=1.8, label="Extreme RI (+60 kt)")
axes[0].set_xlabel("Ground Truth ΔV24 (kt)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Log Density", fontsize=12, fontweight="bold")
axes[0].set_title("Ground Truth ΔV24 Distribution (Log Density)", fontsize=14, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, linestyle="--", alpha=0.4)

# Tail zoomed in CDF (> 20 kt)
tail_vals = np.linspace(20, 105, 150)
train_tail = np.array([(train_seq["delta_v24"] >= v).mean() * 100 for v in tail_vals])
test_tail = np.array([(test_seq["delta_v24"] >= v).mean() * 100 for v in tail_vals])

axes[1].plot(tail_vals, train_tail, lw=2.5, label="Train P(ΔV24 >= x)", color="#2563eb")
axes[1].plot(tail_vals, test_tail, lw=2.5, label="Test P(ΔV24 >= x)", color="#10b981")
axes[1].axvline(45, color="purple", linestyle="--", lw=1.5, label="Severe RI (Train: 2.03%)")
axes[1].axvline(60, color="red", linestyle="--", lw=1.5, label="Extreme RI (Train: 0.61%)")
axes[1].axvline(75, color="darkred", linestyle="--", lw=1.5, label="Super RI (Train: 0.10%)")
axes[1].set_yscale("log")
axes[1].set_xlabel("Threshold x (kt)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Exceedance Probability (%) [Log Scale]", fontsize=12, fontweight="bold")
axes[1].set_title("Extreme Positive Tail Scarcity (Train vs Test)", fontsize=14, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "target_tail_distribution.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# PLOT 2: LOSS GRADIENT COMPARISON (HUBER vs MSE)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

e = np.linspace(-80, 80, 500)
beta = 1.0

# Loss functions
loss_mse = 0.5 * e**2
loss_huber = np.where(np.abs(e) <= beta, 0.5 * (e**2) / beta, np.abs(e) - 0.5 * beta)

axes[0].plot(e, loss_huber, lw=2.5, label="Smooth L1 / Huber (beta=1.0)", color="#ef4444")
axes[0].plot(e, np.clip(loss_mse, 0, 100), lw=2.0, linestyle="--", label="MSE (clipped to 100 for view)", color="#2563eb")
axes[0].axvline(30, color="orange", linestyle=":", label="RI Error (+30 kt)")
axes[0].axvline(60, color="darkred", linestyle=":", label="Extreme RI Error (+60 kt)")
axes[0].set_xlabel("Residual e = y_pred - y_true (kt)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Loss Magnitude", fontsize=12, fontweight="bold")
axes[0].set_title("Loss Profile: Huber (Linear Tail) vs MSE (Quadratic Tail)", fontsize=14, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, linestyle="--", alpha=0.4)

# Gradient functions: dL/d(y_pred)
grad_mse = e
grad_huber = np.where(np.abs(e) <= beta, e / beta, np.sign(e))

axes[1].plot(e, grad_huber, lw=2.8, label="Huber Gradient (Saturates at ±1.0)", color="#ef4444")
axes[1].plot(e, np.clip(grad_mse, -15, 15), lw=2.0, linestyle="--", label="MSE Gradient (Linear in error, clip ±15)", color="#2563eb")
axes[1].axhline(1.0, color="black", linestyle=":", alpha=0.5)
axes[1].axhline(-1.0, color="black", linestyle=":", alpha=0.5)
axes[1].set_xlabel("Residual e = y_pred - y_true (kt)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Gradient dL/d(y_pred)", fontsize=12, fontweight="bold")
axes[1].set_title("Gradient Saturation: Why Huber Cannot Penalize Large Residuals", fontsize=14, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "loss_gradient_comparison.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# PLOT 3: TRAINING VS TEST EXTREME RI SCATTER (PROOF OF CAPACITY CEILING)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

# Test extreme points
test_ext = ultra_test[ultra_test["act_dv24"] >= 45]
ax.scatter(train_ext["actual_dv24"], train_ext["pred_dv24"], color="#ef4444", alpha=0.6, s=35, label=f"Train Extremes (N=738, Max={train_ext['pred_dv24'].max():.2f} kt)")
ax.scatter(test_ext["act_dv24"], test_ext["pred_delta_24h"], color="#10b981", alpha=0.7, s=40, marker="^", label=f"Test Extremes (N=203, Max={test_ext['pred_delta_24h'].max():.2f} kt)")

ref = np.linspace(40, 105, 100)
ax.plot(ref, ref, "k--", lw=2, label="Perfect Forecast (y = x)")
ax.axhline(53.44, color="red", linestyle=":", lw=2, label="Train Maximum Ceiling (+53.44 kt)")
ax.axhline(45.94, color="green", linestyle=":", lw=2, label="Test Maximum Ceiling (+45.94 kt)")

ax.set_xlabel("Ground Truth ΔV24 (kt)", fontsize=13, fontweight="bold")
ax.set_ylabel("Predicted ΔV24 (kt)", fontsize=13, fontweight="bold")
ax.set_title("Proof of Optimization/Loss Ceiling: Train vs Test Extremes (>= +45 kt)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "training_vs_test_extreme_scatter.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# PLOT 4: VCURR STRATIFIED CEILING
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7))

sns.boxplot(
    data=ultra_test,
    x=pd.cut(ultra_test["vmax_curr"], bins=[0, 34, 64, 96, 180], labels=["TD (<34kt)", "TS (34-63kt)", "Cat 1-2 (64-95kt)", "Cat 3-5 (>=96kt)"]),
    y="pred_delta_24h",
    palette="viridis",
    ax=ax,
    boxprops=dict(alpha=0.7),
)

ax.axhline(0, color="gray", linestyle="-", lw=1.2)
ax.axhline(30, color="orange", linestyle="--", lw=1.8, label="RI Threshold (+30 kt)")
ax.axhline(45.94, color="red", linestyle=":", lw=2.0, label="Empirical Ceiling (+45.94 kt)")

ax.set_xlabel("Current Intensity Vcurr Stage", fontsize=12, fontweight="bold")
ax.set_ylabel("Predicted ΔV24 (kt)", fontsize=12, fontweight="bold")
ax.set_title("Predicted ΔV24 Distribution Across Storm Intensity Stages", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "vcurr_stratified_ceiling.png", dpi=300)
plt.close()

print("[PLOTS COMPLETE] Generated 4 diagnostic figures in experiments/ri_stress_test/plots/")
