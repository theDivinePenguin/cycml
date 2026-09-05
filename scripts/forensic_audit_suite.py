"""Comprehensive Forensic Audit Suite for Tropical Cyclone +24h Intensity Ceiling.

Executes Phases 2, 3, 4, 5, 8, 9, 10, 11, 12 without training any new models.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.trend_config import IntensityTrendConfig
from experiments.ri_target_loss.scripts.dataset import build_delta_dataloaders, DeltaSequenceDataset
from experiments.ri_target_loss.scripts.models import DeltaEnvironmentalTemporalClassifier
from experiments.ri_target_loss.scripts.losses import DeltaJointLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[AUDIT] Running on device: {device}")

RESULTS_DIR = repo_root / "experiments" / "ri_stress_test" / "results"
PLOTS_DIR = repo_root / "experiments" / "ri_stress_test" / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# PHASE 2: TARGET FORENSICS
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("PHASE 2: TARGET DISTRIBUTION FORENSICS (TRAIN / VAL / TEST)")
print("="*80)

train_seq = pd.read_csv("data/metadata/forecast_train_sequences_k7.csv")
val_seq = pd.read_csv("data/metadata/forecast_val_sequences_k7.csv")
test_seq = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")

for df, name in [(train_seq, "train"), (val_seq, "val"), (test_seq, "test")]:
    df["delta_v24"] = df["vmax_plus_24h"] - df["vmax_curr"]
    df["split"] = name

summary_rows = []
for name, df in [("Train", train_seq), ("Validation", val_seq), ("Test", test_seq)]:
    dv = df["delta_v24"]
    summary_rows.append({
        "Split": name,
        "N": len(df),
        "Unique_Cyclones": df["cyclone_id"].nunique(),
        "Mean": float(dv.mean()),
        "Std": float(dv.std()),
        "Min": float(dv.min()),
        "p10": float(dv.quantile(0.10)),
        "p25": float(dv.quantile(0.25)),
        "Median": float(dv.median()),
        "p75": float(dv.quantile(0.75)),
        "p90": float(dv.quantile(0.90)),
        "p95": float(dv.quantile(0.95)),
        "p99": float(dv.quantile(0.99)),
        "p99.5": float(dv.quantile(0.995)),
        "Max": float(dv.max()),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULTS_DIR / "phase2_target_summary.csv", index=False)
print(summary_df.to_string(index=False))

bucket_ranges = [
    ("< -30", lambda x: x < -30),
    ("-30 to -15", lambda x: (x >= -30) & (x < -15)),
    ("-15 to 0", lambda x: (x >= -15) & (x < 0)),
    ("0 to +15", lambda x: (x >= 0) & (x < 15)),
    ("+15 to +30", lambda x: (x >= 15) & (x < 30)),
    ("+30 to +45", lambda x: (x >= 30) & (x < 45)),
    ("+45 to +60", lambda x: (x >= 45) & (x < 60)),
    ("+60 to +75", lambda x: (x >= 60) & (x < 75)),
    ("> +75", lambda x: x >= 75),
]

bucket_rows = []
for b_name, b_fn in bucket_ranges:
    row = {"Bucket": b_name}
    for s_name, df in [("Train", train_seq), ("Val", val_seq), ("Test", test_seq)]:
        mask = b_fn(df["delta_v24"])
        sub = df[mask]
        n_samples = len(sub)
        pct = n_samples / len(df) * 100
        n_cyc = sub["cyclone_id"].nunique()
        row[f"{s_name}_N"] = n_samples
        row[f"{s_name}_Pct"] = f"{pct:.2f}%"
        row[f"{s_name}_Cyclones"] = n_cyc
    bucket_rows.append(row)

bucket_df = pd.DataFrame(bucket_rows)
bucket_df.to_csv(RESULTS_DIR / "phase2_target_buckets.csv", index=False)
print("\nTarget Buckets across Splits:")
print(bucket_df.to_string(index=False))

# Extreme Tail Detailed Stats
meta_df = pd.read_csv("data/metadata/metadata_all_basins.csv")
meta_basins = meta_df.drop_duplicates(subset=["cyclone_id"])[["cyclone_id", "region"]].set_index("cyclone_id")["region"].to_dict()

for name, df in [("Train", train_seq), ("Val", val_seq), ("Test", test_seq)]:
    df["basin"] = df["cyclone_id"].map(meta_basins)
    df["year"] = df["target_t_timestamp"].astype(str).str[:4]
    ext_45 = df[df["delta_v24"] >= 45]
    ext_60 = df[df["delta_v24"] >= 60]
    ext_75 = df[df["delta_v24"] >= 75]
    print(f"\n{name} Extreme Counts:")
    print(f"  >= +45 kt: {len(ext_45):4d} ({len(ext_45)/len(df)*100:.2f}%) in {ext_45['cyclone_id'].nunique():3d} cyclones | Basins: {ext_45['basin'].value_counts().to_dict()}")
    print(f"  >= +60 kt: {len(ext_60):4d} ({len(ext_60)/len(df)*100:.2f}%) in {ext_60['cyclone_id'].nunique():3d} cyclones | Basins: {ext_60['basin'].value_counts().to_dict()}")
    print(f"  >= +75 kt: {len(ext_75):4d} ({len(ext_75)/len(df)*100:.2f}%) in {ext_75['cyclone_id'].nunique():3d} cyclones | Basins: {ext_75['basin'].value_counts().to_dict()}")

# -------------------------------------------------------------------------
# PHASE 3: CHECK FOR ARTIFICIAL OUTPUT CEILING
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("PHASE 3: CHECK FOR ARTIFICIAL OUTPUT CEILING")
print("="*80)

ultra_ckpt_path = "experiments/ri_target_loss/checkpoints/exp2_delta_1_6_12/best.pt"
ckpt = torch.load(ultra_ckpt_path, map_location="cpu")
state_dict = ckpt["model_state_dict"]

# Inspect head_delta weights
w0 = state_dict["head_delta.0.weight"]
b0 = state_dict["head_delta.0.bias"]
w3 = state_dict["head_delta.3.weight"]
b3 = state_dict["head_delta.3.bias"]

print("Model Head Architecture: head_delta")
print(f"  Layer 0: Linear(256, 128) | Weight norm: {torch.norm(w0):.4f}, Bias norm: {torch.norm(b0):.4f}")
print(f"  Layer 1: ReLU(inplace=True)")
print(f"  Layer 2: Dropout(p=0.1)")
print(f"  Layer 3: Linear(128, 3)   | Weight norm: {torch.norm(w3):.4f}")
print(f"  Layer 3 Biases: +6h={b3[0]:.4f}, +12h={b3[1]:.4f}, +24h={b3[2]:.4f}")

# Verification of raw output vs CSV
ultra_test_df = pd.read_csv("experiments/ri_target_loss/results/exp2_delta_1_6_12/test_predictions.csv")
print("\nUltra Test CSV Predictions Verification:")
print(f"  Total test predictions: {len(ultra_test_df)}")
print(f"  pred_delta_24h max: {ultra_test_df['pred_delta_24h'].max():.4f} kt")
print(f"  pred_delta_24h min: {ultra_test_df['pred_delta_24h'].min():.4f} kt")
print(f"  pred_delta_24h median: {ultra_test_df['pred_delta_24h'].median():.4f} kt")
print(f"  pred_delta_24h p90: {ultra_test_df['pred_delta_24h'].quantile(0.90):.4f} kt")
print(f"  pred_delta_24h p99: {ultra_test_df['pred_delta_24h'].quantile(0.99):.4f} kt")
print(f"  Number of predictions >= +45 kt: {(ultra_test_df['pred_delta_24h'] >= 45).sum()}")
print(f"  Number of predictions >= +50 kt: {(ultra_test_df['pred_delta_24h'] >= 50).sum()}")
print(f"  Number of predictions >= +60 kt: {(ultra_test_df['pred_delta_24h'] >= 60).sum()}")

# -------------------------------------------------------------------------
# PHASE 4: LOSS FORENSICS & BATCH INSTRUMENTATION
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("PHASE 4: LOSS FORENSICS & BATCH GRADIENT INSTRUMENTATION")
print("="*80)

# Load data loaders for batch testing
with open("data/metadata/normalization_stats_multichannel.json") as f:
    norm_stats = json.load(f)
norm_mean = [norm_stats["mean"][c] for c in [0, 1, 2]]
norm_std = [norm_stats["std"][c] for c in [0, 1, 2]]

env_cache = torch.load("data/metadata/environmental_features_k7.pt")
train_env = env_cache["train"]
val_env = env_cache["val"]
test_env = env_cache["test"]

config = IntensityTrendConfig()
train_loader, val_loader, test_loader = build_delta_dataloaders(
    train_seq_df=train_seq,
    val_seq_df=val_seq,
    test_seq_df=test_seq,
    mean=norm_mean,
    std=norm_std,
    channels=[0, 1, 2],
    batch_size=32,
    num_workers=2,
    config=config,
    train_env_tensor=train_env,
    val_env_tensor=val_env,
    test_env_tensor=test_env,
)

# Load Ultra model onto device
model = DeltaEnvironmentalTemporalClassifier(
    mode="delta_only",
    channels=3,
    num_frames=7,
    d_model=256,
    n_heads=8,
    num_layers=2,
    dropout=0.1,
    use_vis_channel=True,
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Let's find a batch with extreme RI samples from train set
extreme_indices = train_seq[train_seq["delta_v24"] >= 35].index.tolist()
normal_indices = train_seq[(train_seq["delta_v24"] >= -10) & (train_seq["delta_v24"] <= 10)].index.tolist()

# Construct a synthetic controlled batch: 16 normal + 16 extreme
np.random.seed(42)
sampled_ext = np.random.choice(extreme_indices, 16, replace=False)
sampled_norm = np.random.choice(normal_indices, 16, replace=False)
controlled_indices = list(sampled_ext) + list(sampled_norm)

controlled_ds = DeltaSequenceDataset(
    seq_df=train_seq.iloc[controlled_indices],
    mean=norm_mean,
    std=norm_std,
    channels=[0, 1, 2],
    is_training=False,
    config=config,
    env_tensor=train_env[controlled_indices],
)
controlled_loader = DataLoader(controlled_ds, batch_size=32, shuffle=False)

for batch in controlled_loader:
    images, vis_masks, trend_targets, ri_targets, reg_abs_targets, reg_delta_targets, env_vec, meta = batch
    images = images.to(device)
    vis_masks = vis_masks.to(device)
    reg_delta_targets = reg_delta_targets.to(device)
    env_vec = env_vec.to(device)
    break

model.train()
# Forward pass
ri_logits, trend_logits, reg_delta_preds = model(images, vis_masks, env_vec)

# Let's measure gradients per horizon
loss_fn_ultra = DeltaJointLoss(
    mode="delta_only",
    ri_weights=(1.0, 6.0, 12.0),
    huber_beta=1.0,
    lambda_reg_delta=0.1,
)

# Individual horizon Huber losses
huber_elem = F.smooth_l1_loss(reg_delta_preds, reg_delta_targets, beta=1.0, reduction="none") # (32, 3)
residuals = (reg_delta_preds - reg_delta_targets).detach().cpu().numpy()
targets_np = reg_delta_targets.detach().cpu().numpy()

# Calculate gradient norm for each horizon individually
grad_norms_per_horizon = []
for h in range(3):
    model.zero_grad()
    h_loss = huber_elem[:, h].mean()
    h_loss.backward(retain_graph=True)
    # Norm of head_delta.3 weights grad
    gnorm = model.head_delta[3].weight.grad.norm().item()
    grad_norms_per_horizon.append(gnorm)

# Now calculate weighted horizon 2 (+24h)
model.zero_grad()
actual_dv24 = reg_delta_targets[:, 2]
sample_weights = torch.ones_like(actual_dv24) * 1.0
sample_weights[(actual_dv24 >= 15.0) & (actual_dv24 < 30.0)] = 6.0
sample_weights[actual_dv24 >= 30.0] = 12.0
weighted_huber_24 = (huber_elem[:, 2] * sample_weights).mean()
weighted_huber_24.backward(retain_graph=True)
gnorm_w24 = model.head_delta[3].weight.grad.norm().item()

# Now compare extreme vs normal sample gradient contributions for Horizon 2
# Extreme samples are 0..15, Normal are 16..31
model.zero_grad()
ext_loss = (huber_elem[:16, 2] * sample_weights[:16]).mean()
ext_loss.backward(retain_graph=True)
gnorm_ext = model.head_delta[3].weight.grad.norm().item()

model.zero_grad()
norm_loss = (huber_elem[16:, 2] * sample_weights[16:]).mean()
norm_loss.backward(retain_graph=True)
gnorm_norm = model.head_delta[3].weight.grad.norm().item()

# Contrast with MSE Loss
model.zero_grad()
mse_elem = 0.5 * (reg_delta_preds - reg_delta_targets)**2
mse_ext = (mse_elem[:16, 2] * sample_weights[:16]).mean()
mse_ext.backward(retain_graph=True)
gnorm_mse_ext = model.head_delta[3].weight.grad.norm().item()

model.zero_grad()
mse_norm = (mse_elem[16:, 2] * sample_weights[16:]).mean()
mse_norm.backward(retain_graph=True)
gnorm_mse_norm = model.head_delta[3].weight.grad.norm().item()

loss_audit_results = {
    "MAE_h6": float(np.abs(residuals[:, 0]).mean()),
    "MAE_h12": float(np.abs(residuals[:, 1]).mean()),
    "MAE_h24": float(np.abs(residuals[:, 2]).mean()),
    "Huber_Loss_h6": float(huber_elem[:, 0].mean().item()),
    "Huber_Loss_h12": float(huber_elem[:, 1].mean().item()),
    "Huber_Loss_h24_unweighted": float(huber_elem[:, 2].mean().item()),
    "Huber_Loss_h24_weighted": float(weighted_huber_24.item()),
    "GradNorm_h6": grad_norms_per_horizon[0],
    "GradNorm_h12": grad_norms_per_horizon[1],
    "GradNorm_h24_unweighted": grad_norms_per_horizon[2],
    "GradNorm_h24_weighted": gnorm_w24,
    "Huber_GradNorm_Extreme16": gnorm_ext,
    "Huber_GradNorm_Normal16": gnorm_norm,
    "Huber_Ratio_Extreme_to_Normal": gnorm_ext / max(1e-6, gnorm_norm),
    "MSE_GradNorm_Extreme16": gnorm_mse_ext,
    "MSE_GradNorm_Normal16": gnorm_mse_norm,
    "MSE_Ratio_Extreme_to_Normal": gnorm_mse_ext / max(1e-6, gnorm_mse_norm),
}

print("\nLoss & Gradient Audit Results:")
for k, v in loss_audit_results.items():
    print(f"  {k:30s}: {v:.4f}")

pd.DataFrame([loss_audit_results]).to_csv(RESULTS_DIR / "phase4_loss_gradients.csv", index=False)

# -------------------------------------------------------------------------
# PHASE 5: EXTREME RI CONDITIONAL ANALYSIS (E[pred | act bucket])
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("PHASE 5: CONDITIONAL EXPECTATION ANALYSIS (E[pred | act bucket])")
print("="*80)

models_eval = {
    "Baseline": "experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_predictions.csv",
    "Moderate (1/2/4)": "experiments/ri_target_loss/results/exp2_delta_moderate/test_predictions.csv",
    "Ultra (1/6/12)": "experiments/ri_target_loss/results/exp2_delta_1_6_12/test_predictions.csv",
    "Extreme (1/10/20)": "experiments/ri_target_loss/results/exp2_delta_1_10_20/test_predictions.csv",
}

phase5_buckets = [
    ("< 15", lambda x: x < 15),
    ("15 to 30", lambda x: (x >= 15) & (x < 30)),
    ("30 to 45", lambda x: (x >= 30) & (x < 45)),
    ("45 to 60", lambda x: (x >= 45) & (x < 60)),
    ("60 to 75", lambda x: (x >= 60) & (x < 75)),
    ("> 75", lambda x: x >= 75),
]

cond_rows = []
model_dfs = {}

for m_name, path in models_eval.items():
    df = pd.read_csv(path)
    if "pred_delta_24h" in df.columns:
        df["pred_dv24"] = df["pred_delta_24h"]
    elif "pred_plus_24h" in df.columns:
        df["pred_dv24"] = df["pred_plus_24h"] - df["vmax_curr"]
    df["act_dv24"] = df["vmax_plus_24h"] - df["vmax_curr"]
    model_dfs[m_name] = df

    for b_name, b_fn in phase5_buckets:
        sub = df[b_fn(df["act_dv24"])]
        if len(sub) == 0:
            continue
        act = sub["act_dv24"].values
        pred = sub["pred_dv24"].values
        mae = np.mean(np.abs(pred - act))
        bias = np.mean(pred - act)
        slope = np.polyfit(act, pred, deg=1)[0] if len(sub) > 1 and np.std(act) > 1e-4 else np.nan

        cond_rows.append({
            "Model": m_name,
            "Bucket": b_name,
            "N": len(sub),
            "Mean_Actual": float(np.mean(act)),
            "Mean_Predicted": float(np.mean(pred)),
            "Median_Predicted": float(np.median(pred)),
            "Std_Predicted": float(np.std(pred)),
            "Min_Predicted": float(np.min(pred)),
            "Max_Predicted": float(np.max(pred)),
            "MAE": float(mae),
            "Bias": float(bias),
            "Slope": float(slope),
        })

cond_df = pd.DataFrame(cond_rows)
cond_df.to_csv(RESULTS_DIR / "phase5_conditional_expectation.csv", index=False)
print("\nConditional Expectation Table (E[pred | act]):")
print(cond_df[["Model", "Bucket", "N", "Mean_Actual", "Mean_Predicted", "Max_Predicted", "MAE", "Bias"]].to_string(index=False))

# Plotting Conditional Expectation and Saturation
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Subplot 1: E[pred | actual] Curves
palette = {"Baseline": "#555555", "Moderate (1/2/4)": "#3b82f6", "Ultra (1/6/12)": "#10b981", "Extreme (1/10/20)": "#ef4444"}

for m_name in models_eval.keys():
    m_data = cond_df[cond_df["Model"] == m_name]
    axes[0].plot(m_data["Mean_Actual"], m_data["Mean_Predicted"], marker="o", lw=2.5, label=m_name, color=palette[m_name])

act_ref = np.linspace(-40, 85, 100)
axes[0].plot(act_ref, act_ref, "k--", alpha=0.6, label="Ideal (y = x)")
axes[0].axhline(46.0, color="red", linestyle=":", lw=1.8, label="Empirical Ceiling (~46 kt)")
axes[0].set_xlabel("Actual Ground Truth ΔV24 (kt)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Conditional Mean E[Predicted ΔV24 | Actual] (kt)", fontsize=12, fontweight="bold")
axes[0].set_title("Conditional Prediction Saturation Curves", fontsize=14, fontweight="bold")
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(fontsize=10, loc="upper left")

# Subplot 2: Scatter / Saturation on Test Set for Ultra
ultra_df = model_dfs["Ultra (1/6/12)"]
sns.scatterplot(
    data=ultra_df,
    x="act_dv24",
    y="pred_dv24",
    ax=axes[1],
    alpha=0.25,
    color="#10b981",
    s=18,
    label="Ultra Test Points (N=7,901)",
)
axes[1].plot(act_ref, act_ref, "k--", alpha=0.7, label="Ideal (y = x)")
axes[1].axhline(ultra_df["pred_dv24"].max(), color="red", linestyle=":", lw=2.0, label=f"Max Pred = {ultra_df['pred_dv24'].max():.2f} kt")
axes[1].axvline(30.0, color="orange", linestyle="--", alpha=0.7, label="RI Threshold (+30 kt)")

# Fit overall linear regression
p_slope, p_int = np.polyfit(ultra_df["act_dv24"], ultra_df["pred_dv24"], deg=1)
axes[1].plot(act_ref, p_slope * act_ref + p_int, color="#047857", lw=2.2, label=f"Linear Fit (Slope={p_slope:.3f})")

axes[1].set_xlabel("Actual Ground Truth ΔV24 (kt)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Predicted ΔV24 (kt)", fontsize=12, fontweight="bold")
axes[1].set_title("Ultra (1/6/12): Actual vs Predicted ΔV24 Scatter", fontsize=14, fontweight="bold")
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend(fontsize=10, loc="upper left")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "conditional_expectation_saturation.png", dpi=300)
plt.close()
print(f"Saved plot: {PLOTS_DIR / 'conditional_expectation_saturation.png'}")

# -------------------------------------------------------------------------
# PHASE 8 & 9: SENSITIVITY DIAGNOSTICS ON ULTRA
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("PHASE 8 & 9: LIGHTWEIGHT INPUT SENSITIVITY TESTS")
print("="*80)

# Evaluate on first 500 test samples to measure sensitivity without excessive runtime
sens_sub_df = test_seq.iloc[:500].copy()
sens_env = test_env[:500]

sens_ds = DeltaSequenceDataset(
    seq_df=sens_sub_df,
    mean=norm_mean,
    std=norm_std,
    channels=[0, 1, 2],
    is_training=False,
    config=config,
    env_tensor=sens_env,
)
sens_loader = DataLoader(sens_ds, batch_size=32, shuffle=False)

model.eval()

clean_preds = []
rev_preds = []
static_preds = []
zero_env_preds = []
zero_vmax_preds = []
zero_vis_preds = []

with torch.no_grad():
    for batch in sens_loader:
        images, vis_masks, _, _, _, _, env_vec, _ = batch
        images = images.to(device)
        vis_masks = vis_masks.to(device)
        env_vec = env_vec.to(device)

        # 1. Clean
        _, _, p_clean = model(images, vis_masks, env_vec)
        clean_preds.append(p_clean[:, 2].cpu().numpy())

        # 2. Reverse Temporal Order
        images_rev = torch.flip(images, dims=[1]) # flip along K dimension
        vis_rev = torch.flip(vis_masks, dims=[1])
        _, _, p_rev = model(images_rev, vis_rev, env_vec)
        rev_preds.append(p_rev[:, 2].cpu().numpy())

        # 3. Static Latest Frame (repeat frame 6 all 7 times)
        images_static = images[:, -1:, :, :, :].repeat(1, 7, 1, 1, 1)
        vis_static = vis_masks[:, -1:].repeat(1, 7)
        _, _, p_static = model(images_static, vis_static, env_vec)
        static_preds.append(p_static[:, 2].cpu().numpy())

        # 4. Zero Environmental Vector
        _, _, p_zero_env = model(images, vis_masks, torch.zeros_like(env_vec))
        zero_env_preds.append(p_zero_env[:, 2].cpu().numpy())

        # 5. Mask Vmax from Environmental Vector
        env_no_vmax = env_vec.clone()
        env_no_vmax[:, 0] = 0.0 # vmax
        env_no_vmax[:, 6] = 1.0 # missing flag
        _, _, p_no_vmax = model(images, vis_masks, env_no_vmax)
        zero_vmax_preds.append(p_no_vmax[:, 2].cpu().numpy())

        # 6. Zero Visible Channel
        images_no_vis = images.clone()
        images_no_vis[:, :, 2, :, :] = 0.0 # channel 2 is VIS
        vis_zero_mask = torch.zeros_like(vis_masks)
        _, _, p_no_vis = model(images_no_vis, vis_zero_mask, env_vec)
        zero_vis_preds.append(p_no_vis[:, 2].cpu().numpy())

p_clean = np.concatenate(clean_preds)
p_rev = np.concatenate(rev_preds)
p_static = np.concatenate(static_preds)
p_zero_env = np.concatenate(zero_env_preds)
p_no_vmax = np.concatenate(zero_vmax_preds)
p_no_vis = np.concatenate(zero_vis_preds)

sensitivity_results = [
    {
        "Condition": "Baseline Clean Input",
        "Mean_Pred_ΔV24": float(np.mean(p_clean)),
        "Std_Pred_ΔV24": float(np.std(p_clean)),
        "Max_Pred_ΔV24": float(np.max(p_clean)),
        "Shift_MAE_vs_Clean": 0.0,
        "Correlation_with_Clean": 1.0,
    },
    {
        "Condition": "Reverse Temporal Order (t6 -> t0)",
        "Mean_Pred_ΔV24": float(np.mean(p_rev)),
        "Std_Pred_ΔV24": float(np.std(p_rev)),
        "Max_Pred_ΔV24": float(np.max(p_rev)),
        "Shift_MAE_vs_Clean": float(np.mean(np.abs(p_rev - p_clean))),
        "Correlation_with_Clean": float(np.corrcoef(p_rev, p_clean)[0, 1]),
    },
    {
        "Condition": "Static Frame (Repeat Latest Frame 7x)",
        "Mean_Pred_ΔV24": float(np.mean(p_static)),
        "Std_Pred_ΔV24": float(np.std(p_static)),
        "Max_Pred_ΔV24": float(np.max(p_static)),
        "Shift_MAE_vs_Clean": float(np.mean(np.abs(p_static - p_clean))),
        "Correlation_with_Clean": float(np.corrcoef(p_static, p_clean)[0, 1]),
    },
    {
        "Condition": "Zero-Out Entire Environmental Branch",
        "Mean_Pred_ΔV24": float(np.mean(p_zero_env)),
        "Std_Pred_ΔV24": float(np.std(p_zero_env)),
        "Max_Pred_ΔV24": float(np.max(p_zero_env)),
        "Shift_MAE_vs_Clean": float(np.mean(np.abs(p_zero_env - p_clean))),
        "Correlation_with_Clean": float(np.corrcoef(p_zero_env, p_clean)[0, 1]),
    },
    {
        "Condition": "Zero-Out Vmax (Mask Current Intensity)",
        "Mean_Pred_ΔV24": float(np.mean(p_no_vmax)),
        "Std_Pred_ΔV24": float(np.std(p_no_vmax)),
        "Max_Pred_ΔV24": float(np.max(p_no_vmax)),
        "Shift_MAE_vs_Clean": float(np.mean(np.abs(p_no_vmax - p_clean))),
        "Correlation_with_Clean": float(np.corrcoef(p_no_vmax, p_clean)[0, 1]),
    },
    {
        "Condition": "Zero-Out VIS Channel (Night / Missing VIS)",
        "Mean_Pred_ΔV24": float(np.mean(p_no_vis)),
        "Std_Pred_ΔV24": float(np.std(p_no_vis)),
        "Max_Pred_ΔV24": float(np.max(p_no_vis)),
        "Shift_MAE_vs_Clean": float(np.mean(np.abs(p_no_vis - p_clean))),
        "Correlation_with_Clean": float(np.corrcoef(p_no_vis, p_clean)[0, 1]),
    },
]

sens_df = pd.DataFrame(sensitivity_results)
sens_df.to_csv(RESULTS_DIR / "phase9_sensitivity_diagnostics.csv", index=False)
print("\nSensitivity Diagnostics Table:")
print(sens_df.to_string(index=False))

# -------------------------------------------------------------------------
# PHASE 10: TRAINING VS VALIDATION EXTREME RI PERFORMANCE
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("PHASE 10: CHECK WHETHER THE MODEL EVER LEARNS EXTREME RI ON TRAINING SET")
print("="*80)

# Extract extreme training sequences (actual delta_v24 >= 45 kt)
train_extreme_indices = train_seq[train_seq["delta_v24"] >= 45].index.tolist()
print(f"Total training samples with actual ΔV24 >= 45 kt: {len(train_extreme_indices)}")

train_ext_ds = DeltaSequenceDataset(
    seq_df=train_seq.iloc[train_extreme_indices],
    mean=norm_mean,
    std=norm_std,
    channels=[0, 1, 2],
    is_training=False,
    config=config,
    env_tensor=train_env[train_extreme_indices],
)
train_ext_loader = DataLoader(train_ext_ds, batch_size=32, shuffle=False)

train_ext_preds = []
train_ext_acts = []
train_ext_vcurr = []
train_ext_v24 = []
train_ext_cids = []
train_ext_ts = []

with torch.no_grad():
    for batch in train_ext_loader:
        images, vis_masks, _, _, _, reg_delta_targets, env_vec, meta = batch
        images = images.to(device)
        vis_masks = vis_masks.to(device)
        env_vec = env_vec.to(device)

        _, _, p_delta = model(images, vis_masks, env_vec)
        train_ext_preds.append(p_delta[:, 2].cpu().numpy())
        train_ext_acts.append(reg_delta_targets[:, 2].cpu().numpy())
        train_ext_vcurr.extend(meta["vmax_curr"].numpy())
        train_ext_v24.extend(meta["vmax_plus_24h"].numpy())
        train_ext_cids.extend(meta["cyclone_id"])
        train_ext_ts.extend(meta["target_t_timestamp"])

train_ext_preds = np.concatenate(train_ext_preds)
train_ext_acts = np.concatenate(train_ext_acts)

train_ext_df = pd.DataFrame({
    "cyclone_id": train_ext_cids,
    "timestamp": train_ext_ts,
    "vmax_curr": train_ext_vcurr,
    "vmax_plus_24h": train_ext_v24,
    "actual_dv24": train_ext_acts,
    "pred_dv24": train_ext_preds,
})
train_ext_df["error"] = train_ext_df["pred_dv24"] - train_ext_df["actual_dv24"]

print("\nTRAINING SET Extreme RI Predictions Summary (N=738, actual ΔV24 >= 45 kt):")
print(f"  Actual ΔV24: Mean = {train_ext_df['actual_dv24'].mean():.2f} kt, Max = {train_ext_df['actual_dv24'].max():.2f} kt")
print(f"  Predicted ΔV24 on TRAIN: Mean = {train_ext_df['pred_dv24'].mean():.2f} kt, Max = {train_ext_df['pred_dv24'].max():.2f} kt, Min = {train_ext_df['pred_dv24'].min():.2f} kt")
print(f"  Count on TRAIN with pred >= 45 kt: {(train_ext_df['pred_dv24'] >= 45).sum()} / {len(train_ext_df)} ({(train_ext_df['pred_dv24'] >= 45).sum()/len(train_ext_df)*100:.2f}%)")
print(f"  Count on TRAIN with pred >= 50 kt: {(train_ext_df['pred_dv24'] >= 50).sum()} / {len(train_ext_df)}")
print(f"  Count on TRAIN with pred >= 60 kt: {(train_ext_df['pred_dv24'] >= 60).sum()} / {len(train_ext_df)}")
print(f"  MAE on TRAIN extremes: {np.abs(train_ext_df['error']).mean():.2f} kt")
print(f"  Bias on TRAIN extremes: {train_ext_df['error'].mean():.2f} kt")

# Top 10 extreme training examples
print("\nTop 10 Extreme Training Cases (by actual ΔV24):")
top10_train = train_ext_df.sort_values(by="actual_dv24", ascending=False).head(10)
print(top10_train[["cyclone_id", "timestamp", "vmax_curr", "actual_dv24", "pred_dv24", "error"]].to_string(index=False))

train_ext_df.to_csv(RESULTS_DIR / "phase10_train_extreme_fits.csv", index=False)

# -------------------------------------------------------------------------
# PHASE 11: CURRENT INTENSITY DEPENDENCE
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("PHASE 11: CURRENT INTENSITY (Vcurr) DEPENDENCE")
print("="*80)

vcurr_bins = [
    ("TD / Weak (< 34 kt)", lambda v: v < 34),
    ("TS / Moderate (34 to 63 kt)", lambda v: (v >= 34) & (v < 64)),
    ("Cat 1-2 Strong (64 to 95 kt)", lambda v: (v >= 64) & (v < 96)),
    ("Cat 3-5 Major (>= 96 kt)", lambda v: v >= 96),
]

vcurr_rows = []
for b_name, b_fn in vcurr_bins:
    sub = ultra_df[b_fn(ultra_df["vmax_curr"])]
    if len(sub) == 0:
        continue
    act = sub["act_dv24"].values
    pred = sub["pred_dv24"].values
    vcurr_rows.append({
        "Intensity_Bin": b_name,
        "N": len(sub),
        "Mean_Vcurr": float(sub["vmax_curr"].mean()),
        "Mean_Actual_ΔV24": float(np.mean(act)),
        "Mean_Pred_ΔV24": float(np.mean(pred)),
        "Max_Pred_ΔV24": float(np.max(pred)),
        "Min_Pred_ΔV24": float(np.min(pred)),
        "MAE": float(np.mean(np.abs(pred - act))),
        "Bias": float(np.mean(pred - act)),
        "RI_Actual_Pct": float((act >= 30).sum() / len(sub) * 100),
        "RI_Pred_Pct": float((pred >= 30).sum() / len(sub) * 100),
    })

vcurr_df = pd.DataFrame(vcurr_rows)
vcurr_df.to_csv(RESULTS_DIR / "phase11_vcurr_dependence.csv", index=False)
print("\nCurrent Intensity Stratification Table:")
print(vcurr_df.to_string(index=False))

# -------------------------------------------------------------------------
# PHASE 12: COMPARE AGAINST SIMPLE BASELINES
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("PHASE 12: COMPARE AGAINST SIMPLE BASELINES")
print("="*80)

# Baselines on canonical test set (N=7,901)
act_test = ultra_df["act_dv24"].values
vcurr_test = ultra_df["vmax_curr"].values

# 1. Persistence: Pred ΔV24 = 0
pred_persist = np.zeros_like(act_test)

# 2. Climatological Mean ΔV24 (from train set)
train_clim_dv24 = float(train_seq["delta_v24"].mean())
pred_clim = np.full_like(act_test, train_clim_dv24)

# 3. Conditional Climatology E[ΔV24 | Vcurr]
# Discretize train Vcurr into 5 kt bins
train_seq["vcurr_bin"] = (train_seq["vmax_curr"] // 5) * 5
cond_clim_map = train_seq.groupby("vcurr_bin")["delta_v24"].mean().to_dict()
overall_mean = train_seq["delta_v24"].mean()
pred_cond_clim = np.array([cond_clim_map.get((v // 5) * 5, overall_mean) for v in vcurr_test])

# 4. Ultra Model
pred_ultra = ultra_df["pred_dv24"].values

def evaluate_baseline(name, p_arr):
    # Overall
    mae_all = np.mean(np.abs(p_arr - act_test))
    bias_all = np.mean(p_arr - act_test)
    # RI cases (>= 30 kt)
    ri_mask = act_test >= 30
    mae_ri = np.mean(np.abs(p_arr[ri_mask] - act_test[ri_mask]))
    bias_ri = np.mean(p_arr[ri_mask] - act_test[ri_mask])
    # Extreme RI cases (>= 45 kt)
    ext_mask = act_test >= 45
    mae_ext = np.mean(np.abs(p_arr[ext_mask] - act_test[ext_mask]))
    bias_ext = np.mean(p_arr[ext_mask] - act_test[ext_mask])
    # Max predicted
    max_p = np.max(p_arr)

    return {
        "Baseline": name,
        "Overall_MAE": float(mae_all),
        "Overall_Bias": float(bias_all),
        "RI_MAE (>=30kt)": float(mae_ri),
        "RI_Bias": float(bias_ri),
        "Extreme_RI_MAE (>=45kt)": float(mae_ext),
        "Extreme_RI_Bias": float(bias_ext),
        "Max_Predicted_ΔV24": float(max_p),
    }

base_rows = [
    evaluate_baseline("Persistence (ΔV = 0)", pred_persist),
    evaluate_baseline("Climatological Mean", pred_clim),
    evaluate_baseline("Conditional Climatology E[ΔV|Vcurr]", pred_cond_clim),
    evaluate_baseline("Ultra Model (1/6/12)", pred_ultra),
]

base_df = pd.DataFrame(base_rows)
base_df.to_csv(RESULTS_DIR / "phase12_baseline_comparison.csv", index=False)
print("\nBaseline Comparison Table:")
print(base_df.to_string(index=False))

print("\n[ALL AUDIT PHASES COMPLETE SUCCESSFULLY]")
