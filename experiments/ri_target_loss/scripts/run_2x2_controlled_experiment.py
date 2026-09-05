"""2x2 Factorial Controlled Experiment: Intensity Dynamics vs. Power-1.5 Loss.

Strict Factorial Matrix:
  A: Control (Ultra Checkpoint: No Dynamics, Huber 1/6/12) - ALREADY TRAINED
  B: + Intensity Dynamics (15-d env, Huber 1/6/12)
  C: Power-1.5 Loss (12-d env, unweighted Power-1.5, lambda_reg=0.5)
  D: Both (15-d env, unweighted Power-1.5, lambda_reg=0.5)

Zero mid-run tuning. Strictly reproducible.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.trend_config import IntensityTrendConfig
from src.evaluation.classification_metrics import (
    compute_ri_metrics,
    compute_trend_metrics,
    find_optimal_threshold,
)
from experiments.ri_target_loss.scripts.dataset import DeltaSequenceDataset
from experiments.ri_target_loss.scripts.models import DeltaEnvironmentalTemporalClassifier
from experiments.ri_target_loss.scripts.losses import DeltaJointLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[2x2 EXPERIMENT] Device: {device} | Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

EXP_DIR = repo_root / "experiments" / "ri_target_loss"
CKPT_ROOT = EXP_DIR / "checkpoints"
RESULTS_ROOT = EXP_DIR / "results"
PLOTS_DIR = repo_root / "experiments" / "ri_stress_test" / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ULTRA_CKPT = CKPT_ROOT / "exp2_delta_1_6_12" / "best.pt"
assert ULTRA_CKPT.exists(), f"Ultra checkpoint not found at {ULTRA_CKPT}"

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------------------------------------------------
# STEP 1: LOAD METADATA & COMPUTE KINEMATICS WITHOUT LEAKAGE
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("STEP 1: METADATA & KINEMATIC FEATURE EXTRACTION")
print("="*80)

train_df = pd.read_csv("data/metadata/forecast_train_sequences_k7.csv")
val_df = pd.read_csv("data/metadata/forecast_val_sequences_k7.csv")
test_df = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")

def extract_kinematics(df: pd.DataFrame) -> np.ndarray:
    dv6, dv12, acc = [], [], []
    for _, r in df.iterrows():
        v_list = json.loads(r['history_vmax']) if isinstance(r['history_vmax'], str) else r['history_vmax']
        vt = float(v_list[-1])
        vt_3 = float(v_list[-2])
        vt_6 = float(v_list[-3])
        vt_12 = float(v_list[-5])

        dv6.append(vt - vt_6)
        dv12.append(vt - vt_12)
        acc.append((vt - 2.0 * vt_3 + vt_6) / 9.0)
    return np.stack([dv6, dv12, acc], axis=1).astype(np.float32)

k_train = extract_kinematics(train_df)
k_val = extract_kinematics(val_df)
k_test = extract_kinematics(test_df)

k_mu = np.mean(k_train, axis=0)
k_std = np.std(k_train, axis=0) + 1e-7

print(f"Train Kinematics Mean: {k_mu}")
print(f"Train Kinematics Std:  {k_std}")

# Standardize kinematics
k_train_norm = torch.from_numpy((k_train - k_mu) / k_std)
k_val_norm = torch.from_numpy((k_val - k_mu) / k_std)
k_test_norm = torch.from_numpy((k_test - k_mu) / k_std)

# Load existing 12-d environmental tensors
env_cache = torch.load("data/metadata/environmental_features_k7.pt")
train_env_12 = env_cache["train"]
val_env_12 = env_cache["val"]
test_env_12 = env_cache["test"]

# Concatenate to form 15-d environmental tensors
train_env_15 = torch.cat([train_env_12, k_train_norm], dim=1)
val_env_15 = torch.cat([val_env_12, k_val_norm], dim=1)
test_env_15 = torch.cat([test_env_12, k_test_norm], dim=1)

print(f"12-d Environmental shapes: Train={train_env_12.shape}, Val={val_env_12.shape}, Test={test_env_12.shape}")
print(f"15-d Environmental shapes: Train={train_env_15.shape}, Val={val_env_15.shape}, Test={test_env_15.shape}")

# Image normalization stats
with open("data/metadata/normalization_stats_multichannel.json") as f:
    norm_stats = json.load(f)
norm_mean = [norm_stats["mean"][c] for c in [0, 1, 2]]
norm_std = [norm_stats["std"][c] for c in [0, 1, 2]]

trend_config = IntensityTrendConfig()

# Build dataloaders helper
def get_dataloaders(env_dim: int, batch_size: int = 16):
    t_env = train_env_15 if env_dim == 15 else train_env_12
    v_env = val_env_15 if env_dim == 15 else val_env_12
    te_env = test_env_15 if env_dim == 15 else test_env_12

    train_ds = DeltaSequenceDataset(train_df, mean=norm_mean, std=norm_std, channels=[0, 1, 2], is_training=True, config=trend_config, env_tensor=t_env)
    val_ds = DeltaSequenceDataset(val_df, mean=norm_mean, std=norm_std, channels=[0, 1, 2], is_training=False, config=trend_config, env_tensor=v_env)
    test_ds = DeltaSequenceDataset(test_df, mean=norm_mean, std=norm_std, channels=[0, 1, 2], is_training=False, config=trend_config, env_tensor=te_env)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    test_ld = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    # Train extreme loader (N=738)
    mask_ext = (train_df["vmax_plus_24h"] - train_df["vmax_curr"]) >= 45.0
    train_ext_df = train_df[mask_ext].reset_index(drop=True)
    t_ext_env = t_env[mask_ext]
    train_ext_ds = DeltaSequenceDataset(train_ext_df, mean=norm_mean, std=norm_std, channels=[0, 1, 2], is_training=False, config=trend_config, env_tensor=t_ext_env)
    train_ext_ld = DataLoader(train_ext_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return train_ld, val_ld, test_ld, train_ext_ld, train_ext_df

# -------------------------------------------------------------------------
# STEP 2: TRAINING LOOP IMPLEMENTATION
# -------------------------------------------------------------------------
def train_and_eval_model(
    exp_name: str,
    env_dim: int,
    loss_type: str,
    ri_weights: tuple,
    lambda_reg_delta: float,
    epochs: int = 4,
    lr: float = 1e-4,
    batch_size: int = 16,
):
    print("\n" + "="*80)
    print(f"LAUNCHING CONDITION: {exp_name}")
    print(f"Env Dim: {env_dim} | Loss Type: {loss_type} | RI Weights: {ri_weights} | lambda_reg: {lambda_reg_delta} | Epochs: {epochs}")
    print("="*80)

    set_seed(42)
    ckpt_dir = CKPT_ROOT / exp_name
    res_dir = RESULTS_ROOT / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, train_ext_loader, train_ext_df = get_dataloaders(env_dim=env_dim, batch_size=batch_size)

    model = DeltaEnvironmentalTemporalClassifier(
        mode="delta_only",
        channels=3,
        num_frames=7,
        d_model=256,
        n_heads=8,
        num_layers=2,
        dropout=0.1,
        use_vis_channel=True,
        env_in_dim=env_dim,
    ).to(device)

    # Smart warm-start: preserve trained Ultra weights, initialize only new inputs
    model.load_warm_start_with_expanded_env(str(ULTRA_CKPT))

    # Calculate class weights for classification heads (identical to Ultra)
    d24_train = train_df["vmax_plus_24h"].values - train_df["vmax_curr"].values
    n_total = len(d24_train)
    n_ri_pos = int((d24_train >= trend_config.ri_threshold_kt).sum())
    w_pos = (n_total - n_ri_pos) / max(1, n_ri_pos)

    n_t0 = int((d24_train <= trend_config.weakening_threshold_kt).sum())
    n_t1 = int(((d24_train > trend_config.weakening_threshold_kt) & (d24_train < trend_config.intensifying_threshold_kt)).sum())
    n_t2 = int((d24_train >= trend_config.intensifying_threshold_kt).sum())
    trend_weights = n_total / (3.0 * np.array([n_t0, n_t1, n_t2], dtype=np.float32))

    loss_fn = DeltaJointLoss(
        mode="delta_only",
        ri_pos_weight=torch.tensor([w_pos], device=device, dtype=torch.float32),
        trend_class_weights=torch.tensor(trend_weights, device=device, dtype=torch.float32),
        lambda_ri=1.0,
        lambda_trend=1.0,
        lambda_reg_delta=lambda_reg_delta,
        ri_weights=ri_weights,
        delta_loss_type=loss_type,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_score = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            images, vis_masks, trend_targets, ri_targets, _, reg_delta_targets, env_vec, _ = batch
            images = images.to(device, non_blocking=True)
            vis_masks = vis_masks.to(device, non_blocking=True)
            trend_targets = trend_targets.to(device, non_blocking=True)
            ri_targets = ri_targets.to(device, non_blocking=True)
            reg_delta_targets = reg_delta_targets.to(device, non_blocking=True)
            env_vec = env_vec.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                ri_logits, trend_logits, reg_delta_preds = model(images, vis_masks, env_vec)
                loss, _ = loss_fn(
                    ri_logits=ri_logits,
                    trend_logits=trend_logits,
                    ri_targets=ri_targets,
                    trend_targets=trend_targets,
                    reg_delta_preds=reg_delta_preds,
                    reg_delta_targets=reg_delta_targets,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        train_sec = time.time() - t0
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_ri_probs, all_ri_targets, all_pred_d24, all_act_d24 = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                images, vis_masks, _, ri_targets, _, reg_delta_targets, env_vec, _ = batch
                images = images.to(device)
                vis_masks = vis_masks.to(device)
                env_vec = env_vec.to(device)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    ri_logits, _, reg_delta_preds = model(images, vis_masks, env_vec)

                ri_probs = torch.sigmoid(ri_logits).squeeze(-1).cpu().numpy()
                all_ri_probs.append(ri_probs)
                all_ri_targets.append(ri_targets.numpy())
                all_pred_d24.append(reg_delta_preds[:, 2].cpu().numpy())
                all_act_d24.append(reg_delta_targets[:, 2].numpy())

        v_probs = np.concatenate(all_ri_probs)
        v_targets = np.concatenate(all_ri_targets)
        v_pred = np.concatenate(all_pred_d24)
        v_act = np.concatenate(all_act_d24)

        val_m = compute_ri_metrics(v_targets, v_probs)
        val_pr_auc = val_m.get("pr_auc", val_m.get("ri_pr_auc", 0.0))
        mae_24 = float(np.mean(np.abs(v_pred - v_act)))
        max_p = float(np.max(v_pred))

        print(f"Epoch [{epoch:2d}/{epochs:2d}] ({train_sec:.1f}s) - Loss: {avg_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | +24h MAE: {mae_24:.2f} kt | Max Pred: {max_p:.1f} kt", flush=True)

        if val_pr_auc > best_val_score:
            best_val_score = val_pr_auc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_pr_auc": val_pr_auc,
                "best_tau": val_m["optimal_threshold"],
            }, ckpt_dir / "best.pt")

    print(f"Condition {exp_name} Complete! Best Epoch {best_epoch} (Val PR-AUC: {best_val_score:.4f})")

    # Evaluate Test Set
    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_preds, test_p6, test_p12 = [], [], []
    test_ri_probs, test_trend_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            images, vis_masks, _, _, _, _, env_vec, _ = batch
            images = images.to(device)
            vis_masks = vis_masks.to(device)
            env_vec = env_vec.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                ri_logits, trend_logits, reg_delta_preds = model(images, vis_masks, env_vec)

            test_p6.append(reg_delta_preds[:, 0].cpu().numpy())
            test_p12.append(reg_delta_preds[:, 1].cpu().numpy())
            test_preds.append(reg_delta_preds[:, 2].cpu().numpy())
            test_ri_probs.append(torch.sigmoid(ri_logits).squeeze(-1).cpu().numpy())
            test_trend_preds.append(torch.argmax(torch.softmax(trend_logits, dim=-1), dim=-1).cpu().numpy())

    test_df_out = test_df.copy()
    test_df_out["pred_delta_6h"] = np.concatenate(test_p6)
    test_df_out["pred_delta_12h"] = np.concatenate(test_p12)
    test_df_out["pred_delta_24h"] = np.concatenate(test_preds)
    test_df_out["pred_ri_prob"] = np.concatenate(test_ri_probs)
    test_df_out["pred_trend"] = np.concatenate(test_trend_preds)
    test_df_out["recon_plus_24h"] = test_df_out["vmax_curr"] + test_df_out["pred_delta_24h"]

    test_df_out.to_csv(res_dir / "test_predictions.csv", index=False)
    print(f"Saved test predictions to {res_dir / 'test_predictions.csv'}")

    # Evaluate Train Extremes (N=738)
    ext_preds = []
    with torch.no_grad():
        for batch in train_ext_loader:
            images, vis_masks, _, _, _, _, env_vec, _ = batch
            images = images.to(device)
            vis_masks = vis_masks.to(device)
            env_vec = env_vec.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _, _, reg_delta_preds = model(images, vis_masks, env_vec)
            ext_preds.append(reg_delta_preds[:, 2].cpu().numpy())

    train_ext_out = train_ext_df.copy()
    train_ext_out["pred_delta_24h"] = np.concatenate(ext_preds)
    train_ext_out.to_csv(res_dir / "train_extremes_predictions.csv", index=False)
    print(f"Saved train extremes predictions to {res_dir / 'train_extremes_predictions.csv'}")

# -------------------------------------------------------------------------
# STEP 3: RUN THE 3 CONDITIONS SEQUENTIALLY
# -------------------------------------------------------------------------
# Condition B: + Intensity Dynamics (15-d env, Huber loss with 1/6/12 weights, lambda_reg=0.1)
train_and_eval_model(
    exp_name="exp_2x2_B_dynamics",
    env_dim=15,
    loss_type="huber",
    ri_weights=(1.0, 6.0, 12.0),
    lambda_reg_delta=0.1,
    epochs=4,
)

# Condition C: Power-1.5 Loss (12-d env, unweighted Power-1.5, lambda_reg=0.5)
train_and_eval_model(
    exp_name="exp_2x2_C_power_loss",
    env_dim=12,
    loss_type="power_15",
    ri_weights=None,  # strictly unweighted for clean factorial isolation
    lambda_reg_delta=0.5,
    epochs=4,
)

# Condition D: Both Dynamics + Power-1.5 Loss (15-d env, unweighted Power-1.5, lambda_reg=0.5)
train_and_eval_model(
    exp_name="exp_2x2_D_both",
    env_dim=15,
    loss_type="power_15",
    ri_weights=None,  # strictly unweighted
    lambda_reg_delta=0.5,
    epochs=4,
)

print("\n" + "="*80)
print("TRAINING OF CONDITIONS B, C, D COMPLETE! RUNNING UNIFIED EVALUATION...")
print("="*80)

import subprocess
eval_script = str(PROJECT_ROOT / "experiments" / "ri_target_loss" / "scripts" / "evaluate_2x2_factorial.py")
subprocess.run([sys.executable, eval_script], check=True)

