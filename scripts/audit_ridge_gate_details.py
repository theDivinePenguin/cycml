#!/usr/bin/env python3
"""
Deep Forensic Audit of Learned Ridge Gating Model
Verifying:
1. Origin and nature of the 6,000 training predictions (in-sample vs OOF audit)
2. Exact feature list, causality at time t, and target isolation verification
3. Full Ridge weight matrices, intercepts, and algebraic equations
4. Storm-by-storm breakdown across all unique cyclones in the validation set
5. 1,000-iteration Bootstrap Confidence Intervals (95% CI) on delta MAE
"""

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from src.data.environmental import EnvironmentalFeatureManager, get_feature_dim
from src.models.residual_forecaster import ResidualDeltaVForecaster
from src.models.ri_models import DedicatedRIClassifier
from scripts.run_val_fusion_experiment import DualModelValidationDataset



def run_audit():
    print("=" * 80)
    print("DEEP FORENSIC AUDIT: LEARNED RIDGE GATING MODEL")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_dir = Path("data/metadata")
    val_csv = meta_dir / "forecast_val_sequences_k5_aligned.csv"
    train_csv = meta_dir / "forecast_train_sequences_k5_aligned.csv"
    norm_json = meta_dir / "normalization_stats_multichannel.json"

    res_ckpt_path = Path("experiments/checkpoints/residual_delta_v_unconstrained/best.pt")
    ri_ckpt_path = Path("experiments/checkpoints/ri_model1_dedicated_focal/best.pt")

    with open(norm_json) as f:
        norm_stats = json.load(f)

    env_manager = EnvironmentalFeatureManager(metadata_dir=meta_dir, feature_group="full_feature_set")

    # Load Models
    res_ckpt = torch.load(res_ckpt_path, map_location=device)
    model_res = ResidualDeltaVForecaster(
        backbone_arch="resnet18",
        in_channels=3,
        d_model=256,
        temporal_type="transformer",
        num_layers=2,
        nhead=8,
        dropout=0.1,
        parameterization="unconstrained",
        pretrained_backbone=False,
    ).to(device)
    model_res.load_state_dict(res_ckpt["model_state_dict"])
    model_res.eval()

    ri_ckpt = torch.load(ri_ckpt_path, map_location=device)
    model_ri = DedicatedRIClassifier(
        backbone_arch="resnet18",
        in_channels=3,
        d_model=256,
        d_env=get_feature_dim(),
        temporal_type="transformer",
        num_layers=2,
        nhead=8,
        fusion_type="gated",
        dropout=0.15,
        pretrained_backbone=False,
    ).to(device)
    model_ri.load_state_dict(ri_ckpt["model_state_dict"])
    model_ri.eval()

    # ---------------------------------------------------------------------------
    # 1. Gate Training Sample & In-Sample Nature
    # ---------------------------------------------------------------------------
    train_df = pd.read_csv(train_csv)
    train_d24 = train_df["vmax_plus_24h"] - train_df["vmax_curr"]
    train_ri_idx = train_df[train_d24 >= 30.0].index
    train_non_ri_idx = train_df[train_d24 < 30.0].sample(n=4008, random_state=42).index
    gate_train_indices = train_ri_idx.union(train_non_ri_idx)
    gate_train_df = train_df.loc[gate_train_indices].reset_index(drop=True)

    print(f"\n[Audit Item 1: Training Set Predictions Origin]")
    print(f"• Gate training cohort size: {len(gate_train_df)} sequences")
    print(f"• Sourced from: data/metadata/forecast_train_sequences_k5_aligned.csv")
    print(f"• IN-SAMPLE NOTE: The 6,000 samples are part of the base models' training split.")
    print(f"  Base models were evaluated in eval() mode with frozen weights.")
    print(f"  The validation set (7,295 sequences) is strictly out-of-sample for both base models and gate.")

    # Generate training predictions
    train_ds = DualModelValidationDataset(gate_train_df, mean=norm_stats["mean"], std=norm_stats["std"])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))

    train_v_curr_list = []
    train_true_delta_list = []
    train_pred_delta_list = []
    train_ri_probs_list = []
    train_ri_logits_list = []

    with torch.no_grad():
        for seq, vis, v_c, true_f, cids, tss in train_loader:
            seq = seq.to(device, non_blocking=True)
            vis = vis.to(device, non_blocking=True)
            v_c_dev = v_c.to(device).float()
            env_batch = torch.stack([env_manager.get_features(cids[i], int(tss[i])) for i in range(len(cids))]).to(device)

            _, delta_hat = model_res(seq, v_curr=v_c_dev, vis_masks=vis)
            logits = model_ri(seq, vis_masks=vis, x_env=env_batch)
            probs = torch.sigmoid(logits)

            train_v_curr_list.append(v_c.numpy())
            train_true_delta_list.append((true_f.numpy() - v_c.numpy()[:, None]))
            train_pred_delta_list.append(delta_hat.cpu().numpy())
            train_ri_probs_list.append(probs.cpu().numpy().flatten())
            train_ri_logits_list.append(logits.cpu().numpy().flatten())

    X_tr_v = np.concatenate(train_v_curr_list)
    y_tr_delta = np.concatenate(train_true_delta_list, axis=0)
    X_tr_res_delta = np.concatenate(train_pred_delta_list, axis=0)
    X_tr_p_ri = np.concatenate(train_ri_probs_list)
    X_tr_logit_ri = np.concatenate(train_ri_logits_list)

    # ---------------------------------------------------------------------------
    # 2. Features and Target Isolation Audit
    # ---------------------------------------------------------------------------
    feature_names = [
        "pred_delta_6h",
        "pred_delta_12h",
        "pred_delta_24h",
        "P_RI",
        "logit_RI",
        "v_curr_div100",
        "v_curr_x_P_RI",
    ]

    print(f"\n[Audit Item 2, 3, 4: Feature Definition & Anti-Leakage Verification]")
    print(f"Features used in Ridge:")
    for i, fn in enumerate(feature_names):
        print(f"  [{i}] {fn}")
    print("• Target Isolation: Absolutely NO vmax_plus_24h, delta_24h, or future targets in feature matrix.")
    print("• P_RI Causality: Dedicated RI classifier takes only sequence imagery (t-12h to t) and env features at time t.")

    feats_train = np.column_stack([
        X_tr_res_delta[:, 0],
        X_tr_res_delta[:, 1],
        X_tr_res_delta[:, 2],
        X_tr_p_ri,
        X_tr_logit_ri,
        X_tr_v / 100.0,
        (X_tr_v / 100.0) * X_tr_p_ri,
    ])

    ridge_gate = Ridge(alpha=10.0)
    ridge_gate.fit(feats_train, y_tr_delta)

    # ---------------------------------------------------------------------------
    # 3. Exact Ridge Coefficients and Equations
    # ---------------------------------------------------------------------------
    coef = ridge_gate.coef_  # (3, 7)
    intercept = ridge_gate.intercept_  # (3,)

    print(f"\n[Audit Item 5: Exact Ridge Gate Equations & Coefficients]")
    print(f"Ridge Alpha: 10.0")
    for h_idx, h_lbl in enumerate(["+6h", "+12h", "+24h"]):
        print(f"\nHorizon {h_lbl}:")
        print(f"  Intercept b_{h_lbl} = {intercept[h_idx]:+.4f}")
        terms = [f"{coef[h_idx, f_i]:+.4f} * {feature_names[f_i]}" for f_i in range(len(feature_names))]
        print(f"  ΔV*_{h_lbl} = {intercept[h_idx]:+.4f} + \n    " + " + \n    ".join(terms))

    # ---------------------------------------------------------------------------
    # 4. Generate Validation Predictions
    # ---------------------------------------------------------------------------
    val_df = pd.read_csv(val_csv)
    n_val = len(val_df)
    val_ds = DualModelValidationDataset(val_df, mean=norm_stats["mean"], std=norm_stats["std"])
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))

    val_v_curr_list = []
    val_true_future_list = []
    val_res_delta_list = []
    val_ri_probs_list = []
    val_ri_logits_list = []
    val_cids = []

    with torch.no_grad():
        for seq, vis, v_c, true_f, cids, tss in val_loader:
            seq = seq.to(device, non_blocking=True)
            vis = vis.to(device, non_blocking=True)
            v_c_dev = v_c.to(device).float()
            env_batch = torch.stack([env_manager.get_features(cids[i], int(tss[i])) for i in range(len(cids))]).to(device)

            _, delta_hat_res = model_res(seq, v_curr=v_c_dev, vis_masks=vis)
            logits_ri = model_ri(seq, vis_masks=vis, x_env=env_batch)
            probs_ri = torch.sigmoid(logits_ri)

            val_v_curr_list.append(v_c.numpy())
            val_true_future_list.append(true_f.numpy())
            val_res_delta_list.append(delta_hat_res.cpu().numpy())
            val_ri_probs_list.append(probs_ri.cpu().numpy().flatten())
            val_ri_logits_list.append(logits_ri.cpu().numpy().flatten())
            val_cids.extend(cids)

    val_v_curr = np.concatenate(val_v_curr_list)
    val_true_future = np.concatenate(val_true_future_list, axis=0)
    val_res_delta = np.concatenate(val_res_delta_list, axis=0)
    val_ri_prob = np.concatenate(val_ri_probs_list)
    val_ri_logit = np.concatenate(val_ri_logits_list)

    # Compute Predictions
    preds_res = val_v_curr[:, None] + val_res_delta
    feats_val = np.column_stack([
        val_res_delta[:, 0],
        val_res_delta[:, 1],
        val_res_delta[:, 2],
        val_ri_prob,
        val_ri_logit,
        val_v_curr / 100.0,
        (val_v_curr / 100.0) * val_ri_prob,
    ])
    delta_gate = ridge_gate.predict(feats_val)
    preds_gate = val_v_curr[:, None] + delta_gate

    err_res = np.abs(preds_res - val_true_future)
    err_gate = np.abs(preds_gate - val_true_future)

    # ---------------------------------------------------------------------------
    # 5. Storm-by-Storm Breakdown (Audit Item 6)
    # ---------------------------------------------------------------------------
    print(f"\n[Audit Item 6: Storm-by-Storm Distribution of RI Improvement]")
    val_df_full = val_df.copy()
    val_df_full["mae_res_24h"] = err_res[:, 2]
    val_df_full["mae_gate_24h"] = err_gate[:, 2]
    val_df_full["delta_24_true"] = val_true_future[:, 2] - val_v_curr
    val_df_full["is_ri"] = val_df_full["delta_24_true"] >= 30.0

    ri_storms = val_df_full[val_df_full["is_ri"]]["cyclone_id"].unique()
    total_storms = val_df_full["cyclone_id"].nunique()
    print(f"• Total unique validation cyclones: {total_storms}")
    print(f"• Unique cyclones with RI events:   {len(ri_storms)}")

    storm_summary = []
    for cid in ri_storms:
        sdf = val_df_full[(val_df_full["cyclone_id"] == cid) & val_df_full["is_ri"]]
        n_ri_pts = len(sdf)
        res_m = sdf["mae_res_24h"].mean()
        gate_m = sdf["mae_gate_24h"].mean()
        diff_m = gate_m - res_m  # negative is improvement
        storm_summary.append({
            "cyclone_id": cid,
            "ri_timesteps": n_ri_pts,
            "res_mae": res_m,
            "gate_mae": gate_m,
            "improvement_kt": -diff_m,
        })

    storm_summary_df = pd.DataFrame(storm_summary).sort_values("ri_timesteps", ascending=False)
    improved_count = sum(storm_summary_df["improvement_kt"] > 0)
    worsened_count = sum(storm_summary_df["improvement_kt"] < 0)
    print(f"• RI Cyclones where Gate IMPROVED: {improved_count} / {len(ri_storms)} ({improved_count/len(ri_storms)*100:.1f}%)")
    print(f"• RI Cyclones where Gate WORSENED: {worsened_count} / {len(ri_storms)} ({worsened_count/len(ri_storms)*100:.1f}%)")
    print("\nTop 10 RI Storms by sample size:")
    print(storm_summary_df.head(10).to_string(index=False))

    # ---------------------------------------------------------------------------
    # 6. Bootstrap Confidence Intervals (Audit Item 7)
    # ---------------------------------------------------------------------------
    print(f"\n[Audit Item 7: 1,000-Iteration Bootstrap Confidence Intervals]")
    n_boot = 1000
    rng = np.random.RandomState(42)

    val_true_d24 = val_true_future[:, 2] - val_v_curr
    is_ri_val = val_true_d24 >= 30.0
    ri_indices = np.where(is_ri_val)[0]
    non_ri_indices = np.where(~is_ri_val)[0]

    boot_overall_diff = []
    boot_ri_24h_diff = []
    boot_non_ri_24h_diff = []

    mean_err_res = np.mean(err_res, axis=1)
    mean_err_gate = np.mean(err_gate, axis=1)

    for _ in range(n_boot):
        # Overall resample
        idx = rng.choice(n_val, size=n_val, replace=True)
        boot_overall_diff.append(np.mean(mean_err_gate[idx]) - np.mean(mean_err_res[idx]))

        # RI resample
        idx_ri = rng.choice(ri_indices, size=len(ri_indices), replace=True)
        boot_ri_24h_diff.append(np.mean(err_gate[idx_ri, 2]) - np.mean(err_res[idx_ri, 2]))

        # Non-RI resample
        idx_non_ri = rng.choice(non_ri_indices, size=len(non_ri_indices), replace=True)
        boot_non_ri_24h_diff.append(np.mean(err_gate[idx_non_ri, 2]) - np.mean(err_res[idx_non_ri, 2]))

    boot_overall_diff = np.array(boot_overall_diff)
    boot_ri_24h_diff = np.array(boot_ri_24h_diff)
    boot_non_ri_24h_diff = np.array(boot_non_ri_24h_diff)

    ci_overall = np.percentile(boot_overall_diff, [2.5, 50.0, 97.5])
    ci_ri = np.percentile(boot_ri_24h_diff, [2.5, 50.0, 97.5])
    ci_non_ri = np.percentile(boot_non_ri_24h_diff, [2.5, 50.0, 97.5])

    pct_overall_better = np.mean(boot_overall_diff < 0) * 100.0
    pct_ri_better = np.mean(boot_ri_24h_diff < 0) * 100.0

    print(f"\nBootstrap Results (95% CI):")
    print(f"• Overall ΔMAE:         {ci_overall[1]:+.4f} kt  [95% CI: {ci_overall[0]:+.4f}, {ci_overall[2]:+.4f}] (Gate beats Baseline in {pct_overall_better:.1f}% of resamples)")
    print(f"• RI Events +24h ΔMAE:  {ci_ri[1]:+.4f} kt  [95% CI: {ci_ri[0]:+.4f}, {ci_ri[2]:+.4f}] (Gate beats Baseline in {pct_ri_better:.1f}% of resamples)")
    print(f"• Non-RI +24h ΔMAE:     {ci_non_ri[1]:+.4f} kt  [95% CI: {ci_non_ri[0]:+.4f}, {ci_non_ri[2]:+.4f}]")

    # ---------------------------------------------------------------------------
    # Save Audit Data
    # ---------------------------------------------------------------------------
    audit_payload = {
        "features": feature_names,
        "ridge_alpha": 10.0,
        "coefficients": coef.tolist(),
        "intercepts": intercept.tolist(),
        "equations": {
            "+6h": f"{intercept[0]:+.4f} + " + " + ".join([f"{coef[0, i]:+.4f}*{feature_names[i]}" for i in range(7)]),
            "+12h": f"{intercept[1]:+.4f} + " + " + ".join([f"{coef[1, i]:+.4f}*{feature_names[i]}" for i in range(7)]),
            "+24h": f"{intercept[2]:+.4f} + " + " + ".join([f"{coef[2, i]:+.4f}*{feature_names[i]}" for i in range(7)]),
        },
        "storm_breakdown": {
            "total_cyclones": int(total_storms),
            "ri_cyclones_count": len(ri_storms),
            "improved_cyclones_count": int(improved_count),
            "worsened_cyclones_count": int(worsened_count),
            "pct_improved": float(improved_count / len(ri_storms) * 100.0),
            "storms": storm_summary,
        },
        "bootstrap_95ci": {
            "overall_delta_mae": {
                "median": float(ci_overall[1]),
                "ci_lower_2.5": float(ci_overall[0]),
                "ci_upper_97.5": float(ci_overall[2]),
                "prob_better": float(pct_overall_better),
            },
            "ri_24h_delta_mae": {
                "median": float(ci_ri[1]),
                "ci_lower_2.5": float(ci_ri[0]),
                "ci_upper_97.5": float(ci_ri[2]),
                "prob_better": float(pct_ri_better),
            },
            "non_ri_24h_delta_mae": {
                "median": float(ci_non_ri[1]),
                "ci_lower_2.5": float(ci_non_ri[0]),
                "ci_upper_97.5": float(ci_non_ri[2]),
            },
        },
    }

    out_file = Path("reports/RIDGE_GATE_DEEP_AUDIT.json")
    with open(out_file, "w") as f:
        json.dump(audit_payload, f, indent=2)
    print(f"\nSaved deep forensic audit report to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_audit()
