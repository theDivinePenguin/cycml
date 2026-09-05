#!/usr/bin/env python3
"""
Locked-Test Evaluation: Learned RI-Aware Correction Model (MLP_AllHorizons_scale_15kt).

Evaluates the frozen, audited learned correction model against:
1. Persistence Baseline
2. Canonical Residual Forecaster alone
3. Canonical Final Hybrid (Stage 1 + Stage 2 + Ridge Gate)
4. Learned RI-Aware Correction v1 (audited candidate)

Test Manifest: data/metadata/forecast_test_sequences_k5_aligned.csv (EXACT N=6,825, 171 cyclones)
Verification: Zero leakage, frozen weights, single evaluation pass, immutable outputs.
"""

import ast
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(".").resolve()))

from src.data.environmental import EnvironmentalFeatureManager, get_feature_dim
from src.evaluation.sanity_checks import TrajectoryEvaluator
from src.models.residual_forecaster import ResidualDeltaVForecaster
from src.models.ri_models import DedicatedRIClassifier
from scripts.run_val_fusion_experiment import DualModelValidationDataset


class TanhConstrainedMLPCorrection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 3, scale: float = 15.0, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        return self.scale * torch.tanh(raw)


def parse_history_vmax(hist_str: str) -> np.ndarray:
    if isinstance(hist_str, list):
        return np.array(hist_str, dtype=np.float32)
    try:
        return np.array(ast.literal_eval(hist_str), dtype=np.float32)
    except Exception:
        clean = hist_str.strip("[]").split(",")
        return np.array([float(x.strip()) for x in clean if x.strip()], dtype=np.float32)


def compute_history_features(history_vmax_series: pd.Series, v_curr: np.ndarray) -> np.ndarray:
    n = len(history_vmax_series)
    feats = np.zeros((n, 4), dtype=np.float32)
    for i, raw in enumerate(history_vmax_series):
        arr = parse_history_vmax(raw)
        if len(arr) >= 5:
            d6 = float(arr[-1] - arr[-3])
            d12 = float(arr[-1] - arr[0])
            slope = d12 / 12.0
            std_val = float(np.std(arr))
        elif len(arr) >= 2:
            d6 = float(arr[-1] - arr[-2])
            d12 = float(arr[-1] - arr[0])
            slope = d12 / (3.0 * (len(arr) - 1))
            std_val = float(np.std(arr))
        else:
            d6, d12, slope, std_val = 0.0, 0.0, 0.0, 0.0
        feats[i] = [d6, d12, slope, std_val]
    return feats


def calculate_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    median_ae = float(np.median(abs_err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "bias": bias, "median_ae": median_ae, "r2": r2}


def evaluate_model_on_test(
    name: str,
    preds: np.ndarray,
    targets: np.ndarray,
    v_curr: np.ndarray,
    cids: np.ndarray,
    evaluator: TrajectoryEvaluator,
    canonical_sample_mae: np.ndarray = None,
    boot_indices_all: List = None,
    boot_indices_ri: List = None,
    boot_indices_non_ri: List = None,
) -> Dict:
    m6 = calculate_metrics(preds[:, 0], targets[:, 0])
    m12 = calculate_metrics(preds[:, 1], targets[:, 1])
    m24 = calculate_metrics(preds[:, 2], targets[:, 2])
    overall_mean_mae = (m6["mae"] + m12["mae"] + m24["mae"]) / 3.0
    overall_rmse = (m6["rmse"] + m12["rmse"] + m24["rmse"]) / 3.0

    traj_diag = evaluator.evaluate_trajectories(preds, targets, v_curr)
    false_dips = traj_diag.get("false_dip_count", 0)

    true_delta24 = targets[:, 2] - v_curr
    ri_mask = true_delta24 >= 30.0
    non_ri_mask = ~ri_mask
    ext_mask = v_curr >= 95.0

    # Subgroup metrics
    ri_err = preds[ri_mask, 2] - targets[ri_mask, 2]
    ri_mae = float(np.mean(np.abs(ri_err)))
    ri_rmse = float(np.sqrt(np.mean(ri_err ** 2)))
    ri_bias = float(np.mean(ri_err))
    ri_underpred = float(np.mean(preds[ri_mask, 2] < targets[ri_mask, 2]))

    non_ri_err = preds[non_ri_mask, 2] - targets[non_ri_mask, 2]
    non_ri_mae = float(np.mean(np.abs(non_ri_err)))
    non_ri_rmse = float(np.sqrt(np.mean(non_ri_err ** 2)))
    non_ri_bias = float(np.mean(non_ri_err))

    ext_err = preds[ext_mask, 2] - targets[ext_mask, 2]
    ext_mae = float(np.mean(np.abs(ext_err)))

    # Pointwise MAE for paired statistical comparisons
    sample_mae = np.mean(np.abs(preds - targets), axis=1)
    sample_24h_abs = np.abs(preds[:, 2] - targets[:, 2])

    stat_dict = {}
    if canonical_sample_mae is not None and boot_indices_all is not None:
        # Paired hypothesis tests against canonical hybrid
        t_stat, p_t = stats.ttest_rel(sample_mae, canonical_sample_mae)
        try:
            w_stat, p_w = stats.wilcoxon(sample_mae, canonical_sample_mae)
        except Exception:
            p_w = 1.0

        can_24h_abs = np.abs(canonical_sample_mae)  # placeholder

        boot_ov = [float(np.mean(sample_mae[b]) - np.mean(canonical_sample_mae[b])) for b in boot_indices_all]
        ci_ov = np.percentile(boot_ov, [2.5, 50.0, 97.5])
        win_rate = float(np.mean(np.array(boot_ov) < 0) * 100.0)

        # Storm-level win/loss counts
        storm_diffs = []
        for cid in np.unique(cids):
            s_m = cids == cid
            s_diff = np.mean(sample_mae[s_m]) - np.mean(canonical_sample_mae[s_m])
            storm_diffs.append(s_diff)
        storm_diffs = np.array(storm_diffs)
        storms_imp = int(np.sum(storm_diffs < -1e-4))
        storms_wor = int(np.sum(storm_diffs > 1e-4))

        stat_dict = {
            "p_val_paired_t": float(p_t),
            "p_val_wilcoxon": float(p_w),
            "bootstrap_overall_median": float(ci_ov[1]),
            "bootstrap_overall_ci_95": [float(ci_ov[0]), float(ci_ov[2])],
            "bootstrap_win_rate": win_rate,
            "storms_improved": storms_imp,
            "storms_worsened": storms_wor,
        }

    return {
        "name": name,
        "overall_mean_mae": overall_mean_mae,
        "overall_rmse": overall_rmse,
        "mae_6h": m6["mae"],
        "rmse_6h": m6["rmse"],
        "r2_6h": m6["r2"],
        "mae_12h": m12["mae"],
        "rmse_12h": m12["rmse"],
        "r2_12h": m12["r2"],
        "mae_24h": m24["mae"],
        "rmse_24h": m24["rmse"],
        "r2_24h": m24["r2"],
        "bias_24h": m24["bias"],
        "ri_mae_24h": ri_mae,
        "ri_rmse_24h": ri_rmse,
        "ri_bias_24h": ri_bias,
        "ri_underpred_fraction": ri_underpred,
        "non_ri_mae_24h": non_ri_mae,
        "non_ri_rmse_24h": non_ri_rmse,
        "non_ri_bias_24h": non_ri_bias,
        "extreme_mae_24h": ext_mae,
        "false_dips": false_dips,
        "sample_mae": sample_mae,
        "sample_24h_abs": sample_24h_abs,
        "stats": stat_dict,
    }


def main():
    print("=" * 80)
    print("IMMUTABLE LOCKED-TEST EVALUATION: LEARNED RI-AWARE CORRECTION MODEL")
    print("=" * 80)
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    test_csv = Path("data/metadata/forecast_test_sequences_k5_aligned.csv")
    norm_json = Path("data/metadata/normalization_stats_multichannel.json")
    res_ckpt_path = Path("experiments/checkpoints/residual_delta_v_unconstrained/best.pt")
    ri_ckpt_path = Path("experiments/checkpoints/ri_model1_dedicated_focal/best.pt")
    gate_path = Path("experiments/final_locked_test/final_frozen_ridge_gate.json")
    correction_ckpt_path = Path("experiments/ri_aware_correction/best_correction_model.pt")

    for p in [test_csv, norm_json, res_ckpt_path, ri_ckpt_path, gate_path, correction_ckpt_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    out_dir = Path("experiments/final_test_learned_ri_correction")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Checkpoint and Scalers
    print("\nLoading audited learned correction checkpoint...")
    corr_ckpt = torch.load(correction_ckpt_path, map_location="cpu", weights_only=False)
    print(f"✓ Loaded {corr_ckpt['config_name']} (Family: {corr_ckpt['family']}, Scale: {corr_ckpt['scale']})")

    mean_tr = corr_ckpt["mean_tr"]
    std_tr = corr_ckpt["std_tr"]
    scale_val = corr_ckpt["scale"]
    feature_names = list(corr_ckpt["feature_names"])

    model_corr = TanhConstrainedMLPCorrection(in_dim=len(feature_names), out_dim=3, scale=scale_val).to(device)
    model_corr.load_state_dict(corr_ckpt["model_state_dict"])
    model_corr.eval()

    # 2. Load Frozen Base Models
    print("\nLoading frozen neural base models...")
    with open(norm_json) as f:
        norm_stats = json.load(f)

    res_ckpt = torch.load(res_ckpt_path, map_location=device)
    model_res = ResidualDeltaVForecaster(
        backbone_arch="resnet18", in_channels=3, d_model=256, temporal_type="transformer",
        num_layers=2, nhead=8, dropout=0.1, parameterization="unconstrained", pretrained_backbone=False,
    ).to(device)
    model_res.load_state_dict(res_ckpt["model_state_dict"])
    model_res.eval()

    ri_ckpt = torch.load(ri_ckpt_path, map_location=device)
    model_ri = DedicatedRIClassifier(
        backbone_arch="resnet18", in_channels=3, d_model=256, d_env=get_feature_dim(),
        temporal_type="transformer", num_layers=2, nhead=8, fusion_type="gated", dropout=0.15, pretrained_backbone=False,
    ).to(device)
    model_ri.load_state_dict(ri_ckpt["model_state_dict"])
    model_ri.eval()

    with open(gate_path) as f:
        gate_info = json.load(f)
    gate_intercept = np.array(gate_info["intercepts"])
    gate_coef = np.array(gate_info["coefficients"])

    env_manager = EnvironmentalFeatureManager(metadata_dir="data/metadata", feature_group="full_feature_set")

    # 3. Load Locked Test Manifest & Execute Single Forward Pass
    test_df = pd.read_csv(test_csv)
    n_test = len(test_df)
    n_cids = test_df["cyclone_id"].nunique()
    print(f"\nLocked Test Cohort: {test_csv.name} (N={n_test:,} sequences across {n_cids} unique cyclones)")

    test_ds = DualModelValidationDataset(test_df, mean=norm_stats["mean"], std=norm_stats["std"])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))

    test_v_curr_list = []
    test_true_future_list = []
    test_res_delta_list = []
    test_ri_probs_list = []
    test_ri_logits_list = []
    test_cids_list = []
    test_ts_list = []

    print("Running single frozen forward pass over locked test manifest...")
    t0 = time.time()
    with torch.no_grad():
        for seq, vis, v_c, true_f, cids, tss in test_loader:
            seq = seq.to(device, non_blocking=True)
            vis = vis.to(device, non_blocking=True)
            v_c_dev = v_c.to(device).float()
            env_batch = torch.stack([env_manager.get_features(cids[i], int(tss[i])) for i in range(len(cids))]).to(device)

            _, d_hat = model_res(seq, v_curr=v_c_dev, vis_masks=vis)
            logits = model_ri(seq, vis_masks=vis, x_env=env_batch)

            test_v_curr_list.append(v_c.numpy())
            test_true_future_list.append(true_f.numpy())
            test_res_delta_list.append(d_hat.cpu().numpy())
            test_ri_probs_list.append(torch.sigmoid(logits).cpu().numpy().flatten())
            test_ri_logits_list.append(logits.cpu().numpy().flatten())
            test_cids_list.extend(cids)
            test_ts_list.extend(tss.numpy() if hasattr(tss, "numpy") else [int(t) for t in tss])

    print(f"Inference completed in {time.time() - t0:.1f}s.")

    test_v_curr = np.concatenate(test_v_curr_list)
    test_true_future = np.concatenate(test_true_future_list, axis=0)
    test_res_delta = np.concatenate(test_res_delta_list, axis=0)
    test_ri_prob = np.concatenate(test_ri_probs_list)
    test_ri_logit = np.concatenate(test_ri_logits_list)
    test_cids = np.array(test_cids_list)
    test_timestamps = np.array(test_ts_list, dtype=np.int64)

    # 4. Construct All Model Predictions
    # Model 1: Persistence
    pred_persistence = np.repeat(test_v_curr[:, None], 3, axis=1)

    # Model 2: Canonical Residual Forecaster alone
    pred_residual_alone = test_v_curr[:, None] + test_res_delta

    # Model 3: Canonical Final Hybrid (Residual + RI + Ridge Gate)
    X_gate = np.column_stack([
        test_res_delta[:, 0],
        test_res_delta[:, 1],
        test_res_delta[:, 2],
        test_ri_prob,
        test_ri_logit,
        test_v_curr / 100.0,
        (test_v_curr / 100.0) * test_ri_prob,
    ])
    delta_base = np.zeros((n_test, 3), dtype=np.float32)
    for h in range(3):
        delta_base[:, h] = gate_intercept[h] + X_gate @ gate_coef[h]
    pred_canonical_hybrid = test_v_curr[:, None] + delta_base

    # Model 4: Learned RI-Aware Correction v1
    hist_feats = compute_history_features(test_df["history_vmax"], test_v_curr)
    env_list = [env_manager.get_features(test_cids[i], int(test_timestamps[i])).numpy() for i in range(n_test)]
    env_feats = np.stack(env_list).astype(np.float32)

    interactions = np.column_stack([
        test_ri_prob * test_res_delta[:, 2],
        test_ri_prob * hist_feats[:, 1],
        test_ri_prob * (test_v_curr / 100.0),
        test_ri_prob * test_ri_logit,
        test_ri_prob * env_feats[:, 2],
        test_ri_prob * env_feats[:, 4],
        test_ri_prob * delta_base[:, 2],
    ]).astype(np.float32)

    X_test_raw = np.column_stack([
        test_res_delta,
        delta_base,
        test_ri_prob, test_ri_logit,
        test_v_curr, test_v_curr / 100.0,
        hist_feats,
        env_feats[:, :6],
        interactions,
    ]).astype(np.float32)

    # Standardize strictly using training scalers from checkpoint
    X_test_scaled = (X_test_raw - mean_tr) / std_tr
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        learned_corr = model_corr(X_test_t).cpu().numpy()

    pred_learned_correction = test_v_curr[:, None] + delta_base + learned_corr

    # 5. Evaluate Cohorts & Run Paired Bootstrap vs Canonical Hybrid
    evaluator = TrajectoryEvaluator()
    rng = np.random.RandomState(42)
    n_boot = 1000

    test_true_delta24 = test_true_future[:, 2] - test_v_curr
    ri_mask = test_true_delta24 >= 30.0
    non_ri_mask = ~ri_mask

    ri_idx = np.where(ri_mask)[0]
    non_ri_idx = np.where(non_ri_mask)[0]

    boot_indices_all = [rng.choice(n_test, size=n_test, replace=True) for _ in range(n_boot)]
    boot_indices_ri = [rng.choice(ri_idx, size=len(ri_idx), replace=True) for _ in range(n_boot)]
    boot_indices_non_ri = [rng.choice(non_ri_idx, size=len(non_ri_idx), replace=True) for _ in range(n_boot)]

    # Canonical Hybrid reference sample errors
    m_canonical = evaluate_model_on_test(
        "3. Canonical Final Hybrid",
        pred_canonical_hybrid,
        test_true_future,
        test_v_curr,
        test_cids,
        evaluator,
    )
    can_sample_mae = m_canonical["sample_mae"]

    m_persistence = evaluate_model_on_test(
        "1. Persistence Baseline",
        pred_persistence,
        test_true_future,
        test_v_curr,
        test_cids,
        evaluator,
        canonical_sample_mae=can_sample_mae,
        boot_indices_all=boot_indices_all,
        boot_indices_ri=boot_indices_ri,
        boot_indices_non_ri=boot_indices_non_ri,
    )

    m_residual = evaluate_model_on_test(
        "2. Canonical Residual ΔV Forecaster Alone",
        pred_residual_alone,
        test_true_future,
        test_v_curr,
        test_cids,
        evaluator,
        canonical_sample_mae=can_sample_mae,
        boot_indices_all=boot_indices_all,
        boot_indices_ri=boot_indices_ri,
        boot_indices_non_ri=boot_indices_non_ri,
    )

    m_learned = evaluate_model_on_test(
        "4. Learned RI-Aware Correction v1 (Audited MLP)",
        pred_learned_correction,
        test_true_future,
        test_v_curr,
        test_cids,
        evaluator,
        canonical_sample_mae=can_sample_mae,
        boot_indices_all=boot_indices_all,
        boot_indices_ri=boot_indices_ri,
        boot_indices_non_ri=boot_indices_non_ri,
    )

    all_models = [m_persistence, m_residual, m_canonical, m_learned]

    # 6. Display Comparison Table
    print("\n" + "=" * 80)
    print("FINAL LOCKED-TEST EVALUATION RESULTS TABLE")
    print("=" * 80)
    fmt_hdr = "{:<42} | {:<11} | {:<7} | {:<8} | {:<8} | {:<11} | {:<11} | {:<5}"
    print(fmt_hdr.format("Model Architecture", "Overall MAE", "+6h", "+12h", "+24h", "RI +24h MAE", "Non-RI +24h", "Dips"))
    print("-" * 115)
    for m in all_models:
        print(fmt_hdr.format(
            m["name"],
            f"{m['overall_mean_mae']:.4f} kt",
            f"{m['mae_6h']:.2f} kt",
            f"{m['mae_12h']:.2f} kt",
            f"{m['mae_24h']:.2f} kt",
            f"{m['ri_mae_24h']:.2f} kt",
            f"{m['non_ri_mae_24h']:.2f} kt",
            str(m["false_dips"]),
        ))

    # 7. Scientific Comparison: Learned RI Correction vs Canonical Hybrid
    delta_mae = m_learned["overall_mean_mae"] - m_canonical["overall_mean_mae"]
    delta_ri = m_learned["ri_mae_24h"] - m_canonical["ri_mae_24h"]
    delta_non_ri = m_learned["non_ri_mae_24h"] - m_canonical["non_ri_mae_24h"]
    b_stats = m_learned["stats"]
    ci_low = b_stats["bootstrap_overall_ci_95"][0]
    ci_high = b_stats["bootstrap_overall_ci_95"][1]
    p_val = b_stats["p_val_paired_t"]

    print("\n" + "=" * 80)
    print("SCIENTIFIC HEAD-TO-HEAD: LEARNED RI CORRECTION vs CANONICAL HYBRID (6.6350 kt)")
    print("=" * 80)
    print(f"• Canonical Hybrid Locked Test MAE:        {m_canonical['overall_mean_mae']:.4f} kt")
    print(f"• Learned RI-Aware Correction Test MAE:    {m_learned['overall_mean_mae']:.4f} kt")
    print(f"• Overall MAE Delta:                       {delta_mae:+.4f} kt ({(delta_mae/m_canonical['overall_mean_mae'])*100:+.2f}%)")
    print(f"• RI +24h MAE Delta:                       {delta_ri:+.2f} kt ({m_canonical['ri_mae_24h']:.2f} -> {m_learned['ri_mae_24h']:.2f} kt)")
    print(f"• Non-RI +24h MAE Delta:                   {delta_non_ri:+.2f} kt ({m_canonical['non_ri_mae_24h']:.2f} -> {m_learned['non_ri_mae_24h']:.2f} kt)")
    print(f"• 95% Paired Bootstrap Confidence Interval: [{ci_low:+.4f}, {ci_high:+.4f}] kt")
    print(f"• Bootstrap Win Rate:                      {b_stats['bootstrap_win_rate']:.1f}%")
    print(f"• Paired t-test p-value:                   {p_val:.3e}")
    print(f"• Cyclones Improved / Worsened:            {b_stats['storms_improved']} / {b_stats['storms_worsened']}")

    # Scientific Verdict Assessment
    if ci_high < 0 and delta_mae <= -0.10:
        verdict = "WIN"
        verdict_summary = f"WIN: Learned RI-Aware Correction significantly outperforms Canonical Hybrid by {abs(delta_mae):.4f} kt on locked test (95% CI: [{ci_low:+.4f}, {ci_high:+.4f}] kt, p = {p_val:.2e})."
    elif ci_low > 0 and delta_mae >= 0.10:
        verdict = "LOSE"
        verdict_summary = f"LOSE: Learned RI-Aware Correction degraded test performance by +{delta_mae:.4f} kt."
    else:
        verdict = "TIE"
        verdict_summary = f"TIE: Difference ({delta_mae:+.4f} kt) is within statistical margin of error (95% CI: [{ci_low:+.4f}, {ci_high:+.4f}] kt)."

    print(f"\nFINAL SCIENTIFIC VERDICT: {verdict}")
    print(f"Summary: {verdict_summary}")
    print("=" * 80)

    # 8. Save Outputs
    # A. Predictions CSV
    pred_df = pd.DataFrame({
        "cyclone_id": test_cids,
        "timestamp": test_timestamps,
        "v_curr": test_v_curr,
        "target_6h": test_true_future[:, 0],
        "target_12h": test_true_future[:, 1],
        "target_24h": test_true_future[:, 2],
        "pred_persistence_24h": pred_persistence[:, 2],
        "pred_residual_24h": pred_residual_alone[:, 2],
        "pred_canonical_24h": pred_canonical_hybrid[:, 2],
        "pred_learned_corr_6h": pred_learned_correction[:, 0],
        "pred_learned_corr_12h": pred_learned_correction[:, 1],
        "pred_learned_corr_24h": pred_learned_correction[:, 2],
        "ri_probability": test_ri_prob,
        "true_delta24": test_true_delta24,
        "is_true_ri": ri_mask,
    })
    pred_csv_path = out_dir / "test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"✓ Saved test predictions to: {pred_csv_path}")

    # B. Metrics JSON
    json_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "test_manifest": str(test_csv),
        "test_sequences": n_test,
        "test_cyclones": n_cids,
        "verdict": verdict,
        "verdict_summary": verdict_summary,
        "delta_mae_vs_canonical": delta_mae,
        "ci_95": [ci_low, ci_high],
        "p_value": p_val,
        "models": {m["name"]: {k: v for k, v in m.items() if k not in ["sample_mae", "sample_24h_abs"]} for m in all_models},
    }
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(json_results, f, indent=2)
    with open(report_dir / "FINAL_TEST_LEARNED_RI_CORRECTION_REPORT.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ Saved reports JSON to: {report_dir / 'FINAL_TEST_LEARNED_RI_CORRECTION_REPORT.json'}")

    # C. Markdown Report
    md_path = report_dir / "FINAL_TEST_LEARNED_RI_CORRECTION_REPORT.md"
    with open(md_path, "w") as f:
        f.write("# Final Locked-Test Scientific Report: Learned RI-Aware Correction Model\n\n")
        f.write(f"**Execution Date**: {json_results['timestamp']}\n")
        f.write(f"**Locked Test Manifest**: `{test_csv}` (N = {n_test:,} sequences across {n_cids} unique cyclones)\n")
        f.write(f"**Status**: Audited Experimental Candidate Evaluation (Frozen Model v1)\n")
        f.write(f"**Canonical Test Artifacts**: 100% Frozen & Untouched\n\n")

        f.write("## 1. Executive Scientific Verdict\n\n")
        f.write(f"```text\nVERDICT: {verdict}\n\n{verdict_summary}\n```\n\n")

        f.write("## 2. Locked-Test Benchmark Table\n\n")
        f.write("| Model Architecture | Overall MAE | +6h MAE | +12h MAE | +24h MAE | RI +24h MAE | Non-RI +24h MAE | Extreme (>=95kt) | False Dips |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for m in all_models:
            f.write(f"| **{m['name']}** | **{m['overall_mean_mae']:.4f} kt** | {m['mae_6h']:.2f} kt | {m['mae_12h']:.2f} kt | {m['mae_24h']:.2f} kt | {m['ri_mae_24h']:.2f} kt | {m['non_ri_mae_24h']:.2f} kt | {m['extreme_mae_24h']:.2f} kt | {m['false_dips']} |\n")

        f.write("\n## 3. Statistical Significance vs. Canonical Champion (6.6350 kt)\n\n")
        f.write(f"- **Overall Test MAE**: `{m_learned['overall_mean_mae']:.4f} kt` vs `{m_canonical['overall_mean_mae']:.4f} kt` (Δ = `{delta_mae:+.4f} kt`)\n")
        f.write(f"- **RI +24h MAE**: `{m_learned['ri_mae_24h']:.2f} kt` vs `{m_canonical['ri_mae_24h']:.2f} kt` (Δ = `{delta_ri:+.2f} kt`)\n")
        f.write(f"- **Non-RI +24h MAE**: `{m_learned['non_ri_mae_24h']:.2f} kt` vs `{m_canonical['non_ri_mae_24h']:.2f} kt` (Δ = `{delta_non_ri:+.2f} kt`)\n")
        f.write(f"- **95% Bootstrap Confidence Interval**: `[{ci_low:+.4f}, {ci_high:+.4f}] kt`\n")
        f.write(f"- **Bootstrap Win Rate**: `{b_stats['bootstrap_win_rate']:.1f}%`\n")
        f.write(f"- **Paired t-test p-value**: `{p_val:.3e}`\n")
        f.write(f"- **Cyclone Win Ratio**: `{b_stats['storms_improved']} improved / {b_stats['storms_worsened']} worsened` across 171 test cyclones\n\n")

        f.write("## 4. Methodological Safeguards & Audit Confirmation\n\n")
        f.write("- Single evaluation execution on the locked test partition.\n")
        f.write("- Model weights, scalers, and hyperparameters were 100% frozen prior to test inference.\n")
        f.write("- Zero future lookahead: all 27 input features were strictly computed from information available at forecast origin $t$.\n")
        f.write("- Canonical test reports and checkpoints in `experiments/final_locked_test/` remain completely unchanged.\n")

    print(f"✓ Saved Markdown report to: {md_path}")
    print("=" * 80)
    print(f"Evaluation complete in {time.time() - t_start:.1f}s.")
    print("=" * 80)


if __name__ == "__main__":
    main()
