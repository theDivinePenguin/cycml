#!/usr/bin/env python3
"""
Feature Extraction for Learned RI-Aware Correction Model.

Extracts causal time-t features for training and validation splits:
1. Base Residual Forecaster predictions (+6h, +12h, +24h)
2. Canonical Ridge Gate baseline predictions (+6h, +12h, +24h)
3. Dedicated RI Classifier probability and logit
4. Current intensity Vt and Vt / 100
5. Historical intensity evolution (6h delta, 12h delta, slope) from history_vmax
6. Causal environmental features (vmax, mslp, sst, cohc, shrd, rhmd)
7. Physical interaction features

Zero lookahead. Zero target leakage. Locked test set is NEVER accessed.
"""

import ast
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(".").resolve()))

from src.data.environmental import EnvironmentalFeatureManager, get_feature_dim
from src.models.residual_forecaster import ResidualDeltaVForecaster
from src.models.ri_models import DedicatedRIClassifier
from scripts.run_val_fusion_experiment import DualModelValidationDataset


def parse_history_vmax(hist_str: str) -> np.ndarray:
    """Parses history_vmax string e.g. '[45.0, 50.0, 55.0, 60.0, 65.0]'."""
    if isinstance(hist_str, list):
        return np.array(hist_str, dtype=np.float32)
    try:
        return np.array(ast.literal_eval(hist_str), dtype=np.float32)
    except Exception:
        # Fallback split
        clean = hist_str.strip("[]").split(",")
        return np.array([float(x.strip()) for x in clean if x.strip()], dtype=np.float32)


def compute_history_features(history_vmax_series: pd.Series, v_curr: np.ndarray) -> np.ndarray:
    """Computes:
    - recent_delta_6h: V_t - V_{t-6} (index -1 vs index -3 in 3h-sampled series)
    - recent_delta_12h: V_t - V_{t-12} (index -1 vs index 0 in K=5 series)
    - recent_slope: (V_t - V_{t-12}) / 12.0
    - history_std: std dev over observed sequence
    """
    n = len(history_vmax_series)
    feats = np.zeros((n, 4), dtype=np.float32)

    for i, raw in enumerate(history_vmax_series):
        arr = parse_history_vmax(raw)
        if len(arr) >= 5:
            # arr: [t-12, t-9, t-6, t-3, t]
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
            d6 = 0.0
            d12 = 0.0
            slope = 0.0
            std_val = 0.0
        feats[i] = [d6, d12, slope, std_val]

    return feats


def extract_split_features(
    split_name: str,
    manifest_csv: Path,
    device: torch.device,
    model_res: ResidualDeltaVForecaster,
    model_ri: DedicatedRIClassifier,
    env_manager: EnvironmentalFeatureManager,
    norm_stats: Dict,
    gate_intercept: np.ndarray,
    gate_coef: np.ndarray,
    out_cache: Path,
) -> Dict[str, np.ndarray]:
    print(f"\n--- Extracting features for split: {split_name} ({manifest_csv.name}) ---")
    df = pd.read_csv(manifest_csv)
    n_seq = len(df)
    n_cyc = df["cyclone_id"].nunique()
    print(f"Cohort: {n_seq:,} sequences across {n_cyc} cyclones")

    ds = DualModelValidationDataset(df, mean=norm_stats["mean"], std=norm_stats["std"])
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))

    v_curr_list = []
    true_future_list = []
    res_delta_list = []
    ri_probs_list = []
    ri_logits_list = []
    cids_list = []
    timestamps_list = []

    t0 = time.time()
    with torch.no_grad():
        for seq, vis, v_c, true_f, cids, tss in loader:
            seq = seq.to(device, non_blocking=True)
            vis = vis.to(device, non_blocking=True)
            v_c_dev = v_c.to(device).float()
            env_batch = torch.stack([env_manager.get_features(cids[i], int(tss[i])) for i in range(len(cids))]).to(device)

            _, d_hat = model_res(seq, v_curr=v_c_dev, vis_masks=vis)
            logits = model_ri(seq, vis_masks=vis, x_env=env_batch)

            v_curr_list.append(v_c.numpy())
            true_future_list.append(true_f.numpy())
            res_delta_list.append(d_hat.cpu().numpy())
            ri_probs_list.append(torch.sigmoid(logits).cpu().numpy().flatten())
            ri_logits_list.append(logits.cpu().numpy().flatten())
            cids_list.extend(cids)
            timestamps_list.extend(tss.numpy() if hasattr(tss, "numpy") else [int(t) for t in tss])

    print(f"Neural forward inference completed in {time.time() - t0:.1f}s")

    v_curr = np.concatenate(v_curr_list)
    true_future = np.concatenate(true_future_list, axis=0)
    res_delta = np.concatenate(res_delta_list, axis=0)
    ri_prob = np.concatenate(ri_probs_list)
    ri_logit = np.concatenate(ri_logits_list)
    cids = np.array(cids_list)
    timestamps = np.array(timestamps_list, dtype=np.int64)

    # Compute Canonical Ridge Baseline Predictions
    X_gate = np.column_stack([
        res_delta[:, 0],
        res_delta[:, 1],
        res_delta[:, 2],
        ri_prob,
        ri_logit,
        v_curr / 100.0,
        (v_curr / 100.0) * ri_prob,
    ])
    delta_base = np.zeros((n_seq, 3), dtype=np.float32)
    for h in range(3):
        delta_base[:, h] = gate_intercept[h] + X_gate @ gate_coef[h]

    # Compute Historical Evolution Features from history_vmax
    hist_feats = compute_history_features(df["history_vmax"], v_curr)

    # Extract Causal Environmental Features (12-d normalized: 6 values + 6 masks)
    env_list = [env_manager.get_features(cids[i], int(timestamps[i])).numpy() for i in range(n_seq)]
    env_feats = np.stack(env_list).astype(np.float32)  # (N, 12)
    # env_feats[:, :6] are normalized: vmax, mslp, sst, cohc, shrd, rhmd

    # Compute Physical Interaction Features:
    # 1. P_RI * pred_delta_24h
    # 2. P_RI * recent_delta_12h
    # 3. P_RI * (V_t / 100)
    # 4. P_RI * RI_logit
    # 5. P_RI * sst (sst is env_feats[:, 2])
    # 6. P_RI * shrd (shear is env_feats[:, 4])
    interactions = np.column_stack([
        ri_prob * res_delta[:, 2],
        ri_prob * hist_feats[:, 1],          # recent_delta_12h
        ri_prob * (v_curr / 100.0),
        ri_prob * ri_logit,
        ri_prob * env_feats[:, 2],          # sst
        ri_prob * env_feats[:, 4],          # shear
        ri_prob * delta_base[:, 2],         # P_RI * canonical delta24
    ]).astype(np.float32)

    # Build Unified Feature Matrix for Correction Model:
    # [0:3]   Base residual deltas (6h, 12h, 24h)
    # [3:6]   Canonical Ridge baseline deltas (6h, 12h, 24h)
    # [6:8]   RI classifier: P_RI, RI_logit
    # [8:10]  Current intensity: V_t, V_t/100
    # [10:14] Recent history: delta_6h, delta_12h, slope, std
    # [14:20] 6 Normalized Environmental variables: vmax, mslp, sst, cohc, shrd, rhmd
    # [20:27] 7 Interaction features
    X_correction = np.column_stack([
        res_delta,                           # 3
        delta_base,                          # 3
        ri_prob, ri_logit,                   # 2
        v_curr, v_curr / 100.0,              # 2
        hist_feats,                          # 4
        env_feats[:, :6],                    # 6
        interactions,                        # 7
    ]).astype(np.float32)

    feature_names = [
        "res_delta_6h", "res_delta_12h", "res_delta_24h",
        "ridge_base_6h", "ridge_base_12h", "ridge_base_24h",
        "P_RI", "logit_RI",
        "v_curr", "v_curr_div100",
        "recent_delta_6h", "recent_delta_12h", "recent_slope", "history_std",
        "env_vmax", "env_mslp", "env_sst", "env_cohc", "env_shrd", "env_rhmd",
        "interact_pri_x_res24", "interact_pri_x_d12", "interact_pri_x_vcurr",
        "interact_pri_x_logit", "interact_pri_x_sst", "interact_pri_x_shrd",
        "interact_pri_x_ridge24"
    ]

    print(f"Unified correction feature matrix shape: {X_correction.shape} (D = {len(feature_names)} features)")

    # Ground truth targets
    true_deltas = (true_future - v_curr[:, None]).astype(np.float32)
    residual_targets = (true_deltas - delta_base).astype(np.float32)  # Target error for correction model

    out_dict = {
        "X_correction": X_correction,
        "feature_names": np.array(feature_names),
        "v_curr": v_curr,
        "true_future": true_future,
        "true_deltas": true_deltas,
        "delta_base": delta_base,
        "residual_targets": residual_targets,
        "ri_prob": ri_prob,
        "ri_logit": ri_logit,
        "cids": cids,
        "timestamps": timestamps,
    }

    out_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_cache, **out_dict)
    print(f"✓ Cached {split_name} features to: {out_cache}")
    return out_dict


def main():
    print("=" * 80)
    print("FEATURE EXTRACTION PIPELINE FOR LEARNED RI-AWARE CORRECTION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    meta_dir = Path("data/metadata")
    train_csv = meta_dir / "forecast_train_sequences_k5_aligned.csv"
    val_csv = meta_dir / "forecast_val_sequences_k5_aligned.csv"
    norm_json = meta_dir / "normalization_stats_multichannel.json"

    res_ckpt_path = Path("experiments/checkpoints/residual_delta_v_unconstrained/best.pt")
    ri_ckpt_path = Path("experiments/checkpoints/ri_model1_dedicated_focal/best.pt")
    gate_path = Path("experiments/final_locked_test/final_frozen_ridge_gate.json")

    for p in [train_csv, val_csv, norm_json, res_ckpt_path, ri_ckpt_path, gate_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    out_dir = Path("experiments/ri_aware_correction")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(norm_json) as f:
        norm_stats = json.load(f)

    with open(gate_path) as f:
        gate_info = json.load(f)
    gate_intercept = np.array(gate_info["intercepts"])
    gate_coef = np.array(gate_info["coefficients"])

    # Load frozen neural checkpoints
    print("Loading frozen neural models...")
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

    env_manager = EnvironmentalFeatureManager(metadata_dir=meta_dir, feature_group="full_feature_set")

    # Extract Validation features
    val_cache = out_dir / "val_features_cache.npz"
    if val_cache.exists():
        print(f"Validation cache already exists: {val_cache}")
    else:
        extract_split_features(
            "validation", val_csv, device, model_res, model_ri,
            env_manager, norm_stats, gate_intercept, gate_coef, val_cache
        )

    # Extract Training features
    train_cache = out_dir / "train_features_cache.npz"
    if train_cache.exists():
        print(f"Training cache already exists: {train_cache}")
    else:
        extract_split_features(
            "training", train_csv, device, model_res, model_ri,
            env_manager, norm_stats, gate_intercept, gate_coef, train_cache
        )

    print("\nFeature extraction completed successfully.")


if __name__ == "__main__":
    main()
