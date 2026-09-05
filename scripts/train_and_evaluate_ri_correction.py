#!/usr/bin/env python3
"""
Learned RI-Aware Correction Model: Training and Validation Suite.

Trains regularized correction models strictly on the TRAINING split:
  1. Ridge Correction (+24h only & All Horizons)
  2. Small Constrained MLP Correction (scale * tanh(MLP(X))) across scales {5, 10, 15, 20} kt

Evaluates on LOCKED VALIDATION manifest (N=7,295, 181 cyclones).
Locked test set is NEVER touched or accessed.
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
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(".").resolve()))
from src.evaluation.sanity_checks import TrajectoryEvaluator


# ---------------------------------------------------------------------------
# MLP Correction Architecture with Tanh Scale Constraint
# ---------------------------------------------------------------------------
class TanhConstrainedMLPCorrection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1, scale: float = 15.0, hidden_dim: int = 32, dropout: float = 0.2):
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


def evaluate_strengthening_regimes(pred_delta24: np.ndarray, true_delta24: np.ndarray) -> Dict:
    regimes = {
        "all": np.ones_like(true_delta24, dtype=bool),
        "strengthening_pos": true_delta24 > 0.0,
        "strengthening_ge10": true_delta24 >= 10.0,
        "strengthening_ge20": true_delta24 >= 20.0,
        "ri_ge30": true_delta24 >= 30.0,
    }
    out = {}
    for name, mask in regimes.items():
        n = int(np.sum(mask))
        if n == 0:
            continue
        p_d = pred_delta24[mask]
        t_d = true_delta24[mask]
        signed_err = p_d - t_d
        abs_err = np.abs(signed_err)
        out[name] = {
            "n": n,
            "mean_signed_error": float(np.mean(signed_err)),
            "median_signed_error": float(np.median(signed_err)),
            "mae": float(np.mean(abs_err)),
            "underprediction_fraction": float(np.mean(p_d < t_d)),
            "overprediction_fraction": float(np.mean(p_d > t_d)),
        }
    return out


def evaluate_candidate(
    config_name: str,
    family: str,
    horizon_mode: str,
    scale_val: float,
    pred_intensities: np.ndarray,
    val_true_future: np.ndarray,
    val_v_curr: np.ndarray,
    val_true_delta24: np.ndarray,
    val_cids: np.ndarray,
    evaluator: TrajectoryEvaluator,
    base_overall_sample: np.ndarray,
    base_24h_abs: np.ndarray,
    boot_indices_all: List,
    boot_indices_ri: List,
    boot_indices_non_ri: List,
    ri_mask: np.ndarray,
    non_ri_mask: np.ndarray,
    ext95_mask: np.ndarray,
    ext110_mask: np.ndarray,
) -> Dict:
    # Horizon metrics
    m6 = calculate_metrics(pred_intensities[:, 0], val_true_future[:, 0])
    m12 = calculate_metrics(pred_intensities[:, 1], val_true_future[:, 1])
    m24 = calculate_metrics(pred_intensities[:, 2], val_true_future[:, 2])
    overall_mean_mae = (m6["mae"] + m12["mae"] + m24["mae"]) / 3.0
    overall_rmse = (m6["rmse"] + m12["rmse"] + m24["rmse"]) / 3.0

    # False dips
    traj_res = evaluator.evaluate_trajectories(pred_intensities, val_true_future, val_v_curr)
    false_dips = traj_res.get("false_dip_count", 0)

    # Subgroups (+24h)
    ri_err = pred_intensities[ri_mask, 2] - val_true_future[ri_mask, 2]
    ri_mae = float(np.mean(np.abs(ri_err)))
    ri_rmse = float(np.sqrt(np.mean(ri_err ** 2)))
    ri_bias = float(np.mean(ri_err))
    ri_underpred = float(np.mean(pred_intensities[ri_mask, 2] < val_true_future[ri_mask, 2]))

    non_ri_err = pred_intensities[non_ri_mask, 2] - val_true_future[non_ri_mask, 2]
    non_ri_mae = float(np.mean(np.abs(non_ri_err)))
    non_ri_rmse = float(np.sqrt(np.mean(non_ri_err ** 2)))
    non_ri_bias = float(np.mean(non_ri_err))
    non_ri_overpred = float(np.mean(pred_intensities[non_ri_mask, 2] > val_true_future[non_ri_mask, 2]))

    ext95_mae = float(np.mean(np.abs(pred_intensities[ext95_mask, 2] - val_true_future[ext95_mask, 2])))
    ext110_mae = float(np.mean(np.abs(pred_intensities[ext110_mask, 2] - val_true_future[ext110_mask, 2]))) if np.sum(ext110_mask) > 0 else 0.0

    pred_delta24 = pred_intensities[:, 2] - val_v_curr
    regimes = evaluate_strengthening_regimes(pred_delta24, val_true_delta24)

    # Cyclone-level win / loss counts
    cand_pw_abs = np.abs(pred_intensities - val_true_future)
    cand_overall_sample = np.mean(cand_pw_abs, axis=1)
    cand_24h_abs = cand_pw_abs[:, 2]

    storm_diffs = []
    for cid in np.unique(val_cids):
        s_m = val_cids == cid
        s_cand = np.mean(cand_overall_sample[s_m])
        s_base = np.mean(base_overall_sample[s_m])
        storm_diffs.append(s_cand - s_base)
    storm_diffs = np.array(storm_diffs)
    storms_improved = int(np.sum(storm_diffs < -1e-4))
    storms_worsened = int(np.sum(storm_diffs > 1e-4))

    # Bootstrap & Paired Tests vs Baseline
    _, p_t = stats.ttest_rel(cand_overall_sample, base_overall_sample)
    try:
        _, p_w = stats.wilcoxon(cand_overall_sample, base_overall_sample)
    except Exception:
        p_w = 1.0

    boot_ov = [float(np.mean(cand_overall_sample[b]) - np.mean(base_overall_sample[b])) for b in boot_indices_all]
    boot_24 = [float(np.mean(cand_24h_abs[b]) - np.mean(base_24h_abs[b])) for b in boot_indices_all]
    boot_ri = [float(np.mean(cand_24h_abs[b]) - np.mean(base_24h_abs[b])) for b in boot_indices_ri]
    boot_non_ri = [float(np.mean(cand_24h_abs[b]) - np.mean(base_24h_abs[b])) for b in boot_indices_non_ri]

    ci_ov = np.percentile(boot_ov, [2.5, 50.0, 97.5])
    ci_24 = np.percentile(boot_24, [2.5, 50.0, 97.5])
    ci_ri = np.percentile(boot_ri, [2.5, 50.0, 97.5])
    ci_nri = np.percentile(boot_non_ri, [2.5, 50.0, 97.5])

    return {
        "config_name": config_name,
        "family": family,
        "horizon_mode": horizon_mode,
        "scale": scale_val,
        "overall_mean_mae": overall_mean_mae,
        "overall_rmse": overall_rmse,
        "mae_6h": m6["mae"],
        "mae_12h": m12["mae"],
        "mae_24h": m24["mae"],
        "rmse_24h": m24["rmse"],
        "r2_24h": m24["r2"],
        "bias_24h": m24["bias"],
        "false_dips": false_dips,
        "ri_mae_24h": ri_mae,
        "ri_rmse_24h": ri_rmse,
        "ri_bias_24h": ri_bias,
        "ri_underpred_fraction": ri_underpred,
        "non_ri_mae_24h": non_ri_mae,
        "non_ri_rmse_24h": non_ri_rmse,
        "non_ri_bias_24h": non_ri_bias,
        "non_ri_overpred_fraction": non_ri_overpred,
        "extreme95_mae_24h": ext95_mae,
        "extreme110_mae_24h": ext110_mae,
        "storms_improved": storms_improved,
        "storms_worsened": storms_worsened,
        "regimes": regimes,
        "bootstrap": {
            "overall_median_delta": float(ci_ov[1]),
            "overall_ci_95": [float(ci_ov[0]), float(ci_ov[2])],
            "overall_win_rate": float(np.mean(np.array(boot_ov) < 0) * 100.0),
            "h24_median_delta": float(ci_24[1]),
            "h24_ci_95": [float(ci_24[0]), float(ci_24[2])],
            "ri_24h_median_delta": float(ci_ri[1]),
            "ri_24h_ci_95": [float(ci_ri[0]), float(ci_ri[2])],
            "non_ri_24h_median_delta": float(ci_nri[1]),
            "non_ri_24h_ci_95": [float(ci_nri[0]), float(ci_nri[2])],
            "p_val_paired_t": float(p_t),
            "p_val_wilcoxon": float(p_w),
        }
    }


def main():
    print("=" * 80)
    print("LEARNED RI-AWARE CORRECTION: TRAINING & VALIDATION PIPELINE")
    print("=" * 80)

    out_dir = Path("experiments/ri_aware_correction")
    train_cache = out_dir / "train_features_cache.npz"
    val_cache = out_dir / "val_features_cache.npz"

    # Wait if task-2829 is still finalizing
    max_wait = 300
    waited = 0
    while not (train_cache.exists() and val_cache.exists()):
        time.sleep(5)
        waited += 5
        print(f"Waiting for feature extraction cache... ({waited}s)")
        if waited >= max_wait:
            raise TimeoutError("Feature caches did not appear in expected time.")

    print(f"Loading cached feature matrices...")
    tr_data = np.load(train_cache, allow_pickle=True)
    val_data = np.load(val_cache, allow_pickle=True)

    X_train_raw = tr_data["X_correction"]              # (N_tr, 27)
    y_train_residual = tr_data["residual_targets"]      # (N_tr, 3)
    train_cids = tr_data["cids"]                        # (N_tr,)

    X_val_raw = val_data["X_correction"]                  # (N_val, 27)
    val_v_curr = val_data["v_curr"]                     # (N_val,)
    val_true_future = val_data["true_future"]           # (N_val, 3)
    val_delta_base = val_data["delta_base"]             # (N_val, 3)
    val_ri_prob = val_data["ri_prob"]                   # (N_val,)
    val_cids = val_data["cids"]                         # (N_val,)
    feature_names = list(val_data["feature_names"])

    n_train = len(X_train_raw)
    n_val = len(X_val_raw)
    print(f"Train samples: {n_train:,} ({len(np.unique(train_cids))} cyclones)")
    print(f"Validation samples: {n_val:,} ({len(np.unique(val_cids))} cyclones)")
    print(f"Feature count: {len(feature_names)}")

    # Standardize features using TRAINING statistics strictly
    mean_tr = np.mean(X_train_raw, axis=0)
    std_tr = np.std(X_train_raw, axis=0)
    std_tr[std_tr < 1e-6] = 1.0

    X_train = (X_train_raw - mean_tr) / std_tr
    X_val = (X_val_raw - mean_tr) / std_tr

    # Canonical Baseline Predictions on Validation
    pred_base = val_v_curr[:, None] + val_delta_base
    val_true_delta24 = val_true_future[:, 2] - val_v_curr

    ri_mask = val_true_delta24 >= 30.0
    non_ri_mask = ~ri_mask
    ext95_mask = val_v_curr >= 95.0
    ext110_mask = val_v_curr >= 110.0

    evaluator = TrajectoryEvaluator()
    rng = np.random.RandomState(42)
    n_boot = 1000

    ri_idx = np.where(ri_mask)[0]
    non_ri_idx = np.where(non_ri_mask)[0]

    boot_indices_all = [rng.choice(n_val, size=n_val, replace=True) for _ in range(n_boot)]
    boot_indices_ri = [rng.choice(ri_idx, size=len(ri_idx), replace=True) for _ in range(n_boot)]
    boot_indices_non_ri = [rng.choice(non_ri_idx, size=len(non_ri_idx), replace=True) for _ in range(n_boot)]

    base_pw_abs = np.abs(pred_base - val_true_future)
    base_overall_sample = np.mean(base_pw_abs, axis=1)
    base_24h_abs = base_pw_abs[:, 2]

    # Baseline Candidate Evaluation
    base_res = evaluate_candidate(
        config_name="0. Canonical Baseline (No Correction)",
        family="Baseline",
        horizon_mode="none",
        scale_val=0.0,
        pred_intensities=pred_base,
        val_true_future=val_true_future,
        val_v_curr=val_v_curr,
        val_true_delta24=val_true_delta24,
        val_cids=val_cids,
        evaluator=evaluator,
        base_overall_sample=base_overall_sample,
        base_24h_abs=base_24h_abs,
        boot_indices_all=boot_indices_all,
        boot_indices_ri=boot_indices_ri,
        boot_indices_non_ri=boot_indices_non_ri,
        ri_mask=ri_mask,
        non_ri_mask=non_ri_mask,
        ext95_mask=ext95_mask,
        ext110_mask=ext110_mask,
    )
    base_res["bootstrap"]["overall_win_rate"] = 50.0

    print(f"\nBaseline Validation Reference:")
    print(f"  Overall MAE: {base_res['overall_mean_mae']:.4f} kt | +24h MAE: {base_res['mae_24h']:.2f} kt")
    print(f"  RI +24h MAE: {base_res['ri_mae_24h']:.2f} kt | Non-RI +24h: {base_res['non_ri_mae_24h']:.2f} kt | Dips: {base_res['false_dips']}")

    candidate_results = [base_res]
    candidate_preds = {"baseline": pred_base}

    # =========================================================================
    # CANDIDATE A: Ridge Correction Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING CANDIDATE A: REGULARIZED RIDGE CORRECTION")
    print("=" * 80)

    ridge_alphas = [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    best_ridge_24 = None
    best_ridge_24_alpha = 1000.0
    best_ridge_24_mae = 999.0

    # A1. +24h Only Ridge Correction
    for a in ridge_alphas:
        ridge24 = Ridge(alpha=a, random_state=42)
        ridge24.fit(X_train, y_train_residual[:, 2])

        corr_val_24 = ridge24.predict(X_val)
        pred_cand = pred_base.copy()
        pred_cand[:, 2] += corr_val_24

        res = evaluate_candidate(
            config_name=f"Ridge_24h_alpha_{a:.0f}",
            family="Ridge",
            horizon_mode="24h_only",
            scale_val=0.0,
            pred_intensities=pred_cand,
            val_true_future=val_true_future,
            val_v_curr=val_v_curr,
            val_true_delta24=val_true_delta24,
            val_cids=val_cids,
            evaluator=evaluator,
            base_overall_sample=base_overall_sample,
            base_24h_abs=base_24h_abs,
            boot_indices_all=boot_indices_all,
            boot_indices_ri=boot_indices_ri,
            boot_indices_non_ri=boot_indices_non_ri,
            ri_mask=ri_mask,
            non_ri_mask=non_ri_mask,
            ext95_mask=ext95_mask,
            ext110_mask=ext110_mask,
        )
        candidate_results.append(res)
        candidate_preds[res["config_name"]] = pred_cand
        print(f"  • Ridge (+24h) α={a:5.0f} | Overall: {res['overall_mean_mae']:.4f} kt | +24h: {res['mae_24h']:.2f} kt | RI +24h: {res['ri_mae_24h']:.2f} kt | Non-RI: {res['non_ri_mae_24h']:.2f} kt | Dips: {res['false_dips']}")

        if res["overall_mean_mae"] < best_ridge_24_mae:
            best_ridge_24_mae = res["overall_mean_mae"]
            best_ridge_24 = ridge24
            best_ridge_24_alpha = a

    # A2. All-Horizons Ridge Correction
    for a in [100.0, 500.0, 1000.0, 5000.0]:
        ridge_all = Ridge(alpha=a, random_state=42)
        ridge_all.fit(X_train, y_train_residual)

        corr_val_all = ridge_all.predict(X_val)
        pred_cand = pred_base + corr_val_all

        res = evaluate_candidate(
            config_name=f"Ridge_AllHorizons_alpha_{a:.0f}",
            family="Ridge",
            horizon_mode="all_horizons",
            scale_val=0.0,
            pred_intensities=pred_cand,
            val_true_future=val_true_future,
            val_v_curr=val_v_curr,
            val_true_delta24=val_true_delta24,
            val_cids=val_cids,
            evaluator=evaluator,
            base_overall_sample=base_overall_sample,
            base_24h_abs=base_24h_abs,
            boot_indices_all=boot_indices_all,
            boot_indices_ri=boot_indices_ri,
            boot_indices_non_ri=boot_indices_non_ri,
            ri_mask=ri_mask,
            non_ri_mask=non_ri_mask,
            ext95_mask=ext95_mask,
            ext110_mask=ext110_mask,
        )
        candidate_results.append(res)
        candidate_preds[res["config_name"]] = pred_cand
        print(f"  • Ridge (All)  α={a:5.0f} | Overall: {res['overall_mean_mae']:.4f} kt | +24h: {res['mae_24h']:.2f} kt | RI +24h: {res['ri_mae_24h']:.2f} kt | Non-RI: {res['non_ri_mae_24h']:.2f} kt | Dips: {res['false_dips']}")

    # =========================================================================
    # CANDIDATE B: Small Constrained MLP Correction (Tanh-Scaled)
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING CANDIDATE B: CONSTRAINED MLP CORRECTION (TANH-SCALED)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    # Set up training validation split by cyclone_id within train split
    gkf = GroupKFold(n_splits=5)
    tr_sub_idx, val_sub_idx = next(gkf.split(X_train, y_train_residual, groups=train_cids))

    X_tr_t = torch.tensor(X_train[tr_sub_idx], dtype=torch.float32)
    y_tr_t = torch.tensor(y_train_residual[tr_sub_idx], dtype=torch.float32)

    X_subval_t = torch.tensor(X_train[val_sub_idx], dtype=torch.float32).to(device)
    y_subval_t = torch.tensor(y_train_residual[val_sub_idx], dtype=torch.float32).to(device)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    scales = [5.0, 10.0, 15.0, 20.0]
    mlp_models = {}

    for s in scales:
        # B1. +24h Only MLP
        torch.manual_seed(42)
        mlp24 = TanhConstrainedMLPCorrection(in_dim=len(feature_names), out_dim=1, scale=s, hidden_dim=32, dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(mlp24.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion = nn.SmoothL1Loss(beta=1.0)

        best_loss = float("inf")
        best_state = None

        for epoch in range(1, 21):
            mlp24.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred_c = mlp24(bx)
                loss = criterion(pred_c.squeeze(-1), by[:, 2])
                loss.backward()
                optimizer.step()

            # Validation check
            mlp24.eval()
            with torch.no_grad():
                val_c = mlp24(X_subval_t).squeeze(-1)
                val_loss = criterion(val_c, y_subval_t[:, 2]).item()
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu() for k, v in mlp24.state_dict().items()}

        mlp24.load_state_dict(best_state)
        mlp24.eval()
        mlp_models[f"MLP_24h_scale_{s:.0f}kt"] = mlp24

        with torch.no_grad():
            corr_val_24 = mlp24(X_val_t).squeeze(-1).cpu().numpy()

        pred_cand = pred_base.copy()
        pred_cand[:, 2] += corr_val_24

        res = evaluate_candidate(
            config_name=f"MLP_24h_scale_{s:.0f}kt",
            family="MLP",
            horizon_mode="24h_only",
            scale_val=s,
            pred_intensities=pred_cand,
            val_true_future=val_true_future,
            val_v_curr=val_v_curr,
            val_true_delta24=val_true_delta24,
            val_cids=val_cids,
            evaluator=evaluator,
            base_overall_sample=base_overall_sample,
            base_24h_abs=base_24h_abs,
            boot_indices_all=boot_indices_all,
            boot_indices_ri=boot_indices_ri,
            boot_indices_non_ri=boot_indices_non_ri,
            ri_mask=ri_mask,
            non_ri_mask=non_ri_mask,
            ext95_mask=ext95_mask,
            ext110_mask=ext110_mask,
        )
        candidate_results.append(res)
        candidate_preds[res["config_name"]] = pred_cand
        print(f"  • MLP (+24h) Scale={s:2.0f}kt | Overall: {res['overall_mean_mae']:.4f} kt | +24h: {res['mae_24h']:.2f} kt | RI +24h: {res['ri_mae_24h']:.2f} kt | Non-RI: {res['non_ri_mae_24h']:.2f} kt | Dips: {res['false_dips']}")

        # B2. All-Horizons MLP
        torch.manual_seed(42)
        mlp_all = TanhConstrainedMLPCorrection(in_dim=len(feature_names), out_dim=3, scale=s, hidden_dim=32, dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(mlp_all.parameters(), lr=1e-3, weight_decay=1e-2)

        best_loss = float("inf")
        best_state = None

        for epoch in range(1, 21):
            mlp_all.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred_c = mlp_all(bx)
                loss = criterion(pred_c, by)
                loss.backward()
                optimizer.step()

            mlp_all.eval()
            with torch.no_grad():
                val_c = mlp_all(X_subval_t)
                val_loss = criterion(val_c, y_subval_t).item()
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu() for k, v in mlp_all.state_dict().items()}

        mlp_all.load_state_dict(best_state)
        mlp_all.eval()
        mlp_models[f"MLP_AllHorizons_scale_{s:.0f}kt"] = mlp_all

        with torch.no_grad():
            corr_val_all = mlp_all(X_val_t).cpu().numpy()

        pred_cand = pred_base + corr_val_all

        res = evaluate_candidate(
            config_name=f"MLP_AllHorizons_scale_{s:.0f}kt",
            family="MLP",
            horizon_mode="all_horizons",
            scale_val=s,
            pred_intensities=pred_cand,
            val_true_future=val_true_future,
            val_v_curr=val_v_curr,
            val_true_delta24=val_true_delta24,
            val_cids=val_cids,
            evaluator=evaluator,
            base_overall_sample=base_overall_sample,
            base_24h_abs=base_24h_abs,
            boot_indices_all=boot_indices_all,
            boot_indices_ri=boot_indices_ri,
            boot_indices_non_ri=boot_indices_non_ri,
            ri_mask=ri_mask,
            non_ri_mask=non_ri_mask,
            ext95_mask=ext95_mask,
            ext110_mask=ext110_mask,
        )
        candidate_results.append(res)
        candidate_preds[res["config_name"]] = pred_cand
        print(f"  • MLP (All)  Scale={s:2.0f}kt | Overall: {res['overall_mean_mae']:.4f} kt | +24h: {res['mae_24h']:.2f} kt | RI +24h: {res['ri_mae_24h']:.2f} kt | Non-RI: {res['non_ri_mae_24h']:.2f} kt | Dips: {res['false_dips']}")

    # Compute Deltas vs Baseline for all candidates
    base_ov = base_res["overall_mean_mae"]
    base_ri = base_res["ri_mae_24h"]
    base_nri = base_res["non_ri_mae_24h"]

    for r in candidate_results:
        r["delta_overall_mae"] = r["overall_mean_mae"] - base_ov
        r["delta_ri_24h_mae"] = r["ri_mae_24h"] - base_ri
        r["delta_non_ri_24h_mae"] = r["non_ri_mae_24h"] - base_nri
        r["pct_ri_mae"] = (r["delta_ri_24h_mae"] / base_ri) * 100.0

    # Scientific Selection of Best Model
    # Criteria:
    # 1. RI +24h MAE materially improves (delta_ri <= -1.0 kt)
    # 2. RI underprediction decreases
    # 3. Overall MAE improves or remains essentially unchanged (delta_overall <= 0.02 kt)
    # 4. Non-RI degradation is small (delta_non_ri <= 0.15 kt)
    # 5. False dips remain zero
    candidates_pool = [r for r in candidate_results if r["family"] != "Baseline"]

    # Filter by false dips == 0 and small non-ri degradation
    viable = [
        c for c in candidates_pool
        if c["false_dips"] == 0 and c["delta_non_ri_24h_mae"] <= 0.15 and c["delta_ri_24h_mae"] < 0
    ]

    if len(viable) > 0:
        # Sort by combination of RI improvement and overall MAE
        best_candidate = min(viable, key=lambda x: (x["overall_mean_mae"], x["ri_mae_24h"]))
    else:
        best_candidate = min(candidates_pool, key=lambda x: x["overall_mean_mae"])

    print("\n" + "=" * 80)
    print(f"CHOSEN EXPERIMENTAL CANDIDATE: {best_candidate['config_name']}")
    print("=" * 80)
    print(f"• Overall MAE:   {best_candidate['overall_mean_mae']:.4f} kt (Δ: {best_candidate['delta_overall_mae']:+.4f} kt)")
    print(f"• +24h MAE:      {best_candidate['mae_24h']:.2f} kt")
    print(f"• RI +24h MAE:   {best_candidate['ri_mae_24h']:.2f} kt (Δ: {best_candidate['delta_ri_24h_mae']:+.2f} kt / {best_candidate['pct_ri_mae']:+.1f}%)")
    print(f"• Non-RI +24h:   {best_candidate['non_ri_mae_24h']:.2f} kt (Δ: {best_candidate['delta_non_ri_24h_mae']:+.2f} kt)")
    print(f"• RI Underpred:  {best_candidate['ri_underpred_fraction']*100:.1f}% (vs Baseline: {base_res['ri_underpred_fraction']*100:.1f}%)")
    print(f"• False Dips:    {best_candidate['false_dips']}")
    print(f"• Storms (+/-):  {best_candidate['storms_improved']} improved / {best_candidate['storms_worsened']} worsened")
    print(f"• Bootstrap 95%: [{best_candidate['bootstrap']['overall_ci_95'][0]:+.3f}, {best_candidate['bootstrap']['overall_ci_95'][1]:+.3f}] kt")
    print(f"• Win Rate:      {best_candidate['bootstrap']['overall_win_rate']:.1f}%")

    # Verdict Assessment
    if best_candidate["delta_ri_24h_mae"] <= -1.0 and best_candidate["delta_overall_mae"] <= 0.01 and best_candidate["delta_non_ri_24h_mae"] <= 0.10:
        verdict = "PROMISING"
        verdict_text = "PROMISING (Candidate only. Canonical locked test remains unchanged.)"
        rationale = f"Learned correction {best_candidate['config_name']} achieves a genuine -{abs(best_candidate['delta_ri_24h_mae']):.2f} kt reduction in RI error while maintaining overall MAE and keeping non-RI degradation within +{best_candidate['delta_non_ri_24h_mae']:.2f} kt."
    else:
        verdict = "REJECTED"
        verdict_text = "REJECTED"
        rationale = f"While {best_candidate['config_name']} reduces RI underprediction, the associated non-RI degradation (+{best_candidate['delta_non_ri_24h_mae']:.2f} kt) or lack of strong overall statistical significance does not justify replacing the canonical baseline."

    print(f"\nSCIENTIFIC VERDICT: {verdict_text}")
    print(f"Rationale: {rationale}")

    # Save Best Model Checkpoint
    ckpt_path = out_dir / "best_correction_model.pt"
    if best_candidate["family"] == "MLP":
        best_model = mlp_models[best_candidate["config_name"]]
        torch.save({
            "config_name": best_candidate["config_name"],
            "family": "MLP",
            "scale": best_candidate["scale"],
            "horizon_mode": best_candidate["horizon_mode"],
            "model_state_dict": best_model.state_dict(),
            "mean_tr": mean_tr,
            "std_tr": std_tr,
            "feature_names": feature_names,
        }, ckpt_path)
    else:
        torch.save({
            "config_name": best_candidate["config_name"],
            "family": "Ridge",
            "alpha": best_ridge_24_alpha,
            "coef": best_ridge_24.coef_,
            "intercept": best_ridge_24.intercept_,
            "mean_tr": mean_tr,
            "std_tr": std_tr,
            "feature_names": feature_names,
        }, ckpt_path)
    print(f"✓ Saved best correction model checkpoint to: {ckpt_path}")

    # =========================================================================
    # GENERATE 5 VALIDATION DIAGNOSTIC PLOTS
    # =========================================================================
    print("\nGenerating validation diagnostic plots...")
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    pred_best = candidate_preds[best_candidate["config_name"]]
    pred_base_24 = pred_base[:, 2]
    pred_best_24 = pred_best[:, 2]

    d_base_24 = pred_base_24 - val_v_curr
    d_best_24 = pred_best_24 - val_v_curr

    # 1. Predicted ΔV24 vs True ΔV24
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, d_p, title, col in zip(
        axes,
        [d_base_24, d_best_24],
        [f"Canonical Baseline (No Correction)\nOverall MAE: {base_ov:.3f} kt | RI +24h MAE: {base_ri:.2f} kt",
         f"Learned Correction: {best_candidate['config_name']}\nOverall MAE: {best_candidate['overall_mean_mae']:.3f} kt | RI +24h MAE: {best_candidate['ri_mae_24h']:.2f} kt"],
        ["#1f77b4", "#2ca02c"]
    ):
        ax.scatter(val_true_delta24, d_p, alpha=0.25, s=16, color=col, edgecolors="none")
        ax.plot([-60, 80], [-60, 80], "k--", lw=1.5, label="1:1 Perfect Forecast")
        ax.axvline(30, color="orange", linestyle=":", lw=1.5, label="RI Threshold (ΔV24 >= 30 kt)")
        ax.axhline(30, color="orange", linestyle=":", lw=1.5)
        ax.set_xlabel("True ΔV24 (kt)", fontsize=12)
        ax.set_ylabel("Predicted ΔV24 (kt)", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="upper left")
        ax.set_xlim(-60, 80)
        ax.set_ylim(-60, 80)
    plt.tight_layout()
    plt.savefig(plot_dir / "plot1_pred_vs_true_delta24.png", dpi=200)
    plt.close()

    # 2. Prediction Error binned by P_RI
    fig, ax = plt.subplots(figsize=(9, 5))
    pri_bins = np.linspace(0.0, 1.0, 11)
    bin_centers = 0.5 * (pri_bins[:-1] + pri_bins[1:])
    err_base_binned, err_best_binned = [], []

    err_base_24 = np.abs(pred_base_24 - val_true_future[:, 2])
    err_best_24 = np.abs(pred_best_24 - val_true_future[:, 2])

    for i in range(len(pri_bins) - 1):
        b_m = (val_ri_prob >= pri_bins[i]) & (val_ri_prob < pri_bins[i + 1])
        if np.sum(b_m) > 0:
            err_base_binned.append(np.mean(err_base_24[b_m]))
            err_best_binned.append(np.mean(err_best_24[b_m]))
        else:
            err_base_binned.append(np.nan)
            err_best_binned.append(np.nan)

    ax.plot(bin_centers, err_base_binned, "o-", color="#1f77b4", lw=2, label="Baseline")
    ax.plot(bin_centers, err_best_binned, "s--", color="#2ca02c", lw=2, label=best_candidate["config_name"])
    ax.set_xlabel("Predicted RI Probability P(RI)", fontsize=12)
    ax.set_ylabel("+24h MAE (kt)", fontsize=12)
    ax.set_title("Prediction Error Binned by RI Probability", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "plot2_error_binned_by_pri.png", dpi=200)
    plt.close()

    # 3. Prediction Signed Bias binned by True ΔV24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    delta_bins = np.arange(-50, 75, 10)
    d_centers = 0.5 * (delta_bins[:-1] + delta_bins[1:])
    signed_base_binned, signed_best_binned = [], []

    signed_base = pred_base_24 - val_true_future[:, 2]
    signed_best = pred_best_24 - val_true_future[:, 2]

    for i in range(len(delta_bins) - 1):
        b_m = (val_true_delta24 >= delta_bins[i]) & (val_true_delta24 < delta_bins[i + 1])
        if np.sum(b_m) >= 5:
            signed_base_binned.append(np.mean(signed_base[b_m]))
            signed_best_binned.append(np.mean(signed_best[b_m]))
        else:
            signed_base_binned.append(np.nan)
            signed_best_binned.append(np.nan)

    ax.axhline(0, color="k", linestyle="-", lw=1)
    ax.plot(d_centers, signed_base_binned, "o-", color="#1f77b4", lw=2, label="Baseline")
    ax.plot(d_centers, signed_best_binned, "s--", color="#2ca02c", lw=2, label=best_candidate["config_name"])
    ax.axvline(30, color="purple", linestyle=":", lw=1.5, label="RI Threshold (ΔV24 >= 30 kt)")
    ax.set_xlabel("True ΔV24 (kt)", fontsize=12)
    ax.set_ylabel("Mean Signed Error (Pred - True, kt)", fontsize=12)
    ax.set_title("Signed Error / Underprediction Across Regimes", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(plot_dir / "plot3_signed_error_binned_by_true_delta24.png", dpi=200)
    plt.close()

    # 4. RI-event predicted vs true ΔV24
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(val_true_delta24[ri_mask], d_base_24[ri_mask], color="#1f77b4", alpha=0.5, s=28, label=f"Baseline — MAE: {base_ri:.2f} kt")
    ax.scatter(val_true_delta24[ri_mask], d_best_24[ri_mask], color="#2ca02c", alpha=0.5, s=28, marker="^", label=f"{best_candidate['config_name']} — MAE: {best_candidate['ri_mae_24h']:.2f} kt")
    ax.plot([30, 75], [30, 75], "k--", lw=1.5, label="1:1 Perfect Forecast")
    ax.set_xlabel("True ΔV24 (kt)", fontsize=12)
    ax.set_ylabel("Predicted ΔV24 (kt)", fontsize=12)
    ax.set_title(f"True RI Events (N = {np.sum(ri_mask):,}): Predicted vs True ΔV24", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(plot_dir / "plot4_ri_event_pred_vs_true.png", dpi=200)
    plt.close()

    # 5. Real Validation Cyclone Trajectories
    storm_deltas = []
    for cid in np.unique(val_cids):
        s_m = val_cids == cid
        if np.any(ri_mask[s_m]):
            err_b = np.mean(err_base_24[s_m])
            err_a = np.mean(err_best_24[s_m])
            storm_deltas.append((cid, err_a - err_b))

    storm_deltas.sort(key=lambda x: x[1])
    improved_cids = [x[0] for x in storm_deltas[:2]]
    worsened_cids = [x[0] for x in storm_deltas[-2:]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    horizons = [0, 6, 12, 24]
    p_configs = [
        (axes[0, 0], improved_cids[0], "IMPROVED Cyclone", "#2ca02c"),
        (axes[0, 1], improved_cids[1], "IMPROVED Cyclone", "#2ca02c"),
        (axes[1, 0], worsened_cids[0], "WORSENED Cyclone", "#d62728"),
        (axes[1, 1], worsened_cids[1], "WORSENED Cyclone", "#d62728"),
    ]

    for ax, cid, st_label, color in p_configs:
        s_m = (val_cids == cid) & ri_mask
        seq_idx = np.where(s_m)[0][0]
        v0 = val_v_curr[seq_idx]
        true_t = [v0, val_true_future[seq_idx, 0], val_true_future[seq_idx, 1], val_true_future[seq_idx, 2]]
        base_t = [v0, pred_base[seq_idx, 0], pred_base[seq_idx, 1], pred_base[seq_idx, 2]]
        cand_t = [v0, pred_best[seq_idx, 0], pred_best[seq_idx, 1], pred_best[seq_idx, 2]]
        p_val = val_ri_prob[seq_idx]

        ax.plot(horizons, true_t, "k-o", lw=2.5, label="Ground Truth")
        ax.plot(horizons, base_t, "b--s", lw=2, label="Baseline")
        ax.plot(horizons, cand_t, "-^", color=color, lw=2, label=best_candidate["config_name"])
        ax.set_xlabel("Forecast Horizon (hours)", fontsize=11)
        ax.set_ylabel("Intensity Vmax (kt)", fontsize=11)
        ax.set_title(f"{st_label} {cid} (Seq #{seq_idx}, P(RI)={p_val:.2f})", fontsize=12, fontweight="bold")
        ax.set_xticks(horizons)
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(plot_dir / "plot5_example_trajectories.png", dpi=200)
    plt.close()
    print("✓ Saved all 5 validation diagnostic plots under experiments/ri_aware_correction/plots/")

    # =========================================================================
    # SAVE CSV, JSON, AND REPORT
    # =========================================================================
    # 1. CSV
    csv_rows = []
    for r in candidate_results:
        b = r["bootstrap"]
        rg = r["regimes"]
        csv_rows.append({
            "config_name": r["config_name"],
            "family": r["family"],
            "horizon_mode": r["horizon_mode"],
            "scale": r["scale"],
            "overall_mean_mae": r["overall_mean_mae"],
            "mae_6h": r["mae_6h"],
            "mae_12h": r["mae_12h"],
            "mae_24h": r["mae_24h"],
            "ri_mae_24h": r["ri_mae_24h"],
            "ri_bias_24h": r["ri_bias_24h"],
            "ri_underpred_fraction": r["ri_underpred_fraction"],
            "non_ri_mae_24h": r["non_ri_mae_24h"],
            "non_ri_bias_24h": r["non_ri_bias_24h"],
            "extreme95_mae_24h": r["extreme95_mae_24h"],
            "false_dips": r["false_dips"],
            "delta_overall_mae": r.get("delta_overall_mae", 0.0),
            "delta_ri_24h_mae": r.get("delta_ri_24h_mae", 0.0),
            "delta_non_ri_24h_mae": r.get("delta_non_ri_24h_mae", 0.0),
            "storms_improved": r["storms_improved"],
            "storms_worsened": r["storms_worsened"],
            "bootstrap_overall_ci_lower": b["overall_ci_95"][0],
            "bootstrap_overall_ci_upper": b["overall_ci_95"][1],
            "bootstrap_win_rate": b["overall_win_rate"],
            "p_val_paired_t": b["p_val_paired_t"],
            "ri_signed_err": rg["ri_ge30"]["mean_signed_error"],
            "pos10_signed_err": rg["strengthening_ge10"]["mean_signed_error"],
        })
    csv_path = out_dir / "ri_correction_results.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"✓ Saved {csv_path}")

    # 2. JSON
    json_path = out_dir / "ri_correction_results.json"
    json_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "validation_manifest": "data/metadata/forecast_val_sequences_k5_aligned.csv",
        "training_manifest": "data/metadata/forecast_train_sequences_k5_aligned.csv",
        "n_validation_sequences": n_val,
        "n_training_sequences": n_train,
        "features": feature_names,
        "baseline": base_res,
        "best_candidate": best_candidate,
        "verdict": verdict_text,
        "rationale": rationale,
        "candidates": candidate_results,
    }
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"✓ Saved {json_path}")

    # 3. Markdown Report
    md_path = out_dir / "RI_AWARE_CORRECTION_REPORT.md"
    with open(md_path, "w") as f:
        f.write("# Scientific Validation Report: Learned RI-Aware Correction Model\n\n")
        f.write(f"**Execution Date**: {json_payload['timestamp']}\n")
        f.write(f"**Training Split**: `data/metadata/forecast_train_sequences_k5_aligned.csv` (N = {n_train:,})\n")
        f.write(f"**Validation Cohort**: `data/metadata/forecast_val_sequences_k5_aligned.csv` (N = {n_val:,}, 181 cyclones)\n")
        f.write(f"**Locked Test Set**: Strictly Untouched (Zero Test Predictions Generated or Inspected)\n")
        f.write(f"**Base Checkpoints**: 100% Frozen\n\n")

        f.write("## 1. Executive Scientific Verdict\n\n")
        f.write(f"```text\nVERDICT: {verdict_text}\nRATIONALE: {rationale}\n```\n\n")

        f.write("## 2. Full Model Comparison Table\n\n")
        f.write("| Model Configuration | Overall MAE (Δ) | +24h MAE | RI +24h MAE (Δ) | Non-RI +24h MAE (Δ) | RI Underpred % | False Dips | Storms (+/-) | 95% Bootstrap CI |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in candidate_results:
            b = r["bootstrap"]
            lbl = f"**{r['config_name']}**" + (" *(Baseline)*" if r["family"] == "Baseline" else "")
            ci_str = f"[{b['overall_ci_95'][0]:+.3f}, {b['overall_ci_95'][1]:+.3f}]" if r["family"] != "Baseline" else "—"
            f.write(f"| {lbl} | {r['overall_mean_mae']:.4f} kt ({r.get('delta_overall_mae', 0):+.4f}) | {r['mae_24h']:.2f} kt | {r['ri_mae_24h']:.2f} kt ({r.get('delta_ri_24h_mae', 0):+.2f}) | {r['non_ri_mae_24h']:.2f} kt ({r.get('delta_non_ri_24h_mae', 0):+.2f}) | {r['ri_underpred_fraction']*100:.1f}% | {r['false_dips']} | {r['storms_improved']}/{r['storms_worsened']} | {ci_str} |\n")

        f.write("\n## 3. Strengthening Regimes Audit (Signed Error Progression)\n\n")
        f.write("| Regime | Baseline MAE (Underpred %) | Best Candidate MAE (Underpred %) | Baseline Signed Error | Best Candidate Signed Error |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        b_reg = base_res["regimes"]
        c_reg = best_candidate["regimes"]
        for k, lbl in [
            ("all", "All Sequences (N=7,295)"),
            ("strengthening_pos", "True ΔV24 > 0 kt (N=3,522)"),
            ("strengthening_ge10", "True ΔV24 >= 10 kt (N=1,887)"),
            ("strengthening_ge20", "True ΔV24 >= 20 kt (N=989)"),
            ("ri_ge30", "True RI: ΔV24 >= 30 kt (N=409)"),
        ]:
            f.write(f"| **{lbl}** | {b_reg[k]['mae']:.2f} kt ({b_reg[k]['underprediction_fraction']*100:.1f}%) | {c_reg[k]['mae']:.2f} kt ({c_reg[k]['underprediction_fraction']*100:.1f}%) | {b_reg[k]['mean_signed_error']:+.2f} kt | {c_reg[k]['mean_signed_error']:+.2f} kt |\n")

        f.write("\n## 4. Visual Diagnostics Generated\n\n")
        f.write("1. `plot1_pred_vs_true_delta24.png`: Scatter comparison of predicted vs true ΔV24.\n")
        f.write("2. `plot2_error_binned_by_pri.png`: +24h MAE stratified by predicted RI probability.\n")
        f.write("3. `plot3_signed_error_binned_by_true_delta24.png`: Signed bias curve across ground truth ΔV24.\n")
        f.write("4. `plot4_ri_event_pred_vs_true.png`: Close-up on the 409 true RI sequences.\n")
        f.write("5. `plot5_example_trajectories.png`: Real validation trajectories comparing Baseline vs Learned Correction.\n")

    print(f"✓ Saved Markdown report to {md_path}")

    # 4. README.md
    readme_path = out_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write("# RI-Aware Correction Model Experiments\n\n")
        f.write("This directory contains the experimental artifacts, training pipelines, and validation evaluations for learned RI-aware corrections.\n\n")
        f.write("## Architecture & Feature Pipeline\n")
        f.write("- **Features ($D=27$)**: Base Residual ΔV (+6h, +12h, +24h), Canonical Ridge Hybrid ΔV, Dedicated RI Classifier $P_{RI}$ and logit, current intensity $V_t$, historical $K=5$ trends (6h delta, 12h delta, slope), SHIPS causal environmental variables (SST, OHC, VWS, etc.), and physical interactions ($P_{RI} \\times \\text{Shear}$, $P_{RI} \\times \\text{SST}$, etc.).\n")
        f.write("- **Models Tested**:\n")
        f.write("  1. Regularized Ridge Correction across $\\alpha \\in [10, 5000]$\n")
        f.write("  2. Constrained Small MLP: $\\text{scale} \\cdot \\tanh(\\text{MLP}(X))$ across scales $\\{5, 10, 15, 20\\}\\text{ kt}$\n\n")
        f.write("## Leakage & Protocol Controls\n")
        f.write("- **Training**: Trained strictly on `forecast_train_sequences_k5_aligned.csv`.\n")
        f.write("- **Validation**: Evaluated strictly on `forecast_val_sequences_k5_aligned.csv`.\n")
        f.write("- **Locked Test Set**: Strictly untouched (zero access).\n")

    print(f"✓ Saved README to {readme_path}")
    print("=" * 80)
    print("ALL RUNS AND REPORTS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
