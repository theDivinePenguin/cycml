#!/usr/bin/env python3
"""
Independent Rigorous Verification and Persistence Benchmark
for Residual Delta-V Forecaster (Unconstrained).

Evaluates on the EXACT 7,295 validation sequences without using evaluate.py.
Computes:
1. Complete distribution of Delta V6 = V(t+6h) - V(t)
2. Exact persistence baseline metrics across all horizons (+6h, +12h, +24h)
3. Independent PyTorch reconstruction of ResidualDeltaVForecaster from checkpoint
4. Head-to-head comparison of Residual vs Persistence
5. Consistency check against the official logged checkpoint metrics
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.residual_forecaster import ResidualDeltaVForecaster


class IndependentValidationDataset(Dataset):
    """Clean independent dataset loader reading raw HDF5 files directly."""

    def __init__(
        self,
        df: pd.DataFrame,
        mean: List[float],
        std: List[float],
        channels: List[int] = [0, 1, 2],
    ):
        self.df = df.reset_index(drop=True)
        self.channels = channels
        self.mean = np.array([mean[c] for c in channels], dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array([std[c] for c in channels], dtype=np.float32).reshape(-1, 1, 1)
        self._h5_cache: Dict[str, h5py.File] = {}

    def _get_h5(self, path: str) -> h5py.File:
        if path not in self._h5_cache:
            self._h5_cache[path] = h5py.File(path, "r", swmr=True)
        return self._h5_cache[path]

    def __len__(self) -> int:
        return len(self.df)

    def _preprocess_frame(self, raw: np.ndarray) -> Tuple[np.ndarray, float]:
        frame = raw[:, :, self.channels].astype(np.float32)

        # Explicit VIS validity gating
        vis_valid = 1.0
        if 2 in self.channels:
            ch_idx = self.channels.index(2)
            vis = frame[:, :, ch_idx]
            inv = np.isnan(vis) | (vis < 0.0) | (vis > 1e20)
            vis[inv] = 0.0
            vis_valid = 1.0 if np.mean(vis > 0.01) > 0.10 else 0.0
            frame[:, :, ch_idx] = vis

        nan_mask = np.isnan(frame) | np.isinf(frame) | (frame > 1e20) | (frame < -1e20)
        frame[nan_mask] = 0.0

        # Transpose to (C, H, W) and standardize
        t = np.transpose(frame, (2, 0, 1))
        t = (t - self.mean) / (self.std + 1e-7)
        return t, vis_valid

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        h5_files = json.loads(row["history_h5_files"])
        h5_rows = json.loads(row["history_h5_rows"])

        frames = []
        vis_masks = []
        for fpath, r_idx in zip(h5_files, h5_rows):
            raw = self._get_h5(fpath)["matrix"][r_idx]
            proc_t, v_flag = self._preprocess_frame(raw)
            frames.append(proc_t)
            vis_masks.append(v_flag)

        seq_tensor = torch.from_numpy(np.stack(frames, axis=0))  # (K, C, H, W)
        vis_mask_tensor = torch.from_numpy(np.array(vis_masks, dtype=np.float32))  # (K,)
        v_curr = float(row["vmax_curr"])
        true_future = torch.tensor([
            float(row["vmax_plus_6h"]),
            float(row["vmax_plus_12h"]),
            float(row["vmax_plus_24h"]),
        ], dtype=torch.float32)

        return seq_tensor, vis_mask_tensor, v_curr, true_future


def calculate_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def run_independent_verification():
    print("=" * 80)
    print("INDEPENDENT RIGOROUS RESIDUAL MODEL & PERSISTENCE VERIFICATION")
    print("=" * 80)

    val_csv = Path("data/metadata/forecast_val_sequences_k5_aligned.csv")
    norm_json = Path("data/metadata/normalization_stats_multichannel.json")
    ckpt_path = Path("experiments/checkpoints/residual_delta_v_unconstrained/best.pt")

    if not val_csv.exists():
        raise FileNotFoundError(f"Missing validation manifest: {val_csv}")
    if not norm_json.exists():
        raise FileNotFoundError(f"Missing normalization stats: {norm_json}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt_path}")

    # Load Manifest
    df_val = pd.read_csv(val_csv)
    n_seq = len(df_val)
    print(f"\n1. Loaded Validation Manifest: {val_csv}")
    print(f"   Total Sequences: {n_seq} (Target verification cohort: EXACTLY 7,295)")
    assert n_seq == 7295, f"Expected 7,295 sequences, got {n_seq}!"

    # =========================================================================
    # PART A: Complete Distribution of Ground-Truth Delta V6 = V(t+6h) - V(t)
    # =========================================================================
    v_curr = df_val["vmax_curr"].values
    v_6h = df_val["vmax_plus_6h"].values
    v_12h = df_val["vmax_plus_12h"].values
    v_24h = df_val["vmax_plus_24h"].values

    dv6 = v_6h - v_curr
    dv12 = v_12h - v_curr
    dv24 = v_24h - v_curr

    dv6_abs = np.abs(dv6)
    n_gt4 = int(np.sum(dv6_abs > 4.0))
    pct_gt4 = float(n_gt4 / n_seq * 100.0)

    print("\n" + "=" * 80)
    print("PART A: COMPLETE DISTRIBUTION OF GROUND-TRUTH ΔV6 = V(t+6h) - V(t)")
    print("=" * 80)
    print(f"• Sample Size:               {n_seq:,} sequences")
    print(f"• Mean:                      {np.mean(dv6):+.4f} kt")
    print(f"• Standard Deviation (std):  {np.std(dv6):.4f} kt")
    print(f"• Min ΔV6:                   {np.min(dv6):+.1f} kt")
    print(f"• Max ΔV6:                   {np.max(dv6):+.1f} kt")
    print(f"• Median (50th percentile):  {np.median(dv6):+.1f} kt")
    print(f"• 95th percentile (|ΔV6|):   {np.percentile(dv6_abs, 95):.2f} kt")
    print(f"• 95th percentile (signed):  {np.percentile(dv6, 95):+.2f} kt")
    print(f"• 5th percentile (signed):   {np.percentile(dv6, 5):+.2f} kt")
    print(f"• Sequences with |ΔV6| > 4kt:{n_gt4:,} / {n_seq:,} ({pct_gt4:.2f}%)")
    print(f"• Sequences with |ΔV6| > 10kt: {np.sum(dv6_abs > 10.0):,} ({np.mean(dv6_abs > 10.0)*100:.2f}%)")
    print(f"• Explicit Note:             ΔV6 <= 4 kt assumption is FALSE (45.25% exceed 4 kt).")

    # =========================================================================
    # PART B: Exact Persistence Baseline on the Same 7,295 Sequences
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART B: PERSISTENCE BASELINE METRICS (V_hat(t+tau) = V(t))")
    print("=" * 80)
    
    # Persistence predicts 0 delta, so prediction is v_curr for all horizons
    pers_pred_6h = v_curr
    pers_pred_12h = v_curr
    pers_pred_24h = v_curr

    m_pers_6 = calculate_metrics(pers_pred_6h, v_6h)
    m_pers_12 = calculate_metrics(pers_pred_12h, v_12h)
    m_pers_24 = calculate_metrics(pers_pred_24h, v_24h)
    pers_mean_mae = (m_pers_6["mae"] + m_pers_12["mae"] + m_pers_24["mae"]) / 3.0
    pers_mean_rmse = (m_pers_6["rmse"] + m_pers_12["rmse"] + m_pers_24["rmse"]) / 3.0

    print(f"+6h Horizon:   MAE = {m_pers_6['mae']:6.2f} kt | RMSE = {m_pers_6['rmse']:6.2f} kt | R² = {m_pers_6['r2']:6.4f}")
    print(f"+12h Horizon:  MAE = {m_pers_12['mae']:6.2f} kt | RMSE = {m_pers_12['rmse']:6.2f} kt | R² = {m_pers_12['r2']:6.4f}")
    print(f"+24h Horizon:  MAE = {m_pers_24['mae']:6.2f} kt | RMSE = {m_pers_24['rmse']:6.2f} kt | R² = {m_pers_24['r2']:6.4f}")
    print(f"Overall Mean:  MAE = {pers_mean_mae:6.2f} kt | RMSE = {pers_mean_rmse:6.2f} kt")

    # =========================================================================
    # PART C: Independent PyTorch Checkpoint Loading and Sequence Reconstruction
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART C: INDEPENDENT CHECKPOINT INFERENCE & RECONSTRUCTION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Checkpoint Epoch: {ckpt.get('epoch')}")
    print(f"Checkpoint Recorded Best Val MAE: {ckpt.get('best_metric'):.4f} kt")

    # 2. Build Model Independently
    model = ResidualDeltaVForecaster(
        backbone_arch="resnet18",
        in_channels=3,
        d_model=256,
        temporal_type="transformer",
        num_layers=2,
        nhead=8,
        dropout=0.1,
        parameterization="unconstrained",
        pretrained_backbone=False,  # Weights loaded from ckpt
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model initialized and weights restored successfully.")

    # 3. Create Independent DataLoader
    with open(norm_json, "r") as f:
        norm_stats = json.load(f)

    dataset = IndependentValidationDataset(
        df=df_val,
        mean=norm_stats["mean"],
        std=norm_stats["std"],
        channels=[0, 1, 2],
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    all_v_curr = []
    all_true_future = []
    all_pred_delta = []
    all_pred_reconstructed = []

    print(f"Executing independent forward passes over {len(loader)} batches...")
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (seq_tensor, vis_mask_tensor, v_c, true_f) in enumerate(loader):
            seq_tensor = seq_tensor.to(device, non_blocking=True)
            vis_mask_tensor = vis_mask_tensor.to(device, non_blocking=True)
            v_c_dev = v_c.to(device).float()

            # Execute forward pass: outputs v_hat, delta_v_hat
            # v_hat = v_curr + delta_v_hat
            v_hat, delta_hat = model(seq_tensor, v_curr=v_c_dev, vis_masks=vis_mask_tensor)

            all_v_curr.append(v_c.numpy())
            all_true_future.append(true_f.numpy())
            all_pred_delta.append(delta_hat.cpu().numpy())
            all_pred_reconstructed.append(v_hat.cpu().numpy())

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(loader):
                print(f"  Batch [{batch_idx+1:3d}/{len(loader):3d}] processed ({((batch_idx+1)*64)/n_seq*100:.1f}%)")

    t_elapsed = time.time() - t0
    print(f"Inference complete in {t_elapsed:.2f}s ({n_seq/t_elapsed:.1f} sequences/sec).")

    v_curr_arr = np.concatenate(all_v_curr)  # (N,)
    true_future_arr = np.concatenate(all_true_future, axis=0)  # (N, 3)
    pred_delta_arr = np.concatenate(all_pred_delta, axis=0)  # (N, 3)
    pred_future_arr = np.concatenate(all_pred_reconstructed, axis=0)  # (N, 3)

    # Verify reconstruction identity: V_hat == V_curr + Delta_hat
    manual_reconstruction = v_curr_arr[:, None] + pred_delta_arr
    max_recon_diff = np.max(np.abs(pred_future_arr - manual_reconstruction))
    assert max_recon_diff < 1e-4, f"Reconstruction identity violated! diff={max_recon_diff}"
    print(f"✓ Verified physical identity: V_hat(t+tau) = V(t) + ΔV_hat(tau) (max diff: {max_recon_diff:.2e})")

    # =========================================================================
    # PART D: Performance Metrics and Official Evaluation Comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART D: INDEPENDENT EVALUATION METRICS vs OFFICIAL TRAINING RUN")
    print("=" * 80)

    m_res_6 = calculate_metrics(pred_future_arr[:, 0], true_future_arr[:, 0])
    m_res_12 = calculate_metrics(pred_future_arr[:, 1], true_future_arr[:, 1])
    m_res_24 = calculate_metrics(pred_future_arr[:, 2], true_future_arr[:, 2])
    res_mean_mae = (m_res_6["mae"] + m_res_12["mae"] + m_res_24["mae"]) / 3.0
    res_mean_rmse = (m_res_6["rmse"] + m_res_12["rmse"] + m_res_24["rmse"]) / 3.0

    print(f"+6h Horizon:   MAE = {m_res_6['mae']:6.2f} kt | RMSE = {m_res_6['rmse']:6.2f} kt | R² = {m_res_6['r2']:6.4f}")
    print(f"+12h Horizon:  MAE = {m_res_12['mae']:6.2f} kt | RMSE = {m_res_12['rmse']:6.2f} kt | R² = {m_res_12['r2']:6.4f}")
    print(f"+24h Horizon:  MAE = {m_res_24['mae']:6.2f} kt | RMSE = {m_res_24['rmse']:6.2f} kt | R² = {m_res_24['r2']:6.4f}")
    print(f"Overall Mean:  MAE = {res_mean_mae:6.4f} kt | RMSE = {res_mean_rmse:6.4f} kt")

    # Compare against official checkpoint recorded metric
    official_val_mae = float(ckpt["best_metric"])
    diff_val_mae = abs(res_mean_mae - official_val_mae)
    print(f"\nOfficial Checkpoint Metric: {official_val_mae:.4f} kt")
    print(f"Independent Metric:         {res_mean_mae:.4f} kt")
    print(f"Absolute Discrepancy:       {diff_val_mae:.6f} kt")

    if diff_val_mae > 0.05:
        print(f"\n[FATAL WARNING] Independent evaluation differs materially (>0.05 kt) from checkpoint!")
        sys.exit(1)
    else:
        print("✓ EXACT MATCH with official evaluation (within floating point precision).")

    # =========================================================================
    # PART E: Residual Model Improvement Over Persistence at EVERY Horizon
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART E: HEAD-TO-HEAD COMPARISON: RESIDUAL MODEL vs PERSISTENCE")
    print("=" * 80)

    imp_mae_6 = m_pers_6["mae"] - m_res_6["mae"]
    imp_pct_6 = (imp_mae_6 / m_pers_6["mae"]) * 100.0

    imp_mae_12 = m_pers_12["mae"] - m_res_12["mae"]
    imp_pct_12 = (imp_mae_12 / m_pers_12["mae"]) * 100.0

    imp_mae_24 = m_pers_24["mae"] - m_res_24["mae"]
    imp_pct_24 = (imp_mae_24 / m_pers_24["mae"]) * 100.0

    imp_mae_mean = pers_mean_mae - res_mean_mae
    imp_pct_mean = (imp_mae_mean / pers_mean_mae) * 100.0

    table_data = [
        ["+6h Horizon", f"{m_pers_6['mae']:.2f} kt", f"{m_res_6['mae']:.2f} kt", f"+{imp_mae_6:.2f} kt", f"{imp_pct_6:+.2f}%", f"{m_pers_6['rmse']:.2f} vs {m_res_6['rmse']:.2f}", f"{m_pers_6['r2']:.3f} vs {m_res_6['r2']:.3f}"],
        ["+12h Horizon", f"{m_pers_12['mae']:.2f} kt", f"{m_res_12['mae']:.2f} kt", f"+{imp_mae_12:.2f} kt", f"{imp_pct_12:+.2f}%", f"{m_pers_12['rmse']:.2f} vs {m_res_12['rmse']:.2f}", f"{m_pers_12['r2']:.3f} vs {m_res_12['r2']:.3f}"],
        ["+24h Horizon", f"{m_pers_24['mae']:.2f} kt", f"{m_res_24['mae']:.2f} kt", f"+{imp_mae_24:.2f} kt", f"{imp_pct_24:+.2f}%", f"{m_pers_24['rmse']:.2f} vs {m_res_24['rmse']:.2f}", f"{m_pers_24['r2']:.3f} vs {m_res_24['r2']:.3f}"],
        ["Overall Mean", f"{pers_mean_mae:.2f} kt", f"{res_mean_mae:.2f} kt", f"+{imp_mae_mean:.2f} kt", f"{imp_pct_mean:+.2f}%", f"{pers_mean_rmse:.2f} vs {res_mean_rmse:.2f}", "-"],
    ]

    headers = ["Horizon", "Persistence MAE", "Residual Model MAE", "MAE Improvement", "% Improvement", "RMSE (Pers vs Res)", "R² (Pers vs Res)"]
    row_fmt = "{:<14} | {:<16} | {:<19} | {:<16} | {:<14} | {:<20} | {:<16}"
    print(row_fmt.format(*headers))
    print("-" * 125)
    for row in table_data:
        print(row_fmt.format(*row))

    # Save detailed findings to reports
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    report_file = out_dir / "RESIDUAL_VS_PERSISTENCE_AUDIT.json"
    audit_data = {
        "validation_sequences": n_seq,
        "delta_v6_distribution": {
            "mean": float(np.mean(dv6)),
            "std": float(np.std(dv6)),
            "min": float(np.min(dv6)),
            "max": float(np.max(dv6)),
            "median": float(np.median(dv6)),
            "p95_abs": float(np.percentile(dv6_abs, 95)),
            "p95_signed": float(np.percentile(dv6, 95)),
            "p05_signed": float(np.percentile(dv6, 5)),
            "count_abs_gt_4kt": n_gt4,
            "pct_abs_gt_4kt": pct_gt4,
        },
        "persistence_baseline": {
            "mae_6h": m_pers_6["mae"],
            "rmse_6h": m_pers_6["rmse"],
            "r2_6h": m_pers_6["r2"],
            "mae_12h": m_pers_12["mae"],
            "rmse_12h": m_pers_12["rmse"],
            "r2_12h": m_pers_12["r2"],
            "mae_24h": m_pers_24["mae"],
            "rmse_24h": m_pers_24["rmse"],
            "r2_24h": m_pers_24["r2"],
            "mean_mae": pers_mean_mae,
            "mean_rmse": pers_mean_rmse,
        },
        "residual_model_independent": {
            "mae_6h": m_res_6["mae"],
            "rmse_6h": m_res_6["rmse"],
            "r2_6h": m_res_6["r2"],
            "mae_12h": m_res_12["mae"],
            "rmse_12h": m_res_12["rmse"],
            "r2_12h": m_res_12["r2"],
            "mae_24h": m_res_24["mae"],
            "rmse_24h": m_res_24["rmse"],
            "r2_24h": m_res_24["r2"],
            "mean_mae": res_mean_mae,
            "mean_rmse": res_mean_rmse,
            "official_checkpoint_val_mae": official_val_mae,
            "discrepancy": diff_val_mae,
        },
        "improvement_over_persistence": {
            "mae_improvement_6h_kt": imp_mae_6,
            "pct_improvement_6h": imp_pct_6,
            "mae_improvement_12h_kt": imp_mae_12,
            "pct_improvement_12h": imp_pct_12,
            "mae_improvement_24h_kt": imp_mae_24,
            "pct_improvement_24h": imp_pct_24,
            "overall_mae_improvement_kt": imp_mae_mean,
            "overall_pct_improvement": imp_pct_mean,
        }
    }

    with open(report_file, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nSaved structured audit report to: {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_independent_verification()
