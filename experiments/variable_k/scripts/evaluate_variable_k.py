"""Isolated Evaluation Script for Variable-K Model on Held-Out Test Set.

Evaluates Test A (K=3), Test B (K=5), Test C (K=7) and saves:
- experiments/variable_k/results/test_predictions_k3.csv
- experiments/variable_k/results/test_predictions_k5.csv
- experiments/variable_k/results/test_predictions_k7.csv
- experiments/variable_k/results/test_metrics.json
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, r2_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader

from src.data.trend_config import IntensityTrendConfig
from src.evaluation.classification_metrics import (
    compute_ri_metrics,
    compute_trend_metrics,
)
from src.models.environmental_temporal_classifier import EnvironmentalTemporalClassifier
from src.models.temporal_classifier import JointTrendRILoss
from experiments.variable_k.scripts.variable_k_dataset import VariableKDataset, VariableKCollator


def evaluate_test_k(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: JointTrendRILoss,
    device: torch.device,
    threshold: float,
) -> Tuple[Dict, pd.DataFrame]:
    model.eval()
    total_loss = 0.0

    all_cids = []
    all_ts = []
    all_vcurr = []
    all_v24 = []
    all_tr_true = []
    all_tr_pred = []
    all_tr_probs = []
    all_ri_true = []
    all_ri_probs = []
    all_reg_preds = []
    all_reg_targets = []

    with torch.no_grad():
        for batch in loader:
            images, vis_masks, trend_targets, ri_targets, reg_targets, env_vec, meta = batch

            images = images.to(device, non_blocking=True)
            vis_masks = vis_masks.to(device, non_blocking=True)
            trend_targets = trend_targets.to(device, non_blocking=True)
            ri_targets = ri_targets.to(device, non_blocking=True)
            reg_targets = reg_targets.to(device, non_blocking=True)
            env_vec = env_vec.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                ri_logits, trend_logits, reg_preds = model(images, vis_masks, env_vec)
                loss, _ = loss_fn(ri_logits, trend_logits, reg_preds, ri_targets, trend_targets, reg_targets)

            total_loss += loss.item()

            ri_p = torch.sigmoid(ri_logits).squeeze(-1).cpu().numpy()
            tr_p = torch.softmax(trend_logits, dim=-1).cpu().numpy()
            tr_pred = np.argmax(tr_p, axis=-1)

            all_ri_probs.append(ri_p)
            all_ri_true.append(ri_targets.cpu().numpy())
            all_tr_probs.append(tr_p)
            all_tr_pred.append(tr_pred)
            all_tr_true.append(trend_targets.cpu().numpy())
            all_reg_preds.append(reg_preds.cpu().numpy())
            all_reg_targets.append(reg_targets.cpu().numpy())

            all_cids.extend(meta["cyclone_id"])
            all_ts.extend(meta["target_t_timestamp"])
            all_vcurr.extend(meta["vmax_curr"].numpy())
            all_v24.extend(meta["vmax_plus_24h"].numpy())

    ri_probs = np.concatenate(all_ri_probs)
    ri_targets = np.concatenate(all_ri_true)
    trend_probs = np.concatenate(all_tr_probs)
    trend_preds = np.concatenate(all_tr_pred)
    trend_targets = np.concatenate(all_tr_true)
    reg_preds = np.concatenate(all_reg_preds)
    reg_targets = np.concatenate(all_reg_targets)
    vcurr = np.array(all_vcurr)
    v24 = np.array(all_v24)

    tr_m = compute_trend_metrics(trend_targets, trend_preds)
    ri_m = compute_ri_metrics(ri_targets, ri_probs, threshold=threshold)
    pred_ri_flag = (ri_probs >= threshold).astype(int)
    prec_at_tau = float(precision_score(ri_targets, pred_ri_flag, zero_division=0))
    rec_at_tau = float(recall_score(ri_targets, pred_ri_flag, zero_division=0))
    f1_at_tau = float(f1_score(ri_targets, pred_ri_flag, zero_division=0))

    mae = np.mean(np.abs(reg_preds - reg_targets), axis=0)
    rmse = np.sqrt(np.mean((reg_preds - reg_targets) ** 2, axis=0))
    r2 = [float(r2_score(reg_targets[:, i], reg_preds[:, i])) for i in range(3)]

    try:
        brier = float(brier_score_loss(ri_targets, ri_probs))
    except Exception:
        brier = 0.0

    act_dv = v24 - vcurr
    pred_dv = reg_preds[:, 2] - vcurr
    slope, intercept = np.polyfit(act_dv, pred_dv, deg=1)
    corr = float(np.corrcoef(act_dv, pred_dv)[0, 1])

    # Construct DataFrame
    pred_df = pd.DataFrame({
        "cyclone_id": all_cids,
        "target_t_timestamp": all_ts,
        "vmax_curr": vcurr,
        "vmax_plus_24h": v24,
        "actual_trend": trend_targets,
        "pred_trend": trend_preds,
        "prob_weakening": trend_probs[:, 0],
        "prob_stable": trend_probs[:, 1],
        "prob_intensifying": trend_probs[:, 2],
        "actual_ri": ri_targets,
        "pred_ri_prob": ri_probs,
        "pred_ri_flag": pred_ri_flag,
        "pred_plus_6h": reg_preds[:, 0],
        "pred_plus_12h": reg_preds[:, 1],
        "pred_plus_24h": reg_preds[:, 2],
    })

    metrics = {
        "loss": total_loss / max(1, len(loader)),
        "threshold": float(threshold),
        "trend_accuracy": float(tr_m["accuracy"]),
        "trend_macro_f1": float(tr_m["macro_f1"]),
        "confusion_matrix": tr_m.get("confusion_matrix", []),
        "ri_roc_auc": float(ri_m["roc_auc"]),
        "ri_pr_auc": float(ri_m["pr_auc"]),
        "ri_precision": prec_at_tau,
        "ri_recall": rec_at_tau,
        "ri_f1": f1_at_tau,
        "brier_score": float(brier),
        "reg_mae_6h": float(mae[0]),
        "reg_mae_12h": float(mae[1]),
        "reg_mae_24h": float(mae[2]),
        "reg_mae_mean": float(np.mean(mae)),
        "reg_rmse_6h": float(rmse[0]),
        "reg_rmse_12h": float(rmse[1]),
        "reg_rmse_24h": float(rmse[2]),
        "reg_r2_6h": float(r2[0]),
        "reg_r2_12h": float(r2[1]),
        "reg_r2_24h": float(r2[2]),
        "mean_actual_dv24": float(np.mean(act_dv)),
        "mean_pred_dv24": float(np.mean(pred_dv)),
        "slope_dv24": float(slope),
        "intercept_dv24": float(intercept),
        "corr_dv24": float(corr),
    }

    return metrics, pred_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate Variable-K Model on Test Set")
    parser.add_argument("--checkpoint", type=str, default="experiments/variable_k/checkpoints/best.pt")
    parser.add_argument("--results-dir", type=str, default="experiments/variable_k/results")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    tau_val = ckpt.get("best_tau", 0.0161)
    print(f"Validation threshold tau_val = {tau_val:.4f}")

    model = EnvironmentalTemporalClassifier(
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

    test_df = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")
    test_env = torch.load("data/metadata/environmental_features_k7.pt")["test"]

    with open("data/metadata/normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)
    norm_mean = [norm_stats["mean"][c] for c in [0, 1, 2]]
    norm_std = [norm_stats["std"][c] for c in [0, 1, 2]]

    test_ds = VariableKDataset(
        test_df, mean=norm_mean, std=norm_std, channels=[0, 1, 2], is_training=False, env_tensor=test_env
    )

    loss_fn = JointTrendRILoss(
        ri_pos_weight=torch.tensor([13.8], device=device, dtype=torch.float32),
        trend_class_weights=torch.ones(3, device=device, dtype=torch.float32),
    )

    all_eval_metrics = {}

    for k in [3, 5, 7]:
        print(f"\n--- Evaluating Test Set with K={k} ({k*3-3}h history) ---")
        loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            collate_fn=VariableKCollator(mode=k, seed=42),
        )

        tau_k = ckpt.get("tau_by_k", {}).get(k, tau_val) if isinstance(ckpt.get("tau_by_k"), dict) else tau_val
        metrics, pred_df = evaluate_test_k(model, loader, loss_fn, device, threshold=tau_k)

        out_csv = results_dir / f"test_predictions_k{k}.csv"
        pred_df.to_csv(out_csv, index=False)
        print(f"Saved predictions to {out_csv} ({len(pred_df)} rows)")

        print(
            f"Results K={k}:\n"
            f"  • Trend Accuracy:  {metrics['trend_accuracy']*100:.2f}% | Macro F1: {metrics['trend_macro_f1']:.4f}\n"
            f"  • RI PR-AUC:       {metrics['ri_pr_auc']:.4f} | ROC-AUC: {metrics['ri_roc_auc']:.4f}\n"
            f"  • RI Recall:       {metrics['ri_recall']*100:.2f}% | Precision: {metrics['ri_precision']*100:.2f}% | F1: {metrics['ri_f1']:.4f}\n"
            f"  • +6 MAE:          {metrics['reg_mae_6h']:.2f} kt | +12 MAE: {metrics['reg_mae_12h']:.2f} kt | +24 MAE: {metrics['reg_mae_24h']:.2f} kt\n"
            f"  • ΔV24 Slope:      {metrics['slope_dv24']:.4f} (Corr: {metrics['corr_dv24']:.4f})"
        )
        all_eval_metrics[f"test_k{k}"] = metrics

    metrics_out = results_dir / "test_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(all_eval_metrics, f, indent=2)
    print(f"\nSaved comprehensive test metrics to {metrics_out}")


if __name__ == "__main__":
    main()
