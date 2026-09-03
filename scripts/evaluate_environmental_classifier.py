"""
Evaluation script for Multi-Modal Environmental Tropical Cyclone Classifier on Held-Out Test Set.
Uses validation-selected threshold tau_val to evaluate RI performance.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.trend_config import IntensityTrendConfig
from src.data.trend_dataset import build_trend_dataloaders
from src.evaluation.classification_metrics import (
    compute_ri_metrics,
    compute_trend_metrics,
)
from src.models.environmental_temporal_classifier import EnvironmentalTemporalClassifier
from src.models.temporal_classifier import JointTrendRILoss


@torch.no_grad()
def evaluate_test_set(
    model: torch.nn.Module,
    test_loader: DataLoader,
    loss_fn: JointTrendRILoss,
    device: torch.device,
    threshold: float = 0.141,
):
    model.eval()
    total_loss = 0.0

    all_ri_probs = []
    all_ri_targets = []
    all_trend_preds = []
    all_trend_probs = []
    all_trend_targets = []
    all_reg_preds = []
    all_reg_targets = []
    all_cyclone_ids = []
    all_timestamps = []
    all_vmax_curr = []
    all_vmax_24h = []

    for batch in test_loader:
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

        ri_probs = torch.sigmoid(ri_logits).squeeze(-1).cpu().numpy()
        trend_probs = torch.softmax(trend_logits, dim=-1).cpu().numpy()
        trend_preds = np.argmax(trend_probs, axis=-1)

        all_ri_probs.append(ri_probs)
        all_ri_targets.append(ri_targets.cpu().numpy())
        all_trend_probs.append(trend_probs)
        all_trend_preds.append(trend_preds)
        all_trend_targets.append(trend_targets.cpu().numpy())
        all_reg_preds.append(reg_preds.cpu().numpy())
        all_reg_targets.append(reg_targets.cpu().numpy())

        all_cyclone_ids.extend(meta["cyclone_id"])
        all_timestamps.extend(meta["target_t_timestamp"].numpy() if isinstance(meta["target_t_timestamp"], torch.Tensor) else meta["target_t_timestamp"])
        all_vmax_curr.extend(meta["vmax_curr"].numpy() if isinstance(meta["vmax_curr"], torch.Tensor) else meta["vmax_curr"])
        all_vmax_24h.extend(meta["vmax_plus_24h"].numpy() if isinstance(meta["vmax_plus_24h"], torch.Tensor) else meta["vmax_plus_24h"])

    ri_probs = np.concatenate(all_ri_probs)
    ri_targets = np.concatenate(all_ri_targets)
    trend_probs = np.concatenate(all_trend_probs)
    trend_preds = np.concatenate(all_trend_preds)
    trend_targets = np.concatenate(all_trend_targets)
    reg_preds = np.concatenate(all_reg_preds)
    reg_targets = np.concatenate(all_reg_targets)

    trend_metrics = compute_trend_metrics(trend_targets, trend_preds)
    ri_metrics = compute_ri_metrics(ri_targets, ri_probs, threshold=threshold)
    reg_mae = np.mean(np.abs(reg_preds - reg_targets), axis=0)

    results = {
        "trend_metrics": trend_metrics,
        "ri_metrics": ri_metrics,
        "reg_mae_6h": float(reg_mae[0]),
        "reg_mae_12h": float(reg_mae[1]),
        "reg_mae_24h": float(reg_mae[2]),
        "reg_mae_mean": float(np.mean(reg_mae)),
        "predictions_df": pd.DataFrame({
            "cyclone_id": all_cyclone_ids,
            "target_t_timestamp": all_timestamps,
            "vmax_curr": all_vmax_curr,
            "vmax_plus_24h": all_vmax_24h,
            "actual_trend": trend_targets,
            "pred_trend": trend_preds,
            "prob_weakening": trend_probs[:, 0],
            "prob_stable": trend_probs[:, 1],
            "prob_intensifying": trend_probs[:, 2],
            "actual_ri": ri_targets,
            "pred_ri_prob": ri_probs,
            "pred_ri_flag": (ri_probs >= threshold).astype(int),
            "pred_plus_6h": reg_preds[:, 0],
            "pred_plus_12h": reg_preds[:, 1],
            "pred_plus_24h": reg_preds[:, 2],
        }),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Environmental Classifier on Test Set")
    parser.add_argument("--k-history", type=int, default=5, help="Number of history frames (5 or 7)")
    parser.add_argument("--checkpoint", type=str, default="experiments/environmental_fusion/checkpoints/exp_e_full_env/best.pt")
    parser.add_argument("--output-dir", type=str, default="experiments/environmental_fusion/checkpoints/exp_e_full_env")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    tau_val = ckpt.get("best_tau", 0.141)
    print(f"Loaded validation-selected optimal threshold: tau_val = {tau_val:.3f}")

    k = args.k_history
    # 2. Instantiate model
    model = EnvironmentalTemporalClassifier(
        channels=3,
        num_frames=k,
        d_model=256,
        n_heads=8,
        num_layers=2,
        dropout=0.1,
        use_vis_channel=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Load Test Data
    print(f"Loading test data and environmental cache for K={k}...")
    train_df = pd.read_csv(f"data/metadata/forecast_train_sequences_k{k}.csv")
    val_df = pd.read_csv(f"data/metadata/forecast_val_sequences_k{k}.csv")
    test_df = pd.read_csv(f"data/metadata/forecast_test_sequences_k{k}.csv")

    env_cache = torch.load(f"data/metadata/environmental_features_k{k}.pt")
    test_env = env_cache["test"]

    config = IntensityTrendConfig()
    with open("data/metadata/normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)
    norm_mean = [norm_stats["mean"][c] for c in [0, 1, 2]]
    norm_std = [norm_stats["std"][c] for c in [0, 1, 2]]

    _, _, test_loader = build_trend_dataloaders(
        train_seq_df=train_df,
        val_seq_df=val_df,
        test_seq_df=test_df,
        mean=norm_mean,
        std=norm_std,
        channels=[0, 1, 2],
        batch_size=args.batch_size,
        num_workers=4,
        config=config,
        test_env_tensor=test_env,
    )

    loss_fn = JointTrendRILoss(
        ri_pos_weight=torch.tensor([13.8], device=device, dtype=torch.float32),
        trend_class_weights=torch.ones(3, device=device, dtype=torch.float32),
    )

    # 4. Evaluate
    print(f"\nEvaluating on Held-Out Test Set (N = {len(test_df):,} sequences, {test_df['cyclone_id'].nunique()} unseen cyclones)...", flush=True)
    results = evaluate_test_set(model, test_loader, loss_fn, device, threshold=tau_val)

    tr = results["trend_metrics"]
    ri = results["ri_metrics"]

    print("\n" + "=" * 70)
    print(f"EXPERIMENT E (K={k}) HELD-OUT TEST RESULTS")
    print("=" * 70)
    print(f"Test Set Size:     N = {len(test_df):,} sequences from {test_df['cyclone_id'].nunique()} unseen cyclones")
    print(f"RI Prevalence:     {ri.get('prevalence', 0.0)*100:.2f}% ({ri.get('n_ri_events', 0):,} / {ri.get('n_total', 0):,} events)")
    print(f"Trend Accuracy:    {tr['accuracy']*100:.2f}%")
    print(f"Trend Macro F1:    {tr['macro_f1']:.4f}")
    if "class_metrics" in tr:
        for cname, cm_vals in tr["class_metrics"].items():
            print(f"  • {cname:13s}: Prec={cm_vals['precision']*100:5.1f}%, Rec={cm_vals['recall']*100:5.1f}%, F1={cm_vals['f1']:.4f}")
    print(f"RI ROC-AUC:        {ri['roc_auc']:.4f}")
    print(f"RI PR-AUC:         {ri['pr_auc']:.4f}")
    f1_key = f"f1_at_{tau_val:.2f}"
    rec_key = f"recall_at_{tau_val:.2f}"
    prec_key = f"precision_at_{tau_val:.2f}"

    ri_f1 = ri.get(f1_key, ri.get("optimal_f1", 0.0))
    ri_rec = ri.get(rec_key, ri.get("optimal_recall", 0.0))
    ri_prec = ri.get(prec_key, ri.get("optimal_precision", 0.0))
    cm = ri["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    print(f"RI F1 (@ tau_val={tau_val:.3f}): {ri_f1:.4f}")
    print(f"RI Recall:         {ri_rec*100:.1f}% (TP={tp:,}, FN={fn:,})")
    print(f"RI Precision:      {ri_prec*100:.1f}% (FP={fp:,})")
    print(f"RI Optimal F1:     {ri['optimal_f1']:.4f} (@ tau_opt={ri['optimal_threshold']:.3f})")
    print(f"Forecast MAE +6h:  {results['reg_mae_6h']:.2f} kt")
    print(f"Forecast MAE +12h: {results['reg_mae_12h']:.2f} kt")
    print(f"Forecast MAE +24h: {results['reg_mae_24h']:.2f} kt (Mean: {results['reg_mae_mean']:.2f} kt)")
    print("=" * 70)

    # 5. Save Artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "test_predictions.csv")
    results["predictions_df"].to_csv(csv_path, index=False)
    print(f"Saved test predictions to {csv_path}")

    clean_metrics = {
        "trend_accuracy": tr["accuracy"],
        "trend_macro_f1": tr["macro_f1"],
        "ri_roc_auc": ri["roc_auc"],
        "ri_pr_auc": ri["pr_auc"],
        "ri_f1_at_tau": ri_f1,
        "ri_recall_at_tau": ri_rec,
        "ri_precision_at_tau": ri_prec,
        "ri_optimal_f1": ri["optimal_f1"],
        "ri_optimal_recall": ri["optimal_recall"],
        "ri_optimal_precision": ri["optimal_precision"],
        "ri_optimal_threshold": ri["optimal_threshold"],
        "ri_threshold": tau_val,
        "confusion_matrix": cm,
        "reg_mae_6h": results["reg_mae_6h"],
        "reg_mae_12h": results["reg_mae_12h"],
        "reg_mae_24h": results["reg_mae_24h"],
        "reg_mae_mean": results["reg_mae_mean"],
    }
    json_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(json_path, "w") as f:
        json.dump(clean_metrics, f, indent=2)
    print(f"Saved test metrics to {json_path}")


if __name__ == "__main__":
    main()
