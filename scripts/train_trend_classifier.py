"""Training pipeline for Rapid Intensification Prediction & Intensity Trend Classification."""
import argparse
import json
from pathlib import Path
import time
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.trend_config import IntensityTrendConfig
from src.data.trend_dataset import TCIRTrendDataset
from src.evaluation.classification_metrics import (
    compute_ri_metrics,
    compute_trend_metrics,
    find_optimal_threshold,
)
from src.models.temporal_classifier import JointTrendRILoss, TemporalClassifier


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: JointTrendRILoss,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, Dict[str, float], float]:
    """Train model for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    accum_losses = {"loss_ri": 0.0, "loss_trend": 0.0, "loss_reg": 0.0}
    n_batches = len(train_loader)
    start_time = time.time()

    for batch_idx, (images, vis_masks, trend_targets, ri_targets, reg_targets, _) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        vis_masks = vis_masks.to(device, non_blocking=True)
        trend_targets = trend_targets.to(device, non_blocking=True)
        ri_targets = ri_targets.to(device, non_blocking=True)
        reg_targets = reg_targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            ri_logits, trend_logits, reg_preds = model(images, vis_masks)
            loss, loss_dict = loss_fn(
                ri_logits, trend_logits, reg_preds, ri_targets, trend_targets, reg_targets
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k in accum_losses:
            accum_losses[k] += loss_dict.get(k, 0.0)

        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == n_batches:
            curr_lr = optimizer.param_groups[0]["lr"]
            avg_tot = total_loss / (batch_idx + 1)
            avg_ri = accum_losses["loss_ri"] / (batch_idx + 1)
            avg_tr = accum_losses["loss_trend"] / (batch_idx + 1)
            print(
                f"  Epoch [{epoch:2d}/{total_epochs:2d}] Batch [{batch_idx + 1:4d}/{n_batches:4d}] - "
                f"Loss: {loss.item():.4f} (Avg: {avg_tot:.4f} | RI: {avg_ri:.4f} | Tr: {avg_tr:.4f}) - LR: {curr_lr:.2e}",
                flush=True,
            )

    epoch_time = time.time() - start_time
    avg_loss = total_loss / max(n_batches, 1)
    mean_losses = {k: v / max(n_batches, 1) for k, v in accum_losses.items()}
    return avg_loss, mean_losses, epoch_time


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: JointTrendRILoss,
    ri_threshold: float = 0.5,
) -> Tuple[float, Dict[str, any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run deterministic evaluation on validation or test loader."""
    model.eval()
    total_loss = 0.0

    all_ri_probs = []
    all_trend_preds = []
    all_trend_probs = []
    all_reg_preds = []

    all_ri_targets = []
    all_trend_targets = []
    all_reg_targets = []

    n_eval_batches = len(loader)
    for batch_idx, (images, vis_masks, trend_targets, ri_targets, reg_targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        vis_masks = vis_masks.to(device, non_blocking=True)
        trend_gpu = trend_targets.to(device, non_blocking=True)
        ri_gpu = ri_targets.to(device, non_blocking=True)
        reg_gpu = reg_targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            ri_logits, trend_logits, reg_preds = model(images, vis_masks)
            loss, _ = loss_fn(ri_logits, trend_logits, reg_preds, ri_gpu, trend_gpu, reg_gpu)

        total_loss += loss.item() * len(trend_targets)

        ri_prob = torch.sigmoid(ri_logits.squeeze(-1)).cpu().float().numpy()
        tr_prob = torch.softmax(trend_logits, dim=-1).cpu().float().numpy()
        tr_pred = np.argmax(tr_prob, axis=1)

        all_ri_probs.append(ri_prob)
        all_trend_probs.append(tr_prob)
        all_trend_preds.append(tr_pred)
        all_reg_preds.append(reg_preds.cpu().float().numpy())

        all_ri_targets.append(ri_targets.numpy())
        all_trend_targets.append(trend_targets.numpy())
        all_reg_targets.append(reg_targets.numpy())

        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == n_eval_batches:
            print(f"    Evaluating [{batch_idx + 1:3d}/{n_eval_batches:3d}]...", flush=True)

    ri_probs_arr = np.concatenate(all_ri_probs, axis=0)
    trend_preds_arr = np.concatenate(all_trend_preds, axis=0)
    trend_probs_arr = np.concatenate(all_trend_probs, axis=0)
    reg_preds_arr = np.concatenate(all_reg_preds, axis=0)

    ri_targets_arr = np.concatenate(all_ri_targets, axis=0)
    trend_targets_arr = np.concatenate(all_trend_targets, axis=0)
    reg_targets_arr = np.concatenate(all_reg_targets, axis=0)

    avg_loss = total_loss / max(len(trend_targets_arr), 1)

    # Compute evaluation metrics
    ri_metrics = compute_ri_metrics(ri_targets_arr, ri_probs_arr, threshold=ri_threshold)
    trend_metrics = compute_trend_metrics(trend_targets_arr, trend_preds_arr)

    eval_summary = {
        "eval_loss": round(avg_loss, 4),
        "ri_roc_auc": ri_metrics["roc_auc"],
        "ri_pr_auc": ri_metrics["pr_auc"],
        f"ri_precision": ri_metrics[f"precision_at_{ri_threshold:.2f}"],
        f"ri_recall": ri_metrics[f"recall_at_{ri_threshold:.2f}"],
        f"ri_f1": ri_metrics[f"f1_at_{ri_threshold:.2f}"],
        "ri_optimal_threshold": ri_metrics["optimal_threshold"],
        "ri_optimal_f1": ri_metrics["optimal_f1"],
        "ri_brier": ri_metrics["brier_score"],
        "ri_ece": ri_metrics["ece"],
        "trend_accuracy": trend_metrics["accuracy"],
        "trend_macro_f1": trend_metrics["macro_f1"],
        "trend_weighted_f1": trend_metrics["weighted_f1"],
        "trend_per_class": trend_metrics["per_class"],
        "ri_confusion_matrix": ri_metrics["confusion_matrix"],
        "trend_confusion_matrix": trend_metrics["confusion_matrix"],
    }

    return (
        avg_loss,
        eval_summary,
        ri_probs_arr,
        trend_preds_arr,
        trend_probs_arr,
        reg_preds_arr,
        ri_targets_arr,
    )


def run_training(
    save_dir_name: str = "classifier_primary_ri",
    epochs: int = 8,
    lr: float = 1e-4,
    batch_size: int = 16,
    pos_weight_mult: float = 1.0,
    cooldown_seconds: int = 15,
    channels: list = [0, 1, 2],
    warm_start_ckpt: str = "experiments/forecasting/checkpoints/cnn_transformer_k5/best.pt",
):
    """Run full training for unified TemporalClassifier."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path("experiments/trend_classification/checkpoints") / save_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = Path("data/metadata")
    train_seq_df = pd.read_csv(meta_dir / "forecast_train_sequences_k5.csv")
    val_seq_df = pd.read_csv(meta_dir / "forecast_val_sequences_k5.csv")
    test_seq_df = pd.read_csv(meta_dir / "forecast_test_sequences_k5.csv")

    config = IntensityTrendConfig()

    # Calculate empirical class counts from training set
    d24_train = train_seq_df["vmax_plus_24h"].values - train_seq_df["vmax_curr"].values
    n_total = len(d24_train)
    n_ri_pos = int((d24_train >= config.ri_threshold_kt).sum())
    n_ri_neg = n_total - n_ri_pos
    w_calc = n_ri_neg / max(n_ri_pos, 1)  # empirical ratio N_neg / N_pos

    # Determine effective pos_weight for BCE
    if pos_weight_mult <= 0.0:
        eff_pos_weight = 1.0  # unweighted
    else:
        eff_pos_weight = pos_weight_mult * w_calc

    # Calculate Trend inverse frequency weights
    n_t0 = int((d24_train <= config.weakening_threshold_kt).sum())
    n_t1 = int(((d24_train > config.weakening_threshold_kt) & (d24_train < config.intensifying_threshold_kt)).sum())
    n_t2 = int((d24_train >= config.intensifying_threshold_kt).sum())
    trend_weights = n_total / (3.0 * np.array([n_t0, n_t1, n_t2], dtype=np.float32))

    print("=" * 80)
    print(f"STARTING TRAINING: {save_dir_name}")
    print(f"  • Device:                 {device}")
    print(f"  • Training Samples:       {n_total:,}")
    print(f"  • RI Positives:           {n_ri_pos:,} ({n_ri_pos / n_total * 100:.2f}%)")
    print(f"  • Calculated w_pos:       {w_calc:.3f} (N_neg / N_pos)")
    print(f"  • pos_weight_mult:        {pos_weight_mult:.2f} -> Effective pos_weight: {eff_pos_weight:.3f}")
    print(f"  • Trend class counts:     Weakening={n_t0:,}, Stable={n_t1:,}, Intensifying={n_t2:,}")
    print(f"  • Trend class weights:    {trend_weights.round(3).tolist()}")
    print("=" * 80)

    # DataLoaders
    with open(meta_dir / "normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)
    mean = [norm_stats["mean"][c] for c in channels]
    std = [norm_stats["std"][c] for c in channels]

    train_ds = TCIRTrendDataset(train_seq_df, mean=mean, std=std, channels=channels, is_training=True, config=config)
    val_ds = TCIRTrendDataset(val_seq_df, mean=mean, std=std, channels=channels, is_training=False, config=config)
    test_ds = TCIRTrendDataset(test_seq_df, mean=mean, std=std, channels=channels, is_training=False, config=config)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Instantiate Model
    model = TemporalClassifier(
        in_channels=len(channels),
        d_model=256,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        pretrained_cnn=False,
    )

    if warm_start_ckpt and Path(warm_start_ckpt).exists():
        model.load_backbone_from_forecaster(warm_start_ckpt, device)
    model.to(device)

    # Loss Function
    loss_fn = JointTrendRILoss(
        ri_pos_weight=torch.tensor([eff_pos_weight], device=device, dtype=torch.float32),
        trend_class_weights=torch.tensor(trend_weights, device=device, dtype=torch.float32),
        lambda_ri=1.0,
        lambda_trend=1.0,
        lambda_reg=0.1,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_score = -1.0
    best_epoch = -1
    best_ckpt_path = save_dir / "best.pt"
    log_rows = []

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_loss, train_losses, ep_time = train_one_epoch(
            model, train_loader, optimizer, scaler, loss_fn, device, epoch, epochs
        )
        scheduler.step()

        # Validation
        val_loss, val_metrics, _, _, _, _, _ = evaluate_classifier(
            model, val_loader, device, loss_fn, ri_threshold=0.5
        )

        # Composite validation score emphasizing headline RI PR-AUC + Trend Macro F1
        val_score = 0.6 * val_metrics["ri_pr_auc"] + 0.4 * val_metrics["trend_macro_f1"]

        print(
            f"  Epoch {epoch:2d} Complete ({ep_time:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
            f"  • Val RI PR-AUC: {val_metrics['ri_pr_auc']:.4f} | ROC-AUC: {val_metrics['ri_roc_auc']:.4f} | "
            f"Opt F1: {val_metrics['ri_optimal_f1']:.4f} (at thr={val_metrics['ri_optimal_threshold']:.2f})\n"
            f"  • Val Trend Acc: {val_metrics['trend_accuracy']*100:.2f}% | Macro F1: {val_metrics['trend_macro_f1']:.4f} | Score: {val_score:.4f}"
        )

        log_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_ri_pr_auc": val_metrics["ri_pr_auc"],
            "val_ri_roc_auc": val_metrics["ri_roc_auc"],
            "val_ri_f1": val_metrics["ri_f1"],
            "val_ri_opt_f1": val_metrics["ri_optimal_f1"],
            "val_trend_acc": val_metrics["trend_accuracy"],
            "val_trend_macro_f1": val_metrics["trend_macro_f1"],
            "val_composite_score": round(val_score, 4),
        }
        log_rows.append(log_row)

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_score": best_val_score,
                    "val_metrics": val_metrics,
                    "config": config.__dict__,
                    "pos_weight_mult": pos_weight_mult,
                    "eff_pos_weight": eff_pos_weight,
                },
                best_ckpt_path,
            )
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Epoch {epoch}, Score {val_score:.4f})")

        # Thermal Cooldown Gap between epochs to prevent GPU overheating
        if epoch < epochs and cooldown_seconds > 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  [Thermal Safety] Cooling down GPU for {cooldown_seconds}s before epoch {epoch + 1}...")
            time.sleep(cooldown_seconds)

    # Save training log
    pd.DataFrame(log_rows).to_csv(save_dir / "training_log.csv", index=False)

    # Load best checkpoint and evaluate on held-out test set
    print(f"\nLoading best checkpoint from Epoch {best_epoch} for Test Set Evaluation...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Determine optimal threshold from validation set
    val_loss, val_metrics, _, _, _, _, _ = evaluate_classifier(
        model, val_loader, device, loss_fn, ri_threshold=0.5
    )
    opt_ri_thresh = val_metrics["ri_optimal_threshold"]
    print(f"Optimal RI threshold derived from Validation set: {opt_ri_thresh:.3f}")

    # Evaluate on held-out test set
    test_loss, test_metrics, test_ri_probs, test_trend_preds, test_trend_probs, test_reg_preds, _ = (
        evaluate_classifier(model, test_loader, device, loss_fn, ri_threshold=opt_ri_thresh)
    )

    test_metrics["best_epoch"] = best_epoch
    test_metrics["val_opt_ri_threshold"] = opt_ri_thresh
    test_metrics["pos_weight_mult"] = pos_weight_mult
    test_metrics["eff_pos_weight"] = eff_pos_weight

    # Save test metrics JSON
    with open(save_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Save detailed test predictions CSV
    test_pred_df = pd.DataFrame({
        "cyclone_id": test_seq_df["cyclone_id"],
        "target_t_timestamp": test_seq_df["target_t_timestamp"],
        "vmax_curr": test_seq_df["vmax_curr"],
        "vmax_plus_24h": test_seq_df["vmax_plus_24h"],
        "delta_v_24": test_seq_df["vmax_plus_24h"] - test_seq_df["vmax_curr"],
        "actual_trend": [config.compute_trend_label(dv) for dv in (test_seq_df["vmax_plus_24h"] - test_seq_df["vmax_curr"])],
        "actual_ri": [(1 if dv >= config.ri_threshold_kt else 0) for dv in (test_seq_df["vmax_plus_24h"] - test_seq_df["vmax_curr"])],
        "pred_ri_prob": test_ri_probs,
        "pred_ri_flag": (test_ri_probs >= opt_ri_thresh).astype(int),
        "pred_trend": test_trend_preds,
        "prob_weakening": test_trend_probs[:, 0],
        "prob_stable": test_trend_probs[:, 1],
        "prob_intensifying": test_trend_probs[:, 2],
        "pred_plus_6h": test_reg_preds[:, 0],
        "pred_plus_12h": test_reg_preds[:, 1],
        "pred_plus_24h": test_reg_preds[:, 2],
    })
    test_pred_df.to_csv(save_dir / "test_predictions.csv", index=False)

    print("\n" + "=" * 80)
    print("HELD-OUT TEST SET EVALUATION COMPLETE")
    print(f"  • RI PR-AUC:                 {test_metrics['ri_pr_auc']:.4f}")
    print(f"  • RI ROC-AUC:                {test_metrics['ri_roc_auc']:.4f}")
    print(f"  • RI Precision (@ {opt_ri_thresh:.2f}):      {test_metrics[f'ri_precision']:.4f}")
    print(f"  • RI Recall (@ {opt_ri_thresh:.2f}):         {test_metrics[f'ri_recall']:.4f}")
    print(f"  • RI F1 (@ {opt_ri_thresh:.2f}):             {test_metrics[f'ri_f1']:.4f}")
    print(f"  • RI Brier Score:            {test_metrics['ri_brier']:.4f}")
    print(f"  • RI Expected Calib Error:   {test_metrics['ri_ece']:.4f}")
    print(f"  • Trend Accuracy:            {test_metrics['trend_accuracy']*100:.2f}%")
    print(f"  • Trend Macro F1:            {test_metrics['trend_macro_f1']:.4f}")
    print(f"  • Test Predictions saved ->  {save_dir / 'test_predictions.csv'}")
    print("=" * 80)

    train_ds.close()
    val_ds.close()
    test_ds.close()
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Rapid Intensification & Intensity Trend Classifier")
    parser.add_argument("--save-dir", type=str, default="classifier_primary_ri")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pos-weight-mult", type=float, default=1.0, help="Multiplier for empirical N_neg/N_pos (0.0 for 1.0 unweighted)")
    parser.add_argument("--cooldown-seconds", type=int, default=15, help="Thermal cooling gap in seconds between epochs to prevent GPU overheating")
    parser.add_argument("--warm-start", type=str, default="experiments/forecasting/checkpoints/cnn_transformer_k5/best.pt")
    args = parser.parse_args()

    run_training(
        save_dir_name=args.save_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        pos_weight_mult=args.pos_weight_mult,
        cooldown_seconds=args.cooldown_seconds,
        warm_start_ckpt=args.warm_start,
    )
