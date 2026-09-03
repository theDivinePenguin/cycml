"""
Training pipeline for Multi-Modal Environmental Tropical Cyclone Classifier.
Integrates 5 satellite frames with physical environmental predictors (SST, OHC, Shear, RH, Vmax, MSLP).
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.trend_config import IntensityTrendConfig
from src.data.trend_dataset import build_trend_dataloaders
from src.evaluation.classification_metrics import (
    compute_ri_metrics,
    compute_trend_metrics,
    find_optimal_threshold,
)
from src.models.environmental_temporal_classifier import EnvironmentalTemporalClassifier
from src.models.temporal_classifier import JointTrendRILoss


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
    """Train environmental multi-modal model for one epoch."""
    model.train()
    total_loss = 0.0
    accum_losses = {"loss_ri": 0.0, "loss_trend": 0.0, "loss_reg": 0.0}
    n_batches = len(train_loader)
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        images, vis_masks, trend_targets, ri_targets, reg_targets, env_vec, _ = batch

        images = images.to(device, non_blocking=True)
        vis_masks = vis_masks.to(device, non_blocking=True)
        trend_targets = trend_targets.to(device, non_blocking=True)
        ri_targets = ri_targets.to(device, non_blocking=True)
        reg_targets = reg_targets.to(device, non_blocking=True)
        env_vec = env_vec.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            ri_logits, trend_logits, reg_preds = model(images, vis_masks, env_vec)
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

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, n_batches)
    avg_sub_losses = {k: v / max(1, n_batches) for k, v in accum_losses.items()}
    return avg_loss, avg_sub_losses, elapsed


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: JointTrendRILoss,
    device: torch.device,
    threshold: Optional[float] = None,
) -> Tuple[float, Dict[str, float], Dict]:
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0.0
    accum_losses = {"loss_ri": 0.0, "loss_trend": 0.0, "loss_reg": 0.0}

    all_ri_probs = []
    all_ri_targets = []
    all_trend_preds = []
    all_trend_probs = []
    all_trend_targets = []
    all_reg_preds = []
    all_reg_targets = []

    for batch in val_loader:
        images, vis_masks, trend_targets, ri_targets, reg_targets, env_vec, _ = batch

        images = images.to(device, non_blocking=True)
        vis_masks = vis_masks.to(device, non_blocking=True)
        trend_targets = trend_targets.to(device, non_blocking=True)
        ri_targets = ri_targets.to(device, non_blocking=True)
        reg_targets = reg_targets.to(device, non_blocking=True)
        env_vec = env_vec.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            ri_logits, trend_logits, reg_preds = model(images, vis_masks, env_vec)
            loss, loss_dict = loss_fn(
                ri_logits, trend_logits, reg_preds, ri_targets, trend_targets, reg_targets
            )

        total_loss += loss.item()
        for k in accum_losses:
            accum_losses[k] += loss_dict.get(k, 0.0)

        # Collect predictions
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

    n_batches = max(1, len(val_loader))
    avg_loss = total_loss / n_batches
    avg_sub_losses = {k: v / n_batches for k, v in accum_losses.items()}

    ri_probs = np.concatenate(all_ri_probs)
    ri_targets = np.concatenate(all_ri_targets)
    trend_probs = np.concatenate(all_trend_probs)
    trend_preds = np.concatenate(all_trend_preds)
    trend_targets = np.concatenate(all_trend_targets)
    reg_preds = np.concatenate(all_reg_preds)
    reg_targets = np.concatenate(all_reg_targets)

    # Compute comprehensive metrics
    trend_metrics = compute_trend_metrics(trend_targets, trend_preds)

    ri_tau = 0.141 if threshold is None else threshold
    ri_metrics = compute_ri_metrics(ri_targets, ri_probs, threshold=ri_tau)

    # Auxiliary regression MAE
    reg_mae = np.mean(np.abs(reg_preds - reg_targets), axis=0)

    summary = {
        "trend_metrics": trend_metrics,
        "ri_metrics": ri_metrics,
        "reg_mae_6h": float(reg_mae[0]),
        "reg_mae_12h": float(reg_mae[1]),
        "reg_mae_24h": float(reg_mae[2]),
        "reg_mae_mean": float(np.mean(reg_mae)),
        "predictions": {
            "ri_probs": ri_probs,
            "ri_targets": ri_targets,
            "trend_preds": trend_preds,
            "trend_probs": trend_probs,
            "trend_targets": trend_targets,
            "reg_preds": reg_preds,
            "reg_targets": reg_targets,
        },
    }
    return avg_loss, avg_sub_losses, summary


def main():
    parser = argparse.ArgumentParser(description="Train Environmental Multi-Modal Cyclone Classifier")
    parser.add_argument("--k-history", type=int, default=5, help="Number of history frames (e.g. 5 or 7)")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Warmup epochs")
    parser.add_argument("--cooldown-seconds", type=int, default=15, help="Thermal cooling pause between epochs")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/environmental_fusion/checkpoints/exp_e_full_env")
    parser.add_argument("--warm-start", type=str, default="experiments/trend_classification/checkpoints/classifier_primary_ri/best.pt")
    # Feature flags for Experiment E
    parser.add_argument("--use-vmax", action="store_true", default=True)
    parser.add_argument("--use-mslp", action="store_true", default=True)
    parser.add_argument("--use-sst", action="store_true", default=True)
    parser.add_argument("--use-ohc", action="store_true", default=True)
    parser.add_argument("--use-shear", action="store_true", default=True)
    parser.add_argument("--use-rh", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    k = args.k_history
    # 1. Load sequence metadata manifests
    print(f"Loading sequence manifests for K={k}...")
    train_df = pd.read_csv(f"data/metadata/forecast_train_sequences_k{k}.csv")
    val_df = pd.read_csv(f"data/metadata/forecast_val_sequences_k{k}.csv")
    test_df = pd.read_csv(f"data/metadata/forecast_test_sequences_k{k}.csv")

    # 2. Load precomputed PyTorch environmental feature cache
    print(f"Loading precomputed environmental feature cache for K={k}...")
    env_cache_path = f"data/metadata/environmental_features_k{k}.pt"
    env_cache = torch.load(env_cache_path)
    train_env = env_cache['train']
    val_env = env_cache['val']
    test_env = env_cache['test']
    print(f"Environmental features loaded: train={train_env.shape}, val={val_env.shape}, test={test_env.shape}")

    # 3. Build DataLoaders
    config = IntensityTrendConfig()
    with open("data/metadata/normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)
    norm_mean = [norm_stats["mean"][c] for c in [0, 1, 2]]
    norm_std = [norm_stats["std"][c] for c in [0, 1, 2]]
    print(f"Loaded canonical satellite normalization from normalization_stats_multichannel.json:")
    print(f"  • Channels [0, 1, 2] Means: {norm_mean}")
    print(f"  • Channels [0, 1, 2] Stds:  {norm_std}")

    train_loader, val_loader, test_loader = build_trend_dataloaders(
        train_seq_df=train_df,
        val_seq_df=val_df,
        test_seq_df=test_df,
        mean=norm_mean,
        std=norm_std,
        channels=[0, 1, 2],
        batch_size=args.batch_size,
        num_workers=4,
        config=config,
        train_env_tensor=train_env,
        val_env_tensor=val_env,
        test_env_tensor=test_env,
    )

    # 4. Instantiate Multi-Modal Architecture
    print(f"Instantiating EnvironmentalTemporalClassifier (num_frames={k})...")
    model = EnvironmentalTemporalClassifier(
        channels=3,
        num_frames=k,
        d_model=256,
        n_heads=8,
        num_layers=2,
        dropout=0.1,
        use_vis_channel=True,
        use_vmax=args.use_vmax,
        use_mslp=args.use_mslp,
        use_sst=args.use_sst,
        use_ohc=args.use_ohc,
        use_shear=args.use_shear,
        use_rh=args.use_rh,
    ).to(device)

    # Warm-start weights from satellite-only baseline
    if args.warm_start and os.path.exists(args.warm_start):
        model.load_pretrained_backbone(args.warm_start)

    # 5. Loss Function
    d24_train = train_df["vmax_plus_24h"].values - train_df["vmax_curr"].values
    n_total = len(d24_train)
    n_ri_pos = int((d24_train >= config.ri_threshold_kt).sum())
    w_pos = (n_total - n_ri_pos) / max(1, n_ri_pos)

    n_t0 = int((d24_train <= config.weakening_threshold_kt).sum())
    n_t1 = int(((d24_train > config.weakening_threshold_kt) & (d24_train < config.intensifying_threshold_kt)).sum())
    n_t2 = int((d24_train >= config.intensifying_threshold_kt).sum())
    trend_weights = n_total / (3.0 * np.array([n_t0, n_t1, n_t2], dtype=np.float32))

    loss_fn = JointTrendRILoss(
        ri_pos_weight=torch.tensor([w_pos], device=device, dtype=torch.float32),
        trend_class_weights=torch.tensor(trend_weights, device=device, dtype=torch.float32),
        lambda_ri=1.0,
        lambda_trend=1.0,
        lambda_reg=0.1,
    )

    # 6. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_ri_pr_auc = 0.0
    best_ri_f1 = 0.0
    best_trend_f1 = 0.0
    best_tau_ri = 0.141

    print("\n" + "=" * 80)
    print("BEGINNING MULTI-MODAL ENVIRONMENTAL TRAINING (EXPERIMENT E)")
    print(f"Features: Vmax={args.use_vmax}, MSLP={args.use_mslp}, SST={args.use_sst}, OHC={args.use_ohc}, Shear={args.use_shear}, RH={args.use_rh}")
    print("Checkpoint Strategy: Primary scientific checkpoint = best_ri_pr_auc.pt (copied to best.pt)")
    print("=" * 80 + "\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        print(f"--- Epoch [{epoch:2d}/{args.epochs:2d}] ---", flush=True)

        train_loss, train_sub, train_sec = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        scheduler.step()

        print(
            f"Epoch [{epoch:2d}/{args.epochs:2d}] Complete in {train_sec:.1f}s - Train Loss: {train_loss:.4f} "
            f"(RI: {train_sub['loss_ri']:.4f} | Trend: {train_sub['loss_trend']:.4f} | Reg: {train_sub['loss_reg']:.4f})",
            flush=True,
        )

        # Validation
        val_loss, val_sub, val_summary = evaluate_model(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            threshold=None,  # Find optimal threshold on validation set
        )

        tr_m = val_summary["trend_metrics"]
        ri_m = val_summary["ri_metrics"]
        val_macro_f1 = tr_m["macro_f1"]
        val_trend_acc = tr_m["accuracy"]
        val_tau = ri_m["optimal_threshold"]
        val_ri_pr_auc = ri_m["pr_auc"]
        val_ri_roc_auc = ri_m["roc_auc"]
        val_ri_f1 = ri_m["optimal_f1"]
        val_ri_rec = ri_m["optimal_recall"]
        val_ri_prec = ri_m["optimal_precision"]
        reg_6h = val_summary["reg_mae_6h"]
        reg_12h = val_summary["reg_mae_12h"]
        reg_24h = val_summary["reg_mae_24h"]
        reg_mean = val_summary["reg_mae_mean"]

        print(
            f"Validation Results [Epoch {epoch}/{args.epochs}]:\n"
            f"  • Trend Accuracy:  {val_trend_acc*100:.2f}% | Trend Macro F1: {val_macro_f1:.4f}\n"
            f"  • RI PR-AUC:       {val_ri_pr_auc:.4f} | RI ROC-AUC: {val_ri_roc_auc:.4f}\n"
            f"  • RI F1 (optimal): {val_ri_f1:.4f} (Recall: {val_ri_rec*100:.1f}%, Prec: {val_ri_prec*100:.1f}% @ tau={val_tau:.3f})\n"
            f"  • Forecast MAE:    +6h: {reg_6h:.2f} kt | +12h: {reg_12h:.2f} kt | +24h: {reg_24h:.2f} kt (Mean: {reg_mean:.2f} kt)",
            flush=True,
        )

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_tau": val_tau,
            "metrics": {
                "trend_acc": val_trend_acc,
                "trend_macro_f1": val_macro_f1,
                "ri_pr_auc": val_ri_pr_auc,
                "ri_roc_auc": val_ri_roc_auc,
                "ri_optimal_f1": val_ri_f1,
                "ri_optimal_recall": val_ri_rec,
                "ri_optimal_precision": val_ri_prec,
                "reg_mae_6h": reg_6h,
                "reg_mae_12h": reg_12h,
                "reg_mae_24h": reg_24h,
                "reg_mae_mean": reg_mean,
            },
        }

        # 1. Primary Checkpoint for RI: validation RI PR-AUC
        if val_ri_pr_auc > best_ri_pr_auc:
            best_ri_pr_auc = val_ri_pr_auc
            best_tau_ri = val_tau
            torch.save(ckpt_payload, os.path.join(args.checkpoint_dir, "best_ri_pr_auc.pt"))
            torch.save(ckpt_payload, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  >>> SAVED NEW BEST RI PR-AUC CHECKPOINT (best_ri_pr_auc.pt / best.pt: {best_ri_pr_auc:.4f})", flush=True)

        # 2. Checkpoint for RI F1
        if val_ri_f1 > best_ri_f1:
            best_ri_f1 = val_ri_f1
            torch.save(ckpt_payload, os.path.join(args.checkpoint_dir, "best_ri_f1.pt"))
            print(f"  >>> SAVED NEW BEST RI F1 CHECKPOINT (best_ri_f1.pt: {best_ri_f1:.4f})", flush=True)

        # 3. Checkpoint for Trend Macro F1
        if val_macro_f1 > best_trend_f1:
            best_trend_f1 = val_macro_f1
            torch.save(ckpt_payload, os.path.join(args.checkpoint_dir, "best_trend_f1.pt"))
            print(f"  >>> SAVED NEW BEST TREND F1 CHECKPOINT (best_trend_f1.pt: {best_trend_f1:.4f})", flush=True)

        # Thermal cooling pause
        if epoch < args.epochs:
            print(f"  [Thermal Safety] Pausing for {args.cooldown_seconds}s to cool GPU...", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            time.sleep(args.cooldown_seconds)

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE!")
    print(f"  • Best Validation RI PR-AUC:   {best_ri_pr_auc:.4f} (@ tau={best_tau_ri:.3f})")
    print(f"  • Best Validation RI F1:       {best_ri_f1:.4f}")
    print(f"  • Best Validation Trend F1:    {best_trend_f1:.4f}")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
