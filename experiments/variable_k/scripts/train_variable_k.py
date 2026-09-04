"""Isolated Training Script for Variable-Length Temporal Context Model.

Trains ONE unified model using variable sequence lengths K in {3, 5, 7} sampled uniformly at training time,
with deterministic multi-K validation passes (Validation A: K=3, Validation B: K=5, Validation C: K=7).
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import json
import os
import platform
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, r2_score, mean_squared_error
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.trend_config import IntensityTrendConfig
from src.evaluation.classification_metrics import (
    compute_ri_metrics,
    compute_trend_metrics,
    find_optimal_threshold,
)
from src.models.environmental_temporal_classifier import EnvironmentalTemporalClassifier
from src.models.temporal_classifier import JointTrendRILoss
from experiments.variable_k.scripts.variable_k_dataset import build_variable_k_dataloaders, VariableKCollator


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    r2 = [float(r2_score(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
    return {
        "mae_6h": float(mae[0]),
        "mae_12h": float(mae[1]),
        "mae_24h": float(mae[2]),
        "mae_mean": float(np.mean(mae)),
        "rmse_6h": float(rmse[0]),
        "rmse_12h": float(rmse[1]),
        "rmse_24h": float(rmse[2]),
        "r2_6h": r2[0],
        "r2_12h": r2[1],
        "r2_24h": r2[2],
    }


def compute_delta_v24_regression_metrics(actual_dv24: np.ndarray, pred_dv24: np.ndarray) -> Dict[str, float]:
    mean_act = float(np.mean(actual_dv24))
    mean_pred = float(np.mean(pred_dv24))
    if len(actual_dv24) > 1 and np.std(actual_dv24) > 1e-6 and np.std(pred_dv24) > 1e-6:
        slope, intercept = np.polyfit(actual_dv24, pred_dv24, deg=1)
        corr = float(np.corrcoef(actual_dv24, pred_dv24)[0, 1])
    else:
        slope, intercept, corr = 0.0, mean_pred, 0.0
    return {
        "mean_actual_dv24": mean_act,
        "mean_pred_dv24": mean_pred,
        "slope_dv24": float(slope),
        "intercept_dv24": float(intercept),
        "corr_dv24": float(corr),
    }


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    train_collator: VariableKCollator,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: JointTrendRILoss,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, Dict[str, float], float]:
    model.train()
    total_loss = 0.0
    accum_losses = {"loss_ri": 0.0, "loss_trend": 0.0, "loss_reg": 0.0}
    n_batches = len(train_loader)
    start_time = time.time()

    k_counts = {3: 0, 5: 0, 7: 0}

    for batch_idx, batch in enumerate(train_loader):
        images, vis_masks, trend_targets, ri_targets, reg_targets, env_vec, _ = batch

        k_curr = images.shape[1]
        k_counts[k_curr] += images.shape[0]

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

        if (batch_idx + 1) % 400 == 0 or (batch_idx + 1) == n_batches:
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
    return avg_loss, avg_sub_losses, elapsed, k_counts


@torch.no_grad()
def evaluate_deterministic_k(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: JointTrendRILoss,
    device: torch.device,
    threshold: Optional[float] = None,
) -> Dict:
    model.eval()
    total_loss = 0.0

    all_ri_probs = []
    all_ri_targets = []
    all_trend_preds = []
    all_trend_probs = []
    all_trend_targets = []
    all_reg_preds = []
    all_reg_targets = []
    all_vcurr = []
    all_v24 = []
    all_cids = []
    all_timestamps = []

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

        all_vcurr.extend(meta["vmax_curr"].numpy())
        all_v24.extend(meta["vmax_plus_24h"].numpy())
        all_cids.extend(meta["cyclone_id"])
        all_timestamps.extend(meta["target_t_timestamp"])

    ri_probs = np.concatenate(all_ri_probs)
    ri_targets = np.concatenate(all_ri_targets)
    trend_probs = np.concatenate(all_trend_probs)
    trend_preds = np.concatenate(all_trend_preds)
    trend_targets = np.concatenate(all_trend_targets)
    reg_preds = np.concatenate(all_reg_preds)
    reg_targets = np.concatenate(all_reg_targets)
    vcurr = np.array(all_vcurr)
    v24 = np.array(all_v24)

    # Threshold selection
    if threshold is None:
        opt_thresh, opt_f1, opt_p, opt_r = find_optimal_threshold(ri_targets, ri_probs)
        used_thresh = opt_thresh
    else:
        used_thresh = threshold

    trend_metrics = compute_trend_metrics(trend_targets, trend_preds)
    ri_metrics = compute_ri_metrics(ri_targets, ri_probs, threshold=used_thresh)
    reg_metrics = compute_regression_metrics(reg_targets, reg_preds)

    # Brier score
    try:
        brier = float(brier_score_loss(ri_targets, ri_probs))
    except Exception:
        brier = 0.0

    # Delta V24 metrics
    act_dv24 = v24 - vcurr
    pred_dv24 = reg_preds[:, 2] - vcurr
    dv_metrics = compute_delta_v24_regression_metrics(act_dv24, pred_dv24)

    return {
        "loss": total_loss / max(1, len(loader)),
        "threshold_used": float(used_thresh),
        "trend_metrics": trend_metrics,
        "ri_metrics": ri_metrics,
        "regression_metrics": reg_metrics,
        "brier_score": brier,
        "dv24_metrics": dv_metrics,
        "predictions": {
            "cyclone_id": all_cids,
            "target_t_timestamp": all_timestamps,
            "vmax_curr": vcurr,
            "vmax_plus_24h": v24,
            "ri_probs": ri_probs,
            "ri_targets": ri_targets,
            "trend_preds": trend_preds,
            "trend_probs": trend_probs,
            "trend_targets": trend_targets,
            "reg_preds": reg_preds,
            "reg_targets": reg_targets,
            "pred_dv24": pred_dv24,
            "act_dv24": act_dv24,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Train Isolated Variable-K Model")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/variable_k/checkpoints")
    parser.add_argument("--results-dir", type=str, default="experiments/variable_k/results")
    parser.add_argument("--warm-start", type=str, default="experiments/trend_classification/checkpoints/classifier_primary_ri/best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VARIABLE-LENGTH TEMPORAL CONTEXT EXPERIMENT (VARIABLE K IN {3, 5, 7})")
    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"Seed: {args.seed} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 80)

    # 1. Load sequence metadata manifests
    train_df = pd.read_csv("data/metadata/forecast_train_sequences_k7.csv")
    val_df = pd.read_csv("data/metadata/forecast_val_sequences_k7.csv")
    test_df = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")

    env_cache = torch.load("data/metadata/environmental_features_k7.pt")
    train_env = env_cache["train"]
    val_env = env_cache["val"]
    test_env = env_cache["test"]

    with open("data/metadata/normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)
    norm_mean = [norm_stats["mean"][c] for c in [0, 1, 2]]
    norm_std = [norm_stats["std"][c] for c in [0, 1, 2]]

    config = IntensityTrendConfig()

    train_loader, val_loaders, test_loaders, train_collator = build_variable_k_dataloaders(
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
        train_mode="variable",
        seed=args.seed,
    )

    # 2. Instantiate Model
    model = EnvironmentalTemporalClassifier(
        channels=3,
        num_frames=7,
        d_model=256,
        n_heads=8,
        num_layers=2,
        dropout=0.1,
        use_vis_channel=True,
    ).to(device)

    if args.warm_start and os.path.exists(args.warm_start):
        model.load_pretrained_backbone(args.warm_start)

    # 3. Loss Function
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

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Save initial config and run metadata
    run_config = {
        "experiment": "variable_k_training",
        "k_sampling": "uniform_3_5_7",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "architecture": "EnvironmentalTemporalClassifier",
        "channels": [0, 1, 2],
        "norm_stats_file": "data/metadata/normalization_stats_multichannel.json",
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "loss_weights": {"lambda_ri": 1.0, "lambda_trend": 1.0, "lambda_reg": 0.1, "w_pos": float(w_pos)},
        "warm_start": args.warm_start,
    }
    with open(results_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    run_meta = {
        "seed": args.seed,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "none",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git_commit": get_git_commit(),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "manifests": {
            "train": "data/metadata/forecast_train_sequences_k7.csv",
            "val": "data/metadata/forecast_val_sequences_k7.csv",
            "test": "data/metadata/forecast_test_sequences_k7.csv",
            "env_cache": "data/metadata/environmental_features_k7.pt",
        },
    }
    with open(results_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    best_mean_ri_pr_auc = 0.0
    best_tau_k7 = 0.0161
    best_epoch = 0

    print("\nStarting Variable-K Training Loops...\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n==================== EPOCH [{epoch}/{args.epochs}] ====================")
        train_loss, train_sub, train_sec, k_counts = train_one_epoch(
            model=model,
            train_loader=train_loader,
            train_collator=train_collator,
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        scheduler.step()

        c = k_counts
        tot_c = max(1, sum(c.values()))
        print(f"Epoch [{epoch}/{args.epochs}] Complete in {train_sec:.1f}s - Train Loss: {train_loss:.4f}")
        print(f"  • Sample Distribution: K=3: {c[3]:,} ({c[3]/tot_c*100:.1f}%) | K=5: {c[5]:,} ({c[5]/tot_c*100:.1f}%) | K=7: {c[7]:,} ({c[7]/tot_c*100:.1f}%) | Total: {tot_c:,}")

        # Deterministic multi-K validation passes
        print("\nRunning Multi-K Validation Passes...")
        val_results = {}
        for k in [3, 5, 7]:
            val_results[k] = evaluate_deterministic_k(
                model=model,
                loader=val_loaders[k],
                loss_fn=loss_fn,
                device=device,
                threshold=None,  # Find optimal on validation
            )
            v_k = val_results[k]
            tr = v_k["trend_metrics"]
            ri = v_k["ri_metrics"]
            reg = v_k["regression_metrics"]
            print(
                f"  [Val K={k}] Trend Acc: {tr['accuracy']*100:.2f}% (F1: {tr['macro_f1']:.4f}) | "
                f"RI PR-AUC: {ri['pr_auc']:.4f} | ROC-AUC: {ri['roc_auc']:.4f} | "
                f"RI F1: {ri['optimal_f1']:.4f} (Rec: {ri['optimal_recall']*100:.1f}%, Prec: {ri['optimal_precision']*100:.1f}% @ tau={ri['optimal_threshold']:.3f}) | "
                f"+24 MAE: {reg['mae_24h']:.2f} kt"
            )

        mean_ri_pr_auc = np.mean([val_results[k]["ri_metrics"]["pr_auc"] for k in [3, 5, 7]])
        print(f"  --> Composite Mean RI PR-AUC (K=3,5,7): {mean_ri_pr_auc:.4f}")

        # Save checkpoint payload
        tau_k7 = val_results[7]["ri_metrics"]["optimal_threshold"]
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_tau": tau_k7,
            "tau_by_k": {k: val_results[k]["ri_metrics"]["optimal_threshold"] for k in [3, 5, 7]},
            "val_metrics_by_k": {
                k: {
                    "trend_acc": val_results[k]["trend_metrics"]["accuracy"],
                    "trend_macro_f1": val_results[k]["trend_metrics"]["macro_f1"],
                    "ri_pr_auc": val_results[k]["ri_metrics"]["pr_auc"],
                    "ri_roc_auc": val_results[k]["ri_metrics"]["roc_auc"],
                    "ri_optimal_f1": val_results[k]["ri_metrics"]["optimal_f1"],
                    "ri_optimal_recall": val_results[k]["ri_metrics"]["optimal_recall"],
                    "ri_optimal_precision": val_results[k]["ri_metrics"]["optimal_precision"],
                    "reg_mae_6h": val_results[k]["regression_metrics"]["mae_6h"],
                    "reg_mae_12h": val_results[k]["regression_metrics"]["mae_12h"],
                    "reg_mae_24h": val_results[k]["regression_metrics"]["mae_24h"],
                }
                for k in [3, 5, 7]
            },
            "mean_ri_pr_auc": float(mean_ri_pr_auc),
        }

        torch.save(ckpt_payload, ckpt_dir / "latest.pt")

        if mean_ri_pr_auc > best_mean_ri_pr_auc:
            best_mean_ri_pr_auc = mean_ri_pr_auc
            best_tau_k7 = tau_k7
            best_epoch = epoch
            torch.save(ckpt_payload, ckpt_dir / "best_ri_pr_auc.pt")
            torch.save(ckpt_payload, ckpt_dir / "best.pt")
            print(f"  >>> SAVED NEW BEST COMPOSITE RI PR-AUC CHECKPOINT (Epoch {epoch}: {mean_ri_pr_auc:.4f})")

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE! Best Epoch: {best_epoch} with Mean RI PR-AUC: {best_mean_ri_pr_auc:.4f}")
    print(f"Best Checkpoint saved at: {ckpt_dir / 'best.pt'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
