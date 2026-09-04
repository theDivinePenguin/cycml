"""Isolated Training Script for Delta Head and RI-Aware Weighting Experiments."""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is on sys.path
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, r2_score, precision_score, recall_score, f1_score
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
from experiments.ri_target_loss.scripts.dataset import build_delta_dataloaders
from experiments.ri_target_loss.scripts.models import DeltaEnvironmentalTemporalClassifier
from experiments.ri_target_loss.scripts.losses import DeltaJointLoss


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: DeltaJointLoss,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    mode: str,
) -> Tuple[float, Dict[str, float], float]:
    model.train()
    total_loss = 0.0
    accum_losses = {"loss_ri": 0.0, "loss_trend": 0.0, "loss_reg_delta": 0.0}
    if mode == "abs_and_delta":
        accum_losses["loss_reg_abs"] = 0.0
    n_batches = len(train_loader)
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        images, vis_masks, trend_targets, ri_targets, reg_abs_targets, reg_delta_targets, env_vec, _ = batch

        images = images.to(device, non_blocking=True)
        vis_masks = vis_masks.to(device, non_blocking=True)
        trend_targets = trend_targets.to(device, non_blocking=True)
        ri_targets = ri_targets.to(device, non_blocking=True)
        reg_abs_targets = reg_abs_targets.to(device, non_blocking=True)
        reg_delta_targets = reg_delta_targets.to(device, non_blocking=True)
        env_vec = env_vec.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            if mode == "abs_and_delta":
                ri_logits, trend_logits, reg_abs_preds, reg_delta_preds = model(images, vis_masks, env_vec)
                loss, loss_dict = loss_fn(
                    ri_logits=ri_logits,
                    trend_logits=trend_logits,
                    ri_targets=ri_targets,
                    trend_targets=trend_targets,
                    reg_delta_preds=reg_delta_preds,
                    reg_delta_targets=reg_delta_targets,
                    reg_abs_preds=reg_abs_preds,
                    reg_abs_targets=reg_abs_targets,
                )
            else:
                ri_logits, trend_logits, reg_delta_preds = model(images, vis_masks, env_vec)
                loss, loss_dict = loss_fn(
                    ri_logits=ri_logits,
                    trend_logits=trend_logits,
                    ri_targets=ri_targets,
                    trend_targets=trend_targets,
                    reg_delta_preds=reg_delta_preds,
                    reg_delta_targets=reg_delta_targets,
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k in accum_losses:
            accum_losses[k] += loss_dict.get(k, 0.0)

        if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == n_batches:
            curr_lr = optimizer.param_groups[0]["lr"]
            avg_tot = total_loss / (batch_idx + 1)
            print(
                f"  Epoch [{epoch:2d}/{total_epochs:2d}] Batch [{batch_idx + 1:4d}/{n_batches:4d}] - "
                f"Loss: {loss.item():.4f} (Avg: {avg_tot:.4f}) - LR: {curr_lr:.2e}",
                flush=True,
            )

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, n_batches)
    avg_sub_losses = {k: v / max(1, n_batches) for k, v in accum_losses.items()}
    return avg_loss, avg_sub_losses, elapsed


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: DeltaJointLoss,
    device: torch.device,
    mode: str,
    threshold: Optional[float] = None,
) -> Dict:
    model.eval()
    total_loss = 0.0

    all_ri_probs = []
    all_ri_targets = []
    all_trend_preds = []
    all_trend_probs = []
    all_trend_targets = []
    all_delta_preds = []
    all_delta_targets = []
    all_abs_preds = []
    all_abs_targets = []
    all_vcurr = []
    all_v24 = []
    all_cids = []
    all_timestamps = []

    for batch in loader:
        images, vis_masks, trend_targets, ri_targets, reg_abs_targets, reg_delta_targets, env_vec, meta = batch

        images = images.to(device, non_blocking=True)
        vis_masks = vis_masks.to(device, non_blocking=True)
        trend_targets = trend_targets.to(device, non_blocking=True)
        ri_targets = ri_targets.to(device, non_blocking=True)
        reg_abs_targets = reg_abs_targets.to(device, non_blocking=True)
        reg_delta_targets = reg_delta_targets.to(device, non_blocking=True)
        env_vec = env_vec.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            if mode == "abs_and_delta":
                ri_logits, trend_logits, reg_abs_preds, reg_delta_preds = model(images, vis_masks, env_vec)
                loss, _ = loss_fn(
                    ri_logits=ri_logits,
                    trend_logits=trend_logits,
                    ri_targets=ri_targets,
                    trend_targets=trend_targets,
                    reg_delta_preds=reg_delta_preds,
                    reg_delta_targets=reg_delta_targets,
                    reg_abs_preds=reg_abs_preds,
                    reg_abs_targets=reg_abs_targets,
                )
                all_abs_preds.append(reg_abs_preds.cpu().numpy())
            else:
                ri_logits, trend_logits, reg_delta_preds = model(images, vis_masks, env_vec)
                loss, _ = loss_fn(
                    ri_logits=ri_logits,
                    trend_logits=trend_logits,
                    ri_targets=ri_targets,
                    trend_targets=trend_targets,
                    reg_delta_preds=reg_delta_preds,
                    reg_delta_targets=reg_delta_targets,
                )

        total_loss += loss.item()

        ri_probs = torch.sigmoid(ri_logits).squeeze(-1).cpu().numpy()
        trend_probs = torch.softmax(trend_logits, dim=-1).cpu().numpy()
        trend_preds = np.argmax(trend_probs, axis=-1)

        all_ri_probs.append(ri_probs)
        all_ri_targets.append(ri_targets.cpu().numpy())
        all_trend_probs.append(trend_probs)
        all_trend_preds.append(trend_preds)
        all_trend_targets.append(trend_targets.cpu().numpy())
        all_delta_preds.append(reg_delta_preds.cpu().numpy())
        all_delta_targets.append(reg_delta_targets.cpu().numpy())
        all_abs_targets.append(reg_abs_targets.cpu().numpy())

        all_vcurr.extend(meta["vmax_curr"].numpy())
        all_v24.extend(meta["vmax_plus_24h"].numpy())
        all_cids.extend(meta["cyclone_id"])
        all_timestamps.extend(meta["target_t_timestamp"])

    ri_probs = np.concatenate(all_ri_probs).astype(np.float32)
    ri_targets = np.concatenate(all_ri_targets).astype(np.float32)
    trend_probs = np.concatenate(all_trend_probs).astype(np.float32)
    trend_preds = np.concatenate(all_trend_preds)
    trend_targets = np.concatenate(all_trend_targets)
    delta_preds = np.concatenate(all_delta_preds).astype(np.float32)
    delta_targets = np.concatenate(all_delta_targets).astype(np.float32)
    abs_targets = np.concatenate(all_abs_targets).astype(np.float32)
    vcurr = np.array(all_vcurr, dtype=np.float32)
    v24 = np.array(all_v24, dtype=np.float32)

    # Reconstructed intensity from delta: V_hat(h) = V(t) + delta_hat(h)
    recon_v6 = vcurr + delta_preds[:, 0]
    recon_v12 = vcurr + delta_preds[:, 1]
    recon_v24 = vcurr + delta_preds[:, 2]
    recon_preds = np.stack([recon_v6, recon_v12, recon_v24], axis=1)

    # Threshold selection
    if threshold is None:
        opt_thresh, opt_f1, opt_p, opt_r = find_optimal_threshold(ri_targets, ri_probs)
        used_thresh = opt_thresh
    else:
        used_thresh = threshold

    tr_m = compute_trend_metrics(trend_targets, trend_preds)
    ri_m = compute_ri_metrics(ri_targets, ri_probs, threshold=used_thresh)

    pred_ri_flag = (ri_probs >= used_thresh).astype(int)
    prec_at_tau = float(precision_score(ri_targets, pred_ri_flag, zero_division=0))
    rec_at_tau = float(recall_score(ri_targets, pred_ri_flag, zero_division=0))
    f1_at_tau = float(f1_score(ri_targets, pred_ri_flag, zero_division=0))

    # Delta MAE & Reconstructed Absolute MAE
    mae_delta = np.mean(np.abs(delta_preds - delta_targets), axis=0)
    mae_recon = np.mean(np.abs(recon_preds - abs_targets), axis=0)

    # Delta V24 slope & correlation
    act_dv24 = delta_targets[:, 2]
    pred_dv24 = delta_preds[:, 2]
    slope_dv, int_dv = np.polyfit(act_dv24, pred_dv24, deg=1)
    corr_dv = float(np.corrcoef(act_dv24, pred_dv24)[0, 1])

    # RI subset metrics
    ri_mask = act_dv24 >= 30.0
    if np.sum(ri_mask) > 0:
        ri_act = act_dv24[ri_mask]
        ri_pred = pred_dv24[ri_mask]
        ri_mae = float(np.mean(np.abs(recon_preds[ri_mask, 2] - abs_targets[ri_mask, 2])))
        ri_bias = float(np.mean(ri_pred - ri_act))
        ri_slope, _ = np.polyfit(ri_act, ri_pred, deg=1)
        ri_corr = float(np.corrcoef(ri_act, ri_pred)[0, 1])
    else:
        ri_mae, ri_bias, ri_slope, ri_corr = 0.0, 0.0, 0.0, 0.0

    out = {
        "loss": total_loss / max(1, len(loader)),
        "threshold_used": float(used_thresh),
        "trend_acc": float(tr_m["accuracy"]),
        "trend_macro_f1": float(tr_m["macro_f1"]),
        "ri_roc_auc": float(ri_m["roc_auc"]),
        "ri_pr_auc": float(ri_m["pr_auc"]),
        "ri_recall": rec_at_tau,
        "ri_precision": prec_at_tau,
        "ri_f1": f1_at_tau,
        "recon_mae_6h": float(mae_recon[0]),
        "recon_mae_12h": float(mae_recon[1]),
        "recon_mae_24h": float(mae_recon[2]),
        "recon_mae_mean": float(np.mean(mae_recon)),
        "slope_dv24": float(slope_dv),
        "corr_dv24": float(corr_dv),
        "ri_mae_24h": ri_mae,
        "ri_bias": ri_bias,
        "ri_slope": float(ri_slope),
        "ri_corr": float(ri_corr),
        "predictions": {
            "cyclone_id": all_cids,
            "target_t_timestamp": all_timestamps,
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
            "pred_delta_6h": delta_preds[:, 0],
            "pred_delta_12h": delta_preds[:, 1],
            "pred_delta_24h": delta_preds[:, 2],
            "recon_plus_6h": recon_preds[:, 0],
            "recon_plus_12h": recon_preds[:, 1],
            "recon_plus_24h": recon_preds[:, 2],
        }
    }

    if mode == "abs_and_delta" and len(all_abs_preds) > 0:
        abs_preds = np.concatenate(all_abs_preds)
        direct_mae = np.mean(np.abs(abs_preds - abs_targets), axis=0)
        out["direct_mae_6h"] = float(direct_mae[0])
        out["direct_mae_12h"] = float(direct_mae[1])
        out["direct_mae_24h"] = float(direct_mae[2])
        out["predictions"]["direct_plus_6h"] = abs_preds[:, 0]
        out["predictions"]["direct_plus_12h"] = abs_preds[:, 1]
        out["predictions"]["direct_plus_24h"] = abs_preds[:, 2]

    return out


def main():
    parser = argparse.ArgumentParser(description="Train Delta / RI-Aware Experiment")
    parser.add_argument("--mode", type=str, default="abs_and_delta", choices=["abs_and_delta", "delta_only"])
    parser.add_argument("--ri-weight-profile", type=str, default="none", choices=["none", "moderate", "strong", "very_strong", "ultra", "custom"])
    parser.add_argument("--custom-weights", type=str, default=None, help="Comma-separated 3 weights e.g. 1.0,6.0,12.0")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-start", type=str, default="experiments/trend_classification/checkpoints/classifier_primary_ri/best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path("experiments/ri_target_loss")
    ckpt_dir = base_dir / "checkpoints" / args.exp_name
    results_dir = base_dir / "results" / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine RI weights
    weight_map = {
        "none": None,
        "moderate": (1.0, 2.0, 4.0),
        "strong": (1.0, 3.0, 6.0),
        "very_strong": (1.0, 4.0, 8.0),
        "ultra": (1.0, 6.0, 12.0),
    }
    if args.custom_weights:
        ri_weights = tuple(float(x.strip()) for x in args.custom_weights.split(","))
    else:
        ri_weights = weight_map[args.ri_weight_profile]

    print("=" * 80)
    print(f"EXPERIMENT: {args.exp_name}")
    print(f"Mode: {args.mode} | RI Weight Profile: {args.ri_weight_profile} ({ri_weights})")
    print(f"Device: {device} | Epochs: {args.epochs} | Seed: {args.seed}")
    print("=" * 80)

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

    train_loader, val_loader, test_loader = build_delta_dataloaders(
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

    model = DeltaEnvironmentalTemporalClassifier(
        mode=args.mode,
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

    d24_train = train_df["vmax_plus_24h"].values - train_df["vmax_curr"].values
    n_total = len(d24_train)
    n_ri_pos = int((d24_train >= config.ri_threshold_kt).sum())
    w_pos = (n_total - n_ri_pos) / max(1, n_ri_pos)

    n_t0 = int((d24_train <= config.weakening_threshold_kt).sum())
    n_t1 = int(((d24_train > config.weakening_threshold_kt) & (d24_train < config.intensifying_threshold_kt)).sum())
    n_t2 = int((d24_train >= config.intensifying_threshold_kt).sum())
    trend_weights = n_total / (3.0 * np.array([n_t0, n_t1, n_t2], dtype=np.float32))

    loss_fn = DeltaJointLoss(
        mode=args.mode,
        ri_pos_weight=torch.tensor([w_pos], device=device, dtype=torch.float32),
        trend_class_weights=torch.tensor(trend_weights, device=device, dtype=torch.float32),
        lambda_ri=1.0,
        lambda_trend=1.0,
        lambda_reg_abs=0.1,
        lambda_reg_delta=0.1,
        ri_weights=ri_weights,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_score = 0.0
    best_tau = 0.0161
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch [{epoch}/{args.epochs}] ---")
        train_loss, train_sub, train_sec = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            mode=args.mode,
        )
        scheduler.step()

        val_res = evaluate_model(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            mode=args.mode,
            threshold=None,
        )

        print(
            f"Epoch {epoch} Complete ({train_sec:.1f}s) - Train Loss: {train_loss:.4f} | "
            f"Val Trend Acc: {val_res['trend_acc']*100:.2f}% | RI PR-AUC: {val_res['ri_pr_auc']:.4f} | "
            f"Recon +24 MAE: {val_res['recon_mae_24h']:.2f} kt | RI MAE: {val_res['ri_mae_24h']:.2f} kt | "
            f"RI Slope: {val_res['ri_slope']:.4f}"
        )

        val_score = val_res["ri_pr_auc"]
        if val_score > best_val_score:
            best_val_score = val_score
            best_tau = val_res["threshold_used"]
            best_epoch = epoch

            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_tau": best_tau,
                "val_metrics": {k: v for k, v in val_res.items() if k != "predictions"},
            }
            torch.save(payload, ckpt_dir / "best.pt")
            print(f"  >>> Saved new best checkpoint at Epoch {epoch} (Val RI PR-AUC: {val_score:.4f}, tau={best_tau:.4f})")

    print(f"\nTraining Complete! Best Epoch: {best_epoch} (Val RI PR-AUC: {best_val_score:.4f})")

    # Run Test Evaluation using best checkpoint
    print(f"\n--- Evaluating Test Set using Best Checkpoint (tau={best_tau:.4f}) ---")
    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_res = evaluate_model(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        mode=args.mode,
        threshold=best_tau,
    )

    test_preds_df = pd.DataFrame(test_res["predictions"])
    test_preds_df.to_csv(results_dir / "test_predictions.csv", index=False)

    test_metrics = {k: v for k, v in test_res.items() if k != "predictions"}
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Saved test predictions to {results_dir / 'test_predictions.csv'}")
    print(f"Saved test metrics to {results_dir / 'test_metrics.json'}")
    print(
        f"Test Results:\n"
        f"  • Trend Accuracy:  {test_metrics['trend_acc']*100:.2f}% | Macro F1: {test_metrics['trend_macro_f1']:.4f}\n"
        f"  • RI PR-AUC:       {test_metrics['ri_pr_auc']:.4f} | ROC-AUC: {test_metrics['ri_roc_auc']:.4f}\n"
        f"  • RI Recall:       {test_metrics['ri_recall']*100:.2f}% | Precision: {test_metrics['ri_precision']*100:.2f}%\n"
        f"  • Recon +24 MAE:   {test_metrics['recon_mae_24h']:.2f} kt\n"
        f"  • RI-only +24 MAE: {test_metrics['ri_mae_24h']:.2f} kt (Bias: {test_metrics['ri_bias']:.2f} kt)\n"
        f"  • RI-only Slope:   {test_metrics['ri_slope']:.4f} (Corr: {test_metrics['ri_corr']:.4f})"
    )


if __name__ == "__main__":
    main()
