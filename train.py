"""Unified, production-grade experiment training runner for NVIDIA A100 80GB.

STRICT TEST-SET PROTECTION:
  This script NEVER loads, reads, or evaluates test-set data.
  All early stopping, checkpoint selection, and hyperparameter tuning strictly use
  validation metrics (Val MAE for forecasting, Val PR-AUC for RI classification).
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

from src.data.environmental import EnvironmentalFeatureManager, get_feature_dim
from src.data.sequence_dataset import TCIRSequenceDataset
from src.evaluation.classification_metrics import compute_ri_metrics
from src.evaluation.metrics import calculate_metrics
from src.evaluation.sanity_checks import TrajectoryEvaluator
from src.models.backbones import SpatialBackbone
from src.models.probabilistic import PinballLoss, ProbabilisticQuantileForecaster, compute_probabilistic_metrics
from src.models.residual_forecaster import ResidualDeltaVForecaster
from src.models.ri_models import DedicatedRIClassifier, MultiTaskRIIntensityModel, build_ri_loss
from src.models.temporal_forecaster import MultiHorizonHuberLoss, TemporalGRUForecaster, TemporalTransformerForecaster
from src.training.consistency_loss import MultiTaskConsistencyLoss


def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return commit.decode("ascii").strip()
    except Exception:
        return "unknown"


def build_model_from_config(cfg: Dict[str, Any], in_channels: int) -> nn.Module:
    m_cfg = cfg.get("model", {})
    m_type = m_cfg.get("type", "cnn_transformer").lower()
    backbone = m_cfg.get("backbone", "resnet18")
    d_model = m_cfg.get("d_model", 256)
    num_layers = m_cfg.get("num_layers", 2)
    nhead = m_cfg.get("nhead", 8)
    dropout = m_cfg.get("dropout", 0.1)
    pretrained = m_cfg.get("pretrained", True)

    if m_type == "residual":
        return ResidualDeltaVForecaster(
            backbone_arch=backbone,
            in_channels=in_channels,
            d_model=d_model,
            temporal_type=m_cfg.get("temporal_type", "transformer"),
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            parameterization=m_cfg.get("parameterization", "unconstrained"),
            pretrained_backbone=pretrained,
        )
    elif m_type == "ri_dedicated":
        return DedicatedRIClassifier(
            backbone_arch=backbone,
            in_channels=in_channels,
            d_model=d_model,
            d_env=get_feature_dim(),
            temporal_type=m_cfg.get("temporal_type", "transformer"),
            num_layers=num_layers,
            nhead=nhead,
            fusion_type=m_cfg.get("fusion_type", "gated"),
            dropout=dropout,
            pretrained_backbone=pretrained,
        )
    elif m_type in ["ri_multitask", "multitask"]:
        return MultiTaskRIIntensityModel(
            backbone_arch=backbone,
            in_channels=in_channels,
            d_model=d_model,
            d_env=get_feature_dim(),
            temporal_type=m_cfg.get("temporal_type", "transformer"),
            num_layers=num_layers,
            nhead=nhead,
            fusion_type=m_cfg.get("fusion_type", "gated"),
            dropout=dropout,
            pretrained_backbone=pretrained,
        )
    elif m_type == "probabilistic":
        return ProbabilisticQuantileForecaster(
            backbone_arch=backbone,
            in_channels=in_channels,
            d_model=d_model,
            temporal_type=m_cfg.get("temporal_type", "transformer"),
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            monotonic=m_cfg.get("monotonic", True),
            pretrained_backbone=pretrained,
        )
    elif m_type == "cnn_gru":
        return TemporalGRUForecaster(
            in_channels=in_channels,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            pretrained_cnn=pretrained,
        )
    else:
        # Default: Causal CNN + Temporal Transformer
        return TemporalTransformerForecaster(
            in_channels=in_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            pretrained_cnn=pretrained,
        )


def log_experiment_registry(meta: Dict[str, Any], registry_path: str = "experiments/registry.json"):
    reg_p = Path(registry_path)
    reg_p.parent.mkdir(parents=True, exist_ok=True)
    registry = []
    if reg_p.exists():
        try:
            with open(reg_p, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = []

    # Update or append
    exp_id = meta.get("experiment_id")
    registry = [r for r in registry if r.get("experiment_id") != exp_id]
    registry.append(meta)

    with open(reg_p, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train CycML tropical cyclone models.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--priority", type=str, default=None, choices=["essential", "high", "exploratory", "high-risk"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch smoke test without saving")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches per epoch (for quick smoke tests)")
    parser.add_argument("--resume", type=str, nargs="?", const="auto", default=None, help="Resume training from checkpoint (path or 'auto' for {save_dir}/best.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="Override total number of epochs")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.priority:
        cfg["priority"] = args.priority

    exp_id = cfg.get("experiment_id", Path(args.config).stem)
    save_dir = Path(cfg.get("save_dir", f"experiments/checkpoints/{exp_id}"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Hardware acceleration setup
    dev = torch.device(args.device)
    precision = cfg.get("training", {}).get("precision", "bf16").lower()
    if dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_amp = (precision in ["fp16", "bf16"]) and (dev.type == "cuda")
    amp_dtype = torch.bfloat16 if precision == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    # Seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    print("=" * 80)
    print(f"EXPERIMENT RUNNER: {exp_id.upper()}")
    print(f"  • Priority:   {cfg.get('priority', 'standard').upper()}")
    print(f"  • Hypothesis: {cfg.get('hypothesis', 'N/A')}")
    print(f"  • Device:     {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'})")
    print(f"  • Precision:  {precision.upper()} (AMP: {use_amp})")
    print(f"  • Save Dir:   {save_dir}")
    print("=" * 80)

    # Dataset & Manifests
    k_frames = cfg.get("dataset", {}).get("k_history", 5)
    channels = cfg.get("dataset", {}).get("channels", [0, 1, 2])
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    num_workers = cfg.get("training", {}).get("num_workers", 4)
    try:
        import shutil
        shm_usage = shutil.disk_usage("/dev/shm")
        if shm_usage.total < 500 * 1024 * 1024:  # Less than 500MB shm
            num_workers = 0
            print("Notice: /dev/shm is constrained (<500MB). Falling back to num_workers=0 to prevent IPC shm exhaustion.")
    except Exception:
        pass
    meta_dir = Path("data/metadata")
    aligned = cfg.get("dataset", {}).get("aligned", False)
    suffix = "_aligned" if aligned else ""

    train_manifest = meta_dir / f"forecast_train_sequences_k{k_frames}{suffix}.csv"
    val_manifest = meta_dir / f"forecast_val_sequences_k{k_frames}{suffix}.csv"

    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError(f"Sequence manifests for K={k_frames} missing from {meta_dir}. Run build_forecast_sequences.py first.")

    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest)

    with open(meta_dir / "normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)

    mean = [norm_stats["mean"][c] for c in channels]
    std = [norm_stats["std"][c] for c in channels]

    train_ds = TCIRSequenceDataset(train_df, mean=mean, std=std, channels=channels, is_training=True)
    val_ds = TCIRSequenceDataset(val_df, mean=mean, std=std, channels=channels, is_training=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(dev.type == "cuda"), persistent_workers=(num_workers > 0), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(dev.type == "cuda"), persistent_workers=(num_workers > 0), drop_last=False
    )

    # Environmental Manager
    feature_group = cfg.get("features", {}).get("group", "full_feature_set")
    env_manager = EnvironmentalFeatureManager(metadata_dir=meta_dir, feature_group=feature_group)

    # Model
    model = build_model_from_config(cfg, in_channels=len(channels)).to(dev)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[Model Architecture]: {cfg.get('model', {}).get('type')} ({total_params:,} parameters)")

    # Loss formulation
    m_type = cfg.get("model", {}).get("type", "cnn_transformer").lower()
    t_cfg = cfg.get("training", {})
    loss_name = t_cfg.get("loss", "huber").lower()

    if m_type == "probabilistic":
        loss_fn = PinballLoss(quantiles=(0.10, 0.50, 0.90))
    elif m_type == "ri_dedicated":
        loss_fn = build_ri_loss(loss_name, pos_weight=t_cfg.get("pos_weight", 4.0), gamma=t_cfg.get("gamma", 2.0))
    else:
        loss_fn = MultiHorizonHuberLoss(delta=1.0)

    # Optional Consistency Loss
    consistency_wt = float(t_cfg.get("consistency_weight", 0.0))
    consistency_loss_fn = MultiTaskConsistencyLoss(weight=consistency_wt) if consistency_wt > 0 else None

    # Optimizer & Scheduler
    epochs = args.epochs if args.epochs is not None else (1 if args.smoke_test else t_cfg.get("epochs", 25))
    lr = float(t_cfg.get("learning_rate", 1e-4))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=float(t_cfg.get("weight_decay", 1e-4)))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Primary Checkpoint Selection Metric
    eval_metric_name = t_cfg.get("eval_metric", "pr_auc" if "ri" in m_type else "val_mae")
    best_metric_val = -float("inf") if eval_metric_name == "pr_auc" else float("inf")
    best_epoch = -1
    patience = t_cfg.get("early_stopping_patience", 7)
    epochs_no_improve = 0

    start_epoch = 1
    if args.resume:
        resume_path = save_dir / "best.pt" if args.resume == "auto" else Path(args.resume)
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=dev)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as e:
                    print(f"  [Resume Warning] Optimizer state not restored: {e}")
            start_epoch = ckpt.get("epoch", 0) + 1
            best_metric_val = ckpt.get("best_metric", best_metric_val)
            best_epoch = ckpt.get("epoch", -1)
            print(f"  [Resume] Successfully resumed from {resume_path} (epoch {start_epoch-1}, best {eval_metric_name}: {best_metric_val:.4f})")
            for _ in range(1, start_epoch):
                scheduler.step()
        else:
            print(f"  [Resume Warning] Checkpoint {resume_path} not found. Starting from scratch.")

    log_rows = []
    train_start_time = time.time()

    print(f"\nBeginning training from epoch {start_epoch} to {epochs} (Early stopping metric: {eval_metric_name.upper()}, patience={patience})...\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_train_loss = 0.0
        n_train_batches = len(train_loader)
        t_epoch_start = time.time()

        for batch_idx, (images, vis_masks, targets, meta) in enumerate(train_loader):
            images = images.to(dev, non_blocking=True)
            vis_masks = vis_masks.to(dev, non_blocking=True)
            targets = targets.to(dev, non_blocking=True)
            v_curr = meta["vmax_curr"].to(dev).float()

            # Fetch causal environmental vectors
            env_vectors = [
                env_manager.get_features(meta["cyclone_id"][i], int(meta["target_t_timestamp"][i]))
                for i in range(len(images))
            ]
            x_env = torch.stack(env_vectors).to(dev)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                if m_type == "residual":
                    v_hat, delta_hat = model(images, v_curr=v_curr, vis_masks=vis_masks)
                    loss = loss_fn(v_hat, targets)
                elif m_type == "ri_dedicated":
                    ri_labels = (targets[:, 2] - v_curr >= 30.0).float().unsqueeze(1)
                    logits = model(images, vis_masks=vis_masks, x_env=x_env)
                    loss = loss_fn(logits, ri_labels)
                elif m_type in ["ri_multitask", "multitask"]:
                    v_hat, ri_logits, trend_logits = model(images, vis_masks=vis_masks, x_env=x_env)
                    loss_reg = MultiHorizonHuberLoss()(v_hat, targets)

                    ri_labels = (targets[:, 2] - v_curr >= 30.0).float().unsqueeze(1)
                    loss_ri = build_ri_loss(loss_name)(ri_logits, ri_labels)

                    delta_24 = targets[:, 2] - v_curr
                    trend_labels = torch.ones_like(delta_24, dtype=torch.long)
                    trend_labels[delta_24 <= -10.0] = 0
                    trend_labels[delta_24 >= 10.0] = 2
                    loss_trend = nn.CrossEntropyLoss()(trend_logits, trend_labels)

                    loss = loss_reg + 2.0 * loss_ri + 0.5 * loss_trend

                    if consistency_loss_fn:
                        pred_d24 = v_hat[:, 2] - v_curr
                        c_loss, _ = consistency_loss_fn(pred_d24, ri_logits)
                        loss = loss + c_loss
                elif m_type == "probabilistic":
                    q_out = model(images, vis_masks=vis_masks)
                    loss = loss_fn(q_out, targets)
                else:
                    preds = model(images, vis_masks=vis_masks)
                    loss = loss_fn(preds, targets)

            if precision == "fp16":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_train_loss += loss.item()
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_train_batches:
                elapsed_b = time.time() - t_epoch_start
                rate = (batch_idx + 1) * batch_size / max(elapsed_b, 1e-4)
                print(f"  Epoch [{epoch:2d}/{epochs:2d}] Batch [{batch_idx+1:3d}/{n_train_batches:3d}] - Loss: {loss.item():.4f} ({rate:.1f} samples/s)", flush=True)

            if args.max_batches and (batch_idx + 1) >= args.max_batches:
                break

        scheduler.step()
        epoch_dur = time.time() - t_epoch_start
        avg_train_loss = total_train_loss / max(n_train_batches, 1)

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        val_preds_list = []
        val_targets_list = []
        val_probs_list = []
        val_labels_list = []
        val_vcurr_list = []

        with torch.no_grad():
            for images, vis_masks, targets, meta in val_loader:
                images = images.to(dev, non_blocking=True)
                vis_masks = vis_masks.to(dev, non_blocking=True)
                targets_gpu = targets.to(dev, non_blocking=True)
                v_curr = meta["vmax_curr"].to(dev).float()
                val_vcurr_list.append(v_curr.cpu().numpy())

                env_vectors = [
                    env_manager.get_features(meta["cyclone_id"][i], int(meta["target_t_timestamp"][i]))
                    for i in range(len(images))
                ]
                x_env = torch.stack(env_vectors).to(dev)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    if m_type == "residual":
                        v_hat, _ = model(images, v_curr=v_curr, vis_masks=vis_masks)
                        loss = loss_fn(v_hat, targets_gpu)
                        val_preds_list.append(v_hat.float().cpu().numpy())
                    elif m_type == "ri_dedicated":
                        ri_labels = (targets[:, 2] - v_curr.cpu() >= 30.0).float().unsqueeze(1)
                        logits = model(images, vis_masks=vis_masks, x_env=x_env)
                        loss = loss_fn(logits, ri_labels.to(dev))
                        val_probs_list.append(torch.sigmoid(logits).float().cpu().numpy().flatten())
                        val_labels_list.append(ri_labels.numpy().flatten())
                    elif m_type in ["ri_multitask", "multitask"]:
                        v_hat, ri_logits, _ = model(images, vis_masks=vis_masks, x_env=x_env)
                        loss = MultiHorizonHuberLoss()(v_hat, targets_gpu)
                        val_preds_list.append(v_hat.float().cpu().numpy())
                        val_probs_list.append(torch.sigmoid(ri_logits).float().cpu().numpy().flatten())
                        val_labels_list.append((targets[:, 2] - v_curr.cpu() >= 30.0).numpy().flatten())
                    elif m_type == "probabilistic":
                        q_out = model(images, vis_masks=vis_masks)
                        loss = loss_fn(q_out, targets_gpu)
                        val_preds_list.append(q_out.float().cpu().numpy())
                    else:
                        preds = model(images, vis_masks=vis_masks)
                        loss = loss_fn(preds, targets_gpu)
                        val_preds_list.append(preds.float().cpu().numpy())

                total_val_loss += loss.item() * len(targets)
                val_targets_list.append(targets.numpy())
                if args.max_batches and len(val_targets_list) >= args.max_batches:
                    break

        avg_val_loss = total_val_loss / max(sum(len(t) for t in val_targets_list), 1)

        # Compute Validation Metrics
        metric_score = 0.0
        val_summary_str = f"Val Loss: {avg_val_loss:.4f}"

        if m_type == "ri_dedicated":
            all_probs = np.concatenate(val_probs_list)
            all_labels = np.concatenate(val_labels_list)
            ri_metrics = compute_ri_metrics(all_labels, all_probs)
            metric_score = ri_metrics["pr_auc"]
            val_summary_str += f" | Val PR-AUC: {metric_score:.4f} | ROC-AUC: {ri_metrics['roc_auc']:.4f} | Brier: {ri_metrics['brier_score']:.4f} | ECE: {ri_metrics['ece']:.4f} | Opt F1: {ri_metrics['optimal_f1']:.3f} (thr: {ri_metrics['optimal_threshold']:.2f})"
        elif m_type == "probabilistic":
            all_q = np.concatenate(val_preds_list, axis=0)
            all_t = np.concatenate(val_targets_list, axis=0)
            prob_metrics = compute_probabilistic_metrics(all_q, all_t)
            metric_score = prob_metrics["mae_q50_+24h"]
            val_summary_str += f" | Median MAE (+24h): {metric_score:.2f} kt | Cov (+24h): {prob_metrics['coverage_+24h']:.3f} | Cross: {prob_metrics['crossing_rate_+24h']:.3f}"
        else:
            all_preds = np.concatenate(val_preds_list, axis=0)
            all_t = np.concatenate(val_targets_list, axis=0)
            m_6h = calculate_metrics(all_preds[:, 0], all_t[:, 0])
            m_12h = calculate_metrics(all_preds[:, 1], all_t[:, 1])
            m_24h = calculate_metrics(all_preds[:, 2], all_t[:, 2])
            mae_6h, rmse_6h, r2_6h = m_6h["mae"], m_6h["rmse"], m_6h["r2"]
            mae_12h, rmse_12h, r2_12h = m_12h["mae"], m_12h["rmse"], m_12h["r2"]
            mae_24h, rmse_24h, r2_24h = m_24h["mae"], m_24h["rmse"], m_24h["r2"]
            mean_mae = float(np.mean([mae_6h, mae_12h, mae_24h]))
            metric_score = mean_mae

            false_dips = 0
            if val_vcurr_list:
                all_vcurr = np.concatenate(val_vcurr_list).flatten()
                traj_eval = TrajectoryEvaluator().evaluate_trajectories(all_preds, all_t, all_vcurr)
                false_dips = traj_eval.get("false_dip_count", 0)

            val_summary_str += f" | Val MAE: {mean_mae:.2f} kt (+6h: {mae_6h:.2f} [R2: {r2_6h:.2f}], +12h: {mae_12h:.2f} [R2: {r2_12h:.2f}], +24h: {mae_24h:.2f} [R2: {r2_24h:.2f}]) | Dips: {false_dips}"

        print(f"Epoch [{epoch:2d}/{epochs:2d}] - Train Loss: {avg_train_loss:.4f} | {val_summary_str} | Time: {epoch_dur:.1f}s", flush=True)

        # Checkpoint evaluation check
        improved = (metric_score > best_metric_val) if eval_metric_name == "pr_auc" else (metric_score < best_metric_val)

        if improved:
            best_metric_val = metric_score
            best_epoch = epoch
            epochs_no_improve = 0
            if not args.smoke_test:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric": best_metric_val,
                        "metric_name": eval_metric_name,
                        "config": cfg,
                    },
                    save_dir / "best.pt",
                )
                print(f"  [Checkpoint] Saved new best checkpoint -> best.pt ({eval_metric_name}: {best_metric_val:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and not args.smoke_test:
                print(f"\n[Early Stopping] No improvement in {eval_metric_name} for {patience} epochs. Terminating training.")
                break

    total_training_sec = time.time() - train_start_time
    peak_vram = torch.cuda.max_memory_allocated(dev) / (1024 ** 2) if dev.type == "cuda" else 0.0

    print("-" * 80)
    print(f"Training Complete for {exp_id}!")
    print(f"  • Best Epoch:         {best_epoch}")
    print(f"  • Best {eval_metric_name.upper()}:     {best_metric_val:.4f}")
    print(f"  • Total Time:         {total_training_sec / 60.0:.1f} min")
    print(f"  • Peak VRAM:          {peak_vram:.0f} MB")
    print("-" * 80)

    # Save Run Metadata
    meta_info = {
        "experiment_id": exp_id,
        "priority": cfg.get("priority", "standard"),
        "hypothesis": cfg.get("hypothesis", ""),
        "git_commit": get_git_commit(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": seed,
        "model_type": m_type,
        "backbone": cfg.get("model", {}).get("backbone", "resnet18"),
        "k_history": k_frames,
        "channels": channels,
        "environmental_features": feature_group,
        "loss": loss_name,
        "batch_size": batch_size,
        "learning_rate": lr,
        "parameter_count": total_params,
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU",
        "peak_vram_mb": round(peak_vram, 1),
        "total_training_sec": round(total_training_sec, 1),
        "best_epoch": best_epoch,
        "best_metric_name": eval_metric_name,
        "best_metric_val": round(best_metric_val, 4),
        "checkpoint_path": str(save_dir / "best.pt"),
    }

    with open(save_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    log_experiment_registry(meta_info)

    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()
