"""Training and evaluation pipeline for Temporal Forecasting Models (GRU and Transformer)."""
import argparse
import gc
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

from src.data.sequence_dataset import TCIRSequenceDataset
from src.models.temporal_forecaster import (
    MultiHorizonHuberLoss,
    TemporalGRUForecaster,
    TemporalTransformerForecaster,
)


def compute_multi_horizon_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute per-horizon and overall MAE, RMSE, and R2."""
    metrics = {}
    horizons = ["+6h", "+12h", "+24h"]
    maes = []
    rmses = []
    r2s = []

    for idx, h_name in enumerate(horizons):
        p = preds[:, idx]
        t = targets[:, idx]
        err = p - t
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = float(1.0 - (ss_res / max(ss_tot, 1e-8)))
        bias = float(np.mean(err))

        metrics[f"mae_{h_name}"] = round(mae, 3)
        metrics[f"rmse_{h_name}"] = round(rmse, 3)
        metrics[f"r2_{h_name}"] = round(r2, 4)
        metrics[f"bias_{h_name}"] = round(bias, 3)

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    metrics["mean_mae"] = round(float(np.mean(maes)), 3)
    metrics["mean_rmse"] = round(float(np.mean(rmses)), 3)
    metrics["mean_r2"] = round(float(np.mean(r2s)), 4)
    return metrics


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    k_frames: int = 5,
) -> Tuple[float, float, float]:
    """Train model for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    n_batches = len(train_loader)
    start_time = time.time()

    for batch_idx, (images, vis_masks, targets, _) in enumerate(train_loader):
        # images: (B, K, C, H, W) -> slice to k_frames if ablation
        if k_frames < images.shape[1]:
            images = images[:, -k_frames:, :, :, :]
            vis_masks = vis_masks[:, -k_frames:]

        images = images.to(device, non_blocking=True)
        vis_masks = vis_masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images, vis_masks)  # (B, 3)
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        total_loss += loss_val

        if (batch_idx + 1) % 250 == 0 or (batch_idx + 1) == n_batches:
            curr_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch [{epoch:2d}/{total_epochs:2d}] Batch [{batch_idx + 1:4d}/{n_batches:4d}] - "
                f"Loss: {loss_val:.4f} (Avg: {total_loss / (batch_idx + 1):.4f}) - LR: {curr_lr:.2e}"
            )

    epoch_time = time.time() - start_time
    avg_loss = total_loss / max(n_batches, 1)
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    )
    return avg_loss, epoch_time, peak_vram


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    k_frames: int = 5,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """Run deterministic evaluation on validation or test loader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, vis_masks, targets, _ in loader:
        if k_frames < images.shape[1]:
            images = images[:, -k_frames:, :, :, :]
            vis_masks = vis_masks[:, -k_frames:]

        images = images.to(device, non_blocking=True)
        vis_masks = vis_masks.to(device, non_blocking=True)
        targets_gpu = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images, vis_masks)
            loss = loss_fn(outputs, targets_gpu)

        total_loss += loss.item() * len(targets)
        all_preds.append(outputs.cpu().float().numpy())
        all_targets.append(targets.float().numpy())

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss / max(len(targets_arr), 1)

    metrics = compute_multi_horizon_metrics(preds_arr, targets_arr)
    return avg_loss, metrics, preds_arr, targets_arr


def run_forecasting_experiment(
    model_type: str = "transformer",
    k_frames: int = 5,
    save_dir_name: str = "transformer_k5",
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 16,
    channels: list = [0, 1, 2],
):
    """Run full training, validation early stopping, and test evaluation for a forecasting model."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path("experiments/forecasting/checkpoints") / save_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    test_pred_path = save_dir / "test_predictions.csv"
    test_metrics_path = save_dir / "test_metrics.json"
    best_ckpt_path = save_dir / "best.pt"

    # -------------------------------------------------------------
    # LEGACY SCRIPT SAFEGUARD: Test set evaluation is locked
    # -------------------------------------------------------------
    eval_test_confirmed = False
    import sys
    if "--eval-test" in sys.argv and "--confirm-locked-test-eval" in sys.argv:
        eval_test_confirmed = True

    if test_pred_path.exists() and test_metrics_path.exists():
        print(f"\n[{save_dir_name}] Already completed with test predictions and metrics. Skipping.")
        with open(test_metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if best_ckpt_path.exists() and not test_pred_path.exists():
        if not eval_test_confirmed:
            print(f"\n[DEPRECATION / TEST LOCK] {save_dir_name} found best.pt, but TEST SET IS LOCKED.")
            print("  Skipping test evaluation. Canonical training must use `train.py`.")
            print("  To force evaluation, pass: --eval-test --confirm-locked-test-eval")
            return {"status": "TEST_LOCKED", "model_type": model_type}
        print(f"\n[{save_dir_name}] Found saved best.pt checkpoint! Running test set evaluation directly...")
        meta_dir = Path("data/metadata")
        test_seq_df = pd.read_csv(meta_dir / "forecast_test_sequences_k5.csv")
        with open(meta_dir / "normalization_stats_multichannel.json") as f:
            norm_stats = json.load(f)
        mean = [norm_stats["mean"][c] for c in channels]
        std = [norm_stats["std"][c] for c in channels]
        test_ds = TCIRSequenceDataset(test_seq_df, mean=mean, std=std, channels=channels, is_training=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        in_channels = len(channels)
        if model_type.lower() == "gru":
            model = TemporalGRUForecaster(in_channels=in_channels, d_model=256, num_layers=2, dropout=0.1, pretrained_cnn=False)
        else:
            model = TemporalTransformerForecaster(in_channels=in_channels, d_model=256, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.1, pretrained_cnn=False)
        
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model = model.to(device)
        loss_fn = MultiHorizonHuberLoss(delta=1.0)
        
        test_loss, test_metrics, test_preds, test_targets = evaluate_model(model, test_loader, device, loss_fn, k_frames=k_frames)
        test_metrics["best_epoch"] = best_ckpt.get("epoch", -1)
        test_metrics["best_val_mae"] = best_ckpt.get("best_val_mae", -1)
        test_metrics["model_type"] = model_type
        test_metrics["k_frames"] = k_frames
        
        with open(test_metrics_path, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
            
        pred_df = pd.DataFrame({
            "cyclone_id": test_seq_df["cyclone_id"],
            "target_t_timestamp": test_seq_df["target_t_timestamp"],
            "vmax_curr": test_seq_df["vmax_curr"],
            "actual_plus_6h": test_targets[:, 0],
            "actual_plus_12h": test_targets[:, 1],
            "actual_plus_24h": test_targets[:, 2],
            "pred_plus_6h": test_preds[:, 0],
            "pred_plus_12h": test_preds[:, 1],
            "pred_plus_24h": test_preds[:, 2],
        })
        pred_df.to_csv(test_pred_path, index=False)
        print(f"[{save_dir_name}] Saved test predictions -> {test_pred_path}")
        print(f"  • +6h MAE:  {test_metrics['mae_+6h']:.3f} kt")
        print(f"  • +12h MAE: {test_metrics['mae_+12h']:.3f} kt")
        print(f"  • +24h MAE: {test_metrics['mae_+24h']:.3f} kt")
        test_ds.close()
        return test_metrics

    print("\n" + "=" * 90)
    print(f"STARTING FORECASTING EXPERIMENT: {save_dir_name}")
    print(f"  • Model Type:     {model_type.upper()}")
    print(f"  • History Frames: {k_frames} (Spacing: 3h, Range: [t-{3*(k_frames-1)}h, ..., t])")
    print(f"  • Channels:       {channels} (IR1, WV, VIS)")
    print(f"  • Save Dir:       {save_dir}")
    print(f"  • Device:         {device}")
    print("=" * 90)

    # Load sequence manifests
    meta_dir = Path("data/metadata")
    train_seq_df = pd.read_csv(meta_dir / "forecast_train_sequences_k5.csv")
    val_seq_df = pd.read_csv(meta_dir / "forecast_val_sequences_k5.csv")
    test_seq_df = pd.read_csv(meta_dir / "forecast_test_sequences_k5.csv")

    with open(meta_dir / "normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)
    mean = [norm_stats["mean"][c] for c in channels]
    std = [norm_stats["std"][c] for c in channels]

    train_ds = TCIRSequenceDataset(train_seq_df, mean=mean, std=std, channels=channels, is_training=True)
    val_ds = TCIRSequenceDataset(val_seq_df, mean=mean, std=std, channels=channels, is_training=False)
    test_ds = TCIRSequenceDataset(test_seq_df, mean=mean, std=std, channels=channels, is_training=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
    )

    # Instantiate model
    in_channels = len(channels)
    if model_type.lower() == "gru":
        model = TemporalGRUForecaster(
            in_channels=in_channels, d_model=256, num_layers=2, dropout=0.1, pretrained_cnn=True
        )
    elif model_type.lower() == "transformer":
        model = TemporalTransformerForecaster(
            in_channels=in_channels,
            d_model=256,
            nhead=8,
            num_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            pretrained_cnn=True,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)
    loss_fn = MultiHorizonHuberLoss(delta=1.0)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_mae = float("inf")
    best_epoch = -1
    patience = 6
    epochs_no_improve = 0
    log_rows = []

    for epoch in range(1, epochs + 1):
        train_loss, epoch_time, peak_vram = train_one_epoch(
            model, train_loader, optimizer, scaler, loss_fn, device, epoch, epochs, k_frames=k_frames
        )
        val_loss, val_metrics, _, _ = evaluate_model(
            model, val_loader, device, loss_fn, k_frames=k_frames
        )
        scheduler.step()

        val_mean_mae = val_metrics["mean_mae"]
        print(
            f"--> [Epoch {epoch:2d}/{epochs:2d}] Summary: Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Mean MAE: {val_mean_mae:.2f} kt "
            f"(+6h: {val_metrics['mae_+6h']:.2f}, +12h: {val_metrics['mae_+12h']:.2f}, +24h: {val_metrics['mae_+24h']:.2f}) | "
            f"Val Mean R²: {val_metrics['mean_r2']:.3f} | Time: {epoch_time:.1f}s | VRAM: {peak_vram:.0f} MB"
        )

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_mae": val_mean_mae,
            "val_mae_6h": val_metrics["mae_+6h"],
            "val_mae_12h": val_metrics["mae_+12h"],
            "val_mae_24h": val_metrics["mae_+24h"],
            "val_mean_r2": val_metrics["mean_r2"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": round(epoch_time, 2),
            "peak_vram_mb": round(peak_vram, 1),
        }
        log_rows.append(log_row)

        if val_mean_mae < best_val_mae:
            best_val_mae = val_mean_mae
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mae": best_val_mae,
                    "val_metrics": val_metrics,
                },
                save_dir / "best.pt",
            )
            print(f"[Checkpoint] New best model saved (Epoch {epoch}, val_mean_mae: {val_mean_mae:.3f} kt) -> best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Early Stopping] No improvement in validation MAE for {patience} epochs. Stopping.")
                break

        # Inter-epoch cooling buffer (15 seconds) for thermal relief
        if epoch < epochs:
            print("  [Thermal Relief] Cooling pause for 15 seconds...")
            time.sleep(15)

    # -------------------------------------------------------------
    # LEGACY SCRIPT SAFEGUARD: Test set evaluation is locked
    # -------------------------------------------------------------
    if not eval_test_confirmed:
        print(f"\n[TEST LOCK PROTECTED] Training complete. Test set evaluation is locked.")
        print("  To evaluate test set, use canonical runner: python evaluate.py --split test --eval-test --confirm-locked-test-eval")
        train_ds.close()
        val_ds.close()
        if 'test_ds' in locals():
            test_ds.close()
        gc.collect()
        return {"status": "SUCCESS_TRAIN_ONLY_TEST_LOCKED", "best_epoch": best_epoch, "best_val_mae": best_val_mae}

    # Load best model for test evaluation
    print(f"\n[{save_dir_name}] Evaluating best checkpoint (Epoch {best_epoch}) on 8,279 test sequences...")
    best_ckpt = torch.load(save_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_metrics, test_preds, test_targets = evaluate_model(
        model, test_loader, device, loss_fn, k_frames=k_frames
    )
    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_val_mae"] = best_val_mae
    test_metrics["model_type"] = model_type
    test_metrics["k_frames"] = k_frames

    print("\n" + "=" * 80)
    print(f"[{save_dir_name.upper()}] TEST SET PERFORMANCE:")
    print(f"  • +6h Forecast MAE:   {test_metrics['mae_+6h']:5.3f} kt | RMSE: {test_metrics['rmse_+6h']:5.3f} kt | R²: {test_metrics['r2_+6h']:6.3f} | Bias: {test_metrics['bias_+6h']:+5.2f} kt")
    print(f"  • +12h Forecast MAE:  {test_metrics['mae_+12h']:5.3f} kt | RMSE: {test_metrics['rmse_+12h']:5.3f} kt | R²: {test_metrics['r2_+12h']:6.3f} | Bias: {test_metrics['bias_+12h']:+5.2f} kt")
    print(f"  • +24h Forecast MAE:  {test_metrics['mae_+24h']:5.3f} kt | RMSE: {test_metrics['rmse_+24h']:5.3f} kt | R²: {test_metrics['r2_+24h']:6.3f} | Bias: {test_metrics['bias_+24h']:+5.2f} kt")
    print(f"  • Aggregate Mean MAE: {test_metrics['mean_mae']:5.3f} kt | Aggregate R²: {test_metrics['mean_r2']:6.3f}")
    print("=" * 80)

    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Save detailed prediction CSV
    pred_df = pd.DataFrame({
        "cyclone_id": test_seq_df["cyclone_id"],
        "target_t_timestamp": test_seq_df["target_t_timestamp"],
        "vmax_curr": test_seq_df["vmax_curr"],
        "actual_plus_6h": test_targets[:, 0],
        "actual_plus_12h": test_targets[:, 1],
        "actual_plus_24h": test_targets[:, 2],
        "pred_plus_6h": test_preds[:, 0],
        "pred_plus_12h": test_preds[:, 1],
        "pred_plus_24h": test_preds[:, 2],
    })
    pred_df.to_csv(test_pred_path, index=False)
    print(f"[Saved Test Predictions] -> {test_pred_path}")

    train_ds.close()
    val_ds.close()
    test_ds.close()
    gc.collect()

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train temporal forecasting models for TC intensity.")
    parser.add_argument("--model", type=str, default="all", choices=["gru", "transformer", "ablation", "all"])
    args = parser.parse_args()

    experiments = []
    if args.model in ["gru", "all"]:
        experiments.append(("gru", 5, "cnn_gru_k5"))
    if args.model in ["transformer", "all"]:
        experiments.append(("transformer", 5, "cnn_transformer_k5"))
    if args.model in ["ablation", "all"]:
        # Temporal context ablations: 1 frame, 3 frames
        experiments.append(("transformer", 1, "cnn_transformer_k1"))
        experiments.append(("transformer", 3, "cnn_transformer_k3"))

    for m_type, k, s_name in experiments:
        run_forecasting_experiment(
            model_type=m_type,
            k_frames=k,
            save_dir_name=s_name,
            epochs=20,
            lr=1e-4,
            batch_size=16,
            channels=[0, 1, 2]
        )


if __name__ == "__main__":
    main()
