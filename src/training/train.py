"""Training engine with mixed precision, early stopping, and metrics tracking."""
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.evaluation.metrics import calculate_metrics
from src.training.checkpoint import CheckpointManager, CSVLogger


class Trainer:
    """Encapsulates training, validation, mixed precision, and checkpoint management."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        config: Dict[str, Any],
        checkpoint_manager: CheckpointManager
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.ckpt_manager = checkpoint_manager

        self.use_amp = config.get("training", {}).get("use_amp", True) and (device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.early_stopping_patience = config.get("training", {}).get("early_stopping_patience", 8)
        self.log_interval = config.get("logging", {}).get("log_interval", 20)

        # Setup CSV logger
        log_file = self.ckpt_manager.save_dir / "training_log.csv"
        fields = [
            "epoch", "train_loss", "val_loss", "val_mae", "val_rmse", "val_r2",
            "val_median_ae", "val_bias", "lr", "epoch_time_sec", "peak_vram_mb"
        ]
        self.logger = CSVLogger(log_file, fields)

    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Train for one epoch.

        Returns:
            Tuple of (train_loss, epoch_duration_sec, peak_vram_mb).
        """
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)
        start_time = time.time()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for batch_idx, (images, targets, _) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = loss.item()
            total_loss += loss_val

            if (batch_idx + 1) % self.log_interval == 0 or (batch_idx + 1) == n_batches:
                curr_lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Epoch [{epoch:2d}/{self.config.get('training', {}).get('epochs', 30):2d}] "
                      f"Batch [{batch_idx + 1:3d}/{n_batches:3d}] - "
                      f"Loss: {loss_val:.4f} (Avg: {total_loss / (batch_idx + 1):.4f}) - "
                      f"LR: {curr_lr:.2e}")

        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(n_batches, 1)

        peak_vram = 0.0
        if self.device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        return avg_loss, epoch_time, peak_vram

    def validate(self, loader: Optional[DataLoader] = None) -> Tuple[float, Dict[str, float]]:
        """Run deterministic validation.

        Returns:
            Tuple of (val_loss, metrics_dict).
        """
        val_dl = loader or self.val_loader
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_gts = []

        with torch.no_grad():
            for images, targets, _ in val_dl:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, targets)

                total_loss += loss.item() * len(targets)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_gts.extend(targets.cpu().numpy().flatten())

        n_samples = max(len(all_gts), 1)
        avg_loss = total_loss / n_samples
        metrics = calculate_metrics(np.array(all_preds), np.array(all_gts))
        metrics["val_loss"] = avg_loss

        return avg_loss, metrics

    def fit(self) -> Dict[str, Any]:
        """Execute full training loop with early stopping."""
        epochs = self.config.get("training", {}).get("epochs", 30)
        patience = self.early_stopping_patience
        epochs_no_improve = 0

        print(f"\n[Trainer] Starting training for {epochs} epochs on {self.device} (AMP: {self.use_amp})...")
        total_train_start = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, epoch_time, peak_vram = self.train_epoch(epoch)
            val_loss, metrics = self.validate()

            curr_lr = self.optimizer.param_groups[0]["lr"]

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics["mae"])
                else:
                    self.scheduler.step()

            # Checkpoint & Best tracking
            metrics["train_loss"] = train_loss
            metrics["val_mae"] = metrics["mae"]
            is_best = self.ckpt_manager.save(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics=metrics,
                config=self.config
            )

            # Log row
            log_row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_mae": round(metrics["mae"], 4),
                "val_rmse": round(metrics["rmse"], 4),
                "val_r2": round(metrics["r2"], 4),
                "val_median_ae": round(metrics["median_ae"], 4),
                "val_bias": round(metrics["mean_bias"], 4),
                "lr": f"{curr_lr:.6e}",
                "epoch_time_sec": round(epoch_time, 2),
                "peak_vram_mb": round(peak_vram, 1)
            }
            self.logger.log(log_row)

            print(f"--> [Epoch {epoch:2d}/{epochs:2d}] Summary: "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {metrics['mae']:.2f} kt | "
                  f"Val RMSE: {metrics['rmse']:.2f} kt | "
                  f"Val R²: {metrics['r2']:.3f} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Peak VRAM: {peak_vram:.0f} MB")

            if is_best:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\n[Early Stopping] No improvement in validation MAE for {patience} epochs. Stopping training.")
                    break

        total_duration = time.time() - total_train_start
        print(f"\n[Trainer] Training completed in {total_duration / 60:.2f} minutes.")
        print(f"[Trainer] Best validation MAE: {self.ckpt_manager.best_metric:.2f} kt at Epoch {self.ckpt_manager.best_epoch}.")

        return {
            "total_duration_sec": total_duration,
            "best_epoch": self.ckpt_manager.best_epoch,
            "best_val_mae": self.ckpt_manager.best_metric
        }
