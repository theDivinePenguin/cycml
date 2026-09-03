"""Checkpointing and CSV logging utilities."""
import csv
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim


class CheckpointManager:
    """Manages model checkpoints (best.pt, last.pt) and training recovery."""

    def __init__(self, save_dir: str | Path, monitor: str = "val_mae", mode: str = "min"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0

    def is_better(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_metric
        return current > self.best_metric

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> bool:
        current_metric = metrics.get(self.monitor, float("inf") if self.mode == "min" else float("-inf"))
        is_best = self.is_better(current_metric)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config,
            "best_metric": self.best_metric
        }

        # Save last.pt
        last_path = self.save_dir / "last.pt"
        torch.save(state, last_path)

        # Save best.pt if improved
        if is_best:
            self.best_metric = current_metric
            self.best_epoch = epoch
            best_path = self.save_dir / "best.pt"
            torch.save(state, best_path)
            print(f"[Checkpoint] New best model saved (Epoch {epoch}, {self.monitor}: {current_metric:.4f}) -> {best_path.name}")

        return is_best

    def load(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {p}")

        checkpoint = torch.load(p, map_location=device or torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"[Checkpoint] Loaded weights from {p} (Epoch {checkpoint.get('epoch', 0)})")
        return checkpoint


class CSVLogger:
    """Logs training metrics to a CSV file."""

    def __init__(self, log_path: str | Path, fieldnames: list[str]):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames

        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
