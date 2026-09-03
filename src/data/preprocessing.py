"""Image preprocessing, normalization, and physical data augmentation."""
import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class TCIRPreprocessor:
    """Preprocessor for TCIR single-channel (IR1) and multi-channel satellite tensors."""

    def __init__(
        self,
        mean: float | list[float] | torch.Tensor = 0.0,
        std: float | list[float] | torch.Tensor = 1.0,
        target_size: Tuple[int, int] = (224, 224),
        is_training: bool = False,
        augmentation_cfg: Optional[dict] = None,
        channels: Optional[list[int]] = None
    ):
        self.channels = channels if channels is not None else [0]

        # Slice mean and std if full channel list provided
        if isinstance(mean, (list, tuple, np.ndarray, torch.Tensor)):
            if len(mean) >= max(self.channels) + 1 and len(mean) != len(self.channels):
                mean = [mean[c] for c in self.channels]
            m = torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        else:
            m = torch.tensor(float(mean), dtype=torch.float32)

        if isinstance(std, (list, tuple, np.ndarray, torch.Tensor)):
            if len(std) >= max(self.channels) + 1 and len(std) != len(self.channels):
                std = [std[c] for c in self.channels]
            s = torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1)
            s = torch.clamp(s, min=1e-6)
        else:
            s = torch.tensor(float(std) if float(std) > 1e-6 else 1.0, dtype=torch.float32)

        self.mean = m
        self.std = s
        self.target_size = target_size
        self.is_training = is_training
        self.aug_cfg = augmentation_cfg or {}

        # Base resizing transform (bilinear interpolation for continuous physical quantities)
        self.resize = T.Resize(target_size, antialias=True)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing pipeline to image tensor.

        Args:
            tensor: Shape (C, H, W) float tensor.

        Returns:
            Preprocessed, normalized, resized tensor of shape (C, target_size[0], target_size[1]).
        """
        c_count = tensor.shape[0]

        # Channel-aware physical missing-value and fill-value cleaning
        cleaned_channels = []
        for i in range(c_count):
            ch = tensor[i]
            ch_id = self.channels[i] if i < len(self.channels) else i

            if ch_id == 3:
                # PMW: Replace NetCDF fill values (>1e20, <-100) and NaNs with 0.0 (baseline zero precipitation)
                invalid_mask = torch.isnan(ch) | (ch > 1e20) | (ch < -100.0)
                ch = torch.where(invalid_mask, torch.zeros_like(ch), ch)
            elif ch_id == 2:
                # VIS: Replace nighttime NaNs with 0.0 (zero solar reflectance at night)
                invalid_mask = torch.isnan(ch) | (ch < 0.0) | (ch > 1e20)
                ch = torch.where(invalid_mask, torch.zeros_like(ch), ch)
            else:
                # IR1 (0) or WV (1): Replace isolated missing pixels with valid channel spatial mean
                invalid_mask = torch.isnan(ch) | (ch > 1e20) | (ch < 0.0)
                if invalid_mask.any():
                    valid_mask = ~invalid_mask
                    fill_val = ch[valid_mask].mean() if valid_mask.any() else (torch.tensor(267.8) if ch_id == 0 else torch.tensor(236.1))
                    ch = torch.where(invalid_mask, fill_val.to(ch.dtype), ch)

            cleaned_channels.append(ch)

        tensor = torch.stack(cleaned_channels, dim=0)

        # Resize to target resolution (224x224)
        tensor = self.resize(tensor)

        # Apply conservative physical augmentation during training only across all channels jointly
        if self.is_training and self.aug_cfg.get("enabled", False):
            # Mild random rotation (e.g., +/- 15 degrees)
            max_deg = self.aug_cfg.get("rotation_degrees", 15)
            if max_deg > 0:
                angle = float(torch.empty(1).uniform_(-max_deg, max_deg).item())
                tensor = TF.rotate(tensor, angle=angle)

            # Random horizontal flip
            if self.aug_cfg.get("horizontal_flip", False) and torch.rand(1).item() > 0.5:
                tensor = TF.hflip(tensor)

            # Mild intensity perturbation (+/- 5%) applied to thermal/continuous channels
            jitter = self.aug_cfg.get("intensity_jitter", 0.0)
            if jitter > 0:
                scale = 1.0 + float(torch.empty(1).uniform_(-jitter, jitter).item())
                tensor = tensor * scale

        # Standardize using training-set statistics ONLY
        mean = self.mean.to(tensor.device, tensor.dtype)
        std = self.std.to(tensor.device, tensor.dtype)
        tensor = (tensor - mean) / std

        return tensor


def compute_normalization_stats(
    matrix_dataset,
    train_indices: list[int],
    channel_idx: int = 0,
    batch_size: int = 500,
    save_path: Optional[str | Path] = None
) -> Tuple[float, float]:
    """Compute mean and standard deviation strictly over the TRAINING set samples.

    Args:
        matrix_dataset: HDF5 matrix dataset or array of shape (N, H, W, C).
        train_indices: List of row indices belonging exclusively to the training split.
        channel_idx: Channel index to compute stats for (0 for IR1).
        batch_size: Batch size for chunked streaming computation.
        save_path: Optional path to save stats JSON.

    Returns:
        Tuple of (mean, std).
    """
    print(f"[Preprocessing] Computing training-set normalization stats over {len(train_indices)} train frames (Channel {channel_idx})...")

    # Welford's algorithm / streaming sum and sum of squares
    total_pixels = 0
    sum_val = 0.0
    sum_sq_val = 0.0
    min_val = float("inf")
    max_val = float("-inf")

    for i in range(0, len(train_indices), batch_size):
        chunk_idx = train_indices[i:i + batch_size]
        sorted_idx = sorted(chunk_idx)
        batch = matrix_dataset[sorted_idx, :, :, channel_idx]  # Shape (B, H, W)
        valid = batch[~np.isnan(batch)]
        if len(valid) > 0:
            total_pixels += len(valid)
            sum_val += float(np.sum(valid))
            sum_sq_val += float(np.sum(valid ** 2))
            min_val = min(min_val, float(np.min(valid)))
            max_val = max(max_val, float(np.max(valid)))

    if total_pixels == 0:
        raise ValueError("No valid pixels found to compute normalization statistics.")

    mean = sum_val / total_pixels
    variance = (sum_sq_val / total_pixels) - (mean ** 2)
    std = float(np.sqrt(max(variance, 1e-6)))

    print(f"[Preprocessing] Training stats computed (over {total_pixels:,} pixels):")
    print(f"  • Mean: {mean:.4f} | Std: {std:.4f} | Min: {min_val:.2f} | Max: {max_val:.2f}")

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "mean": mean,
                "std": std,
                "min": min_val,
                "max": max_val,
                "channel_idx": channel_idx,
                "n_train_samples": len(train_indices)
            }, f, indent=2)
        print(f"[Preprocessing] Saved normalization statistics to: {p}")

    return mean, std
