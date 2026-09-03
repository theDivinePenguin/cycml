"""PyTorch Dataset and DataLoader builders for TCIR satellite imagery."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import TCIRPreprocessor, compute_normalization_stats
from src.utils.seed import seed_worker


CHANNEL_NAME_TO_IDX = {
    "IR1": 0,
    "WV": 1,
    "VIS": 2,
    "PMW": 3,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3
}


class TCIRDataset(Dataset):
    """PyTorch Dataset for reading TCIR single-channel or multi-channel satellite imagery."""

    def __init__(
        self,
        h5_path: str | Path | None,
        metadata_df: pd.DataFrame,
        channels: int | List[int] | List[str] | None = None,
        channel_idx: Optional[int] = None,
        preprocessor: Optional[TCIRPreprocessor] = None,
        in_memory: bool = False
    ):
        """
        Args:
            h5_path: Default HDF5 file path or None if specified per row in metadata_df['h5_file'].
            metadata_df: DataFrame with 'sample_index', 'cyclone_id', 'wind_speed', and optional 'h5_file', 'h5_row_index'.
            channels: Channel indices or names (e.g. [0] or [0, 1, 2, 3] or ["IR1", "WV"]).
            channel_idx: Deprecated single-channel index for backward compatibility.
            preprocessor: TCIRPreprocessor instance for resizing, augmentation, and normalization.
            in_memory: If True, preload required slice into RAM for fast access.
        """
        self.default_h5_path = str(Path(h5_path).resolve()) if h5_path else None
        self.metadata = metadata_df.reset_index(drop=True)

        # Resolve channels list
        if channels is not None:
            if isinstance(channels, (int, str)):
                ch_list = [channels]
            else:
                ch_list = list(channels)
        elif channel_idx is not None:
            ch_list = [channel_idx]
        else:
            ch_list = [0]

        # Standardize to list of integer indices [0..3]
        self.channels = [
            CHANNEL_NAME_TO_IDX[str(c).upper()] if str(c).upper() in CHANNEL_NAME_TO_IDX else int(c)
            for c in ch_list
        ]
        self.channel_idx = self.channels[0]  # Backward compatibility attribute

        self.preprocessor = preprocessor or TCIRPreprocessor(channels=self.channels)
        self.in_memory = in_memory

        self._h5_handles: Dict[str, h5py.File] = {}
        self.cached_images: Optional[np.ndarray] = None

        if self.in_memory:
            self._preload_images()

    def _preload_images(self) -> None:
        """Preload images for all rows in metadata into memory across single or multiple HDF5 files."""
        n_samples = len(self.metadata)
        h, w = 201, 201
        num_ch = len(self.channels)
        cached = np.empty((n_samples, num_ch, h, w), dtype=np.float32)

        # Group by h5_file to read in contiguous batches
        if "h5_file" in self.metadata.columns:
            grouped = self.metadata.groupby("h5_file")
        else:
            grouped = [(self.default_h5_path, self.metadata)]

        for h5_file_p, group_df in grouped:
            with h5py.File(h5_file_p, "r") as hf:
                matrix_ds = hf["matrix"]
                for orig_idx, row in group_df.iterrows():
                    row_idx = int(row.get("h5_row_index", row["sample_index"]))
                    # Extract channels: shape (H, W, C) -> (C, H, W)
                    raw_data = matrix_ds[row_idx, :, :, self.channels]
                    if len(self.channels) == 1 and raw_data.ndim == 2:
                        cached[orig_idx, 0] = raw_data
                    elif raw_data.ndim == 3:
                        cached[orig_idx] = np.transpose(raw_data, (2, 0, 1))
                    else:
                        cached[orig_idx] = raw_data

        self.cached_images = cached

    def _get_matrix(self, h5_file_path: Optional[str] = None):
        """Lazy worker-safe initialization of HDF5 file handles."""
        target_path = h5_file_path or self.default_h5_path
        if not target_path:
            raise ValueError("No HDF5 file path specified for sample.")

        if target_path not in self._h5_handles:
            self._h5_handles[target_path] = h5py.File(target_path, "r")
        return self._h5_handles[target_path]["matrix"]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.metadata.iloc[index]
        sample_idx = int(row.get("sample_index", index))

        if self.cached_images is not None:
            tensor = torch.from_numpy(self.cached_images[index].copy())  # Shape (C, H, W)
        else:
            h5_file = str(row["h5_file"]) if "h5_file" in row and pd.notna(row["h5_file"]) else self.default_h5_path
            row_idx = int(row.get("h5_row_index", row.get("sample_index", index)))
            matrix = self._get_matrix(h5_file)
            
            # Read channels
            if len(self.channels) == 1:
                img_np = matrix[row_idx, :, :, self.channels[0]]  # (H, W)
                tensor = torch.from_numpy(np.array(img_np, dtype=np.float32)).unsqueeze(0)  # (1, H, W)
            else:
                img_np = matrix[row_idx, :, :, self.channels]  # (H, W, C)
                tensor = torch.from_numpy(np.array(img_np, dtype=np.float32)).permute(2, 0, 1)  # (C, H, W)

        # Apply preprocessing & normalization
        tensor = self.preprocessor(tensor)

        # Target wind speed in knots as scalar tensor
        wind_speed = float(row["wind_speed"])
        target = torch.tensor([wind_speed], dtype=torch.float32)

        meta = {
            "sample_index": sample_idx,
            "cyclone_id": str(row.get("cyclone_id", "")),
            "timestamp": str(row.get("timestamp", "")),
            "wind_speed": wind_speed,
            "region": str(row.get("region", "UNKNOWN"))
        }

        return tensor, target, meta

    def __del__(self):
        for path, hf in self._h5_handles.items():
            try:
                hf.close()
            except Exception:
                pass
        self._h5_handles.clear()


def build_dataloaders(
    h5_path: str | Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    mean: float | list[float],
    std: float | list[float],
    in_memory: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test PyTorch DataLoaders.

    Args:
        h5_path: Path to raw HDF5 file.
        train_df: Training split metadata DataFrame.
        val_df: Validation split metadata DataFrame.
        test_df: Test split metadata DataFrame.
        config: Configuration dictionary.
        mean: Training set mean (scalar or list per channel).
        std: Training set standard deviation (scalar or list per channel).
        in_memory: Whether to cache tensors in RAM.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    ds_cfg = config.get("dataset", {})
    aug_cfg = config.get("augmentation", {})
    batch_size = ds_cfg.get("batch_size", 64)
    num_workers = ds_cfg.get("num_workers", 4)
    pin_memory = ds_cfg.get("pin_memory", True) and torch.cuda.is_available()
    target_size = tuple(ds_cfg.get("input_size", [224, 224]))

    # Resolve channels from config
    raw_channels = ds_cfg.get("channels", [0])
    if isinstance(raw_channels, (int, str)):
        raw_channels = [raw_channels]
    channels = [
        CHANNEL_NAME_TO_IDX[str(c).upper()] if str(c).upper() in CHANNEL_NAME_TO_IDX else int(c)
        for c in raw_channels
    ]

    # Format mean and std for the selected channels
    if isinstance(mean, dict):
        mean_vals = [mean[str(c)]["mean"] if str(c) in mean else mean[c]["mean"] for c in channels]
    elif isinstance(mean, (list, tuple)):
        if len(mean) >= max(channels) + 1 and len(mean) != len(channels):
            mean_vals = [mean[c] for c in channels]
        else:
            mean_vals = list(mean)
    else:
        mean_vals = float(mean)

    if isinstance(std, dict):
        std_vals = [std[str(c)]["std"] if str(c) in std else std[c]["std"] for c in channels]
    elif isinstance(std, (list, tuple)):
        if len(std) >= max(channels) + 1 and len(std) != len(channels):
            std_vals = [std[c] for c in channels]
        else:
            std_vals = list(std)
    else:
        std_vals = float(std)

    # Train preprocessor with augmentation enabled
    train_preprocessor = TCIRPreprocessor(
        mean=mean_vals,
        std=std_vals,
        target_size=target_size,
        is_training=True,
        augmentation_cfg=aug_cfg,
        channels=channels
    )

    # Eval preprocessor: deterministic, NO augmentation
    eval_preprocessor = TCIRPreprocessor(
        mean=mean_vals,
        std=std_vals,
        target_size=target_size,
        is_training=False,
        augmentation_cfg={"enabled": False},
        channels=channels
    )

    train_ds = TCIRDataset(h5_path, train_df, channels=channels, preprocessor=train_preprocessor, in_memory=in_memory)
    val_ds = TCIRDataset(h5_path, val_df, channels=channels, preprocessor=eval_preprocessor, in_memory=in_memory)
    test_ds = TCIRDataset(h5_path, test_df, channels=channels, preprocessor=eval_preprocessor, in_memory=in_memory)

    generator = torch.Generator()
    seed = config.get("training", {}).get("seed", 42)
    generator.manual_seed(seed)

    sampling_cfg = ds_cfg.get("sampling", {})
    sampling_mode = sampling_cfg.get("mode", "natural").lower()
    
    train_sampler = None
    train_shuffle = True

    if sampling_mode == "intensity_aware":
        from src.data.samplers import build_intensity_aware_sampler
        alpha = float(sampling_cfg.get("alpha", 0.5))
        train_sampler, diag = build_intensity_aware_sampler(train_df, alpha=alpha, seed=seed)
        train_shuffle = False  # sampler and shuffle are mutually exclusive
        print(f"[DataLoader] Intensity-aware sampling enabled (alpha={alpha}). Diagnostics:")
        for b_name, b_info in diag.items():
            if b_info["count"] > 0:
                print(f"  • {b_name:<12}: N={b_info['count']:4d} ({b_info['natural_pct']:5.2f}%) -> Eff Prob: {b_info['effective_sampling_pct']:5.2f}% ({b_info['sampling_multiplier']:.2f}x multiplier)")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers if not in_memory else 0,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator if train_sampler is None else None,
        persistent_workers=(num_workers > 0 and not in_memory)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if not in_memory else 0,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0 and not in_memory)
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if not in_memory else 0,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0 and not in_memory)
    )

    return train_loader, val_loader, test_loader
