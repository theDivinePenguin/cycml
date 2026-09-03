"""PyTorch Dataset for Temporal Tropical Cyclone Satellite Sequences with Explicit Validity Gating."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TCIRSequenceDataset(Dataset):
    """PyTorch Dataset yielding historical sequences of multi-channel satellite imagery
    along with explicit validity masks and multi-horizon future intensity targets (+6h, +12h, +24h).
    """

    def __init__(
        self,
        seq_df: pd.DataFrame,
        mean: List[float],
        std: List[float],
        channels: List[int] = [0, 1, 2],
        is_training: bool = False,
    ):
        """
        Args:
            seq_df: DataFrame containing sequence metadata (built by build_forecast_sequences.py).
            mean: Channel means for normalization (length >= len(channels)).
            std: Channel standard deviations for normalization (length >= len(channels)).
            channels: List of channel indices to load (default [0, 1, 2] for IR1, WV, VIS).
            is_training: Whether dataset is used for training (enables random horizontal/vertical flips).
        """
        self.seq_df = seq_df.reset_index(drop=True)
        self.channels = channels
        self.is_training = is_training

        # Slice normalization stats to match active channels
        if len(mean) == 4 and len(channels) != 4:
            self.mean = np.array([mean[c] for c in channels], dtype=np.float32).reshape(-1, 1, 1)
            self.std = np.array([std[c] for c in channels], dtype=np.float32).reshape(-1, 1, 1)
        else:
            self.mean = np.array(mean[:len(channels)], dtype=np.float32).reshape(-1, 1, 1)
            self.std = np.array(std[:len(channels)], dtype=np.float32).reshape(-1, 1, 1)

        # Persistent HDF5 file handles cache (opened lazily per process/worker)
        self._h5_handles: Dict[str, h5py.File] = {}

    def _get_h5(self, path_str: str) -> h5py.File:
        if path_str not in self._h5_handles:
            self._h5_handles[path_str] = h5py.File(path_str, "r", swmr=True)
        return self._h5_handles[path_str]

    def __len__(self) -> int:
        return len(self.seq_df)

    def _preprocess_frame(self, raw_frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Preprocess a single (H, W, C_all) frame into (C_selected, H, W) normalized tensor
        and compute explicit VIS validity flag.
        """
        # Slice active channels
        # raw_frame shape: (201, 201, C_all) -> (H, W, C_sub)
        frame_sub = raw_frame[:, :, self.channels].astype(np.float32)

        # Check VIS validity if Channel 2 is present in active channels
        vis_valid_flag = 1.0
        if 2 in self.channels:
            vis_ch_idx = self.channels.index(2)
            vis_slice = frame_sub[:, :, vis_ch_idx]
            # Missing or night: NaNs, values < 0, or > 1e20
            invalid_mask = np.isnan(vis_slice) | (vis_slice < 0.0) | (vis_slice > 1e20)
            vis_slice[invalid_mask] = 0.0
            # Day threshold: more than 10% of pixels have positive solar reflectance > 0.01
            day_fraction = np.mean(vis_slice > 0.01)
            vis_valid_flag = 1.0 if day_fraction > 0.10 else 0.0
            frame_sub[:, :, vis_ch_idx] = vis_slice

        # Clean NaNs and infs on all channels
        nan_mask = np.isnan(frame_sub) | np.isinf(frame_sub) | (frame_sub > 1e20) | (frame_sub < -1e20)
        frame_sub[nan_mask] = 0.0

        # Transpose to (C, H, W)
        tensor = np.transpose(frame_sub, (2, 0, 1))

        # Standardize: (X - mean) / std
        tensor = (tensor - self.mean) / (self.std + 1e-7)
        return tensor, vis_valid_flag

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        row = self.seq_df.iloc[idx]

        h5_files = json.loads(row["history_h5_files"])
        h5_rows = json.loads(row["history_h5_rows"])
        k_history = len(h5_files)

        seq_frames = []
        vis_masks = []

        for h5_file, h5_row in zip(h5_files, h5_rows):
            h5 = self._get_h5(h5_file)
            raw = h5["matrix"][h5_row]  # shape: (201, 201, 4)
            proc_tensor, vis_flag = self._preprocess_frame(raw)
            seq_frames.append(proc_tensor)
            vis_masks.append(vis_flag)

        # Stack into (K, C, H, W)
        seq_tensor = np.stack(seq_frames, axis=0)  # (K, C, H, W)
        vis_mask_tensor = np.array(vis_masks, dtype=np.float32)  # (K,)

        # Random horizontal/vertical flip augmentation for training
        if self.is_training:
            if np.random.rand() > 0.5:
                seq_tensor = np.flip(seq_tensor, axis=3).copy()  # Horizontal flip across all K frames
            if np.random.rand() > 0.5:
                seq_tensor = np.flip(seq_tensor, axis=2).copy()  # Vertical flip across all K frames

        # Targets: [+6h, +12h, +24h]
        targets = np.array([
            row["vmax_plus_6h"],
            row["vmax_plus_12h"],
            row["vmax_plus_24h"]
        ], dtype=np.float32)

        meta = {
            "cyclone_id": row["cyclone_id"],
            "target_t_timestamp": row["target_t_timestamp"],
            "vmax_curr": float(row["vmax_curr"]),
            "vmax_plus_6h": float(row["vmax_plus_6h"]),
            "vmax_plus_12h": float(row["vmax_plus_12h"]),
            "vmax_plus_24h": float(row["vmax_plus_24h"]),
        }

        return (
            torch.from_numpy(seq_tensor).float(),
            torch.from_numpy(vis_mask_tensor).float(),
            torch.from_numpy(targets).float(),
            meta
        )

    def close(self):
        for h5 in self._h5_handles.values():
            try:
                h5.close()
            except Exception:
                pass
        self._h5_handles.clear()


def build_sequence_dataloaders(
    train_seq_df: pd.DataFrame,
    val_seq_df: pd.DataFrame,
    test_seq_df: pd.DataFrame,
    mean: List[float],
    std: List[float],
    channels: List[int] = [0, 1, 2],
    batch_size: int = 16,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory creating train, val, and test DataLoaders for forecasting sequences."""
    train_ds = TCIRSequenceDataset(train_seq_df, mean=mean, std=std, channels=channels, is_training=True)
    val_ds = TCIRSequenceDataset(val_seq_df, mean=mean, std=std, channels=channels, is_training=False)
    test_ds = TCIRSequenceDataset(test_seq_df, mean=mean, std=std, channels=channels, is_training=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
