"""Isolated Variable-K Dataset and Collation Module.

Loads canonical 7-frame sequences and dynamically slices temporal context:
- K=3: last 3 frames [t-6h, t-3h, t]
- K=5: last 5 frames [t-12h, t-9h, t-6h, t-3h, t]
- K=7: all 7 frames  [t-18h, t-15h, t-12h, t-9h, t-6h, t-3h, t]
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.sequence_dataset import TCIRSequenceDataset
from src.data.trend_config import IntensityTrendConfig


class VariableKDataset(TCIRSequenceDataset):
    """Sequence dataset yielding full 7-frame sequences along with multi-task targets."""

    def __init__(
        self,
        seq_df: pd.DataFrame,
        mean: List[float],
        std: List[float],
        channels: List[int] = [0, 1, 2],
        is_training: bool = False,
        config: Optional[IntensityTrendConfig] = None,
        env_tensor: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            seq_df=seq_df,
            mean=mean,
            std=std,
            channels=channels,
            is_training=is_training,
        )
        self.config = config or IntensityTrendConfig()
        self.env_tensor = env_tensor

        delta_v_24 = self.seq_df["vmax_plus_24h"].values - self.seq_df["vmax_curr"].values
        self.precomputed_trend = self.config.compute_trend_label(delta_v_24)
        self.precomputed_ri = self.config.compute_ri_label(delta_v_24)
        self.delta_v_24 = delta_v_24

    def __getitem__(self, idx: int):
        seq_tensor, vis_mask, reg_targets, meta = super().__getitem__(idx)
        trend_target = torch.tensor(self.precomputed_trend[idx], dtype=torch.long)
        ri_target = torch.tensor(self.precomputed_ri[idx], dtype=torch.float32)

        meta["delta_v_24"] = float(self.delta_v_24[idx])
        meta["trend_label"] = int(self.precomputed_trend[idx])
        meta["ri_label"] = int(self.precomputed_ri[idx])
        meta["trend_name"] = self.config.get_trend_name(self.precomputed_trend[idx])

        if self.env_tensor is not None:
            env_vec = self.env_tensor[idx]
            return seq_tensor, vis_mask, trend_target, ri_target, reg_targets, env_vec, meta
        return seq_tensor, vis_mask, trend_target, ri_target, reg_targets, meta


class VariableKCollator:
    """Collate function that dynamically or deterministically slices the last K frames."""

    def __init__(
        self,
        mode: str = "variable",  # 'variable' or fixed int: 3, 5, 7
        seed: int = 42,
    ):
        self.mode = mode
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.epoch = 0
        self.batch_counter = 0

        # Sample tracking counters
        self.counts = {3: 0, 5: 0, 7: 0}

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.batch_counter = 0
        self.rng = np.random.RandomState(self.seed + epoch * 10000)

    def reset_counts(self):
        self.counts = {3: 0, 5: 0, 7: 0}

    def __call__(self, batch: List[Tuple]) -> Tuple:
        batch_size = len(batch)
        if self.mode == "variable":
            # Sample K from {3, 5, 7} with equal probability (1/3 each)
            k = int(self.rng.choice([3, 5, 7]))
        elif isinstance(self.mode, int):
            k = self.mode
        elif isinstance(self.mode, str) and self.mode.isdigit():
            k = int(self.mode)
        else:
            raise ValueError(f"Unknown collator mode: {self.mode}")

        self.counts[k] += batch_size
        self.batch_counter += 1

        # Slicing: take the last K frames [ -k: ]
        # In the 7-frame sequence, the last index is always t.
        # K=3 -> indices [-3:] = [t-6h, t-3h, t]
        # K=5 -> indices [-5:] = [t-12h, t-9h, t-6h, t-3h, t]
        # K=7 -> indices [-7:] = all 7 frames
        sliced_images = torch.stack([item[0][-k:].contiguous() for item in batch], dim=0)
        sliced_vis = torch.stack([item[1][-k:].contiguous() for item in batch], dim=0)
        trend_targets = torch.stack([item[2] for item in batch], dim=0)
        ri_targets = torch.stack([item[3] for item in batch], dim=0)
        reg_targets = torch.stack([item[4] for item in batch], dim=0)

        has_env = len(batch[0]) == 7
        if has_env:
            env_vecs = torch.stack([item[5] for item in batch], dim=0)
            metas = [item[6] for item in batch]
            # Collate metadata fields into lists or tensors
            collated_meta = {
                "cyclone_id": [m["cyclone_id"] for m in metas],
                "target_t_timestamp": [m["target_t_timestamp"] for m in metas],
                "vmax_curr": torch.tensor([m["vmax_curr"] for m in metas], dtype=torch.float32),
                "vmax_plus_6h": torch.tensor([m["vmax_plus_6h"] for m in metas], dtype=torch.float32),
                "vmax_plus_12h": torch.tensor([m["vmax_plus_12h"] for m in metas], dtype=torch.float32),
                "vmax_plus_24h": torch.tensor([m["vmax_plus_24h"] for m in metas], dtype=torch.float32),
                "delta_v_24": torch.tensor([m["delta_v_24"] for m in metas], dtype=torch.float32),
                "eval_k": k,
            }
            return sliced_images, sliced_vis, trend_targets, ri_targets, reg_targets, env_vecs, collated_meta

        metas = [item[5] for item in batch]
        collated_meta = {
            "cyclone_id": [m["cyclone_id"] for m in metas],
            "target_t_timestamp": [m["target_t_timestamp"] for m in metas],
            "vmax_curr": torch.tensor([m["vmax_curr"] for m in metas], dtype=torch.float32),
            "vmax_plus_6h": torch.tensor([m["vmax_plus_6h"] for m in metas], dtype=torch.float32),
            "vmax_plus_12h": torch.tensor([m["vmax_plus_12h"] for m in metas], dtype=torch.float32),
            "vmax_plus_24h": torch.tensor([m["vmax_plus_24h"] for m in metas], dtype=torch.float32),
            "delta_v_24": torch.tensor([m["delta_v_24"] for m in metas], dtype=torch.float32),
            "eval_k": k,
        }
        return sliced_images, sliced_vis, trend_targets, ri_targets, reg_targets, collated_meta


def build_variable_k_dataloaders(
    train_seq_df: pd.DataFrame,
    val_seq_df: pd.DataFrame,
    test_seq_df: pd.DataFrame,
    mean: List[float],
    std: List[float],
    channels: List[int] = [0, 1, 2],
    batch_size: int = 16,
    num_workers: int = 4,
    config: Optional[IntensityTrendConfig] = None,
    train_env_tensor: Optional[torch.Tensor] = None,
    val_env_tensor: Optional[torch.Tensor] = None,
    test_env_tensor: Optional[torch.Tensor] = None,
    train_mode: str = "variable",
    seed: int = 42,
) -> Tuple[DataLoader, Dict[int, DataLoader], Dict[int, DataLoader], VariableKCollator]:
    """Builds DataLoaders for variable-K training and deterministic evaluation."""
    train_ds = VariableKDataset(
        train_seq_df, mean=mean, std=std, channels=channels, is_training=True, config=config, env_tensor=train_env_tensor
    )
    val_ds = VariableKDataset(
        val_seq_df, mean=mean, std=std, channels=channels, is_training=False, config=config, env_tensor=val_env_tensor
    )
    test_ds = VariableKDataset(
        test_seq_df, mean=mean, std=std, channels=channels, is_training=False, config=config, env_tensor=test_env_tensor
    )

    train_collator = VariableKCollator(mode=train_mode, seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_collator,
    )

    # Validation loaders for K=3, 5, 7
    val_loaders = {}
    for k in [3, 5, 7]:
        val_loaders[k] = DataLoader(
            val_ds,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=VariableKCollator(mode=k, seed=seed),
        )

    # Test loaders for K=3, 5, 7
    test_loaders = {}
    for k in [3, 5, 7]:
        test_loaders[k] = DataLoader(
            test_ds,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=VariableKCollator(mode=k, seed=seed),
        )

    return train_loader, val_loaders, test_loaders, train_collator
