"""Isolated Dataset Module for RI Target Loss and Delta Experiments.

Loads canonical K=7 sequences and provides both absolute and delta intensity targets:
- Absolute targets: [V(t+6h), V(t+12h), V(t+24h)]
- Delta targets:    [V(t+6h)-V(t), V(t+12h)-V(t), V(t+24h)-V(t)]
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.sequence_dataset import TCIRSequenceDataset
from src.data.trend_config import IntensityTrendConfig


class DeltaSequenceDataset(TCIRSequenceDataset):
    """Yields K=7 satellite sequences, explicit VIS validity masks,
    along with trend class, RI binary target, absolute targets, and delta targets.
    """

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

        v_curr = self.seq_df["vmax_curr"].values
        v_6 = self.seq_df["vmax_plus_6h"].values
        v_12 = self.seq_df["vmax_plus_12h"].values
        v_24 = self.seq_df["vmax_plus_24h"].values

        delta_v_6 = v_6 - v_curr
        delta_v_12 = v_12 - v_curr
        delta_v_24 = v_24 - v_curr

        self.deltas = np.stack([delta_v_6, delta_v_12, delta_v_24], axis=1).astype(np.float32)
        self.precomputed_trend = self.config.compute_trend_label(delta_v_24)
        self.precomputed_ri = self.config.compute_ri_label(delta_v_24)
        self.delta_v_24 = delta_v_24

    def __getitem__(self, idx: int):
        seq_tensor, vis_mask, reg_targets_abs, meta = super().__getitem__(idx)

        trend_target = torch.tensor(self.precomputed_trend[idx], dtype=torch.long)
        ri_target = torch.tensor(self.precomputed_ri[idx], dtype=torch.float32)
        reg_targets_delta = torch.from_numpy(self.deltas[idx])

        meta["delta_v_24"] = float(self.delta_v_24[idx])
        meta["trend_label"] = int(self.precomputed_trend[idx])
        meta["ri_label"] = int(self.precomputed_ri[idx])
        meta["trend_name"] = self.config.get_trend_name(self.precomputed_trend[idx])

        if self.env_tensor is not None:
            env_vec = self.env_tensor[idx]
            return (
                seq_tensor,
                vis_mask,
                trend_target,
                ri_target,
                reg_targets_abs,
                reg_targets_delta,
                env_vec,
                meta,
            )

        return (
            seq_tensor,
            vis_mask,
            trend_target,
            ri_target,
            reg_targets_abs,
            reg_targets_delta,
            meta,
        )


def build_delta_dataloaders(
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
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = DeltaSequenceDataset(
        train_seq_df, mean=mean, std=std, channels=channels, is_training=True, config=config, env_tensor=train_env_tensor
    )
    val_ds = DeltaSequenceDataset(
        val_seq_df, mean=mean, std=std, channels=channels, is_training=False, config=config, env_tensor=val_env_tensor
    )
    test_ds = DeltaSequenceDataset(
        test_seq_df, mean=mean, std=std, channels=channels, is_training=False, config=config, env_tensor=test_env_tensor
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )

    return train_loader, val_loader, test_loader
