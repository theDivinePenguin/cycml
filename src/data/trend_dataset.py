"""PyTorch Dataset for Cyclone Intensity Trend and Rapid Intensification Prediction."""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.sequence_dataset import TCIRSequenceDataset
from src.data.trend_config import IntensityTrendConfig


class TCIRTrendDataset(TCIRSequenceDataset):
    """Dataset yielding 5-frame satellite sequences, explicit VIS validity masks,
    along with 24-hour Trend class (0, 1, 2), Rapid Intensification (RI) binary target (0 or 1),
    and auxiliary multi-horizon regression targets [+6h, +12h, +24h].
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

        # Precompute target delta V and labels for fast index access
        delta_v_24 = self.seq_df["vmax_plus_24h"].values - self.seq_df["vmax_curr"].values
        self.precomputed_trend = self.config.compute_trend_label(delta_v_24)
        self.precomputed_ri = self.config.compute_ri_label(delta_v_24)
        self.delta_v_24 = delta_v_24

    def __getitem__(self, idx: int):
        """
        Returns:
            seq_tensor: (K, C, H, W) normalized satellite sequence
            vis_mask: (K,) explicit VIS validity flags
            trend_target: scalar int (0: WEAKENING, 1: STABLE, 2: INTENSIFYING)
            ri_target: scalar float (0.0 or 1.0)
            reg_targets: (3,) float [+6h, +12h, +24h] intensities
            (optional) env_vec: (D_env,) normalized environmental vector + masks
            meta: dictionary of metadata
        """
        seq_tensor, vis_mask, reg_targets, meta = super().__getitem__(idx)

        trend_target = torch.tensor(self.precomputed_trend[idx], dtype=torch.long)
        ri_target = torch.tensor(self.precomputed_ri[idx], dtype=torch.float32)

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
                reg_targets,
                env_vec,
                meta,
            )

        return (
            seq_tensor,
            vis_mask,
            trend_target,
            ri_target,
            reg_targets,
            meta,
        )


def build_trend_dataloaders(
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
    """Factory creating train, val, and test DataLoaders for trend/RI classification."""
    train_ds = TCIRTrendDataset(
        train_seq_df, mean=mean, std=std, channels=channels, is_training=True, config=config,
        env_tensor=train_env_tensor
    )
    val_ds = TCIRTrendDataset(
        val_seq_df, mean=mean, std=std, channels=channels, is_training=False, config=config,
        env_tensor=val_env_tensor
    )
    test_ds = TCIRTrendDataset(
        test_seq_df, mean=mean, std=std, channels=channels, is_training=False, config=config,
        env_tensor=test_env_tensor
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
