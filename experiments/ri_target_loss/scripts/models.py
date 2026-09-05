"""Isolated Model Definition for Delta and Multi-Target Regression Experiments."""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Ensure repo root is on sys.path
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.models.environmental_temporal_classifier import EnvironmentalTemporalClassifier


class DeltaEnvironmentalTemporalClassifier(EnvironmentalTemporalClassifier):
    """Extends canonical EnvironmentalTemporalClassifier with explicit Delta V heads.

    Modes:
    - 'abs_and_delta': Retains head_reg (absolute Vmax) + adds head_delta (Delta V)
    - 'delta_only': Uses head_delta only (predicts Delta V exclusively)
    """

    def __init__(
        self,
        mode: str = "abs_and_delta",  # 'abs_and_delta' or 'delta_only'
        channels: int = 3,
        num_frames: int = 7,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_vis_channel: bool = True,
        env_in_dim: int = 12,
    ):
        super().__init__(
            channels=channels,
            num_frames=num_frames,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_vis_channel=use_vis_channel,
        )
        self.mode = mode
        self.env_in_dim = env_in_dim

        if env_in_dim != 12:
            from src.models.environmental_temporal_classifier import EnvironmentalEncoder
            self.env_encoder = EnvironmentalEncoder(in_dim=env_in_dim, out_dim=64, dropout=dropout)

        # Explicit Delta V Head: predicts [+6h, +12h, +24h] intensity changes
        self.head_delta = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def load_warm_start_with_expanded_env(self, checkpoint_path: str):
        """Warm-start weights preserving all existing layers and expanding env input weights."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model_dict = self.state_dict()

        loaded_keys = []
        for k, v in state_dict.items():
            if k == "env_encoder.net.0.weight" and self.env_in_dim > 12:
                # Shape old: (128, 12), shape new: (128, env_in_dim)
                model_dict[k][:, :12] = v
                loaded_keys.append(f"{k} (expanded 12 -> {self.env_in_dim}, preserved first 12 cols)")
            elif k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k] = v
                loaded_keys.append(k)

        self.load_state_dict(model_dict)
        print(f"Warm-started {len(loaded_keys)} tensor weights from {checkpoint_path} with preserved env weights.")

    def forward(
        self,
        x: torch.Tensor,
        vis_masks: Optional[torch.Tensor] = None,
        x_env: Optional[torch.Tensor] = None,
    ):
        h_vis = self.extract_visual_features(x, vis_masks)

        if x_env is not None:
            gated_env = self.apply_feature_masks(x_env)
            h_env = self.env_encoder(gated_env)
            h_fused = self.fusion(h_vis, h_env)
        else:
            h_fused = h_vis

        ri_logits = self.head_ri(h_fused)
        trend_logits = self.head_trend(h_fused)

        if self.mode == "abs_and_delta":
            reg_abs = self.head_reg(h_fused)
            reg_delta = self.head_delta(h_fused)
            return ri_logits, trend_logits, reg_abs, reg_delta
        elif self.mode == "delta_only":
            reg_delta = self.head_delta(h_fused)
            return ri_logits, trend_logits, reg_delta
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
