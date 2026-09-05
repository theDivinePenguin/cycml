"""Standardized meteorological baseline models for Tropical Cyclone Intensity & Rapid Intensification.

Baselines:
  1. PersistenceForecaster / PersistenceBaseline: V_hat(t+tau) = V(t) for all horizons; P(RI) = 0.
  2. IntensityHistoryExtrapolator / RecentTrendBaseline: Extrapolates recent 6h intensity change linearly.
  3. EnvironmentalOnlyBaseline: Linear/MLP model trained exclusively on causal environmental predictors.
"""
import json
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.trend_config import IntensityTrendConfig


class PersistenceBaseline:
    """Baseline A: Persistence Trend.
    Predicts STABLE for all trend samples and NO for all RI samples.
    """

    def __init__(self, config: Optional[IntensityTrendConfig] = None):
        self.config = config or IntensityTrendConfig()

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns:
        trend_preds: (N,) array of 1 (STABLE)
        trend_probs: (N, 3) array with P(STABLE)=1.0
        ri_probs: (N,) array of 0.0
        """
        n = len(df)
        trend_preds = np.ones(n, dtype=int)
        trend_probs = np.zeros((n, 3), dtype=float)
        trend_probs[:, 1] = 1.0  # 100% confidence on STABLE
        ri_probs = np.zeros(n, dtype=float)
        return trend_preds, trend_probs, ri_probs

    def predict_forecast(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        v_curr = df["vmax_curr"].values.astype(np.float32)
        n = len(df)
        intensity_preds = np.column_stack([v_curr, v_curr, v_curr])
        delta_preds = np.zeros((n, 3), dtype=np.float32)
        ri_probs = np.zeros(n, dtype=np.float32)
        trend_preds = np.ones(n, dtype=np.int64)
        return {
            "intensity_preds": intensity_preds,
            "delta_preds": delta_preds,
            "ri_probs": ri_probs,
            "trend_preds": trend_preds,
        }


class RecentTrendBaseline:
    """Baseline B: Recent Intensity Trend Extrapolation.
    Uses recent 6h historical intensity change (t vs t-6h) to linearly extrapolate 24h change.
    """

    def __init__(self, config: Optional[IntensityTrendConfig] = None):
        self.config = config or IntensityTrendConfig()

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(df)
        hist_vmax_list = []
        for h_str in df["history_vmax"]:
            if isinstance(h_str, str):
                hist_vmax_list.append(json.loads(h_str))
            else:
                hist_vmax_list.append(h_str)
        hist_arr = np.array(hist_vmax_list)

        # Recent 6h change: last frame minus 2 frames prior (3h cadence)
        v_t = hist_arr[:, -1]
        v_t_minus_6 = hist_arr[:, -3] if hist_arr.shape[1] >= 3 else hist_arr[:, 0]
        d_recent_6h = v_t - v_t_minus_6
        extrap_24h = d_recent_6h * 4.0

        trend_preds = np.ones(n, dtype=int)
        trend_preds[extrap_24h <= self.config.weakening_threshold_kt] = 0
        trend_preds[extrap_24h >= self.config.intensifying_threshold_kt] = 2

        logits = np.zeros((n, 3), dtype=float)
        logits[:, 0] = -extrap_24h / 10.0
        logits[:, 1] = 2.0 - np.abs(extrap_24h) / 10.0
        logits[:, 2] = extrap_24h / 10.0
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        trend_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        ri_probs = 1.0 / (1.0 + np.exp(-(extrap_24h - self.config.ri_threshold_kt) / 10.0))
        return trend_preds, trend_probs, ri_probs

    def predict_forecast(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        v_curr = df["vmax_curr"].values.astype(np.float32)
        n = len(df)
        hist_vmax_list = []
        for h_str in df["history_vmax"]:
            if isinstance(h_str, str):
                hist_vmax_list.append(json.loads(h_str))
            else:
                hist_vmax_list.append(h_str)
        hist_arr = np.array(hist_vmax_list)

        v_t = hist_arr[:, -1]
        v_t_minus_6 = hist_arr[:, -3] if hist_arr.shape[1] >= 3 else hist_arr[:, 0]
        d_recent_6h = v_t - v_t_minus_6
        slope = d_recent_6h / 6.0

        d6 = slope * 6.0
        d12 = slope * 12.0
        d24 = slope * 24.0

        intensity_preds = np.column_stack([v_curr + d6, v_curr + d12, v_curr + d24]).astype(np.float32)
        delta_preds = np.column_stack([d6, d12, d24]).astype(np.float32)
        ri_probs = (1.0 / (1.0 + np.exp(-(d24 - 30.0) / 8.0))).astype(np.float32)

        trend_preds = np.ones(n, dtype=np.int64)
        trend_preds[d24 <= -10.0] = 0
        trend_preds[d24 >= 10.0] = 2

        return {
            "intensity_preds": intensity_preds,
            "delta_preds": delta_preds,
            "ri_probs": ri_probs,
            "trend_preds": trend_preds,
        }


# Aliases
PersistenceForecaster = PersistenceBaseline
IntensityHistoryExtrapolator = RecentTrendBaseline


class EnvironmentalOnlyBaseline(nn.Module):
    """Environmental-Only Baseline Model:
    Multi-layer perceptron trained strictly on 12-d causal environmental features (SHIPS).
    Zero satellite image input.
    """

    def __init__(self, in_dim: int = 12, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_intensity = nn.Linear(hidden_dim, 3)
        self.head_ri = nn.Linear(hidden_dim, 1)

    def forward(self, x_env: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x_env)
        intensity = self.head_intensity(h)
        ri_logit = self.head_ri(h)
        return intensity, ri_logit
