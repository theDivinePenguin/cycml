"""Baseline models for Cyclone Intensity Trend and Rapid Intensification Prediction."""
import json
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.data.trend_config import IntensityTrendConfig


class PersistenceBaseline:
    """Baseline A: Persistence Trend.
    Predicts STABLE for all trend samples and NO for all RI samples.
    """

    def __init__(self, config: IntensityTrendConfig = None):
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


class RecentTrendBaseline:
    """Baseline B: Recent Intensity Trend Extrapolation.
    Uses recent 6h historical intensity change (t vs t-6h) to linearly extrapolate 24h change:
    delta_hat_24 = (Vmax(t) - Vmax(t-6h)) * 4.0
    """

    def __init__(self, config: IntensityTrendConfig = None):
        self.config = config or IntensityTrendConfig()

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns:
        trend_preds: (N,) array in {0, 1, 2}
        trend_probs: (N, 3) pseudo-probabilities
        ri_probs: (N,) calibrated sigmoid probabilities based on extrapolated delta V
        """
        n = len(df)

        # Extract history_vmax
        hist_vmax_list = []
        for h_str in df["history_vmax"]:
            if isinstance(h_str, str):
                hist_vmax_list.append(json.loads(h_str))
            else:
                hist_vmax_list.append(h_str)
        hist_arr = np.array(hist_vmax_list)  # (N, 5) -> [t-12h, t-9h, t-6h, t-3h, t]

        # Recent 6h change: t - (t-6h) = index 4 - index 2
        d_recent_6h = hist_arr[:, 4] - hist_arr[:, 2]
        extrap_24h = d_recent_6h * 4.0

        # Class predictions
        trend_preds = np.ones(n, dtype=int)
        trend_preds[extrap_24h <= self.config.weakening_threshold_kt] = 0
        trend_preds[extrap_24h >= self.config.intensifying_threshold_kt] = 2

        # Softmax pseudo-probabilities based on distance to thresholds
        logits = np.zeros((n, 3), dtype=float)
        logits[:, 0] = -extrap_24h / 10.0  # high when negative
        logits[:, 1] = 2.0 - np.abs(extrap_24h) / 10.0  # high near 0
        logits[:, 2] = extrap_24h / 10.0  # high when positive
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        trend_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Sigmoid probability for RI: centered around 30 kt
        # prob = 1 / (1 + exp(-(extrap - 30) / 10))
        ri_probs = 1.0 / (1.0 + np.exp(-(extrap_24h - self.config.ri_threshold_kt) / 10.0))
        return trend_preds, trend_probs, ri_probs


class ThresholdedRegressionBaseline:
    """Baseline C: Continuous Regression Model Thresholded.
    Directly discretizes the continuous +24h Vmax forecasts from the existing
    CNN + Temporal Transformer into Trend classes and RI flags.
    """

    def __init__(self, config: IntensityTrendConfig = None):
        self.config = config or IntensityTrendConfig()

    def predict_from_csv(self, pred_csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        df = pd.read_csv(pred_csv_path)
        pred_delta_24 = df["pred_plus_24h"].values - df["vmax_curr"].values

        trend_preds = np.ones(len(df), dtype=int)
        trend_preds[pred_delta_24 <= self.config.weakening_threshold_kt] = 0
        trend_preds[pred_delta_24 >= self.config.intensifying_threshold_kt] = 2

        # Sigmoid probability around 30 kt
        ri_probs = 1.0 / (1.0 + np.exp(-(pred_delta_24 - self.config.ri_threshold_kt) / 8.0))

        # Approximate trend probabilities
        logits = np.zeros((len(df), 3), dtype=float)
        logits[:, 0] = -pred_delta_24 / 10.0
        logits[:, 1] = 2.0 - np.abs(pred_delta_24) / 10.0
        logits[:, 2] = pred_delta_24 / 10.0
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        trend_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return trend_preds, trend_probs, ri_probs, df
