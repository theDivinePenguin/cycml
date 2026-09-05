"""Probabilistic Tropical Cyclone Intensity Forecasting via Multi-Horizon Quantile Regression.

Predicts predictive intervals/cones for horizons [+6h, +12h, +24h] at quantiles [q10, q50, q90].
Includes Pinball Loss, Monotonic parameterization (guaranteeing zero quantile crossing),
crossing rate diagnostic detection, empirical coverage, sharpness, and Winkler interval score.
"""
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import SpatialBackbone
from src.models.temporal_forecaster import PositionalEncoding


class PinballLoss(nn.Module):
    """Multi-Horizon, Multi-Quantile Pinball (Check) Loss.

    L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """

    def __init__(self, quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90)):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: (B, num_horizons, num_quantiles), e.g. (B, 3, 3)
            targets: (B, num_horizons), e.g. (B, 3)
        Returns:
            Mean pinball loss scalar
        """
        targets_expanded = targets.unsqueeze(-1)  # (B, 3, 1)
        diff = targets_expanded - preds  # (B, 3, 3)

        total_loss = 0.0
        for i, q in enumerate(self.quantiles):
            err = diff[:, :, i]
            loss_q = torch.maximum(q * err, (q - 1.0) * err)
            total_loss = total_loss + loss_q.mean()

        return total_loss / len(self.quantiles)


class ProbabilisticQuantileForecaster(nn.Module):
    """Predicts quantiles (q10, q50, q90) across horizons [+6h, +12h, +24h].

    Supports:
      - monotonic=True: Parameterizes q10 = q50 - softplus(d1), q90 = q50 + softplus(d2)
        guaranteeing strict quantile monotonicity (q10 <= q50 <= q90) by construction.
      - monotonic=False: Unconstrained 3-quantile linear output, allowing empirical measurement
        of quantile crossing rates.
    """

    def __init__(
        self,
        backbone_arch: str = "resnet18",
        in_channels: int = 3,
        d_model: int = 256,
        temporal_type: str = "transformer",
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        monotonic: bool = True,
        quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90),
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.monotonic = monotonic
        self.quantiles = quantiles
        self.temporal_type = temporal_type.lower()
        self.d_model = d_model

        self.spatial = SpatialBackbone(
            architecture=backbone_arch,
            in_channels=in_channels,
            pretrained=pretrained_backbone,
            dropout=dropout,
        )

        self.vis_fusion = nn.Sequential(
            nn.Linear(self.spatial.out_dim + 1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        if self.temporal_type == "gru":
            self.temporal = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=32)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: 3 horizons x 3 parameters = 9 values
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 9),
        )

    def forward(
        self,
        x: torch.Tensor,
        vis_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, K, C, H, W)
            vis_masks: (B, K)
        Returns:
            quantiles: (B, 3, 3) where dim 1 is horizons (+6h, +12h, +24h)
                       and dim 2 is quantiles (q10, q50, q90)
        """
        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)
        spatial_feats = self.spatial(x_flat).view(B, K, -1)

        if vis_masks is None:
            vis_masks = torch.ones(B, K, device=x.device, dtype=x.dtype)
        vis_masks = vis_masks.unsqueeze(-1)

        tokens = self.vis_fusion(torch.cat([spatial_feats, vis_masks], dim=-1))

        if self.temporal_type == "gru":
            gru_out, _ = self.temporal(tokens)
            rep = gru_out[:, -1, :]
        else:
            tokens = self.pos_encoder(tokens)
            enc_out = self.temporal(tokens)
            rep = enc_out[:, -1, :]

        raw_out = self.head(rep).view(B, 3, 3)  # (B, 3 horizons, 3 params)

        if self.monotonic:
            # param 0: q50
            # param 1: delta_lower -> q10 = q50 - softplus(delta_lower)
            # param 2: delta_upper -> q90 = q50 + softplus(delta_upper)
            q50 = raw_out[:, :, 0]
            q10 = q50 - F.softplus(raw_out[:, :, 1])
            q90 = q50 + F.softplus(raw_out[:, :, 2])
            return torch.stack([q10, q50, q90], dim=-1)  # (B, 3, 3)
        else:
            return raw_out


# ---------------------------------------------------------------------------
# Calibration, Coverage, and Crossing Diagnostic Metrics
# ---------------------------------------------------------------------------

def compute_probabilistic_metrics(
    preds: np.ndarray, targets: np.ndarray, nominal_coverage: float = 0.80
) -> Dict[str, float]:
    """Compute empirical coverage, interval width (sharpness), Winkler score,
    and quantile crossing rate across horizons.

    Args:
        preds: (N, 3, 3) array [horizons: +6h, +12h, +24h; quantiles: q10, q50, q90]
        targets: (N, 3) array [horizons: +6h, +12h, +24h]
    """
    horizons = ["+6h", "+12h", "+24h"]
    results = {}

    for h_idx, h_name in enumerate(horizons):
        q10 = preds[:, h_idx, 0]
        q50 = preds[:, h_idx, 1]
        q90 = preds[:, h_idx, 2]
        y = targets[:, h_idx]

        # 1. Empirical Coverage (fraction where y in [q10, q90])
        covered = (y >= q10) & (y <= q90)
        emp_coverage = float(np.mean(covered))

        # 2. Interval Width (Sharpness: narrower is sharper)
        width = q90 - q10
        mean_width = float(np.mean(width))

        # 3. Quantile Crossing Rate (where q10 > q50 or q50 > q90)
        crossings = (q10 > q50) | (q50 > q90)
        crossing_rate = float(np.mean(crossings))

        # 4. Winkler Interval Score (alpha = 1 - nominal_coverage = 0.20)
        # S_alpha(l, u, y) = (u - l) + 2/alpha * (l - y) if y < l
        #                          + 2/alpha * (y - u) if y > u
        alpha = 1.0 - nominal_coverage
        penalty_lower = np.where(y < q10, (2.0 / alpha) * (q10 - y), 0.0)
        penalty_upper = np.where(y > q90, (2.0 / alpha) * (y - q90), 0.0)
        winkler = width + penalty_lower + penalty_upper
        mean_winkler = float(np.mean(winkler))

        # 5. Quantile Pinball Loss decomposition
        diff_q10 = y - q10
        diff_q50 = y - q50
        diff_q90 = y - q90
        loss_q10 = float(np.mean(np.maximum(0.10 * diff_q10, (0.10 - 1.0) * diff_q10)))
        loss_q50 = float(np.mean(np.maximum(0.50 * diff_q50, (0.50 - 1.0) * diff_q50)))
        loss_q90 = float(np.mean(np.maximum(0.90 * diff_q90, (0.90 - 1.0) * diff_q90)))
        mean_pinball = float(np.mean([loss_q10, loss_q50, loss_q90]))

        # 6. Median forecast MAE & RMSE
        mae_median = float(np.mean(np.abs(q50 - y)))
        rmse_median = float(np.sqrt(np.mean((q50 - y) ** 2)))

        results[f"pinball_loss_q10_{h_name}"] = round(loss_q10, 4)
        results[f"pinball_loss_q50_{h_name}"] = round(loss_q50, 4)
        results[f"pinball_loss_q90_{h_name}"] = round(loss_q90, 4)
        results[f"pinball_loss_mean_{h_name}"] = round(mean_pinball, 4)
        results[f"coverage_{h_name}"] = round(emp_coverage, 4)
        results[f"width_{h_name}"] = round(mean_width, 2)
        results[f"winkler_{h_name}"] = round(mean_winkler, 2)
        results[f"crossing_rate_{h_name}"] = round(crossing_rate, 4)
        results[f"mae_q50_{h_name}"] = round(mae_median, 3)
        results[f"rmse_q50_{h_name}"] = round(rmse_median, 3)

    return results

