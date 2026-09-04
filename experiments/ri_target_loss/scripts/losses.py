"""Isolated Loss Functions for Delta Head and RI-Aware Weighting Experiments."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaJointLoss(nn.Module):
    """Multi-task loss supporting absolute Vmax, delta V, and RI-aware sample weighting."""

    def __init__(
        self,
        mode: str = "abs_and_delta",  # 'abs_and_delta' or 'delta_only'
        ri_pos_weight: Optional[torch.Tensor] = None,
        trend_class_weights: Optional[torch.Tensor] = None,
        lambda_ri: float = 1.0,
        lambda_trend: float = 1.0,
        lambda_reg_abs: float = 0.1,
        lambda_reg_delta: float = 0.1,
        ri_weights: Optional[Tuple[float, float, float]] = None,  # (w_low, w_mid, w_high)
        huber_beta: float = 1.0,
    ):
        super().__init__()
        self.mode = mode
        self.lambda_ri = lambda_ri
        self.lambda_trend = lambda_trend
        self.lambda_reg_abs = lambda_reg_abs
        self.lambda_reg_delta = lambda_reg_delta
        self.ri_weights = ri_weights
        self.huber_beta = huber_beta

        self.loss_ri = nn.BCEWithLogitsLoss(pos_weight=ri_pos_weight)
        self.loss_trend = nn.CrossEntropyLoss(weight=trend_class_weights)
        self.loss_reg_abs = nn.SmoothL1Loss(beta=huber_beta)

    def forward(
        self,
        ri_logits: torch.Tensor,
        trend_logits: torch.Tensor,
        ri_targets: torch.Tensor,
        trend_targets: torch.Tensor,
        reg_delta_preds: torch.Tensor,
        reg_delta_targets: torch.Tensor,
        reg_abs_preds: Optional[torch.Tensor] = None,
        reg_abs_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. RI Loss
        l_ri = self.loss_ri(ri_logits.squeeze(-1), ri_targets)

        # 2. Trend Loss
        l_trend = self.loss_trend(trend_logits, trend_targets)

        # 3. Delta Regression Loss (with optional RI-aware sample weighting)
        if self.ri_weights is not None:
            # Weighted Huber on Delta V
            # ri_weights = (w_low, w_mid, w_high)
            w_low, w_mid, w_high = self.ri_weights
            actual_dv24 = reg_delta_targets[:, 2]

            sample_weights = torch.ones_like(actual_dv24) * w_low
            mid_mask = (actual_dv24 >= 15.0) & (actual_dv24 < 30.0)
            high_mask = actual_dv24 >= 30.0
            sample_weights[mid_mask] = w_mid
            sample_weights[high_mask] = w_high

            # Compute elementwise Smooth L1 / Huber
            huber_elementwise = F.smooth_l1_loss(
                reg_delta_preds, reg_delta_targets, beta=self.huber_beta, reduction="none"
            )  # (B, 3)
            # Weight +24h specifically or all delta horizons:
            weighted_huber_24 = huber_elementwise[:, 2] * sample_weights
            l_delta = (huber_elementwise[:, 0].mean() + huber_elementwise[:, 1].mean() + weighted_huber_24.mean()) / 3.0
        else:
            l_delta = F.smooth_l1_loss(reg_delta_preds, reg_delta_targets, beta=self.huber_beta)

        # 4. Absolute Regression Loss (if active)
        if self.mode == "abs_and_delta" and reg_abs_preds is not None and reg_abs_targets is not None:
            l_abs = self.loss_reg_abs(reg_abs_preds, reg_abs_targets)
            total_loss = (
                self.lambda_ri * l_ri +
                self.lambda_trend * l_trend +
                self.lambda_reg_abs * l_abs +
                self.lambda_reg_delta * l_delta
            )
            loss_dict = {
                "loss_ri": l_ri.item(),
                "loss_trend": l_trend.item(),
                "loss_reg_abs": l_abs.item(),
                "loss_reg_delta": l_delta.item(),
            }
        else:
            total_loss = (
                self.lambda_ri * l_ri +
                self.lambda_trend * l_trend +
                self.lambda_reg_delta * l_delta
            )
            loss_dict = {
                "loss_ri": l_ri.item(),
                "loss_trend": l_trend.item(),
                "loss_reg_delta": l_delta.item(),
            }

        return total_loss, loss_dict
