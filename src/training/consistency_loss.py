"""Multi-task physical consistency loss regularizing cross-head agreement between Delta V24 and RI probability."""
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskConsistencyLoss(nn.Module):
    """Soft consistency constraint between continuous +24h intensity change (Delta V_24)
    and Rapid Intensification probability P(RI in 24h).

    Does NOT hard-code RI = positive; instead gently penalizes extreme physical contradiction
    (e.g. predicting Delta V24 = +50 kt while predicting P(RI) = 0.01).
    """

    def __init__(self, ri_threshold_kt: float = 30.0, temperature: float = 6.0, weight: float = 0.1):
        super().__init__()
        self.ri_threshold_kt = ri_threshold_kt
        self.temperature = temperature
        self.weight = weight

    def forward(
        self,
        pred_delta_24: torch.Tensor,
        ri_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_delta_24: (B,) or (B, 1) continuous predicted 24h intensity change in knots
            ri_logits: (B,) or (B, 1) raw logit for P(RI in 24h)
        Returns:
            weighted_loss: scalar consistency loss
            diagnostics: dictionary with cross-head disagreement metrics
        """
        delta = pred_delta_24.view(-1)
        ri_log = ri_logits.view(-1)

        # Continuous proxy logit derived from predicted delta V
        delta_logit = (delta - self.ri_threshold_kt) / self.temperature

        # Probabilities for target labels and metric reporting
        p_ri = torch.sigmoid(ri_log).clamp(min=1e-6, max=1.0 - 1e-6)
        p_delta = torch.sigmoid(delta_logit).clamp(min=1e-6, max=1.0 - 1e-6)

        # Autocast-safe cross-entropy with logits
        loss_p_to_delta = F.binary_cross_entropy_with_logits(ri_log, p_delta.detach())
        loss_delta_to_p = F.binary_cross_entropy_with_logits(delta_logit, p_ri.detach())
        consistency_loss = 0.5 * (loss_p_to_delta + loss_delta_to_p)

        # Cross-head disagreement metric: mean absolute divergence
        disagreement = torch.mean(torch.abs(p_ri - p_delta)).item()

        diagnostics = {
            "consistency_loss": float(consistency_loss.item()),
            "cross_head_disagreement": float(disagreement),
        }

        return self.weight * consistency_loss, diagnostics
