"""Residual Delta-V Forecaster reconstructing future intensity as V_hat(t+tau) = V(t) + Delta_V_hat(tau)."""
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import SpatialBackbone
from src.models.temporal_forecaster import PositionalEncoding


class ResidualDeltaVForecaster(nn.Module):
    """Predicts intensity changes (Delta V6, Delta V12, Delta V24) rather than absolute values,
    then reconstructs V_hat(t+tau) = V(t) + Delta_V_hat(tau).

    Supports:
      - parameterization = 'unconstrained' (direct linear output for Delta V)
      - parameterization = 'bounded' (scaled tanh parameterization, e.g. [-80 kt, +100 kt])
    """

    def __init__(
        self,
        backbone_arch: str = "resnet18",
        in_channels: int = 3,
        d_model: int = 256,
        temporal_type: str = "transformer",  # 'transformer' or 'gru'
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        parameterization: str = "unconstrained",  # 'unconstrained' or 'bounded'
        delta_bounds: Tuple[float, float] = (-80.0, 100.0),  # (min_delta, max_delta) in kt
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.parameterization = parameterization.lower()
        self.min_delta, self.max_delta = delta_bounds
        self.temporal_type = temporal_type.lower()
        self.d_model = d_model

        # 1. Spatial Encoder
        self.spatial = SpatialBackbone(
            architecture=backbone_arch,
            in_channels=in_channels,
            pretrained=pretrained_backbone,
            dropout=dropout,
        )

        # 2. Projection to d_model + VIS validity gating
        self.vis_fusion = nn.Sequential(
            nn.Linear(self.spatial.out_dim + 1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # 3. Temporal Model (Transformer or GRU)
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

        # 4. Residual Delta-V Prediction Head
        # Outputs 3 values: [Delta_V_6h, Delta_V_12h, Delta_V_24h]
        self.delta_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        v_curr: torch.Tensor,
        vis_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Satellite sequence tensor of shape (B, K, C, H, W)
            v_curr: Current observed wind speed at time t (B,) or (B, 1) in knots
            vis_masks: Optional VIS daytime/nighttime flags (B, K)
        Returns:
            v_hat: Reconstructed absolute intensities (B, 3) for [+6h, +12h, +24h]
            delta_v_hat: Predicted intensity deltas (B, 3)
        """
        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)
        spatial_feats = self.spatial(x_flat).view(B, K, -1)

        if vis_masks is None:
            vis_masks = torch.ones(B, K, device=x.device, dtype=x.dtype)
        vis_masks = vis_masks.unsqueeze(-1)  # (B, K, 1)

        fused = torch.cat([spatial_feats, vis_masks], dim=-1)
        tokens = self.vis_fusion(fused)  # (B, K, d_model)

        if self.temporal_type == "gru":
            gru_out, _ = self.temporal(tokens)
            final_rep = gru_out[:, -1, :]
        else:
            tokens = self.pos_encoder(tokens)
            enc_out = self.temporal(tokens)
            final_rep = enc_out[:, -1, :]

        raw_delta = self.delta_head(final_rep)  # (B, 3)

        if self.parameterization == "bounded":
            # Scale tanh from [-1, 1] to [min_delta, max_delta]
            mid = (self.max_delta + self.min_delta) / 2.0
            half_range = (self.max_delta - self.min_delta) / 2.0
            delta_v_hat = mid + half_range * torch.tanh(raw_delta)
        else:
            delta_v_hat = raw_delta

        # Reconstruct absolute future intensity: V_hat(t+tau) = V(t) + Delta_V_hat(tau)
        if v_curr.ndim == 1:
            v_curr = v_curr.unsqueeze(1)  # (B, 1)

        v_hat = v_curr + delta_v_hat  # (B, 3)
        return v_hat, delta_v_hat
