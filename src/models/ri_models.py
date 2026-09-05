"""Dedicated Rapid Intensification (RI) Models and Specialized Imbalance Loss Functions.

RI is defined as delta_V_24 >= 30 kt.

Models:
  - RIModel1_DedicatedClassifier: Focused solely on binary RI discrimination.
  - RIModel2_MultiTask: Shared representation predicting intensities (+6h, +12h, +24h),
    24h intensity trend (3-class), and RI probability.

Loss Functions:
  - WeightedBCELoss
  - FocalLoss (Lin et al.)
  - AsymmetricFocalLoss (Ridnik et al.)
"""
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import SpatialBackbone
from src.models.fusion import build_fusion_layer
from src.models.temporal_forecaster import PositionalEncoding


# ---------------------------------------------------------------------------
# Loss Functions for Imbalanced Classification
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, 1) or (B,), targets: (B, 1) or (B,) with values 0 or 1
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * ((1.0 - p_t) ** self.gamma)

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Loss dynamically decoupling positive and negative focusing gammas."""

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        probs = torch.sigmoid(logits)
        targets_pos = targets
        targets_neg = 1.0 - targets

        # Positive loss
        loss_pos = -targets_pos * torch.log(probs.clamp(min=1e-8)) * ((1.0 - probs) ** self.gamma_pos)

        # Negative loss with probability shifting / clipping
        p_neg = (probs - self.clip).clamp(min=0.0)
        loss_neg = -targets_neg * torch.log((1.0 - p_neg).clamp(min=1e-8)) * (p_neg ** self.gamma_neg)

        loss = loss_pos + loss_neg

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_ri_loss(loss_name: str, pos_weight: float = 4.0, gamma: float = 2.0) -> nn.Module:
    """Factory creating configured RI classification loss."""
    ln = loss_name.lower()
    if ln in ["weighted_bce", "bce"]:
        weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    elif ln in ["focal", "focal_loss"]:
        return FocalLoss(alpha=0.75, gamma=gamma)
    elif ln in ["asymmetric", "asymmetric_focal"]:
        return AsymmetricFocalLoss(gamma_pos=0.0, gamma_neg=gamma)
    else:
        raise ValueError(f"Unknown RI loss '{loss_name}'. Options: weighted_bce, focal, asymmetric.")


# ---------------------------------------------------------------------------
# RI Model 1: Dedicated Image + Temporal RI Classifier
# ---------------------------------------------------------------------------

class DedicatedRIClassifier(nn.Module):
    """RI Model 1: Dedicated binary classifier optimized specifically for P(RI in 24h)."""

    def __init__(
        self,
        backbone_arch: str = "resnet18",
        in_channels: int = 3,
        d_model: int = 256,
        d_env: int = 12,
        temporal_type: str = "transformer",
        num_layers: int = 2,
        nhead: int = 8,
        fusion_type: str = "gated",
        dropout: float = 0.15,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.temporal_type = temporal_type.lower()

        # 1. Spatial Backbone
        self.spatial = SpatialBackbone(
            architecture=backbone_arch,
            in_channels=in_channels,
            pretrained=pretrained_backbone,
            dropout=dropout,
        )

        # 2. Token Projection with VIS Validity
        self.vis_fusion = nn.Sequential(
            nn.Linear(self.spatial.out_dim + 1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # 3. Temporal Model
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

        # 4. Environmental Encoder & Multi-Modal Fusion
        self.env_encoder = nn.Sequential(
            nn.Linear(d_env, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = build_fusion_layer(
            fusion_type=fusion_type,
            d_vis=d_model,
            d_env=64,
            d_fused=d_model,
            dropout=dropout,
        )

        # 5. Dedicated RI Head (outputs single raw logit)
        self.head_ri = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        vis_masks: Optional[torch.Tensor] = None,
        x_env: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns raw logit for P(RI in 24h) of shape (B, 1)."""
        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)
        spatial_feats = self.spatial(x_flat).view(B, K, -1)

        if vis_masks is None:
            vis_masks = torch.ones(B, K, device=x.device, dtype=x.dtype)
        vis_masks = vis_masks.unsqueeze(-1)

        tokens = self.vis_fusion(torch.cat([spatial_feats, vis_masks], dim=-1))

        if self.temporal_type == "gru":
            gru_out, _ = self.temporal(tokens)
            final_vis = gru_out[:, -1, :]
        else:
            tokens = self.pos_encoder(tokens)
            enc_out = self.temporal(tokens)
            final_vis = enc_out[:, -1, :]

        if x_env is not None:
            h_env = self.env_encoder(x_env)
            fused = self.fusion(final_vis, h_env)
        else:
            fused = final_vis

        ri_logit = self.head_ri(fused)
        return ri_logit


# ---------------------------------------------------------------------------
# RI Model 2: Multi-Task RI + Multi-Horizon Intensity Model
# ---------------------------------------------------------------------------

class MultiTaskRIIntensityModel(nn.Module):
    """RI Model 2: Joint multi-task model with shared representation predicting:
      - +6h, +12h, +24h future intensity (regression)
      - RI probability in 24h (binary classification)
      - 24h intensity trend [WEAKENING, STABLE, INTENSIFYING] (3-class classification)
    """

    def __init__(
        self,
        backbone_arch: str = "resnet18",
        in_channels: int = 3,
        d_model: int = 256,
        d_env: int = 12,
        temporal_type: str = "transformer",
        num_layers: int = 2,
        nhead: int = 8,
        fusion_type: str = "gated",
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.temporal_type = temporal_type.lower()

        # Shared Spatial Backbone
        self.spatial = SpatialBackbone(
            architecture=backbone_arch,
            in_channels=in_channels,
            pretrained=pretrained_backbone,
            dropout=dropout,
        )

        # Token Projection
        self.vis_fusion = nn.Sequential(
            nn.Linear(self.spatial.out_dim + 1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Shared Temporal Encoder
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

        # Shared Environmental Projection & Fusion
        self.env_encoder = nn.Sequential(
            nn.Linear(d_env, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = build_fusion_layer(
            fusion_type=fusion_type,
            d_vis=d_model,
            d_env=64,
            d_fused=d_model,
            dropout=dropout,
        )

        # Head 1: Multi-Horizon Continuous Intensity Regression (+6h, +12h, +24h)
        self.head_intensity = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

        # Head 2: Rapid Intensification (RI) Logit (1-d)
        self.head_ri = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # Head 3: 24h Trend Classification Logits (3-d)
        self.head_trend = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        vis_masks: Optional[torch.Tensor] = None,
        x_env: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            intensity_preds: (B, 3) for +6h, +12h, +24h
            ri_logits: (B, 1) raw RI logit
            trend_logits: (B, 3) raw trend logits
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
            final_vis = gru_out[:, -1, :]
        else:
            tokens = self.pos_encoder(tokens)
            enc_out = self.temporal(tokens)
            final_vis = enc_out[:, -1, :]

        if x_env is not None:
            h_env = self.env_encoder(x_env)
            fused = self.fusion(final_vis, h_env)
        else:
            fused = final_vis

        intensity_preds = self.head_intensity(fused)
        ri_logits = self.head_ri(fused)
        trend_logits = self.head_trend(fused)

        return intensity_preds, ri_logits, trend_logits
