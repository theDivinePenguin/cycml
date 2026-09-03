"""Unified Multi-Task Temporal Classifier for Rapid Intensification, Intensity Trend, and Quantitative Forecasting."""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.models.temporal_forecaster import CNNFeatureEncoder, PositionalEncoding


class TemporalClassifier(nn.Module):
    """Unified Spatio-Temporal Model with shared ResNet18 + Temporal Transformer backbone:
    1. Primary Headline Task: Rapid Intensification (RI) binary prediction (P(RI in 24h))
    2. Secondary Task: 24h Intensity Trend classification (WEAKENING, STABLE, INTENSIFYING)
    3. Supporting Task: Auxiliary multi-horizon Vmax regression (+6h, +12h, +24h)
    """

    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pretrained_cnn: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model

        # 1. Spatial CNN Backbone (Shared ResNet-18)
        self.cnn = CNNFeatureEncoder(in_channels=in_channels, pretrained=pretrained_cnn)

        # 2. Explicit VIS Validity Fusion Projection (512 CNN feature + 1 vis_valid flag -> d_model)
        self.vis_fusion = nn.Sequential(
            nn.Linear(512 + 1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # 3. Temporal Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=32)

        # 4. Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Primary Headline Task Head: Rapid Intensification (RI) Prediction
        # Outputs 1 scalar logit for binary classification
        self.head_ri = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # 6. Secondary Task Head: 24-Hour Intensity Trend Classification
        # Outputs 3 logits for [WEAKENING, STABLE, INTENSIFYING]
        self.head_trend = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

        # 7. Supporting Auxiliary Forecast Head: Quantitative Vmax (+6h, +12h, +24h)
        self.head_reg = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def extract_features(self, x: torch.Tensor, vis_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract shared 256-d temporal representations from 5-frame sequence.
        Args:
            x: (B, K, C, H, W)
            vis_masks: (B, K)
        Returns:
            final_token: (B, d_model) representation at time t
        """
        B, K, C, H, W = x.shape

        # Flatten sequence for CNN feature extraction
        x_flat = x.view(B * K, C, H, W)
        cnn_feats = self.cnn(x_flat)  # (B * K, 512)
        cnn_feats = cnn_feats.view(B, K, 512)  # (B, K, 512)

        # VIS validity gating
        if vis_masks is None:
            vis_masks = torch.ones(B, K, device=x.device, dtype=x.dtype)
        vis_masks = vis_masks.unsqueeze(-1)  # (B, K, 1)

        fused_input = torch.cat([cnn_feats, vis_masks], dim=-1)  # (B, K, 513)
        tokens = self.vis_fusion(fused_input)  # (B, K, 256)

        # Temporal positional encoding + Transformer
        tokens = self.pos_encoder(tokens)  # (B, K, 256)
        encoded_tokens = self.transformer_encoder(tokens)  # (B, K, 256)

        # Final representation at time t
        final_token = encoded_tokens[:, -1, :]  # (B, 256)
        return final_token

    def forward(
        self, x: torch.Tensor, vis_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, K, C, H, W)
            vis_masks: (B, K)
        Returns:
            ri_logits: (B, 1) for Rapid Intensification probability
            trend_logits: (B, 3) for [Weakening, Stable, Intensifying]
            reg_preds: (B, 3) for [+6h, +12h, +24h] Vmax intensities
        """
        final_token = self.extract_features(x, vis_masks)
        ri_logits = self.head_ri(final_token)
        trend_logits = self.head_trend(final_token)
        reg_preds = self.head_reg(final_token)
        return ri_logits, trend_logits, reg_preds

    def predict_probabilities(
        self, x: torch.Tensor, vis_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience method returning calibrated probabilities and predictions:
        Returns:
            ri_prob: (B,) float probabilities in [0, 1]
            trend_probs: (B, 3) softmax probabilities
            reg_preds: (B, 3) intensities in knots
        """
        ri_logits, trend_logits, reg_preds = self.forward(x, vis_masks)
        ri_prob = torch.sigmoid(ri_logits.squeeze(-1))
        trend_probs = torch.softmax(trend_logits, dim=-1)
        return ri_prob, trend_probs, reg_preds

    def load_backbone_from_forecaster(self, checkpoint_path: str, device: torch.device):
        """Warm-start CNN encoder, VIS fusion, and Transformer layers from existing trained forecaster."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Filter out old head weights, keep shared backbone
        compatible_dict = {}
        for k, v in state_dict.items():
            if k.startswith("cnn.") or k.startswith("vis_fusion.") or k.startswith("pos_encoder.") or k.startswith("transformer_encoder."):
                compatible_dict[k] = v
            elif k.startswith("head."):
                # Can map old head to head_reg
                compatible_dict[k.replace("head.", "head_reg.")] = v

        msg = self.load_state_dict(compatible_dict, strict=False)
        print(f"Loaded warm-start backbone from {checkpoint_path}:")
        print(f"  • Matched {len(compatible_dict)} tensors.")
        print(f"  • Missing keys (new classification heads): {len(msg.missing_keys)}")
        return msg


class JointTrendRILoss(nn.Module):
    """Joint Multi-Task Loss balancing:
    1. Binary Cross-Entropy with pos_weight for Rapid Intensification
    2. Multi-Class Cross-Entropy for Intensity Trend
    3. Auxiliary SmoothL1 Huber Loss for continuous Vmax
    """

    def __init__(
        self,
        ri_pos_weight: Optional[torch.Tensor] = None,
        trend_class_weights: Optional[torch.Tensor] = None,
        lambda_ri: float = 1.0,
        lambda_trend: float = 1.0,
        lambda_reg: float = 0.1,
    ):
        super().__init__()
        self.bce_ri = nn.BCEWithLogitsLoss(pos_weight=ri_pos_weight)
        self.ce_trend = nn.CrossEntropyLoss(weight=trend_class_weights)
        self.smooth_l1 = nn.SmoothL1Loss(beta=1.0)

        self.lambda_ri = lambda_ri
        self.lambda_trend = lambda_trend
        self.lambda_reg = lambda_reg

    def forward(
        self,
        ri_logits: torch.Tensor,
        trend_logits: torch.Tensor,
        reg_preds: torch.Tensor,
        ri_targets: torch.Tensor,
        trend_targets: torch.Tensor,
        reg_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # ri_targets: (B,), ri_logits: (B, 1)
        loss_ri = self.bce_ri(ri_logits.squeeze(-1), ri_targets)
        loss_trend = self.ce_trend(trend_logits, trend_targets)
        loss_reg = self.smooth_l1(reg_preds, reg_targets)

        total_loss = (
            self.lambda_ri * loss_ri
            + self.lambda_trend * loss_trend
            + self.lambda_reg * loss_reg
        )

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_ri": float(loss_ri.item()),
            "loss_trend": float(loss_trend.item()),
            "loss_reg": float(loss_reg.item()),
        }
        return total_loss, loss_dict
