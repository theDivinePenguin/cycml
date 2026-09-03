"""Temporal Deep Learning Architectures for Multi-Horizon Tropical Cyclone Intensity Forecasting."""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class CNNFeatureEncoder(nn.Module):
    """ResNet18 Feature Encoder extracting 512-dimensional spatial representations from multi-channel satellite imagery."""

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Adapt first convolution for in_channels with principled weight initialization
        if in_channels != 3:
            orig_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    orig_weight = orig_conv.weight.data  # shape: (64, 3, 7, 7)
                    if in_channels == 1:
                        new_conv.weight.data = orig_weight.mean(dim=1, keepdim=True)
                    else:
                        weight_scale = 3.0 / in_channels
                        for c in range(in_channels):
                            new_conv.weight.data[:, c : c + 1, :, :] = (
                                orig_weight[:, (c % 3) : (c % 3) + 1, :, :] * weight_scale
                            )
            backbone.conv1 = new_conv

        # Replace classification head with Identity
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B * K, C, H, W) -> (B * K, 512)
        return self.encoder(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, K, d_model)
        return x + self.pe[:, : x.size(1), :]


class TemporalGRUForecaster(nn.Module):
    """Causal Unidirectional GRU Multi-Horizon Intensity Forecaster (+6h, +12h, +24h)."""

    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        pretrained_cnn: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model

        # 1. Spatial CNN Backbone
        self.cnn = CNNFeatureEncoder(in_channels=in_channels, pretrained=pretrained_cnn)

        # 2. Explicit VIS Validity Fusion Projection
        # 512 (CNN feature) + 1 (vis_valid flag) -> d_model (256)
        self.vis_fusion = nn.Sequential(
            nn.Linear(512 + 1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # 3. Causal Unidirectional GRU
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 4. Multi-Horizon Forecasting Head (+6h, +12h, +24h)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor, vis_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input sequence of satellite frames (B, K, C, H, W).
            vis_masks: Optional VIS validity flag per timestep (B, K). If None, defaults to 1.0.

        Returns:
            Predicted intensities (B, 3) for [+6h, +12h, +24h].
        """
        B, K, C, H, W = x.shape

        # Flatten sequence into batch for CNN feature extraction
        x_flat = x.view(B * K, C, H, W)
        cnn_feats = self.cnn(x_flat)  # (B * K, 512)
        cnn_feats = cnn_feats.view(B, K, 512)  # (B, K, 512)

        # Prepare VIS validity mask
        if vis_masks is None:
            vis_masks = torch.ones(B, K, device=x.device, dtype=x.dtype)
        vis_masks = vis_masks.unsqueeze(-1)  # (B, K, 1)

        # Concatenate spatial features with explicit VIS validity flag
        fused_input = torch.cat([cnn_feats, vis_masks], dim=-1)  # (B, K, 513)
        tokens = self.vis_fusion(fused_input)  # (B, K, 256)

        # Causal GRU forward pass
        gru_out, h_n = self.gru(tokens)  # gru_out: (B, K, 256), h_n: (num_layers, B, 256)
        final_state = gru_out[:, -1, :]  # State at final observed timestep t: (B, 256)

        # Multi-horizon forecast
        predictions = self.head(final_state)  # (B, 3)
        return predictions


class TemporalTransformerForecaster(nn.Module):
    """Causal Temporal Transformer Multi-Horizon Intensity Forecaster (+6h, +12h, +24h)."""

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

        # 1. Spatial CNN Backbone
        self.cnn = CNNFeatureEncoder(in_channels=in_channels, pretrained=pretrained_cnn)

        # 2. Explicit VIS Validity Fusion Projection
        self.vis_fusion = nn.Sequential(
            nn.Linear(512 + 1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # 3. Temporal Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=32)

        # 4. Transformer Encoder
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

        # 5. Multi-Horizon Forecasting Head (+6h, +12h, +24h)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor, vis_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input sequence of satellite frames (B, K, C, H, W).
            vis_masks: Optional VIS validity flag per timestep (B, K).

        Returns:
            Predicted intensities (B, 3) for [+6h, +12h, +24h].
        """
        B, K, C, H, W = x.shape

        # Flatten sequence into batch for CNN feature extraction
        x_flat = x.view(B * K, C, H, W)
        cnn_feats = self.cnn(x_flat)  # (B * K, 512)
        cnn_feats = cnn_feats.view(B, K, 512)  # (B, K, 512)

        # Prepare VIS validity mask
        if vis_masks is None:
            vis_masks = torch.ones(B, K, device=x.device, dtype=x.dtype)
        vis_masks = vis_masks.unsqueeze(-1)  # (B, K, 1)

        # Fused temporal tokens
        fused_input = torch.cat([cnn_feats, vis_masks], dim=-1)  # (B, K, 513)
        tokens = self.vis_fusion(fused_input)  # (B, K, 256)

        # Add positional encoding
        tokens = self.pos_encoder(tokens)  # (B, K, 256)

        # Transformer Encoding
        encoded_tokens = self.transformer_encoder(tokens)  # (B, K, 256)

        # Final observed timestep t representation
        final_token = encoded_tokens[:, -1, :]  # (B, 256)

        # Multi-horizon forecast
        predictions = self.head(final_token)  # (B, 3)
        return predictions


class MultiHorizonHuberLoss(nn.Module):
    """Multi-Horizon Huber Loss for joint prediction of [+6h, +12h, +24h] intensity."""

    def __init__(self, delta: float = 1.0, weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=delta, reduction="none")
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds: (B, 3), targets: (B, 3)
        losses_per_horizon = self.smooth_l1(preds, targets)  # (B, 3)
        weights = self.weights.to(preds.device)
        weighted_loss = (losses_per_horizon * weights).mean(dim=0).sum() / weights.sum()
        return weighted_loss
