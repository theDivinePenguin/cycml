"""
Multi-Modal Environmental Temporal Classifier for Tropical Cyclone RI and Trend Prediction.
Combines 5-frame multi-spectral satellite imagery with physical environmental thermodynamic
and kinematic variables (SST, OHC, Vertical Wind Shear, Mid-Level RH, Vmax, MSLP).

Architecture:
  - Satellite branch: Shared ResNet-18 + VIS validity fusion + Temporal Transformer -> h_vis (256-d)
  - Environmental branch: Modular gating -> Dense MLP -> LayerNorm -> h_env (64-d)
  - Multi-Modal Fusion: Gated projection with residual connection -> h_fused (256-d)
  - Multi-Task Heads:
      * Primary: Rapid Intensification probability P(RI in 24h)
      * Secondary: 3-Class Dynamic Intensity Trend [Weakening, Stable, Intensifying]
      * Auxiliary: Multi-Horizon Numerical Guidance (+6h, +12h, +24h)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.temporal_forecaster import CNNFeatureEncoder, PositionalEncoding


class EnvironmentalEncoder(nn.Module):
    """Encodes physical environmental and storm-state features with missingness gating."""

    def __init__(
        self,
        in_dim: int = 12,  # 6 continuous features + 6 missingness masks
        out_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x_env: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_env: (B, in_dim) tensor of normalized environmental features + missing masks
        Returns:
            (B, out_dim) environmental embedding
        """
        return self.net(x_env)


class MultiModalFusionLayer(nn.Module):
    """Gated fusion of visual temporal representations (256-d) and environmental embeddings (64-d)."""

    def __init__(self, d_vis: int = 256, d_env: int = 64, d_fused: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_vis + d_env, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fused, d_fused),
        )
        self.norm = nn.LayerNorm(d_fused)

    def forward(self, h_vis: torch.Tensor, h_env: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_vis, h_env], dim=-1)
        fused = self.proj(combined)
        # Residual shortcut from visual representation
        return self.norm(h_vis + fused)


class EnvironmentalTemporalClassifier(nn.Module):
    """Full Multi-Modal AI System fusing satellite sequences with physical environmental data."""

    def __init__(
        self,
        channels: int = 3,
        num_frames: int = 5,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_vis_channel: bool = True,
        # Modular Environmental Feature Flags
        use_vmax: bool = True,
        use_mslp: bool = True,
        use_sst: bool = True,
        use_ohc: bool = True,
        use_shear: bool = True,
        use_rh: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.d_model = d_model
        self.use_vis_channel = use_vis_channel

        # Environmental Feature Toggles
        self.use_vmax = use_vmax
        self.use_mslp = use_mslp
        self.use_sst = use_sst
        self.use_ohc = use_ohc
        self.use_shear = use_shear
        self.use_rh = use_rh

        # 1. Shared Spatial CNN Encoder (ResNet-18)
        self.cnn = CNNFeatureEncoder(in_channels=channels, pretrained=True)

        # 2. Linear projection from CNN features (512) to d_model (256)
        self.feature_proj = nn.Linear(512, d_model)

        # 3. Explicit VIS Validity Mask Embedding (Learned Gating)
        self.vis_gate = nn.Linear(1, d_model) if use_vis_channel else None

        # 4. Sinusoidal Positional Encoding & Temporal Transformer
        self.pos_encoder = PositionalEncoding(d_model, max_len=10)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Environmental Branch (12-dim input -> 64-dim embedding)
        self.env_encoder = EnvironmentalEncoder(in_dim=12, out_dim=64, dropout=dropout)

        # 6. Multi-Modal Fusion Layer (256-d vis + 64-d env -> 256-d fused)
        self.fusion = MultiModalFusionLayer(d_vis=d_model, d_env=64, d_fused=d_model, dropout=dropout)

        # 7. Primary Task Head: Rapid Intensification (RI in next 24h)
        self.head_ri = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # 8. Secondary Task Head: 24-Hour Intensity Trend Classification (3 classes)
        self.head_trend = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

        # 9. Supporting Auxiliary Forecast Head: Quantitative Vmax (+6h, +12h, +24h)
        self.head_reg = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def extract_visual_features(self, x: torch.Tensor, vis_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract shared 256-d temporal visual representations from 5-frame sequence."""
        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)
        cnn_feats = self.cnn(x_flat)
        cnn_feats = cnn_feats.view(B, K, 512)

        # Visible validity embedding
        if self.vis_gate is not None and vis_masks is not None:
            vis_emb = self.vis_gate(vis_masks.unsqueeze(-1))
        else:
            vis_emb = 0.0

        tokens = self.feature_proj(cnn_feats) + vis_emb
        tokens = self.pos_encoder(tokens)
        temporal_out = self.transformer_encoder(tokens)
        return temporal_out[:, -1, :]  # (B, d_model)

    def apply_feature_masks(self, x_env: torch.Tensor) -> torch.Tensor:
        """Zero-out disabled features based on modular configuration flags."""
        masked_env = x_env.clone()
        # Indices: 0: vmax, 1: mslp, 2: sst, 3: cohc, 4: shrd, 5: rhmd
        # Corresponding missing mask indices: 6..11
        if not self.use_vmax:
            masked_env[:, 0] = 0.0
            masked_env[:, 6] = 1.0
        if not self.use_mslp:
            masked_env[:, 1] = 0.0
            masked_env[:, 7] = 1.0
        if not self.use_sst:
            masked_env[:, 2] = 0.0
            masked_env[:, 8] = 1.0
        if not self.use_ohc:
            masked_env[:, 3] = 0.0
            masked_env[:, 9] = 1.0
        if not self.use_shear:
            masked_env[:, 4] = 0.0
            masked_env[:, 10] = 1.0
        if not self.use_rh:
            masked_env[:, 5] = 0.0
            masked_env[:, 11] = 1.0
        return masked_env

    def forward(
        self,
        x: torch.Tensor,
        vis_masks: Optional[torch.Tensor] = None,
        x_env: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, K, C, H, W) satellite sequence
            vis_masks: (B, K) daytime/nighttime flags
            x_env: (B, 12) normalized environmental features + missing masks
        Returns:
            ri_logits: (B, 1) raw logit for Rapid Intensification
            trend_logits: (B, 3) raw logits for [WEAKENING, STABLE, INTENSIFYING]
            reg_preds: (B, 3) continuous intensity predictions for [+6h, +12h, +24h]
        """
        h_vis = self.extract_visual_features(x, vis_masks)

        if x_env is not None:
            gated_env = self.apply_feature_masks(x_env)
            h_env = self.env_encoder(gated_env)
            h_fused = self.fusion(h_vis, h_env)
        else:
            h_fused = h_vis

        ri_logits = self.head_ri(h_fused)
        trend_logits = self.head_trend(h_fused)
        reg_preds = self.head_reg(h_fused)

        return ri_logits, trend_logits, reg_preds

    def load_pretrained_backbone(self, checkpoint_path: str):
        """Warm-start weights from existing satellite-only classifier checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Filter out keys that match
        model_dict = self.state_dict()
        loaded_keys = []
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k] = v
                loaded_keys.append(k)

        self.load_state_dict(model_dict)
        print(f"Warm-started {len(loaded_keys)} tensor weights from {checkpoint_path}")
