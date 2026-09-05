"""Multi-modal fusion architectures for satellite visual representations and environmental predictors."""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LateConcatFusion(nn.Module):
    """Simple linear projection over concatenated visual and environmental embeddings."""

    def __init__(self, d_vis: int, d_env: int, d_fused: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_vis + d_env, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h_vis: torch.Tensor, h_env: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_vis, h_env], dim=-1)
        return self.proj(combined)


class GatedResidualFusion(nn.Module):
    """Gated projection with residual connection from visual representation."""

    def __init__(self, d_vis: int, d_env: int, d_fused: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_vis + d_env, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fused, d_fused),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_vis + d_env, d_fused),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_fused)

    def forward(self, h_vis: torch.Tensor, h_env: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_vis, h_env], dim=-1)
        fused = self.proj(combined)
        g = self.gate(combined)
        # Gated modulation of environmental branch + residual visual shortcut
        return self.norm(h_vis + g * fused)


class CrossAttentionFusion(nn.Module):
    """Bidirectional / Query-Key-Value Cross-Attention between visual tokens and environmental representations."""

    def __init__(self, d_vis: int, d_env: int, d_fused: int, nheads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Project environmental embedding to match visual dimension
        self.env_proj = nn.Linear(d_env, d_vis)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_vis, num_heads=nheads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_vis)
        self.norm2 = nn.LayerNorm(d_fused)
        self.ffn = nn.Sequential(
            nn.Linear(d_vis, d_fused),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fused, d_fused),
        )

    def forward(self, h_vis: torch.Tensor, h_env: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_vis: (B, d_vis) or (B, K, d_vis)
            h_env: (B, d_env)
        Returns:
            fused: (B, d_fused)
        """
        # Ensure 3D shapes: (B, seq_len, dim)
        if h_vis.ndim == 2:
            q = h_vis.unsqueeze(1)  # (B, 1, d_vis)
        else:
            q = h_vis

        env_k = self.env_proj(h_env).unsqueeze(1)  # (B, 1, d_vis)

        attn_out, _ = self.cross_attn(query=q, key=env_k, value=env_k)
        x = self.norm1(q + attn_out)
        out = self.norm2(self.ffn(x))

        if out.shape[1] == 1:
            return out.squeeze(1)
        return out[:, -1, :]  # pool final timestep


def build_fusion_layer(
    fusion_type: str,
    d_vis: int,
    d_env: int,
    d_fused: int,
    dropout: float = 0.1,
) -> nn.Module:
    """Factory creating configured multi-modal fusion layer."""
    ft = fusion_type.lower()
    if ft in ["late", "concat", "late_concat"]:
        return LateConcatFusion(d_vis, d_env, d_fused, dropout)
    elif ft in ["gated", "gated_residual"]:
        return GatedResidualFusion(d_vis, d_env, d_fused, dropout)
    elif ft in ["cross_attention", "cross_attn", "attention"]:
        return CrossAttentionFusion(d_vis, d_env, d_fused, nheads=4, dropout=dropout)
    else:
        raise ValueError(f"Unknown fusion type '{fusion_type}'. Options: 'late', 'gated', 'cross_attention'.")
