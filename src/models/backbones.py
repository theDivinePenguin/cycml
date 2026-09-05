"""Spatial backbones (ResNet and ConvNeXt) with principled multi-spectral channel adaptation."""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
)


class SpatialBackbone(nn.Module):
    """Unified spatial feature extractor supporting ResNet18/34/50 and ConvNeXt-Tiny/Small
    with variance-preserving input channel adaptation.
    """

    def __init__(
        self,
        architecture: str = "resnet18",
        in_channels: int = 3,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.architecture = architecture.lower()
        self.in_channels = in_channels

        if self.architecture == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            self.out_dim = backbone.fc.in_features  # 512
            backbone.fc = nn.Identity()
            self._adapt_resnet_conv1(backbone, in_channels, pretrained)
            self.encoder = backbone

        elif self.architecture == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
            self.out_dim = backbone.fc.in_features  # 512
            backbone.fc = nn.Identity()
            self._adapt_resnet_conv1(backbone, in_channels, pretrained)
            self.encoder = backbone

        elif self.architecture == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            self.out_dim = backbone.fc.in_features  # 2048
            backbone.fc = nn.Identity()
            self._adapt_resnet_conv1(backbone, in_channels, pretrained)
            self.encoder = backbone

        elif self.architecture == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            self.out_dim = backbone.classifier[2].in_features  # 768
            backbone.classifier[2] = nn.Identity()
            self._adapt_convnext_stem(backbone, in_channels, pretrained)
            self.encoder = backbone

        elif self.architecture == "convnext_small":
            weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_small(weights=weights)
            self.out_dim = backbone.classifier[2].in_features  # 768
            backbone.classifier[2] = nn.Identity()
            self._adapt_convnext_stem(backbone, in_channels, pretrained)
            self.encoder = backbone

        else:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. "
                f"Supported: resnet18, resnet34, resnet50, convnext_tiny, convnext_small."
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _adapt_resnet_conv1(self, backbone: nn.Module, in_channels: int, pretrained: bool):
        orig_conv = backbone.conv1
        if in_channels == 3:
            return

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
                orig_weight = orig_conv.weight.data  # (64, 3, 7, 7)
                if in_channels == 1:
                    new_conv.weight.data = orig_weight.mean(dim=1, keepdim=True)
                elif in_channels == 2:
                    new_conv.weight.data = orig_weight[:, :2, :, :] * (3.0 / 2.0)
                elif in_channels == 4:
                    new_conv.weight.data[:, 0:3, :, :] = orig_weight * (3.0 / 4.0)
                    new_conv.weight.data[:, 3:4, :, :] = orig_weight.mean(dim=1, keepdim=True) * (3.0 / 4.0)
                else:
                    scale = 3.0 / in_channels
                    new_conv.weight.data[:, :min(3, in_channels), :, :] = orig_weight[:, :min(3, in_channels), :, :] * scale
                    if in_channels > 3:
                        mean_w = orig_weight.mean(dim=1, keepdim=True)
                        new_conv.weight.data[:, 3:, :, :] = mean_w.repeat(1, in_channels - 3, 1, 1) * scale
        backbone.conv1 = new_conv

    def _adapt_convnext_stem(self, backbone: nn.Module, in_channels: int, pretrained: bool):
        orig_conv = backbone.features[0][0]  # stem convolution: (out_channels, 3, 4, 4)
        if in_channels == 3:
            return

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                orig_weight = orig_conv.weight.data
                if in_channels == 1:
                    new_conv.weight.data = orig_weight.mean(dim=1, keepdim=True)
                elif in_channels == 2:
                    new_conv.weight.data = orig_weight[:, :2, :, :] * (3.0 / 2.0)
                elif in_channels == 4:
                    new_conv.weight.data[:, 0:3, :, :] = orig_weight * (3.0 / 4.0)
                    new_conv.weight.data[:, 3:4, :, :] = orig_weight.mean(dim=1, keepdim=True) * (3.0 / 4.0)
                else:
                    scale = 3.0 / in_channels
                    new_conv.weight.data[:, :min(3, in_channels), :, :] = orig_weight[:, :min(3, in_channels), :, :] * scale
                    if in_channels > 3:
                        mean_w = orig_weight.mean(dim=1, keepdim=True)
                        new_conv.weight.data[:, 3:, :, :] = mean_w.repeat(1, in_channels - 3, 1, 1) * scale
                if orig_conv.bias is not None:
                    new_conv.bias.data = orig_conv.bias.data
        backbone.features[0][0] = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Embedding of shape (B, out_dim)
        """
        feats = self.encoder(x)
        return self.dropout(feats)
