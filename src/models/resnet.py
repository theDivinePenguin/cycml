"""ResNet architecture modified for single-channel satellite cyclone intensity regression."""
from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


class CycloneResNet(nn.Module):
    """ResNet backbone adapted for 1-channel satellite imagery regression."""

    def __init__(
        self,
        architecture: str = "resnet18",
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        """
        Args:
            architecture: "resnet18" or "resnet50".
            in_channels: Number of input channels (1 for IR1).
            pretrained: Whether to load ImageNet pretrained weights.
            dropout: Dropout probability before the final regression head.
        """
        super().__init__()
        self.architecture = architecture.lower()
        self.in_channels = in_channels

        # Load backbone
        if self.architecture == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base_model = resnet18(weights=weights)
            in_features = base_model.fc.in_features  # 512
        elif self.architecture == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base_model = resnet50(weights=weights)
            in_features = base_model.fc.in_features  # 2048
        else:
            raise ValueError(f"Unsupported architecture '{architecture}'. Use 'resnet18' or 'resnet50'.")

        # Adapt first convolution for single-channel or multi-channel count
        orig_conv1 = base_model.conv1
        if in_channels != 3:
            new_conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=orig_conv1.out_channels,
                kernel_size=orig_conv1.kernel_size,
                stride=orig_conv1.stride,
                padding=orig_conv1.padding,
                bias=orig_conv1.bias is not None
            )

            # Principled weight transfer preserving activation variance
            if pretrained:
                with torch.no_grad():
                    if in_channels == 1:
                        # Single-channel: channel-wise mean of pretrained RGB filters
                        new_conv1.weight.data = orig_conv1.weight.data.mean(dim=1, keepdim=True)
                    elif in_channels == 2:
                        # 2 channels: first two RGB filters scaled by 3/2
                        new_conv1.weight.data = orig_conv1.weight.data[:, :2, :, :] * (3.0 / 2.0)
                    elif in_channels == 4:
                        # 4 channels: Channels 0, 1, 2 receive ImageNet RGB filters;
                        # Channel 3 (PMW) is initialized with the spatial channel-mean of RGB filters.
                        # Total activation energy is normalized by factor (3/4)
                        new_conv1.weight.data[:, 0:3, :, :] = orig_conv1.weight.data * (3.0 / 4.0)
                        new_conv1.weight.data[:, 3:4, :, :] = orig_conv1.weight.data.mean(dim=1, keepdim=True) * (3.0 / 4.0)
                    else:
                        # General C-channel fallback
                        scale = 3.0 / in_channels
                        new_conv1.weight.data[:, :min(3, in_channels), :, :] = orig_conv1.weight.data[:, :min(3, in_channels), :, :] * scale
                        if in_channels > 3:
                            rgb_mean = orig_conv1.weight.data.mean(dim=1, keepdim=True)
                            new_conv1.weight.data[:, 3:, :, :] = rgb_mean.repeat(1, in_channels - 3, 1, 1) * scale

            base_model.conv1 = new_conv1

        # Extract features up to global average pooling
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        # Regression Head
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Predicted wind speed tensor of shape (B, 1).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        output = self.fc(x)

        return output
