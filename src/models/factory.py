"""Model factory for building regression architectures."""
from typing import Dict, Any
import torch
import torch.nn as nn
from src.models.resnet import CycloneResNet


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Build and initialize model from configuration dictionary.

    Args:
        config: Configuration dictionary.

    Returns:
        Instantiated PyTorch neural network model.
    """
    model_cfg = config.get("model", {})
    ds_cfg = config.get("dataset", {})
    architecture = model_cfg.get("architecture", "resnet18")
    
    # Infer in_channels from model_cfg or dataset_cfg
    if "in_channels" in model_cfg:
        in_channels = int(model_cfg["in_channels"])
    elif "channels" in ds_cfg:
        channels_spec = ds_cfg["channels"]
        in_channels = len(channels_spec) if isinstance(channels_spec, (list, tuple)) else 1
    else:
        in_channels = 1
    pretrained = model_cfg.get("pretrained", True)
    dropout = model_cfg.get("dropout", 0.2)

    model = CycloneResNet(
        architecture=architecture,
        in_channels=in_channels,
        pretrained=pretrained,
        dropout=dropout
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[Model Factory] Built {architecture.upper()} (in_channels={in_channels}, pretrained={pretrained}):")
    print(f"  • Total Parameters:     {total_params:,}")
    print(f"  • Trainable Parameters: {trainable_params:,}")

    return model
