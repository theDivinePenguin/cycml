"""Loss functions for regression."""
from typing import Dict, Any
import torch
import torch.nn as nn


def build_loss_fn(config: Dict[str, Any]) -> nn.Module:
    """Build loss function based on configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        PyTorch loss module.
    """
    loss_type = config.get("training", {}).get("loss", "mse").lower()

    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type in ("huber", "smooth_l1"):
        delta = config.get("training", {}).get("huber_delta", 1.0)
        return nn.HuberLoss(delta=delta)
    elif loss_type == "l1":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss function '{loss_type}'. Use 'mse', 'huber', or 'l1'.")
