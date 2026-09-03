"""Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN regression models."""
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionGradCAM:
    """Grad-CAM implementation tailored for continuous regression backbones."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles.extend([h1, h2])

    def generate_cam(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Generate Grad-CAM heatmap for input tensor.

        Args:
            input_tensor: Shape (1, C, H, W) on model's device.

        Returns:
            2D numpy array heatmap normalized to [0, 1] with shape (H, W).
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        target_score = output[0, 0]

        # Backward pass targeting predicted intensity score
        target_score.backward(retain_graph=True)

        # Global average pooling of gradients: weights alpha_k
        # gradients shape: (1, K, H', W')
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)  # Only consider positive influence

        # Upsample to match original image resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def plot_gradcam_explanation(
    image_np: np.ndarray,
    cam_heatmap: np.ndarray,
    predicted_kt: float,
    ground_truth_kt: Optional[float],
    storm_name: str,
    save_path: str | Path,
    alpha: float = 0.5
) -> None:
    """Plot original satellite image, Grad-CAM heatmap, and overlay side-by-side."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # 1. Original Satellite Image
    im0 = axes[0].imshow(image_np, cmap="inferno")
    axes[0].set_title(f"Satellite IR1 (Brightness Temp)\n{storm_name}", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Kelvin")

    # 2. Grad-CAM Activation Heatmap
    im1 = axes[1].imshow(cam_heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Attention Map\n(Regions Influencing Intensity)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Importance")

    # 3. Superimposed Overlay
    axes[2].imshow(image_np, cmap="gray")
    im2 = axes[2].imshow(cam_heatmap, cmap="jet", alpha=alpha)
    title_str = f"Superimposed Eye/Eyewall Attention\nPredicted: {predicted_kt:.1f} kt"
    if ground_truth_kt is not None:
        title_str += f" | Actual: {ground_truth_kt:.1f} kt (Err: {predicted_kt - ground_truth_kt:+.1f} kt)"
    axes[2].set_title(title_str, fontsize=11, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Activation")

    plt.tight_layout()
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"[Grad-CAM] Saved explanation visual to: {p}")
