"""Unit tests for ResNet single-channel regression model."""
import pytest
import torch
from src.models.resnet import CycloneResNet


def test_resnet18_single_channel_forward():
    """Verify input shape (B, 1, 224, 224) produces output shape (B, 1)."""
    model = CycloneResNet(architecture="resnet18", in_channels=1, pretrained=False)
    dummy_input = torch.randn(4, 1, 224, 224)
    output = model(dummy_input)

    assert output.shape == (4, 1), f"Expected shape (4, 1), got {output.shape}"
    assert not torch.isnan(output).any(), "Model produced NaN output"


def test_resnet18_backward_pass():
    """Verify backward gradient computation."""
    model = CycloneResNet(architecture="resnet18", in_channels=1, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    dummy_input = torch.randn(2, 1, 224, 224)
    dummy_target = torch.tensor([[45.0], [85.0]])

    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0.0
