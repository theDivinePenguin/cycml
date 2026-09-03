"""Comprehensive unit tests for TCIR Multi-Channel Satellite Pipeline.

Validates:
1. Dataset tensor output shapes for single-channel and multi-channel modes.
2. Channel ordering and value ranges.
3. Missing-value and fill-value imputation without residual NaNs/Infs.
4. Normalization tensor dimensions and numerical bounds.
5. Target label equivalence across single and multi-channel modes.
6. ResNet18 multi-channel forward pass and backward pass gradient flow.
7. Principled pretrained conv1 weight initialization.
8. Grouped cyclone split integrity and zero-leakage verification.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.data.dataset import TCIRDataset, build_dataloaders
from src.data.preprocessing import TCIRPreprocessor
from src.models.factory import build_model
from src.models.resnet import CycloneResNet
from src.utils.config import load_config


@pytest.fixture
def mock_metadata_df():
    """Create realistic mock metadata DataFrame."""
    return pd.DataFrame({
        "sample_index": [0, 1, 2, 3],
        "h5_row_index": [0, 1, 2, 3],
        "cyclone_id": ["TEST01", "TEST01", "TEST02", "TEST02"],
        "timestamp": ["2020010100", "2020010106", "2020010200", "2020010206"],
        "wind_speed": [45.0, 65.0, 90.0, 120.0],
        "region": ["IO", "IO", "WPAC", "WPAC"]
    })


def test_tcir_preprocessor_single_and_multichannel():
    """Test TCIRPreprocessor handles 1-channel and 4-channel tensors cleanly."""
    # 1-channel preprocessor
    prep1 = TCIRPreprocessor(mean=267.8, std=27.0, channels=[0])
    raw1 = torch.full((1, 201, 201), 267.8, dtype=torch.float32)
    raw1[0, 10, 10] = float("nan")
    out1 = prep1(raw1)
    
    assert out1.shape == (1, 224, 224)
    assert not torch.isnan(out1).any()
    assert not torch.isinf(out1).any()
    assert abs(out1.mean().item()) < 0.1

    # 4-channel preprocessor with fill values and NaNs
    means = [267.8, 236.1, 0.30, 0.48]
    stds = [27.0, 11.9, 0.61, 1.47]
    prep4 = TCIRPreprocessor(mean=means, std=stds, channels=[0, 1, 2, 3])
    
    raw4 = torch.zeros((4, 201, 201), dtype=torch.float32)
    raw4[0] = 267.8 # IR1
    raw4[1] = 236.1 # WV
    raw4[2] = float("nan") # VIS nighttime NaN
    raw4[3] = 9.96921e36 # PMW NetCDF fill value
    
    out4 = prep4(raw4)
    assert out4.shape == (4, 224, 224)
    assert not torch.isnan(out4).any(), "Residual NaNs found after preprocessing"
    assert not torch.isinf(out4).any(), "Residual Infs found after preprocessing"
    assert out4[3].abs().max() < 10.0, "PMW fill value not cleaned properly"


def test_dataset_channel_shapes_and_targets(mock_metadata_df):
    """Verify TCIRDataset output shapes and target equivalence."""
    h5_path = "data/raw/TCIR-CPAC_IO_SH.h5"
    if not Path(h5_path).exists():
        pytest.skip(f"HDF5 file {h5_path} not found")

    # Single-channel dataset
    ds_single = TCIRDataset(h5_path=h5_path, metadata_df=mock_metadata_df, channels=[0])
    img1, target1, meta1 = ds_single[0]
    assert img1.shape == (1, 224, 224)
    assert target1.item() == 45.0
    assert meta1["cyclone_id"] == "TEST01"

    # 4-channel dataset
    ds_multi = TCIRDataset(h5_path=h5_path, metadata_df=mock_metadata_df, channels=[0, 1, 2, 3])
    img4, target4, meta4 = ds_multi[0]
    assert img4.shape == (4, 224, 224)
    assert target4.item() == 45.0
    assert meta4["cyclone_id"] == "TEST01"

    # Target equivalence
    assert torch.equal(target1, target4)
    assert meta1 == meta4


def test_multichannel_resnet18_forward_and_backward():
    """Verify ResNet18 forward pass, backward pass, and parameter gradient flow for 4 channels."""
    model = CycloneResNet(architecture="resnet18", in_channels=4, pretrained=True, dropout=0.2)
    model.train()
    
    batch_size = 4
    x = torch.randn(batch_size, 4, 224, 224, requires_grad=True)
    target = torch.tensor([[45.0], [75.0], [105.0], [135.0]], dtype=torch.float32)
    
    # Forward
    output = model(x)
    assert output.shape == (batch_size, 1)
    assert not torch.isnan(output).any()

    # Loss and Backward
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    loss.backward()

    # Verify input gradients and conv1 weight gradients
    assert x.grad is not None
    assert model.conv1.weight.grad is not None
    assert not torch.isnan(model.conv1.weight.grad).any()
    assert (model.conv1.weight.grad.abs() > 0).any()


def test_principled_weight_initialization():
    """Verify principled conv1 weight initialization for 4-channel ResNet18."""
    model_4ch = CycloneResNet(architecture="resnet18", in_channels=4, pretrained=True)
    w = model_4ch.conv1.weight.data
    
    assert w.shape == (64, 4, 7, 7)
    assert not torch.isnan(w).any()
    assert not torch.isinf(w).any()
    
    # Check that channel 3 is initialized from the mean of channels 0, 1, 2
    # Before scaling, ch3 = mean(ch0, ch1, ch2)
    mean_ch012 = w[:, 0:3, :, :].mean(dim=1)
    ch3 = w[:, 3, :, :]
    
    # Both are scaled by 3/4, so ch3 should match the mean of channels 0..2
    diff = (ch3 - mean_ch012).abs().max().item()
    assert diff < 1e-5, f"Channel 3 initialization deviates from mean of channels 0..2 by {diff}"


def test_splits_zero_leakage():
    """Verify zero cyclone overlap across splits_all_basins.json."""
    splits_path = Path("data/metadata/splits_all_basins.json")
    if not splits_path.exists():
        pytest.skip(f"{splits_path} not found")
        
    with open(splits_path, "r") as f:
        splits = json.load(f)
        
    train_cids = set(splits["train"]["cyclone_ids"])
    val_cids = set(splits["val"]["cyclone_ids"])
    test_cids = set(splits["test"]["cyclone_ids"])
    
    assert len(train_cids & val_cids) == 0, "Data leakage: Train and Val share cyclones!"
    assert len(train_cids & test_cids) == 0, "Data leakage: Train and Test share cyclones!"
    assert len(val_cids & test_cids) == 0, "Data leakage: Val and Test share cyclones!"


def test_normalization_stats_integrity():
    """Verify data/metadata/normalization_stats_multichannel.json has valid values for all 4 channels."""
    stats_path = Path("data/metadata/normalization_stats_multichannel.json")
    if not stats_path.exists():
        pytest.skip(f"{stats_path} not found")
        
    with open(stats_path, "r") as f:
        stats = json.load(f)
        
    assert "mean" in stats and "std" in stats
    assert len(stats["mean"]) == 4
    assert len(stats["std"]) == 4
    for c in range(4):
        assert stats["std"][c] > 0.01, f"Std for channel {c} is too small: {stats['std'][c]}"
        assert not np.isnan(stats["mean"][c])
        assert not np.isnan(stats["std"][c])
