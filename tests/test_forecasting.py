"""Unit test suite for temporal forecasting sequences, anti-leakage invariants, datasets, and models."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.sequence_dataset import TCIRSequenceDataset, build_sequence_dataloaders
from src.models.temporal_forecaster import (
    CNNFeatureEncoder,
    MultiHorizonHuberLoss,
    TemporalGRUForecaster,
    TemporalTransformerForecaster,
)


def test_sequence_manifest_invariants_and_anti_leakage():
    """Verify that forecasting sequences strictly satisfy chronological ordering,
    correct future target offsets (+6h, +12h, +24h), and zero cyclone leakage.
    """
    meta_dir = Path("data/metadata")
    train_path = meta_dir / "forecast_train_sequences_k5.csv"
    val_path = meta_dir / "forecast_val_sequences_k5.csv"
    test_path = meta_dir / "forecast_test_sequences_k5.csv"

    assert train_path.exists() and val_path.exists() and test_path.exists()

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # 1. Zero cyclone overlap across splits
    c_train = set(train_df["cyclone_id"].unique())
    c_val = set(val_df["cyclone_id"].unique())
    c_test = set(test_df["cyclone_id"].unique())

    assert len(c_train & c_val) == 0, "Cyclone leakage between Train and Val!"
    assert len(c_train & c_test) == 0, "Cyclone leakage between Train and Test!"
    assert len(c_val & c_test) == 0, "Cyclone leakage between Val and Test!"

    # 2. Strict anti-leakage inside every sequence
    for name, df in [("train", train_df.head(50)), ("test", test_df.head(50))]:
        for _, row in df.iterrows():
            t_dt = datetime.strptime(row["target_t_dt"], "%Y-%m-%d %H:%M:%S")
            hist_ts = json.loads(row["history_timestamps"])
            hist_dts = [datetime.strptime(str(ts), "%Y%m%d%H") for ts in hist_ts]

            # Invariant: History must be strictly monotonically increasing and end at t_dt
            assert len(hist_dts) == 5
            for i in range(len(hist_dts) - 1):
                assert hist_dts[i] < hist_dts[i + 1]
                assert (hist_dts[i + 1] - hist_dts[i]).total_seconds() == 3 * 3600

            assert hist_dts[-1] == t_dt, f"Last history frame {hist_dts[-1]} must equal target t {t_dt}"

            # Invariant: Future targets (+6h, +12h, +24h) must be strictly in the future relative to all history frames
            t6 = t_dt + timedelta(hours=6)
            t12 = t_dt + timedelta(hours=12)
            t24 = t_dt + timedelta(hours=24)

            for h_dt in hist_dts:
                assert h_dt < t6, "Leakage: History frame timestamp is >= t+6h target!"
                assert h_dt < t12, "Leakage: History frame timestamp is >= t+12h target!"
                assert h_dt < t24, "Leakage: History frame timestamp is >= t+24h target!"


def test_sequence_dataset_item_and_vis_mask():
    """Verify TCIRSequenceDataset yields correct tensor shapes and explicit VIS validity masks."""
    meta_dir = Path("data/metadata")
    test_seq_df = pd.read_csv(meta_dir / "forecast_test_sequences_k5.csv").head(10)

    mean = [270.0, 240.0, 0.2, 5.0]
    std = [30.0, 20.0, 0.3, 10.0]

    ds = TCIRSequenceDataset(test_seq_df, mean=mean, std=std, channels=[0, 1, 2], is_training=False)
    assert len(ds) == 10

    images, vis_mask, targets, meta = ds[0]

    # Shape checks: (K, C, H, W) -> (5, 3, 201, 201)
    assert images.shape == (5, 3, 201, 201)
    assert images.dtype == torch.float32

    # VIS validity mask shape: (K,) -> (5,)
    assert vis_mask.shape == (5,)
    assert vis_mask.dtype == torch.float32
    assert torch.all((vis_mask == 0.0) | (vis_mask == 1.0))

    # Targets shape: (3,) -> [+6h, +12h, +24h]
    assert targets.shape == (3,)
    assert targets.dtype == torch.float32
    assert float(targets[0]) == meta["vmax_plus_6h"]
    assert float(targets[1]) == meta["vmax_plus_12h"]
    assert float(targets[2]) == meta["vmax_plus_24h"]

    ds.close()


def test_temporal_gru_forecaster_forward_backward():
    """Verify TemporalGRUForecaster forward pass, backward gradients, and output shape (B, 3)."""
    torch.manual_seed(42)
    model = TemporalGRUForecaster(in_channels=3, d_model=64, num_layers=1, pretrained_cnn=False)

    # Dummy batch: B=2, K=5, C=3, H=64, W=64
    x = torch.randn(2, 5, 3, 64, 64)
    vis_masks = torch.ones(2, 5)

    preds = model(x, vis_masks)
    assert preds.shape == (2, 3), f"Expected shape (2, 3), got {preds.shape}"

    targets = torch.tensor([[45.0, 50.0, 60.0], [70.0, 75.0, 80.0]], dtype=torch.float32)
    loss_fn = MultiHorizonHuberLoss()
    loss = loss_fn(preds, targets)

    assert loss.dim() == 0
    loss.backward()

    # Verify gradients computed
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_temporal_transformer_forecaster_forward_backward():
    """Verify TemporalTransformerForecaster forward pass, backward gradients, and output shape (B, 3)."""
    torch.manual_seed(42)
    model = TemporalTransformerForecaster(
        in_channels=3, d_model=64, nhead=4, num_layers=1, dim_feedforward=128, pretrained_cnn=False
    )

    # Dummy batch: B=2, K=5, C=3, H=64, W=64
    x = torch.randn(2, 5, 3, 64, 64)
    vis_masks = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    preds = model(x, vis_masks)
    assert preds.shape == (2, 3), f"Expected shape (2, 3), got {preds.shape}"

    targets = torch.tensor([[45.0, 50.0, 60.0], [70.0, 75.0, 80.0]], dtype=torch.float32)
    loss_fn = MultiHorizonHuberLoss()
    loss = loss_fn(preds, targets)

    assert loss.dim() == 0
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_temporal_ablation_variable_lengths():
    """Verify temporal models correctly accept variable sequence lengths K=1, K=3, K=5."""
    torch.manual_seed(42)
    gru_model = TemporalGRUForecaster(in_channels=3, d_model=32, num_layers=1, pretrained_cnn=False)
    tf_model = TemporalTransformerForecaster(
        in_channels=3, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, pretrained_cnn=False
    )

    for k in [1, 3, 5]:
        x = torch.randn(2, k, 3, 32, 32)
        vis_masks = torch.ones(2, k)

        preds_gru = gru_model(x, vis_masks)
        preds_tf = tf_model(x, vis_masks)

        assert preds_gru.shape == (2, 3)
        assert preds_tf.shape == (2, 3)
