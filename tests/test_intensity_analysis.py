"""Unit tests for intensity binning, boundary handling, and distribution metrics."""
import json
import numpy as np
import pandas as pd
import pytest

from src.evaluation.intensity_bins import (
    INTENSITY_BINS,
    REGIME_BINS,
    assign_intensity_bin,
    compute_binned_distribution
)


def test_intensity_bin_boundaries():
    """Verify that boundary values are assigned to the correct intervals [lower, upper)."""
    assert assign_intensity_bin(15.0) == "15–30 kt"
    assert assign_intensity_bin(29.99) == "15–30 kt"
    assert assign_intensity_bin(30.0) == "30–50 kt"
    assert assign_intensity_bin(49.99) == "30–50 kt"
    assert assign_intensity_bin(50.0) == "50–70 kt"
    assert assign_intensity_bin(70.0) == "70–90 kt"
    assert assign_intensity_bin(90.0) == "90–110 kt"
    assert assign_intensity_bin(110.0) == "110–130 kt"
    assert assign_intensity_bin(130.0) == "130–150 kt"
    assert assign_intensity_bin(150.0) == "> 150 kt"
    assert assign_intensity_bin(175.0) == "> 150 kt"


def test_regime_bin_boundaries():
    """Verify regime bins for cross-basin comparison."""
    assert assign_intensity_bin(25.0, REGIME_BINS) == "<60 kt"
    assert assign_intensity_bin(59.9, REGIME_BINS) == "<60 kt"
    assert assign_intensity_bin(60.0, REGIME_BINS) == "60–100 kt"
    assert assign_intensity_bin(100.0, REGIME_BINS) == "100–130 kt"
    assert assign_intensity_bin(130.0, REGIME_BINS) == "> 130 kt"
    assert assign_intensity_bin(165.0, REGIME_BINS) == "> 130 kt"


def test_binned_distribution_percentage_sum():
    """Verify that binned frame percentages sum to exactly 100% and count matches."""
    sample_winds = [20.0, 35.0, 45.0, 65.0, 85.0, 105.0, 125.0, 140.0, 160.0]
    df = pd.DataFrame({
        "wind_speed": sample_winds,
        "cyclone_id": [f"storm_{i}" for i in range(len(sample_winds))]
    })

    binned = compute_binned_distribution(df, intensity_col="wind_speed", cyclone_id_col="cyclone_id")

    total_frames = sum(item["frames"] for item in binned)
    total_pct = sum(item["percent_frames"] for item in binned)

    assert total_frames == len(sample_winds)
    assert abs(total_pct - 100.0) < 0.1


def test_training_split_isolation():
    """Verify that training distribution analysis uses strictly disjoint training indices."""
    splits_path = "data/metadata/splits_CPAC_IO_SH.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_cids = set(splits["train"]["cyclone_ids"])
    val_cids = set(splits["val"]["cyclone_ids"])
    test_cids = set(splits["test"]["cyclone_ids"])

    assert train_cids.isdisjoint(val_cids), "Train and Val cyclone sets must be disjoint"
    assert train_cids.isdisjoint(test_cids), "Train and Test cyclone sets must be disjoint"
    assert val_cids.isdisjoint(test_cids), "Val and Test cyclone sets must be disjoint"
