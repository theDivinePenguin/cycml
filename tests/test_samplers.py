"""Unit tests for intensity-aware sampler."""
import numpy as np
import pandas as pd
import torch

from src.data.samplers import compute_intensity_sampling_weights, build_intensity_aware_sampler


def test_intensity_sampling_weights():
    # Create synthetic dataset with heavy skew (many 25 kt, few 140 kt)
    winds = [25.0] * 100 + [45.0] * 50 + [80.0] * 20 + [120.0] * 5 + [145.0] * 2
    df = pd.DataFrame({"wind_speed": winds})

    weights, diagnostics = compute_intensity_sampling_weights(df, alpha=0.5)

    assert len(weights) == len(df)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)

    # Extreme bin (130-150 kt with N=2) should have higher per-sample weight than dense bin (15-30 kt with N=100)
    idx_rare = len(df) - 1
    idx_dense = 0
    assert weights[idx_rare] > weights[idx_dense]

    # Effective probability of rare bin should be boosted relative to its natural percentage
    assert diagnostics["130–150 kt"]["effective_sampling_pct"] > diagnostics["130–150 kt"]["natural_pct"]
    assert diagnostics["15–30 kt"]["effective_sampling_pct"] < diagnostics["15–30 kt"]["natural_pct"]


def test_intensity_aware_sampler_deterministic():
    winds = [25.0] * 50 + [60.0] * 30 + [115.0] * 10
    df = pd.DataFrame({"wind_speed": winds})

    sampler1, _ = build_intensity_aware_sampler(df, alpha=0.5, seed=123)
    sampler2, _ = build_intensity_aware_sampler(df, alpha=0.5, seed=123)

    indices1 = list(iter(sampler1))
    indices2 = list(iter(sampler2))

    assert len(indices1) == len(df)
    assert indices1 == indices2
