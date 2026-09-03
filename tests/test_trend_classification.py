"""Unit test suite for Cyclone Intensity Trend and Rapid Intensification prediction."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.trend_config import IntensityTrendConfig
from src.evaluation.baselines import PersistenceBaseline, RecentTrendBaseline
from src.evaluation.classification_metrics import (
    compute_expected_calibration_error,
    compute_ri_metrics,
    compute_trend_metrics,
    find_optimal_threshold,
)
from src.models.temporal_classifier import JointTrendRILoss, TemporalClassifier


def test_trend_config_thresholds_and_labels():
    """Verify configurable threshold logic for 24h intensity trend and RI."""
    cfg = IntensityTrendConfig(weakening_threshold_kt=-10.0, intensifying_threshold_kt=10.0, ri_threshold_kt=30.0)

    # Scalar checks
    assert cfg.compute_trend_label(-20.0) == 0  # WEAKENING
    assert cfg.compute_trend_label(-10.0) == 0  # WEAKENING boundary
    assert cfg.compute_trend_label(0.0) == 1    # STABLE
    assert cfg.compute_trend_label(5.0) == 1    # STABLE
    assert cfg.compute_trend_label(10.0) == 2   # INTENSIFYING boundary
    assert cfg.compute_trend_label(35.0) == 2   # INTENSIFYING

    assert cfg.compute_ri_label(29.0) == 0      # Non-RI
    assert cfg.compute_ri_label(30.0) == 1      # RI boundary
    assert cfg.compute_ri_label(45.0) == 1      # RI

    # Array checks
    deltas = np.array([-15.0, -5.0, 0.0, 15.0, 35.0])
    trends = cfg.compute_trend_label(deltas)
    np.testing.assert_array_equal(trends, [0, 1, 1, 2, 2])

    ris = cfg.compute_ri_label(deltas)
    np.testing.assert_array_equal(ris, [0, 0, 0, 0, 1])

    # Risk level mapping
    assert cfg.get_ri_risk_level(0.10) == "LOW"
    assert cfg.get_ri_risk_level(0.40) == "MEDIUM"
    assert cfg.get_ri_risk_level(0.78) == "HIGH"


def test_temporal_classifier_forward_backward():
    """Verify TemporalClassifier outputs correct shapes and computes backward gradients."""
    torch.manual_seed(42)
    model = TemporalClassifier(
        in_channels=3,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.0,
        pretrained_cnn=False,
    )

    B, K, C, H, W = 2, 5, 3, 64, 64
    x = torch.randn(B, K, C, H, W)
    vis_masks = torch.ones(B, K)

    ri_logits, trend_logits, reg_preds = model(x, vis_masks)

    # Shape checks
    assert ri_logits.shape == (B, 1), f"Expected (2, 1), got {ri_logits.shape}"
    assert trend_logits.shape == (B, 3), f"Expected (2, 3), got {trend_logits.shape}"
    assert reg_preds.shape == (B, 3), f"Expected (2, 3), got {reg_preds.shape}"

    # Probability checks
    ri_prob, trend_probs, reg_preds_prob = model.predict_probabilities(x, vis_masks)
    assert ri_prob.shape == (B,)
    assert torch.all((ri_prob >= 0.0) & (ri_prob <= 1.0))
    assert trend_probs.shape == (B, 3)
    np.testing.assert_allclose(trend_probs.sum(dim=-1).detach().numpy(), np.ones(B), rtol=1e-5)

    # Loss and backward pass
    loss_fn = JointTrendRILoss(
        ri_pos_weight=torch.tensor([2.0]),
        trend_class_weights=torch.tensor([1.0, 1.0, 1.0]),
        lambda_ri=1.0,
        lambda_trend=1.0,
        lambda_reg=0.1,
    )

    ri_targets = torch.tensor([1.0, 0.0])
    trend_targets = torch.tensor([2, 1])
    reg_targets = torch.tensor([[45.0, 50.0, 60.0], [70.0, 75.0, 80.0]])

    total_loss, loss_dict = loss_fn(
        ri_logits, trend_logits, reg_preds, ri_targets, trend_targets, reg_targets
    )

    assert total_loss.dim() == 0
    assert total_loss.item() > 0
    assert "loss_ri" in loss_dict and "loss_trend" in loss_dict and "loss_reg" in loss_dict

    total_loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_classification_metrics_and_calibration():
    """Verify metrics calculation on synthetic test cases."""
    # Synthetic ground truth
    y_true_trend = np.array([0, 0, 1, 1, 2, 2])
    y_pred_trend = np.array([0, 1, 1, 1, 2, 0])
    t_metrics = compute_trend_metrics(y_true_trend, y_pred_trend)

    assert 0.0 <= t_metrics["accuracy"] <= 1.0
    assert 0.0 <= t_metrics["macro_f1"] <= 1.0
    assert len(t_metrics["confusion_matrix"]) == 3

    # RI metrics
    y_true_ri = np.array([0, 0, 0, 1, 1, 1])
    y_prob_ri = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    r_metrics = compute_ri_metrics(y_true_ri, y_prob_ri, threshold=0.5)

    assert r_metrics["roc_auc"] == 1.0
    assert r_metrics["pr_auc"] == 1.0
    assert r_metrics["precision_at_0.50"] == 1.0
    assert r_metrics["recall_at_0.50"] == 1.0

    # Calibration error
    ece, accs, confs, counts = compute_expected_calibration_error(y_true_ri, y_prob_ri, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert len(counts) == 5

    # Optimal threshold finder
    best_th, best_f1, p_th, r_th = find_optimal_threshold(y_true_ri, y_prob_ri)
    assert 0.0 <= best_th <= 1.0
    assert best_f1 == 1.0


def test_baselines_prediction():
    """Verify Baseline A (Persistence) and Baseline B (Recent Trend) produce expected formats."""
    dummy_df = pd.DataFrame({
        "cyclone_id": ["201003I", "201003I"],
        "history_vmax": [
            "[20.0, 20.0, 25.0, 30.0, 35.0]",
            "[50.0, 50.0, 50.0, 50.0, 50.0]",
        ],
        "vmax_curr": [35.0, 50.0],
        "vmax_plus_24h": [65.0, 50.0],
    })

    # Baseline A: Persistence
    base_a = PersistenceBaseline()
    t_pred_a, t_prob_a, ri_prob_a = base_a.predict(dummy_df)
    assert np.all(t_pred_a == 1)  # All STABLE
    assert np.all(ri_prob_a == 0.0)  # All 0.0

    # Baseline B: Recent trend
    base_b = RecentTrendBaseline()
    t_pred_b, t_prob_b, ri_prob_b = base_b.predict(dummy_df)
    # First storm went from 25.0 (t-6h) to 35.0 (t) -> +10kt in 6h -> +40kt in 24h -> INTENSIFYING & RI!
    assert t_pred_b[0] == 2
    assert ri_prob_b[0] > 0.5
    # Second storm was constant 50.0 -> STABLE & non-RI
    assert t_pred_b[1] == 1
    assert ri_prob_b[1] < 0.1
