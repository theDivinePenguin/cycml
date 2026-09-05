"""Comprehensive unit test suite for A100 models, loss functions, baselines, and sanity checks."""
import numpy as np
import pytest
import torch

from src.evaluation.baselines import IntensityHistoryExtrapolator, PersistenceForecaster
from src.evaluation.sanity_checks import PhysicalSanityChecker
from src.inference.predict_api import StandardizedPredictor
from src.models.backbones import SpatialBackbone
from src.models.probabilistic import PinballLoss, ProbabilisticQuantileForecaster, compute_probabilistic_metrics
from src.models.residual_forecaster import ResidualDeltaVForecaster
from src.models.ri_models import (
    AsymmetricFocalLoss,
    DedicatedRIClassifier,
    FocalLoss,
    MultiTaskRIIntensityModel,
)
from src.training.consistency_loss import MultiTaskConsistencyLoss


def test_spatial_backbones():
    """Verify channel adaptation and output shapes across ResNet and ConvNeXt backbones."""
    for arch in ["resnet18", "resnet34", "convnext_tiny"]:
        for ch in [1, 3, 4]:
            backbone = SpatialBackbone(architecture=arch, in_channels=ch, pretrained=False)
            x = torch.randn(2, ch, 64, 64)
            feats = backbone(x)
            assert feats.shape[0] == 2
            assert feats.shape[1] == backbone.out_dim


def test_residual_forecaster():
    """Verify unconstrained and bounded residual delta-V forecasters."""
    for param in ["unconstrained", "bounded"]:
        model = ResidualDeltaVForecaster(
            backbone_arch="resnet18",
            in_channels=3,
            d_model=64,
            temporal_type="transformer",
            num_layers=1,
            nhead=2,
            parameterization=param,
            pretrained_backbone=False,
        )
        x = torch.randn(2, 3, 3, 64, 64)  # (B=2, K=3, C=3, H=64, W=64)
        v_curr = torch.tensor([50.0, 75.0])

        v_hat, delta_hat = model(x, v_curr=v_curr)
        assert v_hat.shape == (2, 3)
        assert delta_hat.shape == (2, 3)

        # Gradient check
        loss = v_hat.sum()
        loss.backward()
        assert model.delta_head[0].weight.grad is not None


def test_ri_models_and_losses():
    """Verify RI Model 1, RI Model 2, and Focal Loss functions."""
    # RI Model 1
    model_ri1 = DedicatedRIClassifier(
        backbone_arch="resnet18",
        in_channels=3,
        d_model=64,
        d_env=12,
        num_layers=1,
        nhead=2,
        pretrained_backbone=False,
    )
    x = torch.randn(2, 3, 3, 64, 64)
    x_env = torch.randn(2, 12)
    logit = model_ri1(x, x_env=x_env)
    assert logit.shape == (2, 1)

    # Focal Loss
    focal = FocalLoss(gamma=2.0)
    target = torch.tensor([1.0, 0.0])
    l_focal = focal(logit, target)
    assert l_focal.item() > 0.0

    # Asymmetric Focal Loss
    asym = AsymmetricFocalLoss(gamma_neg=2.0)
    l_asym = asym(logit, target)
    assert l_asym.item() > 0.0

    # RI Model 2 (Multi-Task)
    model_ri2 = MultiTaskRIIntensityModel(
        backbone_arch="resnet18",
        in_channels=3,
        d_model=64,
        d_env=12,
        num_layers=1,
        nhead=2,
        pretrained_backbone=False,
    )
    intensities, ri_logits, trend_logits = model_ri2(x, x_env=x_env)
    assert intensities.shape == (2, 3)
    assert ri_logits.shape == (2, 1)
    assert trend_logits.shape == (2, 3)


def test_probabilistic_quantiles():
    """Verify monotonic quantile forecaster guarantees zero quantile crossings."""
    model = ProbabilisticQuantileForecaster(
        backbone_arch="resnet18",
        in_channels=3,
        d_model=64,
        num_layers=1,
        nhead=2,
        monotonic=True,
        pretrained_backbone=False,
    )
    x = torch.randn(4, 3, 3, 64, 64)
    q_out = model(x)  # (B=4, horizons=3, quantiles=3)
    assert q_out.shape == (4, 3, 3)

    q10 = q_out[:, :, 0]
    q50 = q_out[:, :, 1]
    q90 = q_out[:, :, 2]

    # Monotonic invariant: q10 <= q50 <= q90
    assert torch.all(q10 <= q50 + 1e-5)
    assert torch.all(q50 <= q90 + 1e-5)

    # Pinball Loss
    pinball = PinballLoss()
    targets = torch.randn(4, 3)
    p_loss = pinball(q_out, targets)
    assert p_loss.item() > 0.0

    # Probabilistic Metrics
    metrics = compute_probabilistic_metrics(q_out.detach().numpy(), targets.numpy())
    assert "coverage_+24h" in metrics
    assert "crossing_rate_+24h" in metrics
    assert metrics["crossing_rate_+24h"] == 0.0  # Zero crossings guaranteed by construction!


def test_consistency_loss():
    """Verify multi-task consistency loss and cross-head divergence."""
    cons_loss_fn = MultiTaskConsistencyLoss(ri_threshold_kt=30.0, weight=1.0)
    pred_d24 = torch.tensor([50.0, 10.0])  # Storm 1: +50kt (clear RI); Storm 2: +10kt (non-RI)
    ri_logits = torch.tensor([2.0, -2.0])  # Compatible logits

    loss_val, diag = cons_loss_fn(pred_d24, ri_logits)
    assert loss_val.item() > 0.0
    assert "cross_head_disagreement" in diag
    assert diag["cross_head_disagreement"] < 0.5  # Should show low disagreement for compatible predictions


def test_physical_sanity_checker():
    """Verify sanity checker identifies violations without modifying data."""
    checker = PhysicalSanityChecker(min_intensity_kt=0.0, max_plausible_kt=200.0, large_step_change_kt=45.0)

    clean_preds = np.array([[30.0, 35.0, 45.0], [50.0, 55.0, 60.0]])
    clean_copy = clean_preds.copy()
    rep_clean = checker.inspect(clean_preds, v_curr=np.array([25.0, 45.0]))
    assert rep_clean["status"] == "CLEAN"
    assert np.array_equal(clean_preds, clean_copy)  # Zero modification invariant!

    # Dirty predictions: negative value and large jump
    bad_preds = np.array([[-5.0, 35.0, 45.0], [50.0, 110.0, 120.0]])  # -5 is negative; 50->110 is +60kt in 6h
    rep_bad = checker.inspect(bad_preds, v_curr=np.array([25.0, 45.0]))
    assert rep_bad["status"] == "ANOMALIES_DETECTED"
    assert rep_bad["negative_intensity_count"] == 1
    assert rep_bad["large_single_step_count"] == 1


def test_standardized_predict_api():
    """Verify standardized prediction API outputs expected dictionary schema."""
    model = ResidualDeltaVForecaster(
        backbone_arch="resnet18",
        in_channels=3,
        d_model=64,
        num_layers=1,
        nhead=2,
        parameterization="unconstrained",
        pretrained_backbone=False,
    )
    predictor = StandardizedPredictor(model, model_type="residual", device="cpu")
    seq = np.random.randn(3, 3, 64, 64).astype(np.float32)
    res = predictor.predict(seq, v_curr=45.0)

    assert "current_intensity" in res
    assert "predicted_6h" in res
    assert "predicted_12h" in res
    assert "predicted_24h" in res
    assert "delta_24h" in res
    assert res["current_intensity"] == 45.0


def test_baselines():
    """Verify Persistence and History Extrapolator baselines."""
    import pandas as pd
    df = pd.DataFrame({
        "vmax_curr": [40.0, 60.0],
        "history_vmax": ["[30, 35, 40]", "[50, 55, 60]"],
    })

    # Persistence
    p_model = PersistenceForecaster()
    p_res = p_model.predict_forecast(df)
    assert np.all(p_res["intensity_preds"][:, 0] == df["vmax_curr"].values)
    assert np.all(p_res["ri_probs"] == 0.0)

    # History Extrapolator
    h_model = IntensityHistoryExtrapolator()
    h_res = h_model.predict_forecast(df)
    assert h_res["intensity_preds"].shape == (2, 3)
    # Since +10kt in recent 6h, slope = 10/6 kt/h -> 24h delta = +40kt -> should be > 60kt + 40kt = 100kt
    assert h_res["intensity_preds"][1, 2] == pytest.approx(100.0, abs=1.0)


def test_trajectory_evaluator_and_pr_auc_ci():
    """Verify TrajectoryEvaluator false-dip detection and PR-AUC bootstrap CI."""
    from src.evaluation.sanity_checks import TrajectoryEvaluator
    from src.evaluation.stratified import compute_pr_auc_bootstrap_ci

    evaluator = TrajectoryEvaluator(dip_dip_tolerance_kt=5.0)

    # Test sample with artificial false dip: 65 -> 43 -> 70 -> 75
    # Ground truth is steady: 65 -> 67 -> 70 -> 75
    preds = np.array([[43.0, 70.0, 75.0], [66.0, 72.0, 85.0]])
    targets = np.array([[67.0, 70.0, 75.0], [66.0, 72.0, 85.0]])
    v_curr = np.array([65.0, 60.0])

    report = evaluator.evaluate_trajectories(preds, targets, v_curr)
    assert report["total_samples"] == 2
    assert report["false_dip_count"] == 1
    assert report["false_dip_rate_pct"] == 50.0
    assert report["second_diff_error_kt"] > 0.0

    # Test bootstrap CI for PR-AUC
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.1, 0.8, 0.9, 0.3, 0.7, 0.2, 0.1, 0.85, 0.4, 0.75])
    pr_auc, ci_l, ci_u = compute_pr_auc_bootstrap_ci(y_true, y_prob, n_bootstrap=200)
    assert 0.0 <= ci_l <= pr_auc <= ci_u <= 1.0

