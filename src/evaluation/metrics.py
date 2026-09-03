"""Evaluation metrics computation for cyclone intensity estimation."""
from typing import Dict
import numpy as np
import torch
from sklearn.metrics import r2_score


def calculate_metrics(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor
) -> Dict[str, float]:
    """Calculate all standard regression metrics.

    Args:
        predictions: Array or tensor of predicted wind speeds in knots.
        targets: Array or tensor of ground truth wind speeds in knots.

    Returns:
        Dictionary of calculated metric values.
    """
    if isinstance(predictions, torch.Tensor):
        preds = predictions.detach().cpu().numpy().flatten()
    else:
        preds = np.asarray(predictions).flatten()

    if isinstance(targets, torch.Tensor):
        gts = targets.detach().cpu().numpy().flatten()
    else:
        gts = np.asarray(targets).flatten()

    assert len(preds) == len(gts), f"Shape mismatch: preds={len(preds)}, gts={len(gts)}"
    if len(preds) == 0:
        return {}

    errors = preds - gts
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    median_ae = float(np.median(abs_errors))
    mean_error = float(np.mean(errors))  # Bias / Signed Error
    max_error = float(np.max(abs_errors))

    # R-squared calculation
    if np.var(gts) > 1e-8:
        r2 = float(r2_score(gts, preds))
    else:
        r2 = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_ae": median_ae,
        "mean_bias": mean_error,
        "max_ae": max_error,
        "n_samples": len(preds)
    }
