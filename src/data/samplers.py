"""Intensity-aware sampler for continuous cyclone regression."""
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from src.evaluation.intensity_bins import INTENSITY_BINS, assign_intensity_bin


def compute_intensity_sampling_weights(
    df: pd.DataFrame,
    alpha: float = 0.5,
    wind_speed_col: str = "wind_speed"
) -> Tuple[np.ndarray, Dict[str, dict]]:
    """Compute intensity-aware sample weights using square-root inverse-frequency weighting.
    
    Formula:
        For bin b with count N_b:
        Bin weight: w_b = 1.0 / (N_b ** alpha)
        Sample weight: s_i = w_{b(i)} / N_{b(i)} = 1.0 / (N_{b(i)} ** (1 + alpha))
        
        Effective expected probability of drawing a sample from bin b:
        P_eff(b) = N_b * s_i = N_b^{1 - alpha} / sum_{b'} N_{b'}^{1 - alpha}
        
    Args:
        df: Training DataFrame containing wind speed column.
        alpha: Damping power (alpha=0.0 -> natural, alpha=0.5 -> sqrt-inverse, alpha=1.0 -> strictly balanced).
        wind_speed_col: Name of wind speed column.
        
    Returns:
        Tuple of (sample_weights_array, bin_weight_diagnostics_dict).
    """
    bins = df[wind_speed_col].apply(assign_intensity_bin).values
    bin_counts = pd.Series(bins).value_counts().to_dict()
    total_samples = len(df)

    # Calculate bin weights and per-sample weights
    bin_weights = {}
    effective_bin_probs = {}
    
    raw_bin_w_sum = sum(N ** (1.0 - alpha) for N in bin_counts.values() if N > 0)

    for _, _, label in INTENSITY_BINS:
        n_b = bin_counts.get(label, 0)
        if n_b > 0:
            sample_w = 1.0 / (n_b ** alpha)
            eff_p = (n_b ** (1.0 - alpha)) / raw_bin_w_sum
        else:
            sample_w = 0.0
            eff_p = 0.0
            
        bin_weights[label] = sample_w
        effective_bin_probs[label] = {
            "count": n_b,
            "natural_pct": (n_b / total_samples) * 100.0 if total_samples > 0 else 0.0,
            "effective_sampling_pct": eff_p * 100.0,
            "sampling_multiplier": (eff_p / (n_b / total_samples)) if (n_b > 0 and total_samples > 0) else 0.0
        }

    # Assign weight to each individual sample
    sample_weights = np.array([bin_weights[b] for b in bins], dtype=np.float64)
    # Normalize sample weights so they sum to 1.0
    sample_weights = sample_weights / sample_weights.sum()

    return sample_weights, effective_bin_probs


def build_intensity_aware_sampler(
    df: pd.DataFrame,
    alpha: float = 0.5,
    seed: int = 42,
    wind_speed_col: str = "wind_speed"
) -> Tuple[WeightedRandomSampler, Dict[str, dict]]:
    """Build a deterministic PyTorch WeightedRandomSampler using intensity-aware weights.
    
    Args:
        df: Training DataFrame.
        alpha: Damping power (default 0.5 for sqrt-inverse frequency).
        seed: Random seed for deterministic reproducibility.
        wind_speed_col: Name of wind speed column.
        
    Returns:
        Tuple of (WeightedRandomSampler, bin_diagnostics_dict).
    """
    sample_weights, diagnostics = compute_intensity_sampling_weights(
        df=df,
        alpha=alpha,
        wind_speed_col=wind_speed_col
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(df),
        replacement=True,
        generator=generator
    )

    return sampler, diagnostics
