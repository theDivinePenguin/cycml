"""Standardized intensity bin definitions and helper functions for cyclone analysis."""
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

INTENSITY_BINS: List[Tuple[float, float, str]] = [
    (15.0, 30.0, "15–30 kt"),
    (30.0, 50.0, "30–50 kt"),
    (50.0, 70.0, "50–70 kt"),
    (70.0, 90.0, "70–90 kt"),
    (90.0, 110.0, "90–110 kt"),
    (110.0, 130.0, "110–130 kt"),
    (130.0, 150.0, "130–150 kt"),
    (150.0, float("inf"), "> 150 kt"),
]

REGIME_BINS: List[Tuple[float, float, str]] = [
    (0.0, 60.0, "<60 kt"),
    (60.0, 100.0, "60–100 kt"),
    (100.0, 130.0, "100–130 kt"),
    (130.0, float("inf"), "> 130 kt"),
]


def assign_intensity_bin(wind_speed: float, bins: List[Tuple[float, float, str]] = INTENSITY_BINS) -> str:
    """Assign a wind speed value (in knots) to its corresponding intensity bin label.

    Interval logic:
    - Intermediate bins are [lower, upper)
    - The first bin includes values equal to or slightly below lower (e.g. <= lower)
    - The last bin includes values >= 150.0
    """
    for lower, upper, label in bins:
        if upper == float("inf"):
            if wind_speed >= lower:
                return label
        else:
            if lower <= wind_speed < upper:
                return label
            # Edge case for minimum values below the first bin boundary
            if lower == bins[0][0] and wind_speed < lower:
                return label
    return bins[-1][2]


def assign_regime_bin(wind_speed: float) -> str:
    """Assign a wind speed value to aggregate regime bin (<60, 60-100, 100-130, >130 kt)."""
    return assign_intensity_bin(wind_speed, bins=REGIME_BINS)


def compute_binned_distribution(
    df: pd.DataFrame,
    intensity_col: str = "wind_speed",
    cyclone_id_col: str = "cyclone_id",
    bins: List[Tuple[float, float, str]] = INTENSITY_BINS
) -> List[Dict[str, Any]]:
    """Compute frame counts, percentages, and unique cyclone counts for each intensity bin."""
    total_frames = len(df)
    results = []

    for lower, upper, label in bins:
        if upper == float("inf"):
            mask = df[intensity_col] >= lower
        elif lower == bins[0][0]:
            mask = df[intensity_col] < upper
        else:
            mask = (df[intensity_col] >= lower) & (df[intensity_col] < upper)

        bin_df = df[mask]
        frames_count = len(bin_df)
        pct_frames = (frames_count / total_frames * 100.0) if total_frames > 0 else 0.0
        unique_cyclones = int(bin_df[cyclone_id_col].nunique()) if frames_count > 0 else 0

        results.append({
            "bin": label,
            "lower_kt": lower,
            "upper_kt": upper if upper != float("inf") else None,
            "frames": frames_count,
            "percent_frames": round(pct_frames, 2),
            "unique_cyclones": unique_cyclones
        })

    return results
