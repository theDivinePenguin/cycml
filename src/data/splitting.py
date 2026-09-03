"""Leakage-free cyclone-level grouped data splitting."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def split_by_cyclone_id(
    df: pd.DataFrame,
    split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    stratify_by_intensity: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform leak-free grouped split by cyclone ID.

    Ensures that ALL frames belonging to any single cyclone ID reside
    exclusively in either train, validation, or test split.

    Args:
        df: Cleaned metadata DataFrame containing 'cyclone_id' and 'wind_speed'.
        split_ratio: Tuple of (train_ratio, val_ratio, test_ratio).
        seed: Random seed for reproducibility.
        stratify_by_intensity: If True, bins cyclones by their maximum wind speed
            to ensure balanced intensity distributions across splits.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert len(split_ratio) == 3 and np.isclose(sum(split_ratio), 1.0), "Split ratios must sum to 1.0"
    train_pct, val_pct, test_pct = split_ratio

    # Aggregate by cyclone ID to compute storm-level characteristics
    storm_stats = df.groupby("cyclone_id").agg(
        max_wind=("wind_speed", "max"),
        mean_wind=("wind_speed", "mean"),
        n_frames=("sample_index", "count")
    ).reset_index()

    rng = np.random.RandomState(seed)

    if stratify_by_intensity:
        # Bin cyclones into 4 intensity quartiles based on max wind speed
        storm_stats["intensity_bin"] = pd.qcut(
            storm_stats["max_wind"], q=4, labels=False, duplicates="drop"
        )
        train_ids, val_ids, test_ids = [], [], []

        for _, group in storm_stats.groupby("intensity_bin"):
            c_ids = list(group["cyclone_id"].values)
            rng.shuffle(c_ids)
            n = len(c_ids)
            n_train = int(round(n * train_pct))
            n_val = int(round(n * val_pct))

            train_ids.extend(c_ids[:n_train])
            val_ids.extend(c_ids[n_train:n_train + n_val])
            test_ids.extend(c_ids[n_train + n_val:])
    else:
        c_ids = list(storm_stats["cyclone_id"].values)
        rng.shuffle(c_ids)
        n = len(c_ids)
        n_train = int(round(n * train_pct))
        n_val = int(round(n * val_pct))

        train_ids = list(c_ids[:n_train])
        val_ids = list(c_ids[n_train:n_train + n_val])
        test_ids = list(c_ids[n_train + n_val:])

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    # Mandatory Invariant Verification
    assert train_set.isdisjoint(val_set), "CRITICAL: Cyclone overlap between Train and Val!"
    assert train_set.isdisjoint(test_set), "CRITICAL: Cyclone overlap between Train and Test!"
    assert val_set.isdisjoint(test_set), "CRITICAL: Cyclone overlap between Val and Test!"

    train_df = df[df["cyclone_id"].isin(train_set)].copy().reset_index(drop=True)
    val_df = df[df["cyclone_id"].isin(val_set)].copy().reset_index(drop=True)
    test_df = df[df["cyclone_id"].isin(test_set)].copy().reset_index(drop=True)

    print("[Splitting] Grouped cyclone split complete:")
    print(f"  • Train: {len(train_df):5d} frames across {len(train_set):3d} cyclones ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  • Val:   {len(val_df):5d} frames across {len(val_set):3d} cyclones ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  • Test:  {len(test_df):5d} frames across {len(test_set):3d} cyclones ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def split_chronologically(
    df: pd.DataFrame,
    split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform chronological split by cyclone start year/timestamp.

    Args:
        df: Metadata DataFrame with 'cyclone_id' and 'year' or 'timestamp'.
        split_ratio: Tuple of (train_ratio, val_ratio, test_ratio).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train_pct, val_pct, test_pct = split_ratio
    storm_years = df.groupby("cyclone_id")["year"].min().sort_values().reset_index()

    n = len(storm_years)
    n_train = int(round(n * train_pct))
    n_val = int(round(n * val_pct))

    train_ids = set(storm_years["cyclone_id"].iloc[:n_train])
    val_ids = set(storm_years["cyclone_id"].iloc[n_train:n_train + n_val])
    test_ids = set(storm_years["cyclone_id"].iloc[n_train + n_val:])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    train_df = df[df["cyclone_id"].isin(train_ids)].copy().reset_index(drop=True)
    val_df = df[df["cyclone_id"].isin(val_ids)].copy().reset_index(drop=True)
    test_df = df[df["cyclone_id"].isin(test_ids)].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def save_splits_json(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: str | Path
) -> None:
    """Save split sample indices and cyclone IDs to a JSON file."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    splits_data = {
        "train": {
            "cyclone_ids": sorted(list(train_df["cyclone_id"].unique())),
            "sample_indices": train_df["sample_index"].tolist(),
            "n_frames": len(train_df),
            "n_cyclones": int(train_df["cyclone_id"].nunique())
        },
        "val": {
            "cyclone_ids": sorted(list(val_df["cyclone_id"].unique())),
            "sample_indices": val_df["sample_index"].tolist(),
            "n_frames": len(val_df),
            "n_cyclones": int(val_df["cyclone_id"].nunique())
        },
        "test": {
            "cyclone_ids": sorted(list(test_df["cyclone_id"].unique())),
            "sample_indices": test_df["sample_index"].tolist(),
            "n_frames": len(test_df),
            "n_cyclones": int(test_df["cyclone_id"].nunique())
        }
    }

    with open(p, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)

    print(f"[Splitting] Saved split index definition to: {p}")
