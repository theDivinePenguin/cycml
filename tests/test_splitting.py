"""Unit tests for cyclone-level grouped data splitting."""
import pandas as pd
import numpy as np
import pytest
from src.data.splitting import split_by_cyclone_id


def test_cyclone_grouped_split_invariants():
    """Verify that no cyclone ID is shared across train, val, and test splits."""
    # Create synthetic dataset with multiple frames per cyclone
    np.random.seed(42)
    cyclone_ids = [f"STORM_{i:03d}" for i in range(50)]
    records = []
    sample_idx = 0
    for cid in cyclone_ids:
        n_frames = np.random.randint(5, 30)
        for _ in range(n_frames):
            records.append({
                "sample_index": sample_idx,
                "cyclone_id": cid,
                "wind_speed": float(np.random.uniform(20, 140)),
                "region": "IO"
            })
            sample_idx += 1

    df = pd.DataFrame(records)

    train_df, val_df, test_df = split_by_cyclone_id(
        df=df,
        split_ratio=(0.70, 0.15, 0.15),
        seed=42,
        stratify_by_intensity=True
    )

    train_cids = set(train_df["cyclone_id"].unique())
    val_cids = set(val_df["cyclone_id"].unique())
    test_cids = set(test_df["cyclone_id"].unique())

    # Mandatory Invariant Checks
    assert train_cids.isdisjoint(val_cids), "Train and Val share cyclone IDs!"
    assert train_cids.isdisjoint(test_cids), "Train and Test share cyclone IDs!"
    assert val_cids.isdisjoint(test_cids), "Val and Test share cyclone IDs!"

    # Frame count check
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
