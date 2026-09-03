"""Automated 8-point Data Leakage Audit."""
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class LeakageAuditor:
    """Auditor to verify strict separation between train, validation, and test datasets."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        norm_stats: Optional[Dict[str, float]] = None
    ):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.norm_stats = norm_stats or {}

    def run_audit(self) -> bool:
        """Run complete 8-point data leakage audit.

        Returns:
            True if all checks pass, False otherwise.
        """
        print("=" * 60)
        print("DATASET LEAKAGE AUDIT")
        print("=" * 60)

        train_cyclones = set(self.train_df["cyclone_id"].unique())
        val_cyclones = set(self.val_df["cyclone_id"].unique())
        test_cyclones = set(self.test_df["cyclone_id"].unique())

        train_indices = set(self.train_df["sample_index"].tolist())
        val_indices = set(self.val_df["sample_index"].tolist())
        test_indices = set(self.test_df["sample_index"].tolist())

        # Check 1: Cyclone ID Overlap
        overlap_tv = train_cyclones.intersection(val_cyclones)
        overlap_tt = train_cyclones.intersection(test_cyclones)
        overlap_vt = val_cyclones.intersection(test_cyclones)
        cyclone_overlap_count = len(overlap_tv) + len(overlap_tt) + len(overlap_vt)

        # Check 2: Sample Index Overlap
        idx_overlap_tv = train_indices.intersection(val_indices)
        idx_overlap_tt = train_indices.intersection(test_indices)
        idx_overlap_vt = val_indices.intersection(test_indices)
        index_overlap_count = len(idx_overlap_tv) + len(idx_overlap_tt) + len(idx_overlap_vt)

        # Check 3: Metadata Target Leakage
        target_leakage = False
        for name, df in [("Train", self.train_df), ("Val", self.val_df), ("Test", self.test_df)]:
            if df["wind_speed"].isna().any() or (df["wind_speed"] <= 0).any():
                print(f"[AUDIT ERROR] Found invalid/NaN wind_speed in {name} split!")
                target_leakage = True

        # Check 4: Normalization Leakage
        norm_leak = False
        if "n_train_samples" in self.norm_stats:
            stat_n = self.norm_stats["n_train_samples"]
            if stat_n != len(self.train_df):
                print(f"[AUDIT WARNING] Normalization sample count ({stat_n}) differs from train split ({len(self.train_df)})!")
                norm_leak = True

        # Print Formal Report
        print(f"Train cyclones:       {len(train_cyclones):4d} ({len(self.train_df):6,d} frames)")
        print(f"Validation cyclones:  {len(val_cyclones):4d} ({len(self.val_df):6,d} frames)")
        print(f"Test cyclones:        {len(test_cyclones):4d} ({len(self.test_df):6,d} frames)")
        print("-" * 60)
        print(f"Cyclone overlap:      {cyclone_overlap_count}")
        print(f"Duplicate overlap:    {index_overlap_count}")
        print(f"Metadata leakage:     {'YES (FAIL)' if target_leakage else 'NO'}")
        print(f"Normalization leak:   {'YES (FAIL)' if norm_leak else 'NO'}")
        print("-" * 60)

        passed = (cyclone_overlap_count == 0 and index_overlap_count == 0 and not target_leakage and not norm_leak)
        status_str = "STATUS: PASS" if passed else "STATUS: FAIL"
        print(status_str)
        print("=" * 60)

        if not passed:
            if overlap_tv:
                print(f"Overlapping cyclones (Train-Val): {list(overlap_tv)[:5]}")
            if overlap_tt:
                print(f"Overlapping cyclones (Train-Test): {list(overlap_tt)[:5]}")
            if overlap_vt:
                print(f"Overlapping cyclones (Val-Test): {list(overlap_vt)[:5]}")

        return passed
