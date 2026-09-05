"""Causal environmental feature extraction, normalization, and modular ablation gating."""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch


# Fixed environmental feature names
FEATURE_NAMES = ["vmax", "mslp", "sst", "cohc", "shrd", "rhmd"]

# Feature group configurations for automated ablations
FEATURE_GROUPS = {
    "satellite_only": [],
    "satellite_plus_sst": ["sst"],
    "satellite_plus_ohc": ["cohc"],
    "satellite_plus_vws": ["shrd"],
    "all_environmental": ["sst", "cohc", "shrd", "rhmd"],
    "full_feature_set": ["vmax", "mslp", "sst", "cohc", "shrd", "rhmd"],
}


class EnvironmentalFeatureManager:
    """Manages causal SHIPS environmental predictors with strict train-only normalization,
    missingness gating, and zero-lookahead guarantees.
    """

    def __init__(
        self,
        metadata_dir: str | Path = "data/metadata",
        norm_stats: Optional[Dict] = None,
        feature_group: str = "full_feature_set",
    ):
        self.meta_dir = Path(metadata_dir)
        self.feature_group = feature_group
        self.active_features = FEATURE_GROUPS.get(feature_group, FEATURE_NAMES)

        # Load or compute strict training-set normalization statistics
        self.norm_stats = norm_stats or self._load_or_compute_norm_stats()

        # Cache of lookup tables: (cyclone_id, timestamp) -> 12-d normalized vector
        self._lookup_cache: Dict[Tuple[str, int], np.ndarray] = {}
        self._load_caches()

    def _load_or_compute_norm_stats(self) -> Dict[str, Dict[str, float]]:
        stats_path = self.meta_dir / "environmental_norm_stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Compute strictly on training split
        train_cache_path = self.meta_dir / "environmental_cache_k7_train.csv"
        if not train_cache_path.exists():
            train_cache_path = self.meta_dir / "environmental_cache_train.csv"

        if not train_cache_path.exists():
            raise FileNotFoundError(f"Cannot find training environmental cache in {self.meta_dir}")

        train_df = pd.read_csv(train_cache_path)
        stats = {}
        for col in FEATURE_NAMES:
            valid_vals = train_df[col].dropna()
            mean_v = float(valid_vals.mean())
            std_v = float(valid_vals.std()) if valid_vals.std() > 1e-6 else 1.0
            stats[col] = {
                "mean": mean_v,
                "std": std_v,
                "missing_pct": float(100.0 * (1.0 - len(valid_vals) / len(train_df)))
            }

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        return stats

    def _load_caches(self) -> None:
        """Populate in-memory lookup dictionary from existing environmental cache CSVs."""
        cache_files = list(self.meta_dir.glob("environmental_cache_k7_*.csv"))
        if not cache_files:
            cache_files = list(self.meta_dir.glob("environmental_cache_*.csv"))

        for c_file in cache_files:
            df = pd.read_csv(c_file)
            for _, row in df.iterrows():
                cid = str(row["cyclone_id"])
                ts = int(row["timestamp"])
                key = (cid, ts)
                if key in self._lookup_cache:
                    continue

                feat_vals = []
                mask_vals = []

                for col in FEATURE_NAMES:
                    val = row.get(col, np.nan)
                    is_missing = pd.isna(val)
                    mask_vals.append(1.0 if is_missing else 0.0)

                    mean_c = self.norm_stats[col]["mean"]
                    std_c = self.norm_stats[col]["std"]

                    norm_val = 0.0 if is_missing else (float(val) - mean_c) / std_c
                    feat_vals.append(norm_val)

                # 12-dim vector: 6 normalized features + 6 missingness masks
                vec = np.array(feat_vals + mask_vals, dtype=np.float32)
                self._lookup_cache[key] = vec

    def get_features(
        self,
        cyclone_id: str,
        timestamp: int,
        feature_group: Optional[str] = None
    ) -> torch.Tensor:
        """Retrieve 12-d normalized tensor with missingness gating for given storm & synoptic time."""
        key = (str(cyclone_id), int(timestamp))
        vec = self._lookup_cache.get(key)
        if vec is None:
            # Fallback: all features missing
            vec = np.zeros(12, dtype=np.float32)
            vec[6:] = 1.0  # missingness masks all set to 1.0

        vec_copy = vec.copy()
        group = feature_group or self.feature_group
        active = FEATURE_GROUPS.get(group, FEATURE_NAMES)

        # Apply feature group gating (disable inactive features)
        for idx, name in enumerate(FEATURE_NAMES):
            if name not in active:
                vec_copy[idx] = 0.0
                vec_copy[idx + 6] = 1.0  # mark as masked

        return torch.from_numpy(vec_copy)


def get_feature_dim() -> int:
    """Returns 12 (6 continuous features + 6 binary missingness masks)."""
    return len(FEATURE_NAMES) * 2
