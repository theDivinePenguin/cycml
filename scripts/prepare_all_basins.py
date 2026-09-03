"""Prepare, split, normalize, and audit the expanded all-basins TCIR dataset."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

from src.data.metadata import build_unified_multi_hdf5_metadata
from src.data.preprocessing import compute_normalization_stats
from src.data.splitting import split_by_cyclone_id, save_splits_json
from src.utils.config import load_config


def prepare_all_basins_dataset(config_path: str | Path = "configs/all_basins.yaml") -> dict:
    """End-to-end preparation for all-basins TCIR dataset."""
    config = load_config(config_path)
    ds_cfg = config.get("dataset", {})

    meta_dir = Path(ds_cfg.get("metadata_dir", "data/metadata"))
    meta_dir.mkdir(parents=True, exist_ok=True)

    h5_files = [Path(f) for f in ds_cfg.get("h5_files", ["data/raw/TCIR-CPAC_IO_SH.h5", "data/raw/TCIR-ATLN_EPAC_WPAC.h5"])]
    for hf in h5_files:
        if not hf.exists():
            raise FileNotFoundError(f"Required HDF5 file not found: {hf}. Ensure it is downloaded and extracted.")

    meta_csv_path = meta_dir / "metadata_all_basins.csv"
    splits_json_path = meta_dir / "splits_all_basins.json"
    norm_stats_path = meta_dir / "normalization_stats_all_basins.json"

    # 1. Ingest & Unify Metadata
    print("=" * 70)
    print("STEP 1: UNIFYING METADATA FROM ALL TCIR OCEAN BASINS")
    print("=" * 70)
    df_meta = build_unified_multi_hdf5_metadata(h5_paths=h5_files, save_path=meta_csv_path)

    # 2. Cyclone-Level Grouped Splitting
    print("\n" + "=" * 70)
    print("STEP 2: EXECUTING GROUPED CYCLONE-LEVEL 70/15/15 SPLIT")
    print("=" * 70)
    split_ratio = tuple(ds_cfg.get("split_ratio", [0.70, 0.15, 0.15]))
    split_seed = ds_cfg.get("split_seed", 42)

    from src.data.splitting import split_by_cyclone_id, save_splits_json

    train_df, val_df, test_df = split_by_cyclone_id(
        df_meta,
        split_ratio=split_ratio,
        seed=split_seed,
        stratify_by_intensity=True
    )

    save_splits_json(train_df, val_df, test_df, splits_json_path)

    # 3. Compute Training-Only Normalization Statistics
    print("\n" + "=" * 70)
    print("STEP 3: COMPUTING TRAINING-ONLY NORMALIZATION STATISTICS")
    print("=" * 70)

    # Accumulate mean and std over training frames across multiple HDF5 files
    print(f"[Preprocessing] Computing stats over {len(train_df):,} training frames...")
    total_pixels = 0
    sum_val = 0.0
    sum_sq_val = 0.0
    min_val = float("inf")
    max_val = float("-inf")

    # Group by h5_file to stream efficiently
    for h5_file_p, group_df in train_df.groupby("h5_file"):
        print(f"  • Streaming {len(group_df):,} frames from {Path(h5_file_p).name}...")
        with h5py.File(h5_file_p, "r") as hf:
            matrix_ds = hf["matrix"]
            row_indices = group_df["h5_row_index"].tolist()
            
            # Process in chunks of 500 frames to keep RAM low
            chunk_size = 500
            for i in range(0, len(row_indices), chunk_size):
                chunk_rows = row_indices[i:i + chunk_size]
                chunk_data = matrix_ds[chunk_rows, :, :, 0]  # (B, H, W)
                
                # Replace NaNs
                nan_mask = np.isnan(chunk_data)
                if np.any(nan_mask):
                    chunk_data = np.nan_to_num(chunk_data, nan=270.0)

                total_pixels += chunk_data.size
                sum_val += float(np.sum(chunk_data, dtype=np.float64))
                sum_sq_val += float(np.sum(chunk_data ** 2, dtype=np.float64))
                min_val = min(min_val, float(np.min(chunk_data)))
                max_val = max(max_val, float(np.max(chunk_data)))

    mean_val = sum_val / total_pixels
    std_val = float(np.sqrt((sum_sq_val / total_pixels) - (mean_val ** 2)))

    norm_stats = {
        "channel": "IR1",
        "channel_idx": 0,
        "n_samples": len(train_df),
        "total_pixels": total_pixels,
        "mean": float(mean_val),
        "std": float(std_val),
        "min": float(min_val),
        "max": float(max_val)
    }

    with open(norm_stats_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"[Preprocessing] Normalization stats saved: Mean={mean_val:.4f} K, Std={std_val:.4f} K, Min={min_val:.2f} K, Max={max_val:.2f} K")
    print(f"[Preprocessing] Saved to: {norm_stats_path}")

    # 4. Leakage Audit
    print("\n" + "=" * 70)
    print("STEP 4: AUTOMATED 8-POINT DATA LEAKAGE AUDIT")
    print("=" * 70)

    # Save split metadata CSVs
    train_df.to_csv(meta_dir / "train_metadata_all_basins.csv", index=False)
    val_df.to_csv(meta_dir / "val_metadata_all_basins.csv", index=False)
    test_df.to_csv(meta_dir / "test_metadata_all_basins.csv", index=False)

    from src.data.leakage import LeakageAuditor
    auditor = LeakageAuditor(train_df=train_df, val_df=val_df, test_df=test_df, norm_stats=norm_stats)
    passed = auditor.run_audit()

    return {
        "metadata_path": str(meta_csv_path),
        "splits_path": str(splits_json_path),
        "norm_stats_path": str(norm_stats_path),
        "passed": passed
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare all-basins TCIR dataset.")
    parser.add_argument("--config", type=str, default="configs/all_basins.yaml", help="Path to all_basins config")
    args = parser.parse_args()

    prepare_all_basins_dataset(config_path=args.config)


if __name__ == "__main__":
    main()
