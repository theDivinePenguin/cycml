"""Prepare Indian Ocean (IO) only dataset, grouped split, normalization, and leakage audit."""
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

from src.data.leakage import LeakageAuditor
from src.data.metadata import load_tcir_info_table, parse_and_normalize_metadata
from src.data.splitting import split_by_cyclone_id, save_splits_json


def prepare_io_dataset(
    h5_path: str | Path = "data/raw/TCIR-CPAC_IO_SH.h5",
    metadata_dir: str | Path = "data/metadata",
    split_ratio: tuple = (0.70, 0.15, 0.15),
    seed: int = 42
) -> dict:
    """Extract IO metadata, execute 70/15/15 cyclone split, compute normalization, and audit."""
    h5_p = Path(h5_path)
    meta_dir = Path(metadata_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)

    if not h5_p.exists():
        raise FileNotFoundError(f"Authoritative HDF5 not found at {h5_p}")

    print("=" * 70)
    print("STEP 1: EXTRACTING INDIAN OCEAN (IO) DATASET METADATA")
    print("=" * 70)

    df_io = parse_and_normalize_metadata(h5_p, target_regions=["IO"])
    df_io["h5_file"] = str(h5_p)
    df_io["h5_row_index"] = df_io["sample_index"]  # original row in TCIR-CPAC_IO_SH.h5

    meta_io_path = meta_dir / "metadata_IO.csv"
    df_io.to_csv(meta_io_path, index=False)
    print(f"[Metadata] Extracted {len(df_io):,} IO frames across {df_io['cyclone_id'].nunique()} unique cyclones.")
    print(f"[Metadata] Saved to: {meta_io_path}")

    # Step 2: Cyclone-Level Grouped Splitting
    print("\n" + "=" * 70)
    print("STEP 2: EXECUTING GROUPED CYCLONE-LEVEL 70/15/15 SPLIT (IO ONLY)")
    print("=" * 70)
    splits_json_path = meta_dir / "splits_IO.json"

    train_df, val_df, test_df = split_by_cyclone_id(
        df_io,
        split_ratio=split_ratio,
        seed=seed,
        stratify_by_intensity=True
    )

    save_splits_json(train_df, val_df, test_df, splits_json_path)

    # Save split metadata CSVs
    train_path = meta_dir / "train_metadata_IO.csv"
    val_path = meta_dir / "val_metadata_IO.csv"
    test_path = meta_dir / "test_metadata_IO.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Step 3: Compute Training-Only Normalization Statistics
    print("\n" + "=" * 70)
    print("STEP 3: COMPUTING TRAINING-ONLY NORMALIZATION STATISTICS (IO)")
    print("=" * 70)
    print(f"[Preprocessing] Computing stats over {len(train_df):,} IO training frames...")

    with h5py.File(h5_p, "r") as hf:
        matrix_ds = hf["matrix"]
        train_indices = train_df["h5_row_index"].tolist()
        
        chunk_size = 500
        total_pixels = 0
        sum_val = 0.0
        sum_sq_val = 0.0
        min_val = float("inf")
        max_val = float("-inf")

        for i in range(0, len(train_indices), chunk_size):
            chunk_rows = train_indices[i:i + chunk_size]
            chunk_data = matrix_ds[chunk_rows, :, :, 0]  # (B, H, W)
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

    norm_path = meta_dir / "normalization_stats_IO.json"
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"[Preprocessing] IO Normalization stats saved: Mean={mean_val:.4f} K, Std={std_val:.4f} K, Min={min_val:.2f} K, Max={max_val:.2f} K")
    print(f"[Preprocessing] Saved to: {norm_path}")

    # Step 4: Automated Leakage Audit
    print("\n" + "=" * 70)
    print("STEP 4: AUTOMATED 8-POINT DATA LEAKAGE AUDIT (IO)")
    print("=" * 70)
    auditor = LeakageAuditor(train_df=train_df, val_df=val_df, test_df=test_df, norm_stats=norm_stats)
    passed = auditor.run_audit()

    return {
        "metadata_path": str(meta_io_path),
        "splits_path": str(splits_json_path),
        "norm_stats_path": str(norm_path),
        "passed": passed
    }


def main():
    prepare_io_dataset()


if __name__ == "__main__":
    main()
