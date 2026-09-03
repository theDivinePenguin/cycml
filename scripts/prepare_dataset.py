"""Preparation pipeline: metadata parsing, grouped splitting, normalization calculation, and audit."""
import argparse
import json
from pathlib import Path
import h5py
import pandas as pd

from src.data.downloader import download_tcir_archive
from src.data.leakage import LeakageAuditor
from src.data.metadata import parse_and_normalize_metadata
from src.data.preprocessing import compute_normalization_stats
from src.data.splitting import save_splits_json, split_by_cyclone_id
from src.utils.config import load_config


def prepare_dataset(config: dict) -> None:
    """Execute complete dataset preparation pipeline."""
    ds_cfg = config.get("dataset", {})
    regions = ds_cfg.get("regions", ["CPAC", "IO", "SH"])
    raw_dir = Path(ds_cfg.get("raw_dir", "data/raw"))
    metadata_dir = Path(ds_cfg.get("metadata_dir", "data/metadata"))
    split_ratio = tuple(ds_cfg.get("split_ratio", [0.70, 0.15, 0.15]))
    split_seed = ds_cfg.get("split_seed", 42)

    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download & Extract if needed
    h5_path = download_tcir_archive(key="CPAC_IO_SH", destination_dir=raw_dir, extract=True)

    # 2. Parse & Normalize Metadata
    regions_str = "_".join(sorted(regions))
    norm_meta_path = metadata_dir / f"metadata_{regions_str}.csv"
    df = parse_and_normalize_metadata(h5_path, target_regions=regions, save_path=norm_meta_path)

    # 3. Cyclone-level Grouped Split
    train_df, val_df, test_df = split_by_cyclone_id(
        df=df,
        split_ratio=split_ratio,
        seed=split_seed,
        stratify_by_intensity=True
    )

    # Save split metadata CSVs
    train_meta_path = metadata_dir / f"train_metadata_{regions_str}.csv"
    val_meta_path = metadata_dir / f"val_metadata_{regions_str}.csv"
    test_meta_path = metadata_dir / f"test_metadata_{regions_str}.csv"

    train_df.to_csv(train_meta_path, index=False)
    val_df.to_csv(val_meta_path, index=False)
    test_df.to_csv(test_meta_path, index=False)

    # Save splits.json
    splits_json_path = metadata_dir / f"splits_{regions_str}.json"
    save_splits_json(train_df, val_df, test_df, splits_json_path)

    # 4. Compute Normalization Statistics (TRAIN SET ONLY)
    stats_path = metadata_dir / f"normalization_stats_{regions_str}.json"
    with h5py.File(h5_path, "r") as hf:
        matrix = hf["matrix"]
        train_indices = train_df["sample_index"].tolist()
        mean, std = compute_normalization_stats(
            matrix_dataset=matrix,
            train_indices=train_indices,
            channel_idx=0,
            save_path=stats_path
        )

    # 5. Run Automated Data Leakage Audit
    norm_stats = {"mean": mean, "std": std, "n_train_samples": len(train_df)}
    auditor = LeakageAuditor(train_df, val_df, test_df, norm_stats)
    audit_passed = auditor.run_audit()

    if not audit_passed:
        raise RuntimeError("CRITICAL ERROR: Data leakage audit failed!")

    print(f"\n[Preparation Complete] Dataset is ready for training and evaluation.")


def main():
    parser = argparse.ArgumentParser(description="Prepare TCIR dataset: parsing, grouped splitting, normalization.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    prepare_dataset(config)


if __name__ == "__main__":
    main()
