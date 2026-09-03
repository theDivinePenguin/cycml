"""CLI script to run standalone 8-point data leakage audit."""
import argparse
import json
from pathlib import Path
import pandas as pd

from src.data.leakage import LeakageAuditor
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run 8-point data leakage audit on dataset splits.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    ds_cfg = config.get("dataset", {})
    regions = ds_cfg.get("regions", ["CPAC", "IO", "SH"])
    regions_str = "_".join(sorted(regions))
    metadata_dir = Path(ds_cfg.get("metadata_dir", "data/metadata"))

    train_path = metadata_dir / f"train_metadata_{regions_str}.csv"
    val_path = metadata_dir / f"val_metadata_{regions_str}.csv"
    test_path = metadata_dir / f"test_metadata_{regions_str}.csv"
    stats_path = metadata_dir / f"normalization_stats_{regions_str}.json"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(f"Split metadata files not found in {metadata_dir}. Run prepare_dataset.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    norm_stats = {}
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            norm_stats = json.load(f)

    auditor = LeakageAuditor(train_df, val_df, test_df, norm_stats)
    passed = auditor.run_audit()

    if not passed:
        exit(1)


if __name__ == "__main__":
    main()
