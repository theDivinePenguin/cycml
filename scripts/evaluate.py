"""Standalone evaluation CLI script."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch

from src.data.dataset import build_dataloaders
from src.data.downloader import download_tcir_archive
from src.evaluation.evaluate import evaluate_model_on_dataset, generate_evaluation_artifacts
from src.models.factory import build_model
from src.training.checkpoint import CheckpointManager
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoint on held-out test set.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (optional, defaults to checkpoint's config)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation artifacts")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint data
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = load_config(args.config) if args.config else checkpoint.get("config", load_config("configs/baseline.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_cfg = config.get("dataset", {})
    regions = ds_cfg.get("regions", ["CPAC", "IO", "SH"])
    regions_str = "_".join(sorted(regions))
    raw_dir = Path(ds_cfg.get("raw_dir", "data/raw"))
    metadata_dir = Path(ds_cfg.get("metadata_dir", "data/metadata"))

    h5_path = download_tcir_archive(key="CPAC_IO_SH", destination_dir=raw_dir, extract=False)

    train_path = metadata_dir / f"train_metadata_{regions_str}.csv"
    val_path = metadata_dir / f"val_metadata_{regions_str}.csv"
    test_path = metadata_dir / f"test_metadata_{regions_str}.csv"
    stats_path = metadata_dir / f"normalization_stats_{regions_str}.json"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    with open(stats_path, "r", encoding="utf-8") as f:
        norm_stats = json.load(f)
    mean, std = norm_stats["mean"], norm_stats["std"]

    _, _, test_loader = build_dataloaders(
        h5_path=h5_path,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config,
        mean=mean,
        std=std,
        in_memory=False
    )

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent

    eval_results = evaluate_model_on_dataset(
        model=model,
        data_loader=test_loader,
        device=device,
        use_amp=config.get("training", {}).get("use_amp", True)
    )

    generate_evaluation_artifacts(
        eval_results=eval_results,
        output_dir=out_dir,
        experiment_name=f"Evaluation ({ckpt_path.name})"
    )


if __name__ == "__main__":
    main()
