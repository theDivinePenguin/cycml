"""Main training CLI supporting overfit test, smoke test, and full baseline training."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
import torch.optim as optim

from src.data.dataset import build_dataloaders
from src.data.downloader import download_tcir_archive
from src.evaluation.evaluate import evaluate_model_on_dataset, generate_evaluation_artifacts
from src.models.factory import build_model
from src.training.checkpoint import CheckpointManager
from src.training.losses import build_loss_fn
from src.training.train import Trainer
from src.utils.config import load_config, save_config
from src.utils.seed import seed_everything


def run_overfit_test(config: dict, h5_path: Path, train_df: pd.DataFrame, mean: float, std: float, device: torch.device) -> bool:
    """Step 8: Overfit test on 16-32 samples to verify optimization capability."""
    print("\n" + "=" * 60)
    print("STEP 8: OVERFIT DIAGNOSTIC TEST (16-32 SAMPLES)")
    print("=" * 60)

    overfit_df = train_df.iloc[:32].copy()
    print(f"Testing overfit on {len(overfit_df)} samples across {overfit_df['cyclone_id'].nunique()} cyclones.")

    # Override config for overfit test
    test_cfg = config.copy()
    test_cfg["training"] = test_cfg["training"].copy()
    test_cfg["training"]["epochs"] = 150
    test_cfg["training"]["learning_rate"] = 5e-4
    test_cfg["dataset"] = test_cfg["dataset"].copy()
    test_cfg["dataset"]["batch_size"] = 16
    test_cfg["dataset"]["num_workers"] = 0
    test_cfg["augmentation"] = {"enabled": False}

    train_loader, _, _ = build_dataloaders(
        h5_path=h5_path,
        train_df=overfit_df,
        val_df=overfit_df,
        test_df=overfit_df,
        config=test_cfg,
        mean=mean,
        std=std,
        in_memory=True
    )

    model = build_model(test_cfg).to(device)
    loss_fn = build_loss_fn(test_cfg)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)

    model.train()
    initial_loss = None
    final_loss = None

    for step in range(1, 151):
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()

        if step % 25 == 0 or step == 1:
            print(f"  Step [{step:3d}/150] Overfit MSE Loss: {final_loss:.4f}")

    print(f"\nInitial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    passed = final_loss < 2.0  # MSE loss should drop to near zero (< 2.0 knots^2, i.e. < 1.4 kt error)
    if passed:
        print("OVERFIT DIAGNOSTIC: PASS (Model easily memorized tiny batch).")
    else:
        print("OVERFIT DIAGNOSTIC: FAIL (Optimization stalled).")
    print("=" * 60)
    return passed


def run_smoke_test(config: dict, h5_path: Path, train_df: pd.DataFrame, val_df: pd.DataFrame, mean: float, std: float, device: torch.device) -> bool:
    """Step 9: Small smoke training test (~100-300 samples, 2 epochs)."""
    print("\n" + "=" * 60)
    print("STEP 9: SMALL SMOKE TEST (100-300 SAMPLES, 2 EPOCHS)")
    print("=" * 60)

    smoke_train_df = train_df.iloc[:200].copy()
    smoke_val_df = val_df.iloc[:50].copy()

    smoke_cfg = config.copy()
    smoke_cfg["training"] = smoke_cfg["training"].copy()
    smoke_cfg["training"]["epochs"] = 2
    smoke_cfg["dataset"] = smoke_cfg["dataset"].copy()
    smoke_cfg["dataset"]["batch_size"] = 32
    smoke_cfg["dataset"]["num_workers"] = 2

    train_loader, val_loader, _ = build_dataloaders(
        h5_path=h5_path,
        train_df=smoke_train_df,
        val_df=smoke_val_df,
        test_df=smoke_val_df,
        config=smoke_cfg,
        mean=mean,
        std=std,
        in_memory=True
    )

    save_dir = Path("experiments/smoke_test_run")
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_manager = CheckpointManager(save_dir=save_dir)

    model = build_model(smoke_cfg).to(device)
    loss_fn = build_loss_fn(smoke_cfg)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        config=smoke_cfg,
        checkpoint_manager=ckpt_manager
    )

    trainer.fit()
    print("SMOKE TEST: PASS (Pipeline, DataLoader, AMP, and Checkpointing verified).")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Train Cyclone Intensity Estimation baseline.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    parser.add_argument("--overfit-test", action="store_true", help="Run 16-32 sample overfit diagnostic only")
    parser.add_argument("--smoke-test", action="store_true", help="Run small smoke test only")
    parser.add_argument("--evaluate-only", action="store_true", help="Run evaluation on test set only")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.get("training", {}).get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using computing device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    ds_cfg = config.get("dataset", {})
    regions = ds_cfg.get("regions", ["CPAC", "IO", "SH"])
    regions_str = "_".join(sorted(regions))
    raw_dir = Path(ds_cfg.get("raw_dir", "data/raw"))
    metadata_dir = Path(ds_cfg.get("metadata_dir", "data/metadata"))

    # Determine if all_basins or standard baseline
    if ds_cfg.get("name") == "TCIR_ALL_BASINS" or "ATLN" in regions:
        h5_path = None
        train_path = metadata_dir / "train_metadata_all_basins.csv"
        val_path = metadata_dir / "val_metadata_all_basins.csv"
        test_path = metadata_dir / "test_metadata_all_basins.csv"
        stats_path = Path(ds_cfg.get("normalization_file", metadata_dir / "normalization_stats_all_basins.json"))

        if not (train_path.exists() and val_path.exists() and test_path.exists() and stats_path.exists()):
            from scripts.prepare_all_basins import prepare_all_basins_dataset
            print("[Train] All-basins metadata splits not found. Running preparation...")
            prepare_all_basins_dataset(args.config)
            
            # Split metadata into train/val/test CSVs for dataloader
            df_all = pd.read_csv(metadata_dir / "metadata_all_basins.csv")
            with open(metadata_dir / "splits_all_basins.json", "r") as f:
                s = json.load(f)
            train_cids = set(s["train"]["cyclone_ids"])
            val_cids = set(s["val"]["cyclone_ids"])
            test_cids = set(s["test"]["cyclone_ids"])

            df_all[df_all["cyclone_id"].isin(train_cids)].to_csv(train_path, index=False)
            df_all[df_all["cyclone_id"].isin(val_cids)].to_csv(val_path, index=False)
            df_all[df_all["cyclone_id"].isin(test_cids)].to_csv(test_path, index=False)
    elif regions == ["IO"]:
        h5_path = Path(ds_cfg.get("h5_path", "data/raw/TCIR-CPAC_IO_SH.h5"))
        train_path = metadata_dir / "train_metadata_IO.csv"
        val_path = metadata_dir / "val_metadata_IO.csv"
        test_path = metadata_dir / "test_metadata_IO.csv"
        stats_path = Path(ds_cfg.get("normalization_file", metadata_dir / "normalization_stats_IO.json"))

        if not (train_path.exists() and val_path.exists() and test_path.exists() and stats_path.exists()):
            from scripts.prepare_io_dataset import prepare_io_dataset
            print("[Train] IO metadata splits not found. Automatically running IO dataset preparation...")
            prepare_io_dataset()
    else:
        h5_path = download_tcir_archive(key="CPAC_IO_SH", destination_dir=raw_dir, extract=True)
        train_path = metadata_dir / f"train_metadata_{regions_str}.csv"
        val_path = metadata_dir / f"val_metadata_{regions_str}.csv"
        test_path = metadata_dir / f"test_metadata_{regions_str}.csv"
        stats_path = Path(ds_cfg.get("normalization_file", metadata_dir / f"normalization_stats_{regions_str}.json"))

        if not (train_path.exists() and val_path.exists() and test_path.exists() and stats_path.exists()):
            from scripts.prepare_dataset import prepare_dataset
            print("[Train] Metadata splits not found. Automatically running dataset preparation...")
            prepare_dataset(config)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    with open(stats_path, "r", encoding="utf-8") as f:
        norm_stats = json.load(f)
    mean, std = norm_stats["mean"], norm_stats["std"]

    # Step 8: Overfit Diagnostic
    if args.overfit_test:
        passed = run_overfit_test(config, h5_path, train_df, mean, std, device)
        exit(0 if passed else 1)

    # Step 9: Smoke Test
    if args.smoke_test:
        passed = run_smoke_test(config, h5_path, train_df, val_df, mean, std, device)
        exit(0 if passed else 1)

    # Step 10: Full Baseline Training
    save_dir = Path(config.get("training", {}).get("save_dir", "experiments/baseline_resnet18_cpac_io_sh"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, save_dir / "config.yaml")

    train_loader, val_loader, test_loader = build_dataloaders(
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
    loss_fn = build_loss_fn(config)

    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_type = train_cfg.get("scheduler", "cosine").lower()
    epochs = train_cfg.get("epochs", 30)
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(train_cfg.get("min_lr", 1e-6)))
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    else:
        scheduler = None

    ckpt_manager = CheckpointManager(save_dir=save_dir)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        checkpoint_manager=ckpt_manager
    )

    print("\n" + "=" * 60)
    print("STEP 10: FULL BASELINE TRAINING (TCIR-CPAC_IO_SH)")
    print(f"  • Train: {len(train_df):,} frames ({train_df['cyclone_id'].nunique()} cyclones)")
    print(f"  • Val:   {len(val_df):,} frames ({val_df['cyclone_id'].nunique()} cyclones)")
    print(f"  • Test:  {len(test_df):,} frames ({test_df['cyclone_id'].nunique()} cyclones)")
    print("=" * 60)

    train_results = trainer.fit()

    # Step 11: Final Evaluation on Held-Out Test Set
    print("\n" + "=" * 60)
    print("STEP 11: TEST SET EVALUATION ON BEST CHECKPOINT")
    print("=" * 60)

    best_ckpt_path = save_dir / "best.pt"
    ckpt_manager.load(best_ckpt_path, model, device=device)

    eval_results = evaluate_model_on_dataset(
        model=model,
        data_loader=test_loader,
        device=device,
        use_amp=config.get("training", {}).get("use_amp", True)
    )

    test_metrics = generate_evaluation_artifacts(
        eval_results=eval_results,
        output_dir=save_dir,
        experiment_name=f"ResNet18 Baseline ({regions_str})"
    )

    print(f"\nBaseline run complete. Artifacts stored in: {save_dir}")


if __name__ == "__main__":
    main()
