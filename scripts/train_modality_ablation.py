"""Orchestrator for sequential training and test evaluation of the TCIR 8-Way Modality Ablation Study."""
import gc
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import build_dataloaders
from src.evaluation.metrics import calculate_metrics
from src.models.factory import build_model
from src.training.losses import build_loss_fn
from src.training.checkpoint import CheckpointManager
from src.training.train import Trainer
from src.utils.config import load_config, save_config


ABLATION_RUNS = [
    {
        "name": "Exp B: IR1 + WV",
        "config_path": "configs/ablation_ir1_wv.yaml",
        "save_dir": "experiments/modality_ablation/ir1_wv",
        "channels": [0, 1],
    },
    {
        "name": "Exp C: IR1 + VIS",
        "config_path": "configs/ablation_ir1_vis.yaml",
        "save_dir": "experiments/modality_ablation/ir1_vis",
        "channels": [0, 2],
    },
    {
        "name": "Exp D: IR1 + PMW",
        "config_path": "configs/ablation_ir1_pmw.yaml",
        "save_dir": "experiments/modality_ablation/ir1_pmw",
        "channels": [0, 3],
    },
    {
        "name": "Exp E: IR1 + WV + VIS",
        "config_path": "configs/ablation_ir1_wv_vis.yaml",
        "save_dir": "experiments/modality_ablation/ir1_wv_vis",
        "channels": [0, 1, 2],
    },
    {
        "name": "Exp F: IR1 + WV + PMW",
        "config_path": "configs/ablation_ir1_wv_pmw.yaml",
        "save_dir": "experiments/modality_ablation/ir1_wv_pmw",
        "channels": [0, 1, 3],
    },
    {
        "name": "Exp G: IR1 + VIS + PMW",
        "config_path": "configs/ablation_ir1_vis_pmw.yaml",
        "save_dir": "experiments/modality_ablation/ir1_vis_pmw",
        "channels": [0, 2, 3],
    },
]


def train_and_evaluate_ablation_model(run_info: dict, device: torch.device):
    name = run_info["name"]
    cfg_path = run_info["config_path"]
    save_dir = Path(run_info["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    pred_csv_path = save_dir / "test_predictions.csv"
    metrics_json_path = save_dir / "test_metrics.json"
    if pred_csv_path.exists() and metrics_json_path.exists():
        print(f"\n[{name}] Already completed with test predictions and metrics. Skipping.")
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("\n" + "=" * 90)
    print(f"STARTING ABLATION TRAINING: {name}")
    print(f"  • Config:     {cfg_path}")
    print(f"  • Save Dir:   {save_dir}")
    print(f"  • Channels:   {run_info['channels']}")
    print(f"  • Device:     {device}")
    print("=" * 90)

    config = load_config(cfg_path)
    save_config(config, save_dir / "config.yaml")

    ds_cfg = config.get("dataset", {})
    metadata_dir = Path(ds_cfg.get("metadata_dir", "data/metadata"))
    train_path = metadata_dir / "train_metadata_all_basins.csv"
    val_path = metadata_dir / "val_metadata_all_basins.csv"
    test_path = metadata_dir / "test_metadata_all_basins.csv"
    stats_path = Path(ds_cfg.get("normalization_file", metadata_dir / "normalization_stats_multichannel.json"))

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    with open(stats_path, "r", encoding="utf-8") as f:
        norm_stats = json.load(f)
    mean, std = norm_stats["mean"], norm_stats["std"]

    train_loader, val_loader, test_loader = build_dataloaders(
        h5_path=None,
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

    t_cfg = config.get("training", {})
    lr = float(t_cfg.get("learning_rate", 1e-4))
    wd = float(t_cfg.get("weight_decay", 1e-4))
    epochs = int(t_cfg.get("epochs", 30))
    min_lr = float(t_cfg.get("min_lr", 1e-6))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    checkpoint_manager = CheckpointManager(save_dir=save_dir)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        checkpoint_manager=checkpoint_manager
    )

    t0 = time.time()
    history = trainer.fit()
    duration_min = (time.time() - t0) / 60.0

    # -------------------------------------------------------------
    # LEGACY SCRIPT SAFEGUARD: Test set evaluation is locked
    # -------------------------------------------------------------
    import sys
    best_ckpt_path = save_dir / "best.pt"
    best_ckpt = torch.load(best_ckpt_path, map_location=device)

    if not ("--eval-test" in sys.argv and "--confirm-locked-test-eval" in sys.argv):
        print(f"\n[TEST LOCK PROTECTED] Training of {name} finished in {duration_min:.2f} min. Test set evaluation is locked.")
        print("  To evaluate test set, use canonical runner: python evaluate.py --split test --eval-test --confirm-locked-test-eval")
        return {"status": "SUCCESS_TRAIN_ONLY_TEST_LOCKED", "val_mae": float(best_ckpt.get("val_mae", -1.0))}

    print(f"\n[{name}] Training finished in {duration_min:.2f} min. Loading best checkpoint for test evaluation...")
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            all_preds.extend(outputs.view(-1).cpu().numpy())
            all_targets.extend(targets.view(-1).numpy())

    preds_np = np.array(all_preds, dtype=np.float32)
    targets_np = np.array(all_targets, dtype=np.float32)

    # Save Predictions CSV
    test_preds_df = test_df.copy()
    test_preds_df["predicted_wind_speed"] = preds_np
    test_preds_df["absolute_error"] = np.abs(preds_np - targets_np)
    test_preds_df["error"] = preds_np - targets_np
    preds_csv_path = save_dir / "test_predictions.csv"
    test_preds_df.to_csv(preds_csv_path, index=False)

    # Calculate Test Metrics
    test_metrics = calculate_metrics(targets_np, preds_np)
    test_metrics["best_epoch"] = int(best_ckpt.get("epoch", -1))
    test_metrics["best_val_mae"] = float(best_ckpt.get("val_mae", -1.0))
    test_metrics["training_duration_minutes"] = round(duration_min, 2)
    test_metrics["channels"] = run_info["channels"]

    metrics_json_path = save_dir / "test_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"[{name}] Test Set Performance:")
    print(f"  • Test MAE:       {test_metrics['mae']:.3f} kt")
    print(f"  • Test RMSE:      {test_metrics['rmse']:.3f} kt")
    print(f"  • Test R²:        {test_metrics['r2']:.4f}")
    print(f"  • Test Bias:      {test_metrics['mean_bias']:+.3f} kt")
    print(f"  • Best Epoch:     {test_metrics['best_epoch']}")
    print(f"  • Best Val MAE:   {test_metrics['best_val_mae']:.3f} kt")
    print("=" * 90)

    # Clean up GPU memory
    del model, optimizer, scheduler, trainer, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    return test_metrics


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Modality Ablation] Starting execution on device: {device}")

    all_results = {}
    for run in ABLATION_RUNS:
        res = train_and_evaluate_ablation_model(run, device)
        all_results[run["name"]] = res

    print("\n" + "=" * 90)
    print("ALL ABLATION TRAINING RUNS COMPLETED SUCCESSFULLY!")
    print("EXECUTING STATISTICAL EVALUATION, BOOTSTRAP SIGNIFICANCE & FIGURES...")
    print("=" * 90)

    from scripts.evaluate_modality_ablation import main as run_evaluation
    run_evaluation()

    print("\n" + "=" * 90)
    print("GENERATING COMPREHENSIVE WORD DOCUMENT REPORT...")
    print("=" * 90)
    from scripts.generate_word_report import build_word_document
    docx_path = build_word_document()
    print(f"\n[Ablation Pipeline Finished] Complete Word Report generated: {docx_path}")


if __name__ == "__main__":
    main()
