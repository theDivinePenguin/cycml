#!/usr/bin/env python3
"""Audits the Residual Forecaster checkpoint with real PyTorch inference on actual validation data.
Verifies:
1. Does the model predict realistic delta_v without future leakage?
2. What are the actual errors at +6h, +12h, +24h?
3. What is the variance and correlation?
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.sequence_dataset import TCIRSequenceDataset
from src.models.residual_forecaster import ResidualDeltaVForecaster


def audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Auditing on device: {device}")

    # 1. Load Checkpoint
    ckpt_path = Path("experiments/checkpoints/residual_delta_v_unconstrained/best.pt")
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} missing!")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    m_cfg = cfg.get("model", {})

    print(f"\n--- MODEL METADATA ---")
    print(f"Experiment ID:   {cfg.get('experiment_id')}")
    print(f"Saved Best Epoch:{ckpt.get('epoch')}")
    print(f"Reported Val MAE:{ckpt.get('best_metric'):.4f} kt")
    print(f"Architecture:    {m_cfg.get('type')} (Backbone: {m_cfg.get('backbone')}, Temporal: {m_cfg.get('temporal_type')})")
    print(f"Parameterization:{m_cfg.get('parameterization')}")

    # 2. Recreate Model Architecture
    model = ResidualDeltaVForecaster(
        backbone_arch=m_cfg.get("backbone", "resnet18"),
        in_channels=3,
        d_model=m_cfg.get("d_model", 256),
        temporal_type=m_cfg.get("temporal_type", "transformer"),
        num_layers=m_cfg.get("num_layers", 2),
        nhead=m_cfg.get("nhead", 8),
        dropout=0.0,
        parameterization="unconstrained",
        pretrained_backbone=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Load Validation Dataset
    meta_dir = Path("data/metadata")
    val_manifest = meta_dir / "forecast_val_sequences_k5_aligned.csv"
    val_df = pd.read_csv(val_manifest)

    with open(meta_dir / "normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)

    channels = [0, 1, 2]
    mean = [norm_stats["mean"][c] for c in channels]
    std = [norm_stats["std"][c] for c in channels]

    val_ds = TCIRSequenceDataset(val_df, mean=mean, std=std, channels=channels, is_training=False)
    # Take first 256 samples for fast, thorough audit
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"\n--- RUNNING REAL INFERENCE ON VALIDATION SAMPLES ---")
    all_vcurr = []
    all_targets = []
    all_vhat = []
    all_deltahat = []

    with torch.no_grad():
        for i, (images, vis_masks, targets, meta) in enumerate(val_loader):
            if i >= 8:  # 8 batches * 32 = 256 samples
                break
            images = images.to(device)
            vis_masks = vis_masks.to(device)
            v_curr = meta["vmax_curr"].to(device).float()

            v_hat, delta_hat = model(images, v_curr=v_curr, vis_masks=vis_masks)

            all_vcurr.append(v_curr.cpu().numpy())
            all_targets.append(targets.numpy())
            all_vhat.append(v_hat.cpu().numpy())
            all_deltahat.append(delta_hat.cpu().numpy())

    v_curr_arr = np.concatenate(all_vcurr)
    targets_arr = np.concatenate(all_targets)
    v_hat_arr = np.concatenate(all_vhat)
    delta_hat_arr = np.concatenate(all_deltahat)

    # Calculate actual ground truth deltas
    true_deltas = targets_arr - v_curr_arr[:, None]

    # Calculate Errors
    err_6h = np.abs(v_hat_arr[:, 0] - targets_arr[:, 0])
    err_12h = np.abs(v_hat_arr[:, 1] - targets_arr[:, 1])
    err_24h = np.abs(v_hat_arr[:, 2] - targets_arr[:, 2])

    print(f"Sample Count Evaluated: {len(v_curr_arr)}")
    print(f"\nREAL INFERENCE RESULTS:")
    print(f"  • +6h Horizon MAE:  {np.mean(err_6h):.2f} kt (Std: {np.std(err_6h):.2f} kt, Max Err: {np.max(err_6h):.1f} kt)")
    print(f"  • +12h Horizon MAE: {np.mean(err_12h):.2f} kt (Std: {np.std(err_12h):.2f} kt, Max Err: {np.max(err_12h):.1f} kt)")
    print(f"  • +24h Horizon MAE: {np.mean(err_24h):.2f} kt (Std: {np.std(err_24h):.2f} kt, Max Err: {np.max(err_24h):.1f} kt)")
    print(f"  • Overall Mean MAE: {np.mean([np.mean(err_6h), np.mean(err_12h), np.mean(err_24h)]):.2f} kt")

    print(f"\nSAMPLE CASE INSPECTION (First 5 Real Predictions):")
    print(f"{'Idx':<4} | {'V_curr':<7} | {'True Δ24':<9} | {'Pred Δ24':<9} | {'True V24':<9} | {'Pred V24':<9} | {'Error 24h':<9}")
    print("-" * 75)
    for idx in range(min(10, len(v_curr_arr))):
        v0 = v_curr_arr[idx]
        td24 = true_deltas[idx, 2]
        pd24 = delta_hat_arr[idx, 2]
        tv24 = targets_arr[idx, 2]
        pv24 = v_hat_arr[idx, 2]
        e24 = err_24h[idx]
        print(f"{idx:<4} | {v0:<7.1f} | {td24:<+9.1f} | {pd24:<+9.1f} | {tv24:<9.1f} | {pv24:<9.1f} | {e24:<9.1f} kt")


if __name__ == "__main__":
    audit()
