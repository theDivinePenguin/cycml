"""Forensic Audit Script: Controlled Inference-Time Ablation Studies (No Retraining).

Evaluates:
1. Exact replication of test benchmark on held-out test set
2. Temporal contribution ablation:
   - Normal K=7
   - Static frame repeated (all frames = frame t)
   - Reversed frames (t, t-3, ..., t-18)
   - Shuffled historical frames
   - History zeroed (only frame t retained)
3. Environmental contribution ablation:
   - Normal environment
   - Shuffled environment between test samples
   - Zeroed environment with missing masks set to 1
   - Satellite-only (x_env=None)
   - Feature knockout (zeroing Vmax, MSLP, Shear, SST, etc.)
"""
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc

from src.data.trend_config import IntensityTrendConfig
from src.data.trend_dataset import build_trend_dataloaders
from src.models.environmental_temporal_classifier import EnvironmentalTemporalClassifier

@torch.no_grad()
def eval_subset(model, loader, device, tau=0.0161, mode="normal", max_batches=None):
    model.eval()
    all_p24, all_t24 = [], []
    all_ri_probs, all_ri_true = [], []
    all_tr_preds, all_tr_true = [], []
    
    rng = np.random.RandomState(42)

    for b_idx, batch in enumerate(loader):
        if max_batches and b_idx >= max_batches:
            break
            
        images, vis_masks, trend_targets, ri_targets, reg_targets, env_vec, meta = batch
        images = images.to(device, non_blocking=True)
        vis_masks = vis_masks.to(device, non_blocking=True)
        env_vec = env_vec.to(device, non_blocking=True)
        
        # Apply ablation modes
        if mode == "normal":
            pass
        elif mode == "temp_repeat_last":
            # Replace all frames with frame t (last frame index 6)
            images = images[:, -1:, :, :, :].repeat(1, images.shape[1], 1, 1, 1)
            vis_masks = vis_masks[:, -1:].repeat(1, vis_masks.shape[1])
        elif mode == "temp_reverse":
            # Reverse along sequence dimension (dim 1)
            images = torch.flip(images, dims=[1])
            vis_masks = torch.flip(vis_masks, dims=[1])
        elif mode == "temp_shuffle_history":
            # Permute indices 0..5
            perm = torch.randperm(images.shape[1] - 1)
            perm_full = torch.cat([perm, torch.tensor([images.shape[1] - 1])])
            images = images[:, perm_full, :, :, :]
            vis_masks = vis_masks[:, perm_full]
        elif mode == "temp_zero_history":
            # Zero out frames 0..5
            images[:, :-1, :, :, :] = 0.0
            vis_masks[:, :-1] = 0.0
        elif mode == "env_zero":
            # Set continuous features to 0.0, missing masks to 1.0
            env_vec[:, :6] = 0.0
            env_vec[:, 6:] = 1.0
        elif mode == "env_shuffle":
            # Permute env_vec rows across batch
            perm = torch.randperm(env_vec.size(0))
            env_vec = env_vec[perm]
        elif mode == "env_none":
            env_vec = None
        elif mode == "env_no_vmax":
            env_vec[:, 0] = 0.0 # vmax
            env_vec[:, 6] = 1.0 # mask_vmax
        elif mode == "env_no_thermo_kinematic":
            # Zero out SST, OHC, Shear, RH (indices 2, 3, 4, 5)
            env_vec[:, 2:6] = 0.0
            env_vec[:, 8:12] = 1.0

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            ri_logits, trend_logits, reg_preds = model(images, vis_masks, env_vec)

        ri_probs = torch.sigmoid(ri_logits).squeeze(-1).cpu().numpy()
        tr_preds = np.argmax(trend_logits.cpu().numpy(), axis=-1)
        p24 = reg_preds[:, 2].cpu().numpy()

        all_p24.append(p24)
        all_t24.append(reg_targets[:, 2].numpy())
        all_ri_probs.append(ri_probs)
        all_ri_true.append(ri_targets.numpy())
        all_tr_preds.append(tr_preds)
        all_tr_true.append(trend_targets.numpy())

    p24_arr = np.concatenate(all_p24)
    t24_arr = np.concatenate(all_t24)
    ri_p_arr = np.concatenate(all_ri_probs)
    ri_t_arr = np.concatenate(all_ri_true)
    tr_p_arr = np.concatenate(all_tr_preds)
    tr_t_arr = np.concatenate(all_tr_true)

    mae24 = mean_absolute_error(t24_arr, p24_arr)
    acc_tr = accuracy_score(tr_t_arr, tr_p_arr)
    f1_tr = f1_score(tr_t_arr, tr_p_arr, average="macro")
    
    prec, rec, _ = precision_recall_curve(ri_t_arr, ri_p_arr)
    pr_auc = auc(rec, prec)
    roc_auc = roc_auc_score(ri_t_arr, ri_p_arr)
    
    # F1 at tau
    ri_pred_bin = (ri_p_arr >= tau).astype(int)
    f1_ri = f1_score(ri_t_arr, ri_pred_bin, zero_division=0)
    rec_ri = ((ri_t_arr == 1) & (ri_pred_bin == 1)).sum() / max(1, ri_t_arr.sum())

    return {
        "mae_24": float(mae24),
        "trend_acc": float(acc_tr),
        "trend_macro_f1": float(f1_tr),
        "ri_pr_auc": float(pr_auc),
        "ri_roc_auc": float(roc_auc),
        "ri_f1_tau": float(f1_ri),
        "ri_rec_tau": float(rec_ri)
    }

def main():
    print("=" * 80)
    print("FORENSIC AUDIT 4: INFERENCE-TIME ABLATION SUITE (NO RETRAINING)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt_path = "experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    tau = ckpt.get("best_tau", 0.0161)
    print(f"Loaded checkpoint {ckpt_path} (best_tau = {tau:.4f})")

    model = EnvironmentalTemporalClassifier(
        channels=3,
        num_frames=7,
        d_model=256,
        n_heads=8,
        num_layers=2,
        dropout=0.1,
        use_vis_channel=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load data
    train_df = pd.read_csv("data/metadata/forecast_train_sequences_k7.csv")
    val_df = pd.read_csv("data/metadata/forecast_val_sequences_k7.csv")
    test_df = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")
    env_cache = torch.load("data/metadata/environmental_features_k7.pt")
    test_env = env_cache["test"]

    with open("data/metadata/normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)
    norm_mean = [norm_stats["mean"][c] for c in [0, 1, 2]]
    norm_std = [norm_stats["std"][c] for c in [0, 1, 2]]

    config = IntensityTrendConfig()
    _, _, test_loader = build_trend_dataloaders(
        train_seq_df=train_df,
        val_seq_df=val_df,
        test_seq_df=test_df,
        mean=norm_mean,
        std=norm_std,
        channels=[0, 1, 2],
        batch_size=32,
        num_workers=4,
        config=config,
        test_env_tensor=test_env,
    )

    modes = [
        ("Normal (Full Model)", "normal"),
        ("Ablation: Static Frame Repeated (No History)", "temp_repeat_last"),
        ("Ablation: Reversed Time Order", "temp_reverse"),
        ("Ablation: Shuffled History Frames", "temp_shuffle_history"),
        ("Ablation: Zeroed History (t only)", "temp_zero_history"),
        ("Ablation: Zeroed Environment (Masks=1)", "env_zero"),
        ("Ablation: Shuffled Environment", "env_shuffle"),
        ("Ablation: Satellite-Only (x_env=None)", "env_none"),
        ("Ablation: Environment Without Vmax", "env_no_vmax"),
        ("Ablation: Environment Without SST/OHC/Shear/RH", "env_no_thermo_kinematic"),
    ]

    results = {}
    print(f"\n{'Condition':<46} {'+24h MAE':<10} {'Tr Acc':<9} {'Tr F1':<9} {'RI PR-AUC':<11} {'RI F1':<8}")
    print("-" * 95)

    for desc, mode in modes:
        t0 = time.time()
        res = eval_subset(model, test_loader, device, tau=tau, mode=mode)
        elapsed = time.time() - t0
        print(f"{desc:<46} {res['mae_24']:<10.2f} {res['trend_acc']*100:<8.1f}% {res['trend_macro_f1']:<9.4f} {res['ri_pr_auc']:<11.4f} {res['ri_f1_tau']:<8.4f} ({elapsed:.1f}s)")
        results[mode] = {
            "description": desc,
            "metrics": res,
            "runtime_sec": round(elapsed, 2)
        }

    out_file = Path("experiments/forensic_audit/inference_ablations.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {out_file}")

if __name__ == "__main__":
    main()
