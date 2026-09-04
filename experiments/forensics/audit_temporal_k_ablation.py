"""Script 4: Temporal Horizon Ablation (K=1, 3, 5, 7) Specifically on Rapid Intensification (RI) Episodes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, precision_recall_curve, auc, roc_auc_score, accuracy_score, f1_score

from src.models.environmental_temporal_classifier import EnvironmentalTemporalClassifier
from src.data.sequence_dataset import TCIRSequenceDataset

@torch.no_grad()
def run():
    print("=" * 80)
    print("FORENSIC INVESTIGATION: TEMPORAL SEQUENCE LENGTH (K) SPECIFICALLY ON RI")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load canonical K=7 model
    ckpt_path = "experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    tau_val = 0.0161

    model = EnvironmentalTemporalClassifier(
        channels=3, num_frames=7, d_model=256, n_heads=8, num_layers=2, dropout=0.1, use_vis_channel=True
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load test data
    test_seq = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")
    test_seq["actual_dv24"] = test_seq["vmax_plus_24h"] - test_seq["vmax_curr"]
    env_cache = torch.load("data/metadata/environmental_features_k7.pt")
    test_env = env_cache["test"].to(device)

    with open("data/metadata/normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)
    norm_mean = [norm_stats["mean"][c] for c in [0, 1, 2]]
    norm_std = [norm_stats["std"][c] for c in [0, 1, 2]]

    ds = TCIRSequenceDataset(test_seq, mean=norm_mean, std=norm_std, channels=[0, 1, 2], is_training=False)

    # Filter to RI indices (dv24 >= 30)
    ri_indices = test_seq.index[test_seq["actual_dv24"] >= 30].tolist()
    print(f"Total RI test sequences to evaluate: {len(ri_indices)}")

    # We test K=1, K=3, K=5, K=7 on these RI sequences
    # For K=1: frames 0..5 zeroed, frame 6 (t) active
    # For K=3: frames 0..3 zeroed, frames 4..6 (t-6h, t-3h, t) active
    # For K=5: frames 0..1 zeroed, frames 2..6 (t-12h, ..., t) active
    # For K=7: all frames active
    k_configs = [
        ("K=7 (18h History, Canonical)", 0),
        ("K=5 (12h History)", 2),
        ("K=3 (6h History)", 4),
        ("K=1 (0h History, Single Frame)", 6),
    ]

    k_results = {}
    print(f"\n{'Temporal Context':<32} {'RI +24h MAE':<14} {'RI Mean Pred ΔV':<18} {'RI Recall':<12} {'RI Mean Prob':<14} {'Trend Acc':<10}")
    print("-" * 100)

    for desc, zero_up_to in k_configs:
        all_pred24 = []
        all_ri_probs = []
        all_trend_preds = []

        # Batch inference over RI sequences
        batch_size = 64
        for start_i in range(0, len(ri_indices), batch_size):
            batch_idxs = ri_indices[start_i:start_i + batch_size]
            b_tensors = []
            b_masks = []
            for b_idx in batch_idxs:
                st, vm, _, _ = ds[b_idx]
                b_tensors.append(st)
                b_masks.append(vm)

            img_batch = torch.stack(b_tensors).to(device)
            mask_batch = torch.stack(b_masks).to(device)
            env_batch = test_env[batch_idxs]

            # Zero out history before active window
            if zero_up_to > 0:
                img_batch[:, :zero_up_to, :, :, :] = 0.0
                mask_batch[:, :zero_up_to] = 0.0

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                ri_l, tr_l, reg_p = model(img_batch, mask_batch, env_batch)

            ri_p = torch.sigmoid(ri_l).squeeze(-1).cpu().numpy()
            tr_p = np.argmax(tr_l.cpu().numpy(), axis=-1)
            p24 = reg_p[:, 2].cpu().numpy()

            all_pred24.extend(p24)
            all_ri_probs.extend(ri_p)
            all_trend_preds.extend(tr_p)

        all_pred24 = np.array(all_pred24)
        all_ri_probs = np.array(all_ri_probs)
        all_trend_preds = np.array(all_trend_preds)

        act24 = test_seq.loc[ri_indices, "vmax_plus_24h"].values
        vcurr = test_seq.loc[ri_indices, "vmax_curr"].values
        act_dv = act24 - vcurr
        pred_dv = all_pred24 - vcurr

        mae_24 = mean_absolute_error(act24, all_pred24)
        mean_p_dv = np.mean(pred_dv)
        ri_rec = np.mean(all_ri_probs >= tau_val) * 100
        ri_mean_p = np.mean(all_ri_probs)
        # Trend is actual intensifying (class 2)
        tr_acc = np.mean(all_trend_preds == 2) * 100

        print(f"{desc:<32} {mae_24:<14.2f} {mean_p_dv:<+18.1f} {ri_rec:<11.1f}% {ri_mean_p:<14.4f} {tr_acc:<9.1f}%")

        k_results[desc] = {
            "mae_24": float(mae_24),
            "mean_pred_dv": float(mean_p_dv),
            "ri_recall": float(ri_rec),
            "ri_mean_prob": float(ri_mean_p),
            "trend_acc": float(tr_acc)
        }

    ds.close()

    out_file = Path("experiments/forensics/temporal_k_ri_ablation.json")
    with open(out_file, "w") as f:
        json.dump(k_results, f, indent=2)
    print(f"\nSaved K ablation on RI to {out_file}")

if __name__ == "__main__":
    run()
