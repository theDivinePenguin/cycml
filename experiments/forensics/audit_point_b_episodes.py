"""Script 2: Deep Dive into Point B Failure Cases (Cyclone Ingrid 200522S, Hurricane Javier 200413E, etc.)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.models.environmental_temporal_classifier import EnvironmentalTemporalClassifier

def run():
    print("=" * 80)
    print("FORENSIC INVESTIGATION: DEEP DIVE INTO POINT B CASES")
    print("=" * 80)

    pred_csv = "experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_predictions.csv"
    df = pd.read_csv(pred_csv)
    test_seq = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")
    test_env_csv = pd.read_csv("data/metadata/environmental_cache_k7_test.csv")
    env_cache_pt = torch.load("data/metadata/environmental_features_k7.pt")
    test_env_tensor = env_cache_pt["test"]

    # Combine metadata
    df["actual_dv24"] = df["vmax_plus_24h"] - df["vmax_curr"]
    df["pred_dv24"] = df["pred_plus_24h"] - df["vmax_curr"]
    df["sst"] = test_env_csv["sst"]
    df["ohc"] = test_env_csv["cohc"]
    df["shear"] = test_env_csv["shrd"]
    df["rh"] = test_env_csv["rhmd"]
    df["mslp"] = test_env_csv["mslp"]
    df["has_env"] = test_env_csv["has_env_data"]
    df["missing_shear"] = test_env_csv["missing_shrd"]

    past_6 = []
    past_12 = []
    for _, r in test_seq.iterrows():
        hv = json.loads(r["history_vmax"])
        past_6.append(hv[6] - hv[4])
        past_12.append(hv[6] - hv[2])
    df["past_6h_dv"] = past_6
    df["past_12h_dv"] = past_12

    # Focus on Cyclone Ingrid (200522S) and Hurricane Javier (200413E)
    target_storms = ["200522S", "200413E", "201516W"]

    for cid in target_storms:
        sdf = df[df["cyclone_id"] == cid].sort_values("target_t_timestamp").reset_index(drop=True)
        print(f"\n{'='*80}\nSTORM ANALYSIS: CYCLONE {cid}\n{'='*80}")
        print(f"{'Timestamp':<12} {'Vcurr':<6} {'P6':<5} {'P12':<5} {'Act24':<6} {'ActΔV':<6} {'Pr24':<6} {'PrΔV':<6} {'Trend':<10} {'RI Prob':<8} {'SST':<5} {'Shear':<6} {'RH':<5} {'MSLP':<6}")
        print("-" * 95)
        for _, r in sdf.iterrows():
            if r["actual_dv24"] >= 20 or r["pred_trend"] == 0 and r["vmax_curr"] >= 50:
                t_name = ["WEAK", "STAB", "INTE"][int(r["pred_trend"])]
                sst_s = f"{r['sst']:.1f}" if pd.notna(r['sst']) else "NaN"
                sh_s = f"{r['shear']:.1f}" if pd.notna(r['shear']) else "NaN"
                rh_s = f"{r['rh']:.0f}" if pd.notna(r['rh']) else "NaN"
                mslp_s = f"{r['mslp']:.0f}" if pd.notna(r['mslp']) else "NaN"
                p_b = " <--- POINT B!" if (r["actual_dv24"] >= 30 and r["pred_trend"] == 0) else ""
                print(f"{int(r['target_t_timestamp']):<12} {r['vmax_curr']:<6.0f} {r['past_6h_dv']:<+5.0f} {r['past_12h_dv']:<+5.0f} {r['vmax_plus_24h']:<6.0f} {r['actual_dv24']:<+6.0f} {r['pred_plus_24h']:<6.1f} {r['pred_dv24']:<+6.1f} {t_name:<10} {r['pred_ri_prob']:<8.4f} {sst_s:<5} {sh_s:<6} {rh_s:<5} {mslp_s:<6}{p_b}")

    # Now investigate WHY the model predicts WEAKENING during these steps:
    # Let's run a feature knockout on the exact Point B rows!
    print("\n" + "=" * 80)
    print("MECHANISTIC FEATURE KNOCKOUT ON POINT B ROWS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = EnvironmentalTemporalClassifier(
        channels=3, num_frames=7, d_model=256, n_heads=8, num_layers=2, dropout=0.1, use_vis_channel=True
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load dataset to get satellite tensors
    from src.data.sequence_dataset import TCIRSequenceDataset
    with open("data/metadata/normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)
    norm_mean = [norm_stats["mean"][c] for c in [0, 1, 2]]
    norm_std = [norm_stats["std"][c] for c in [0, 1, 2]]

    ds = TCIRSequenceDataset(test_seq, mean=norm_mean, std=norm_std, channels=[0, 1, 2], is_training=False)

    # Find the Point B indices in test_seq
    point_b_indices = df[(df["actual_dv24"] >= 30) & (df["pred_trend"] == 0)].index.tolist()
    print(f"Testing {len(point_b_indices)} Point B sequences under 5 controlled input states:")

    # Select representative 5 cases:
    # e.g. Ingrid step 2005031103 (index), Javier step 2004091300
    sample_pb = point_b_indices[:8]

    print(f"\n{'Idx':<5} {'Cyclone':<8} {'Timestamp':<11} {'Vcurr':<6} {'Act24':<6} {'Condition':<25} {'Pred24':<8} {'PredTrend':<10} {'RI Prob':<8}")
    print("-" * 90)

    for idx in sample_pb:
        r = df.iloc[idx]
        seq_tensor, vis_mask, targets, meta = ds[idx]
        seq_tensor = seq_tensor.unsqueeze(0).to(device)
        vis_mask = vis_mask.unsqueeze(0).to(device)
        env_orig = test_env_tensor[idx:idx+1].clone().to(device)

        conditions = [
            ("1. Full Model (Baseline)", env_orig.clone()),
            ("2. Zero Environment", None), # satellite-only
            ("3. No Vmax in Env", env_orig.clone()),
            ("4. No Shear in Env", env_orig.clone()),
            ("5. Low Shear Injection (5kt)", env_orig.clone()),
            ("6. High SST Injection (30C)", env_orig.clone()),
        ]
        
        # Modify condition 3: zero vmax
        conditions[2][1][:, 0] = 0.0 # vmax normed
        conditions[2][1][:, 6] = 1.0 # mask_vmax

        # Modify condition 4: zero shear
        conditions[3][1][:, 4] = 0.0 # shear normed
        conditions[3][1][:, 10] = 1.0 # mask_shear

        # Modify condition 5: set shear to 5 kt (very favorable for RI!)
        # shear mean=16.02, std=9.76 -> (5 - 16.02)/9.76 = -1.129
        conditions[4][1][:, 4] = (5.0 - 16.0158) / 9.7637
        conditions[4][1][:, 10] = 0.0

        # Modify condition 6: set SST to 30.5 C (very favorable for RI!)
        # sst mean=28.29, std=1.55 -> (30.5 - 28.29)/1.55 = +1.425
        conditions[5][1][:, 2] = (30.5 - 28.2877) / 1.5477
        conditions[5][1][:, 8] = 0.0

        for cond_name, e_vec in conditions:
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    ri_l, tr_l, reg_p = model(seq_tensor, vis_mask, e_vec)
                ri_p = torch.sigmoid(ri_l).item()
                tr_p = int(torch.argmax(tr_l, dim=-1).item())
                p24 = reg_p[0, 2].item()
                t_name = ["WEAK", "STAB", "INTE"][tr_p]
                print(f"{idx:<5} {r['cyclone_id']:<8} {int(r['target_t_timestamp']):<11} {r['vmax_curr']:<6.0f} {r['vmax_plus_24h']:<6.0f} {cond_name:<25} {p24:<8.1f} {t_name:<10} {ri_p:<8.4f}")
        print("-" * 90)

    ds.close()

if __name__ == "__main__":
    run()
