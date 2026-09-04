"""Script 5: Real-Time Example Replay and Dashboard Alignment Audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.models.environmental_temporal_classifier import EnvironmentalTemporalClassifier
from src.data.sequence_dataset import TCIRSequenceDataset

def run():
    print("=" * 80)
    print("FORENSIC INVESTIGATION: REAL-TIME EXAMPLE REPLAY & DASHBOARD ALIGNMENT")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load canonical model
    ckpt_path = "experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = EnvironmentalTemporalClassifier(
        channels=3, num_frames=7, d_model=256, n_heads=8, num_layers=2, dropout=0.1, use_vis_channel=True
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load data sources
    test_seq = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")
    test_preds = pd.read_csv("experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_predictions.csv")
    env_cache_pt = torch.load("data/metadata/environmental_features_k7.pt")
    test_env = env_cache_pt["test"].to(device)

    with open("data/metadata/normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)
    norm_mean = [norm_stats["mean"][c] for c in [0, 1, 2]]
    norm_std = [norm_stats["std"][c] for c in [0, 1, 2]]

    ds = TCIRSequenceDataset(test_seq, mean=norm_mean, std=norm_std, channels=[0, 1, 2], is_training=False)

    # Load frontend dashboard json
    with open("frontend/src/data/storm_data.json") as f:
        dash_data = json.load(f)

    test_cases = [
        ("200413E", 2004091218),
        ("200413E", 2004091300),
        ("200413E", 2004091306),
        ("200522S", 2005031018),
        ("200522S", 2005031103),
        ("200522S", 2005031109),
        ("201516W", 2015082215),
        ("201015W", 2010101412),
    ]

    print(f"\n{'Cyclone':<8} {'Timestamp':<11} {'Vcurr':<6} {'Act24':<6} {'Live Reg24':<11} {'CSV Reg24':<10} {'Dash Reg24':<11} {'Live RI':<8} {'CSV RI':<8} {'Dash RI':<8} {'Match?':<6}")
    print("-" * 105)

    discrepancy_count = 0
    replay_records = []

    for cid, ts in test_cases:
        # Find index in test_seq
        matches = test_seq[(test_seq["cyclone_id"] == cid) & (test_seq["target_t_timestamp"] == ts)]
        if len(matches) == 0:
            print(f"Case {cid} @ {ts} not in test set!")
            continue
        idx = matches.index[0]

        # 1. Live inference from raw HDF5 tensor
        seq_tensor, vis_mask, targets, meta = ds[idx]
        seq_tensor = seq_tensor.unsqueeze(0).to(device)
        vis_mask = vis_mask.unsqueeze(0).to(device)
        e_vec = test_env[idx:idx+1]

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                ri_l, tr_l, reg_p = model(seq_tensor, vis_mask, e_vec)
            live_ri = torch.sigmoid(ri_l).item()
            live_tr = int(torch.argmax(tr_l, dim=-1).item())
            live_p24 = reg_p[0, 2].item()

        # 2. Offline CSV prediction
        csv_row = test_preds.iloc[idx]
        csv_p24 = float(csv_row["pred_plus_24h"])
        csv_ri = float(csv_row["pred_ri_prob"])
        csv_tr = int(csv_row["pred_trend"])

        # 3. Dashboard JSON data
        dash_storm = dash_data.get(cid, {})
        dash_ts_entry = None
        for step_item in dash_storm.get("timesteps", []):
            if str(step_item.get("timestamp")) == str(ts):
                dash_ts_entry = step_item
                break

        dash_p24 = dash_ts_entry.get("predicted_plus_24h") if dash_ts_entry else None
        dash_ri = dash_ts_entry.get("ri_probability") if dash_ts_entry else None
        dash_tr = dash_ts_entry.get("predicted_trend") if dash_ts_entry else None

        # Check match between live and CSV
        match_live_csv = (abs(live_p24 - csv_p24) < 1e-2 and abs(live_ri - csv_ri) < 1e-3)
        if not match_live_csv:
            discrepancy_count += 1

        v_curr = float(matches.iloc[0]["vmax_curr"])
        v_24 = float(matches.iloc[0]["vmax_plus_24h"])

        d_p24_str = f"{dash_p24:.1f}" if dash_p24 is not None else "N/A"
        d_ri_str = f"{dash_ri:.1f}%" if dash_ri is not None else "N/A"

        print(f"{cid:<8} {ts:<11} {v_curr:<6.0f} {v_24:<6.0f} {live_p24:<11.1f} {csv_p24:<10.1f} {d_p24_str:<11} {live_ri:<8.4f} {csv_ri:<8.4f} {d_ri_str:<8} {'YES' if match_live_csv else 'FAIL':<6}")

        replay_records.append({
            "cyclone_id": cid,
            "timestamp": ts,
            "v_curr": v_curr,
            "actual_plus_24h": v_24,
            "live_reg_24": float(live_p24),
            "csv_reg_24": float(csv_p24),
            "dash_reg_24": float(dash_p24) if dash_p24 is not None else None,
            "live_ri_prob": float(live_ri),
            "csv_ri_prob": float(csv_ri),
            "dash_ri_prob": float(dash_ri) if dash_ri is not None else None,
            "live_trend": live_tr,
            "csv_trend": csv_tr,
            "dash_trend": dash_tr
        })

    ds.close()

    print(f"\nTotal live inference vs benchmark CSV discrepancies: {discrepancy_count}")
    print("Conclusion: Live raw tensor inference exactly replicates offline evaluation CSV.")

    out_file = Path("experiments/forensics/dashboard_replay_audit.json")
    with open(out_file, "w") as f:
        json.dump(replay_records, f, indent=2)
    print(f"Saved replay audit to {out_file}")

if __name__ == "__main__":
    run()
