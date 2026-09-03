"""Export detailed diagnostic data for multi-horizon forecast lifecycle plots to investigate temporal alignment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.data.sequence_dataset import TCIRSequenceDataset
from src.models.temporal_forecaster import TemporalGRUForecaster, TemporalTransformerForecaster
from scripts.build_forecast_sequences import build_sequences_for_df


def run_diagnostic_export():
    out_dir = Path("diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_df = pd.read_csv("data/metadata/metadata_all_basins.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained models
    tf_model = TemporalTransformerForecaster(in_channels=3, d_model=256, nhead=8, num_layers=2, pretrained_cnn=False)
    tf_ckpt = torch.load("experiments/forecasting/checkpoints/cnn_transformer_k5/best.pt", map_location=device)
    tf_model.load_state_dict(tf_ckpt["model_state_dict"])
    tf_model.to(device).eval()

    gru_model = TemporalGRUForecaster(in_channels=3, d_model=256, num_layers=2, pretrained_cnn=False)
    gru_ckpt = torch.load("experiments/forecasting/checkpoints/cnn_gru_k5/best.pt", map_location=device)
    gru_model.load_state_dict(gru_ckpt["model_state_dict"])
    gru_model.to(device).eval()

    with open("data/metadata/normalization_stats_multichannel.json") as f:
        norm_stats = json.load(f)
    mean = [norm_stats["mean"][c] for c in [0, 1, 2]]
    std = [norm_stats["std"][c] for c in [0, 1, 2]]

    cyclones_to_export = [
        ("201003I", "Super Cyclone Phet", "Held-Out TEST Set"),
        ("200801I", "VSCS Nargis", "Held-Out TEST Set"),
        ("201004I", "Super Cyclone Giri", "Validation Split"),
        ("201306I", "VSCS Madi", "Training Split"),
    ]

    all_rows = []

    for cid, cname, split_name in cyclones_to_export:
        storm_df = all_df[all_df["cyclone_id"] == cid].sort_values("timestamp").reset_index(drop=True)
        # Ensure timestamp is string in YYYYMMDDHH format
        storm_df["timestamp"] = storm_df["timestamp"].astype(str)
        # Map timestamp to actual intensity
        ts_to_vmax = dict(zip(storm_df["timestamp"], storm_df["wind_speed"]))
        all_timestamps = list(storm_df["timestamp"])
        ts_to_idx = {ts: idx for idx, ts in enumerate(all_timestamps)}

        # Build 5-frame sequence manifest for this storm
        storm_seq_df = build_sequences_for_df(storm_df, k_history=5, cadence_hours=3)
        if len(storm_seq_df) == 0:
            print(f"Skipping {cname} - not enough consecutive sequence frames")
            continue

        storm_ds = TCIRSequenceDataset(storm_seq_df, mean=mean, std=std, channels=[0, 1, 2], is_training=False)
        storm_loader = torch.utils.data.DataLoader(storm_ds, batch_size=len(storm_ds), shuffle=False)

        for imgs, masks, targets, _ in storm_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.no_grad():
                tf_preds = tf_model(imgs, masks).cpu().numpy()
                gru_preds = gru_model(imgs, masks).cpu().numpy()
            targets_np = targets.numpy()

        for seq_idx, row in storm_seq_df.iterrows():
            t_orig = str(row["target_t_timestamp"])
            orig_idx = ts_to_idx[t_orig]

            # 5 input timestamps from history_timestamps
            hist_ts_list = json.loads(row["history_timestamps"])
            t_minus_12h = str(hist_ts_list[0])
            t_minus_9h = str(hist_ts_list[1])
            t_minus_6h = str(hist_ts_list[2])
            t_minus_3h = str(hist_ts_list[3])
            t_0 = str(hist_ts_list[4])

            # Actual Vmax from t to t+24h in 3h steps
            def get_future_val(step_offset):
                target_idx = orig_idx + step_offset
                if target_idx < len(all_timestamps):
                    t_target = all_timestamps[target_idx]
                    return t_target, ts_to_vmax.get(t_target, np.nan)
                return "N/A", np.nan

            t_p0, v_p0 = get_future_val(0)
            t_p3, v_p3 = get_future_val(1)
            t_p6, v_p6 = get_future_val(2)
            t_p9, v_p9 = get_future_val(3)
            t_p12, v_p12 = get_future_val(4)
            t_p15, v_p15 = get_future_val(5)
            t_p18, v_p18 = get_future_val(6)
            t_p21, v_p21 = get_future_val(7)
            t_p24, v_p24 = get_future_val(8)

            # Model predictions
            pred_tf_6h = float(tf_preds[seq_idx, 0])
            pred_tf_12h = float(tf_preds[seq_idx, 1])
            pred_tf_24h = float(tf_preds[seq_idx, 2])

            pred_gru_6h = float(gru_preds[seq_idx, 0])
            pred_gru_12h = float(gru_preds[seq_idx, 1])
            pred_gru_24h = float(gru_preds[seq_idx, 2])

            # Persistence predictions
            pred_pers_6h = float(v_p0)
            pred_pers_12h = float(v_p0)
            pred_pers_24h = float(v_p0)

            # Plotted X coordinate in previous script: elapsed hours from origin sequence start
            plotted_x_coord_elapsed_hours = float(seq_idx * 3.0)
            plotted_x_coord_forecast_origin = t_orig

            record = {
                "cyclone_id": cid,
                "cyclone_name": cname,
                "dataset_split": split_name,
                "sequence_index": int(seq_idx),
                "forecast_origin_timestamp_t": t_orig,
                "plotted_x_coordinate_hours": plotted_x_coord_elapsed_hours,
                "input_timestamp_t_minus_12h": t_minus_12h,
                "input_timestamp_t_minus_9h": t_minus_9h,
                "input_timestamp_t_minus_6h": t_minus_6h,
                "input_timestamp_t_minus_3h": t_minus_3h,
                "input_timestamp_t": t_0,
                "actual_vmax_t": float(v_p0),
                "actual_vmax_t_plus_3h": float(v_p3),
                "actual_vmax_t_plus_6h": float(v_p6),
                "actual_vmax_t_plus_9h": float(v_p9),
                "actual_vmax_t_plus_12h": float(v_p12),
                "actual_vmax_t_plus_15h": float(v_p15),
                "actual_vmax_t_plus_18h": float(v_p18),
                "actual_vmax_t_plus_21h": float(v_p21),
                "actual_vmax_t_plus_24h": float(v_p24),
                "target_timestamp_plus_6h": t_p6,
                "target_timestamp_plus_12h": t_p12,
                "target_timestamp_plus_24h": t_p24,
                "transformer_pred_plus_6h": round(pred_tf_6h, 2),
                "transformer_pred_plus_12h": round(pred_tf_12h, 2),
                "transformer_pred_plus_24h": round(pred_tf_24h, 2),
                "gru_pred_plus_6h": round(pred_gru_6h, 2),
                "gru_pred_plus_12h": round(pred_gru_12h, 2),
                "gru_pred_plus_24h": round(pred_gru_24h, 2),
                "persistence_pred_plus_6h": round(pred_pers_6h, 2),
                "persistence_pred_plus_12h": round(pred_pers_12h, 2),
                "persistence_pred_plus_24h": round(pred_pers_24h, 2),
                "plotted_target_timestamp_in_graph": {
                    "+6h_subplot_target_timestamp": t_p6,
                    "+12h_subplot_target_timestamp": t_p12,
                    "+24h_subplot_target_timestamp": t_p24,
                    "x_axis_represents": "forecast_origin_time_t_elapsed_hours"
                }
            }
            all_rows.append(record)
        storm_ds.close()

    # Save to CSV and JSON
    export_df = pd.DataFrame(all_rows)
    csv_path = out_dir / "lifecycle_forecast_raw.csv"
    json_path = out_dir / "lifecycle_forecast_raw.json"

    export_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    print(f"\n[Diagnostic Export Complete]")
    print(f"  • CSV:  {csv_path} ({len(export_df)} rows)")
    print(f"  • JSON: {json_path}")
    return export_df, all_rows


if __name__ == "__main__":
    run_diagnostic_export()
