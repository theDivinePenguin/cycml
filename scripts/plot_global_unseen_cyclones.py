"""Generate multi-horizon forecast lifecycle graphs for 4 major 100% unseen test cyclones across global ocean basins:
1. 201015W: Super Typhoon Megi (West Pacific, 160 kt Category 5)
2. 201614L: Hurricane Matthew (North Atlantic, 145 kt Category 5)
3. 200413E: Hurricane Javier (East Pacific, 130 kt Category 4)
4. 200519S: Cyclone Percy (South Pacific / SH, 145 kt Category 5)
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.sequence_dataset import TCIRSequenceDataset
from src.models.temporal_forecaster import TemporalTransformerForecaster, TemporalGRUForecaster
from scripts.build_forecast_sequences import build_sequences_for_df


def generate_global_unseen_plots():
    out_dir = Path("diagnostics/global_cyclones")
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

    global_cyclones = [
        ("201015W", "Super Typhoon Megi", "West Pacific (WPAC)", "160 kt (Category 5)"),
        ("201614L", "Hurricane Matthew", "North Atlantic (ATLN)", "145 kt (Category 5)"),
        ("200413E", "Hurricane Javier", "East Pacific (EPAC)", "130 kt (Category 4)"),
        ("200519S", "Cyclone Percy", "South Pacific (SH)", "145 kt (Category 5)"),
    ]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    summary_stats = []

    for cid, s_name, basin, peak_info in global_cyclones:
        storm_df = all_df[all_df["cyclone_id"] == cid].sort_values("timestamp").reset_index(drop=True)
        storm_seq_df = build_sequences_for_df(storm_df, k_history=5, cadence_hours=3)
        if len(storm_seq_df) == 0:
            print(f"Skipping {s_name} - not enough sequence frames")
            continue

        storm_ds = TCIRSequenceDataset(storm_seq_df, mean=mean, std=std, channels=[0, 1, 2], is_training=False)
        storm_loader = torch.utils.data.DataLoader(storm_ds, batch_size=len(storm_ds), shuffle=False)

        for imgs, masks, targets, _ in storm_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.no_grad():
                tf_preds = tf_model(imgs, masks).cpu().numpy()
                gru_preds = gru_model(imgs, masks).cpu().numpy()
            targets_np = targets.numpy()
            v_curr_np = storm_seq_df["vmax_curr"].values

        t_origin = np.arange(len(storm_seq_df)) * 3.0

        v_act_6h = targets_np[:, 0]
        v_act_12h = targets_np[:, 1]
        v_act_24h = targets_np[:, 2]

        pred_6h = tf_preds[:, 0]
        pred_12h = tf_preds[:, 1]
        pred_24h = tf_preds[:, 2]

        mae_6h = float(np.mean(np.abs(pred_6h - v_act_6h)))
        mae_12h = float(np.mean(np.abs(pred_12h - v_act_12h)))
        mae_24h = float(np.mean(np.abs(pred_24h - v_act_24h)))

        pers_mae_6h = float(np.mean(np.abs(v_curr_np - v_act_6h)))
        pers_mae_12h = float(np.mean(np.abs(v_curr_np - v_act_12h)))
        pers_mae_24h = float(np.mean(np.abs(v_curr_np - v_act_24h)))

        summary_stats.append({
            "cyclone_id": cid,
            "name": s_name,
            "basin": basin,
            "peak": peak_info,
            "sequences": len(storm_seq_df),
            "mae_6h_tf": round(mae_6h, 2),
            "mae_6h_pers": round(pers_mae_6h, 2),
            "mae_12h_tf": round(mae_12h, 2),
            "mae_12h_pers": round(pers_mae_12h, 2),
            "mae_24h_tf": round(mae_24h, 2),
            "mae_24h_pers": round(pers_mae_24h, 2),
            "gain_24h": round(pers_mae_24h - mae_24h, 2)
        })

        # Generate 3-Panel Multi-Horizon Graph for this storm
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=180)

        # Panel A: +6h Forecast
        ax_a = axes[0]
        ax_a.axhspan(0, 34, color="#E2E8F0", alpha=0.35, label="TD (<34 kt)")
        ax_a.axhspan(34, 63, color="#FEF08A", alpha=0.3, label="TS (34-63 kt)")
        ax_a.axhspan(64, 82, color="#FED7AA", alpha=0.3, label="Cat 1 (64-82 kt)")
        ax_a.axhspan(83, 95, color="#FDBA74", alpha=0.3, label="Cat 2 (83-95 kt)")
        ax_a.axhspan(96, 112, color="#FCA5A5", alpha=0.3, label="Cat 3 (96-112 kt)")
        ax_a.axhspan(113, 175, color="#F87171", alpha=0.35, label="Cat 4/5 (113+ kt)")

        ax_a.plot(t_origin, v_act_6h, color="#0F172A", linewidth=2.8, marker="o", markersize=5, label="Actual Ground Truth (+6h)", zorder=5)
        ax_a.plot(t_origin, v_curr_np, color="#64748B", linewidth=1.8, linestyle="--", marker="s", markersize=4, label=f"Persistence (MAE: {pers_mae_6h:.1f} kt)", zorder=4)
        ax_a.plot(t_origin, pred_6h, color="#2563EB", linewidth=2.4, marker="^", markersize=5, label=f"ML Model (+6h) (MAE: {mae_6h:.1f} kt)", zorder=6)

        ax_a.set_title(f"(A) +6-Hour Short-Term Forecast — {s_name} ({basin})", fontsize=13, fontweight="bold")
        ax_a.set_xlabel("Elapsed Observation Time (Hours)", fontsize=11, fontweight="bold")
        ax_a.set_ylabel("Wind Speed (knots)", fontsize=11, fontweight="bold")
        ax_a.set_ylim(15, 175)
        ax_a.set_xlim(0, t_origin.max() + 3)
        ax_a.grid(True, linestyle="--", alpha=0.6)
        ax_a.legend(loc="upper left", frameon=True, fontsize=8.5, ncol=2)

        # Panel B: +24h Forecast
        ax_b = axes[1]
        ax_b.axhspan(0, 34, color="#E2E8F0", alpha=0.35)
        ax_b.axhspan(34, 63, color="#FEF08A", alpha=0.3)
        ax_b.axhspan(64, 82, color="#FED7AA", alpha=0.3)
        ax_b.axhspan(83, 95, color="#FDBA74", alpha=0.3)
        ax_b.axhspan(96, 112, color="#FCA5A5", alpha=0.3)
        ax_b.axhspan(113, 175, color="#F87171", alpha=0.35)

        ax_b.plot(t_origin, v_act_24h, color="#0F172A", linewidth=2.8, marker="o", markersize=5, label="Actual Ground Truth (+24h)", zorder=5)
        ax_b.plot(t_origin, v_curr_np, color="#64748B", linewidth=1.8, linestyle="--", marker="s", markersize=4, label=f"Persistence (MAE: {pers_mae_24h:.1f} kt)", zorder=4)
        ax_b.plot(t_origin, pred_24h, color="#DC2626", linewidth=2.5, marker="D", markersize=5, label=f"ML Model (+24h) (MAE: {mae_24h:.1f} kt)", zorder=6)

        gain_str = f"-{pers_mae_24h - mae_24h:.1f} kt" if pers_mae_24h > mae_24h else f"+{mae_24h - pers_mae_24h:.1f} kt"
        ax_b.set_title(f"(B) +24-Hour Day-Ahead Forecast [ML Gain: {gain_str}]", fontsize=13, fontweight="bold")
        ax_b.set_xlabel("Elapsed Observation Time (Hours)", fontsize=11, fontweight="bold")
        ax_b.set_ylabel("Wind Speed (knots)", fontsize=11, fontweight="bold")
        ax_b.set_ylim(15, 175)
        ax_b.set_xlim(0, t_origin.max() + 3)
        ax_b.grid(True, linestyle="--", alpha=0.6)
        ax_b.legend(loc="upper left", frameon=True, fontsize=9)

        clean_fn = s_name.lower().replace(" ", "_")
        plt.suptitle(f"100% Unseen Test Cyclone: {s_name} ({cid}, {basin} — Peak: {peak_info})", fontsize=15, fontweight="bold", y=0.98)
        p_path = out_dir / f"{clean_fn}_6h_vs_24h_forecast.png"
        plt.tight_layout()
        plt.savefig(p_path)
        plt.close()
        print(f"[Generated: {s_name} Graph] -> {p_path}")
        storm_ds.close()

    # Save summary stats to JSON
    with open(out_dir / "global_unseen_cyclones_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    return summary_stats


if __name__ == "__main__":
    generate_global_unseen_plots()
