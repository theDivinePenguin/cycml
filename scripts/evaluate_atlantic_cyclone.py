"""Select a random (or specified) Atlantic cyclone from TCIR, evaluate its entire lifecycle, and plot results."""
import argparse
import json
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.metadata import load_tcir_info_table
from src.data.preprocessing import TCIRPreprocessor
from src.models.factory import build_model
from src.utils.config import load_config


def wind_speed_to_category(wind_speed: float) -> str:
    """Convert wind speed in knots into standard meteorological intensity category."""
    if wind_speed < 34:
        return "Tropical Depression"
    elif 34 <= wind_speed < 48:
        return "Tropical Storm"
    elif 48 <= wind_speed < 64:
        return "Severe Tropical Storm"
    elif 64 <= wind_speed < 83:
        return "Category 1 Hurricane"
    elif 83 <= wind_speed < 96:
        return "Category 2 Hurricane"
    elif 96 <= wind_speed < 113:
        return "Category 3 Major Hurricane"
    elif 113 <= wind_speed < 137:
        return "Category 4 Major Hurricane"
    else:
        return "Category 5 Major Hurricane"


def evaluate_atlantic_cyclone(
    h5_path: Path,
    checkpoint_path: Path,
    stats_path: Path,
    config_path: Path,
    cyclone_id: str | None = None,
    output_dir: Path = Path("experiments/atlantic_eval")
) -> dict:
    """Evaluate a complete Atlantic cyclone lifecycle through the trained baseline model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Atlantic Metadata
    print(f"[Atlantic Evaluation] Loading metadata from {h5_path.name}...")
    df_raw = load_tcir_info_table(h5_path)
    df_raw = df_raw.reset_index(drop=True)
    df_raw["sample_index"] = df_raw.index

    col_map = {"ID": "cyclone_id", "time": "timestamp", "lat": "latitude", "lon": "longitude", "Vmax": "wind_speed", "vmax": "wind_speed", "data_set": "data_set"}
    df = df_raw.rename(columns=col_map).copy()

    for col in ["cyclone_id", "timestamp", "data_set"]:
        if col in df.columns and len(df) > 0 and isinstance(df[col].iloc[0], (bytes, bytearray)):
            df[col] = df[col].str.decode("utf-8")
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Filter for Atlantic Ocean (ATLN)
    df_atln = df[(df.get("data_set", "") == "ATLN") | (df["cyclone_id"].str.endswith("L"))].copy()
    print(f"[Atlantic Evaluation] Found {len(df_atln)} total Atlantic frames across {df_atln['cyclone_id'].nunique()} unique cyclones.")

    # 2. Select Cyclone
    available_cyclones = df_atln["cyclone_id"].unique()
    if cyclone_id and cyclone_id in available_cyclones:
        selected_cid = cyclone_id
    else:
        # Prefer cyclones with good length (> 20 frames) and significant intensity (> 60 kt)
        storm_stats = df_atln.groupby("cyclone_id").agg(n_frames=("sample_index", "count"), max_wind=("wind_speed", "max"))
        interesting = storm_stats[(storm_stats["n_frames"] >= 25) & (storm_stats["max_wind"] >= 65)].index.tolist()
        if interesting:
            selected_cid = str(np.random.choice(interesting))
        else:
            selected_cid = str(np.random.choice(available_cyclones))

    storm_df = df_atln[df_atln["cyclone_id"] == selected_cid].sort_values("timestamp").reset_index(drop=True)
    print(f"\n[Selected Atlantic Cyclone: {selected_cid}]")
    print(f"  • Total frames:    {len(storm_df)}")
    print(f"  • Time span:       {storm_df['timestamp'].iloc[0]} to {storm_df['timestamp'].iloc[-1]}")
    print(f"  • Actual peak:     {storm_df['wind_speed'].max():.1f} knots")

    # 3. Load Model and Preprocessor
    config = load_config(config_path)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]

    preprocessor = TCIRPreprocessor(
        mean=mean,
        std=std,
        target_size=tuple(config.get("dataset", {}).get("input_size", [224, 224])),
        is_training=False,
        augmentation_cfg={"enabled": False}
    )

    model = build_model(config).to(dev)
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Run Inference on All Lifecycle Frames
    predictions = []
    actuals = []
    timestamps = []
    images = []

    with h5py.File(h5_path, "r") as hf:
        matrix = hf["matrix"]
        for _, row in storm_df.iterrows():
            s_idx = int(row["sample_index"])
            img_np = matrix[s_idx, :, :, 0]  # IR1 channel
            images.append(img_np)

            tensor = torch.from_numpy(np.array(img_np, dtype=np.float32)).unsqueeze(0)
            tensor = preprocessor(tensor).unsqueeze(0).to(dev)

            with torch.no_grad():
                pred = model(tensor).item()

            actual = float(row["wind_speed"])
            predictions.append(pred)
            actuals.append(actual)
            timestamps.append(str(row["timestamp"]))

    preds_arr = np.array(predictions)
    acts_arr = np.array(actuals)
    errors = np.abs(preds_arr - acts_arr)

    storm_mae = float(np.mean(errors))
    storm_rmse = float(np.sqrt(np.mean((preds_arr - acts_arr) ** 2)))
    pred_peak = float(np.max(preds_arr))
    actual_peak = float(np.max(acts_arr))

    # 5. Print Frame-by-Frame Timeline Table
    print("\n" + "=" * 80)
    print(f"LIFECYCLE INFERENCE TIMELINE: Cyclone {selected_cid}")
    print("=" * 80)
    print(f"{'Step':<5} | {'Timestamp (UTC)':<12} | {'Actual (kt)':<11} | {'Pred (kt)':<10} | {'Error (kt)':<10} | {'Intensity Category'}")
    print("-" * 80)

    for i in range(len(storm_df)):
        err_str = f"{preds_arr[i] - acts_arr[i]:+.1f}"
        cat = wind_speed_to_category(preds_arr[i])
        print(f"{i+1:3d}   | {timestamps[i]:<12} | {acts_arr[i]:6.1f} kt   | {preds_arr[i]:6.1f} kt  | {err_str:>7s} kt | {cat}")

    print("-" * 80)
    print(f"SUMMARY FOR ATLANTIC CYCLONE {selected_cid}:")
    print(f"  • Lifecycle MAE:      {storm_mae:.2f} knots")
    print(f"  • Lifecycle RMSE:     {storm_rmse:.2f} knots")
    print(f"  • Actual Peak Wind:   {actual_peak:.1f} knots ({wind_speed_to_category(actual_peak)})")
    print(f"  • Predicted Peak:     {pred_peak:.1f} knots ({wind_speed_to_category(pred_peak)})")
    print("=" * 80)

    # 6. Plot Lifecycle Intensity Curve
    plt.figure(figsize=(12, 6), dpi=150)
    time_indices = np.arange(len(timestamps))

    plt.plot(time_indices, acts_arr, "o-", color="black", linewidth=2.5, label=f"Ground Truth (HURDAT2 Peak: {actual_peak:.0f} kt)", zorder=4)
    plt.plot(time_indices, preds_arr, "s--", color="#1f77b4", linewidth=2, alpha=0.9, label=f"ResNet18 Prediction (Peak: {pred_peak:.1f} kt)", zorder=3)

    # Category Thresholds
    plt.axhline(34, color="orange", linestyle=":", alpha=0.7, label="Tropical Storm (34 kt)")
    plt.axhline(64, color="red", linestyle=":", alpha=0.7, label="Hurricane (64 kt)")
    if max(actual_peak, pred_peak) >= 96:
        plt.axhline(96, color="darkmagenta", linestyle=":", alpha=0.7, label="Major Hurricane (96 kt)")

    # X-axis formatting: show sample of timestamps
    step = max(1, len(timestamps) // 8)
    plt.xticks(time_indices[::step], timestamps[::step], rotation=30, ha="right", fontsize=9)
    plt.xlabel("Observation Timestamp (YYYYMMDDHH UTC)", fontsize=11, fontweight="bold")
    plt.ylabel("Maximum Sustained Wind Speed (knots)", fontsize=11, fontweight="bold")
    plt.title(f"Atlantic Cyclone {selected_cid} — Intensity Lifetime Estimation\n(Model trained strictly on CPAC/IO/SH, zero Atlantic training data)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)

    text_box = f"Cyclone ID: {selected_cid}\nTotal Frames: {len(storm_df)}\nLifecycle MAE: {storm_mae:.2f} kt\nLifecycle RMSE: {storm_rmse:.2f} kt"
    plt.gca().text(
        0.02, 0.95, text_box,
        transform=plt.gca().transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
    )

    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    curve_plot_path = output_dir / f"atlantic_{selected_cid}_lifetime_curve.png"
    plt.savefig(curve_plot_path)
    plt.close()
    print(f"\n[Visualization] Saved lifecycle curve plot to: {curve_plot_path}")

    # 7. Plot Satellite IR Frame Montage (Initial, Peak, Decay)
    peak_idx = int(np.argmax(acts_arr))
    sample_indices = [
        0,                                   # Genesis / Initial
        len(acts_arr) // 4,                  # Intensifying
        peak_idx,                            # Peak Intensity
        min(len(acts_arr) - 1, peak_idx + len(acts_arr)//4), # Weakening
        len(acts_arr) - 1                    # Final Frame
    ]
    # Remove duplicates while preserving order
    unique_indices = []
    for idx in sample_indices:
        if idx not in unique_indices and idx < len(images):
            unique_indices.append(idx)

    fig, axes = plt.subplots(1, len(unique_indices), figsize=(4 * len(unique_indices), 4.5), dpi=150)
    if len(unique_indices) == 1:
        axes = [axes]

    for ax_i, s_i in enumerate(unique_indices):
        ax = axes[ax_i]
        im = ax.imshow(images[s_i], cmap="inferno")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        stage_label = "Genesis" if s_i == 0 else ("Peak Intensity" if s_i == peak_idx else ("Final Stage" if s_i == len(images)-1 else "Observation"))
        ax.set_title(
            f"[{stage_label}]\n{timestamps[s_i]}\nActual: {acts_arr[s_i]:.0f} kt | Pred: {preds_arr[s_i]:.1f} kt\nError: {preds_arr[s_i]-acts_arr[s_i]:+.1f} kt",
            fontsize=9
        )
        ax.axis("off")

    plt.suptitle(f"Satellite IR1 Observations & Predictions for Atlantic Cyclone {selected_cid}", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    montage_plot_path = output_dir / f"atlantic_{selected_cid}_satellite_montage.png"
    plt.savefig(montage_plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved satellite montage to: {montage_plot_path}")

    return {
        "cyclone_id": selected_cid,
        "n_frames": len(storm_df),
        "mae": storm_mae,
        "rmse": storm_rmse,
        "actual_peak": actual_peak,
        "predicted_peak": pred_peak,
        "curve_plot": str(curve_plot_path),
        "montage_plot": str(montage_plot_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Atlantic cyclone from TCIR on trained baseline model.")
    parser.add_argument("--h5-path", type=str, default="data/raw/TCIR-ATLN_EPAC_WPAC.h5", help="Path to ATLN_EPAC_WPAC HDF5")
    parser.add_argument("--cyclone-id", type=str, default=None, help="Specific Atlantic Cyclone ID (e.g. 200512L)")
    parser.add_argument("--checkpoint", type=str, default="experiments/baseline_resnet18_cpac_io_sh/best.pt", help="Path to best.pt")
    parser.add_argument("--stats", type=str, default="data/metadata/normalization_stats_CPAC_IO_SH.json", help="Path to normalization stats")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default="experiments/atlantic_eval", help="Output directory for plots")
    args = parser.parse_args()

    evaluate_atlantic_cyclone(
        h5_path=Path(args.h5_path),
        checkpoint_path=Path(args.checkpoint),
        stats_path=Path(args.stats),
        config_path=Path(args.config),
        cyclone_id=args.cyclone_id,
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
