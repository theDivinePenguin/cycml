"""Analyze the training dataset intensity distribution and quantify data imbalance."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.intensity_bins import INTENSITY_BINS, compute_binned_distribution


def analyze_training_distribution(
    metadata_path: str | Path = "data/metadata/metadata_CPAC_IO_SH.csv",
    splits_path: str | Path = "data/metadata/splits_CPAC_IO_SH.json",
    output_dir: str | Path = "experiments/analysis"
) -> dict:
    """Analyze training intensity distribution on training data only."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Load authoritative metadata & splits
    df_meta = pd.read_csv(metadata_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    if isinstance(splits["train"], dict):
        train_indices = set(splits["train"]["sample_indices"])
        df_train = df_meta[df_meta["sample_index"].isin(train_indices)].copy()
    else:
        df_train = df_meta[df_meta["sample_index"].isin(set(splits["train"]))].copy()
    total_train_frames = len(df_train)
    total_train_cyclones = df_train["cyclone_id"].nunique()

    # 2. Compute binned distribution
    binned_stats = compute_binned_distribution(
        df_train,
        intensity_col="wind_speed",
        cyclone_id_col="cyclone_id",
        bins=INTENSITY_BINS
    )

    # 3. Print clean report
    print("\n" + "=" * 65)
    print("TRAINING INTENSITY DISTRIBUTION (TRAINING DATA ONLY)")
    print(f"Total Training Frames: {total_train_frames:,} | Unique Cyclones: {total_train_cyclones}")
    print("=" * 65)
    print(f"{'Bin':<14} | {'Frames':<10} | {'% Frames':<12} | {'Cyclones':<10}")
    print("-" * 65)

    for item in binned_stats:
        print(f"{item['bin']:<14} | {item['frames']:<10,d} | {item['percent_frames']:<10.2f}% | {item['unique_cyclones']:<10,d}")

    print("=" * 65)

    # 4. Save JSON
    json_data = {
        "dataset": "TCIR-CPAC_IO_SH (Training Split)",
        "total_train_frames": total_train_frames,
        "total_train_cyclones": total_train_cyclones,
        "mean_wind_speed_kt": float(df_train["wind_speed"].mean()),
        "median_wind_speed_kt": float(df_train["wind_speed"].median()),
        "p90_wind_speed_kt": float(df_train["wind_speed"].quantile(0.90)),
        "p95_wind_speed_kt": float(df_train["wind_speed"].quantile(0.95)),
        "p99_wind_speed_kt": float(df_train["wind_speed"].quantile(0.99)),
        "max_wind_speed_kt": float(df_train["wind_speed"].max()),
        "distribution": binned_stats
    }

    json_path = out_p / "intensity_distribution.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"\n[Saved JSON] {json_path}")

    # 5. Generate High-Quality Visual Plot
    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=150)

    bins_labels = [item["bin"] for item in binned_stats]
    frame_counts = [item["frames"] for item in binned_stats]
    pct_labels = [item["percent_frames"] for item in binned_stats]
    cyclone_counts = [item["unique_cyclones"] for item in binned_stats]

    x = np.arange(len(bins_labels))
    width = 0.55

    # Color gradient from blue (weak) to deep red (extreme)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(bins_labels)))
    bars = ax1.bar(x, frame_counts, width, color=colors, edgecolor="black", alpha=0.85, zorder=3)

    ax1.set_xlabel("Tropical Cyclone Intensity Bins (Maximum Sustained Wind Speed)", fontsize=11, fontweight="bold", labelpad=10)
    ax1.set_ylabel("Number of Training Frames", fontsize=11, fontweight="bold", color="#1f2937")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bins_labels, rotation=25, ha="right", fontsize=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

    # Annotate bars with Frame Count and Percentage
    for bar, pct in zip(bars, pct_labels):
        h = bar.get_height()
        ax1.annotate(
            f"{h:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold"
        )

    # Add secondary line for unique cyclone counts
    ax2 = ax1.twinx()
    ax2.plot(x, cyclone_counts, color="#dc2626", marker="o", linewidth=2.5, markersize=8, label="Unique Cyclones", zorder=4)
    ax2.set_ylabel("Number of Unique Cyclones", fontsize=11, fontweight="bold", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax2.set_ylim(0, max(cyclone_counts) * 1.25)

    for i, count in enumerate(cyclone_counts):
        ax2.annotate(
            f"{count} storms",
            xy=(x[i], count),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8, color="#dc2626", fontweight="bold"
        )

    plt.title(
        f"TCIR-CPAC_IO_SH Training Intensity Distribution (N = {total_train_frames:,} frames, {total_train_cyclones} cyclones)\nSevere Long-Tail Imbalance Toward Low-Intensity Observations",
        fontsize=12, fontweight="bold", pad=15
    )

    fig.tight_layout()
    plot_path = out_p / "intensity_distribution.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved Plot] {plot_path}")

    return json_data


def main():
    parser = argparse.ArgumentParser(description="Analyze training intensity distribution.")
    parser.add_argument("--metadata", type=str, default="data/metadata/metadata_CPAC_IO_SH.csv", help="Path to metadata CSV")
    parser.add_argument("--splits", type=str, default="data/metadata/splits_CPAC_IO_SH.json", help="Path to splits JSON")
    parser.add_argument("--output-dir", type=str, default="experiments/analysis", help="Output directory")
    args = parser.parse_args()

    analyze_training_distribution(
        metadata_path=args.metadata,
        splits_path=args.splits,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
