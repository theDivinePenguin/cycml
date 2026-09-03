"""Analyze Indian Ocean training intensity distribution and expected sampling shift."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.samplers import compute_intensity_sampling_weights
from src.evaluation.intensity_bins import INTENSITY_BINS, assign_intensity_bin


def analyze_io_distribution(
    train_metadata_path: str | Path = "data/metadata/train_metadata_IO.csv",
    output_dir: str | Path = "experiments/io_balancing_study"
) -> dict:
    """Analyze IO training distribution and compute expected intensity-aware sampling shift."""
    meta_p = Path(train_metadata_path)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(meta_p)
    df_train["bin"] = df_train["wind_speed"].apply(assign_intensity_bin)
    total_frames = len(df_train)
    total_cyclones = df_train["cyclone_id"].nunique()

    # Percentiles
    mean_speed = float(df_train["wind_speed"].mean())
    median_speed = float(df_train["wind_speed"].median())
    p90 = float(df_train["wind_speed"].quantile(0.90))
    p95 = float(df_train["wind_speed"].quantile(0.95))
    p98 = float(df_train["wind_speed"].quantile(0.98))
    p99 = float(df_train["wind_speed"].quantile(0.99))
    max_speed = float(df_train["wind_speed"].max())

    # Sampling weights & expected probabilities
    sample_weights, sampling_diag = compute_intensity_sampling_weights(df_train, alpha=0.5)

    distribution_data = []
    print("=" * 85)
    print(f"INDIAN OCEAN (IO) TRAINING INTENSITY DISTRIBUTION (N={total_frames:,} frames, {total_cyclones} cyclones)")
    print("=" * 85)
    print(f"{'Intensity Bin':<14} | {'Frames':<8} | {'% Frames':<10} | {'Cyclones':<10} | {'Eff Sample %':<14} | {'Multiplier':<10}")
    print("-" * 85)

    for lower, upper, label in INTENSITY_BINS:
        bin_df = df_train[df_train["bin"] == label]
        n_frames = len(bin_df)
        pct_frames = (n_frames / total_frames) * 100.0 if total_frames > 0 else 0.0
        n_cyclones = int(bin_df["cyclone_id"].nunique())
        eff_pct = sampling_diag[label]["effective_sampling_pct"]
        mult = sampling_diag[label]["sampling_multiplier"]

        distribution_data.append({
            "bin": label,
            "lower_kt": lower,
            "upper_kt": upper,
            "frames": n_frames,
            "pct_frames": round(pct_frames, 2),
            "cyclones": n_cyclones,
            "effective_sampling_pct": round(eff_pct, 2),
            "sampling_multiplier": round(mult, 2)
        })

        print(f"{label:<14} | {n_frames:<8,d} | {pct_frames:7.2f} %  | {n_cyclones:<10d} | {eff_pct:10.2f} %   | {mult:6.2f}x")

    print("=" * 85)
    print(f"Percentile Summary:")
    print(f"  • Mean:        {mean_speed:.2f} kt")
    print(f"  • Median:      {median_speed:.2f} kt")
    print(f"  • 90th %:      {p90:.2f} kt")
    print(f"  • 95th %:      {p95:.2f} kt")
    print(f"  • 98th %:      {p98:.2f} kt")
    print(f"  • 99th %:      {p99:.2f} kt")
    print(f"  • Max Speed:   {max_speed:.2f} kt")
    print("=" * 85)

    summary_json = {
        "dataset": "TCIR_IO_TRAIN",
        "total_frames": total_frames,
        "unique_cyclones": total_cyclones,
        "mean_kt": round(mean_speed, 2),
        "median_kt": round(median_speed, 2),
        "p90_kt": round(p90, 2),
        "p95_kt": round(p95, 2),
        "p98_kt": round(p98, 2),
        "p99_kt": round(p99, 2),
        "max_kt": round(max_speed, 2),
        "distribution": distribution_data
    }

    json_path = out_p / "io_distribution_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    # Plot 1: Natural IO Training Intensity Distribution
    plt.figure(figsize=(10, 6), dpi=150)
    bin_labels = [d["bin"] for d in distribution_data]
    frame_counts = [d["frames"] for d in distribution_data]
    
    x = np.arange(len(bin_labels))
    bars = plt.bar(x, frame_counts, color="#0284c7", edgecolor="black", alpha=0.85)

    for bar, d in zip(bars, distribution_data):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 15,
                     f"{d['frames']}\n({d['pct_frames']}%)",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.xlabel("Intensity Bins", fontsize=11, fontweight="bold")
    plt.ylabel("Training Frame Count", fontsize=11, fontweight="bold")
    plt.xticks(x, bin_labels, rotation=25, ha="right")
    plt.title(f"Indian Ocean (IO) Training Set Distribution (N={total_frames:,} frames, {total_cyclones} cyclones)\nSevere Low-End Skewness (90% ≤ {p90:.0f} kt)", fontsize=12, fontweight="bold", pad=15)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.ylim(0, max(frame_counts) * 1.22)
    plt.tight_layout()

    plot1_path = out_p / "io_training_intensity_distribution.png"
    plt.savefig(plot1_path)
    plt.close()

    # Plot 2: Natural vs Expected Intensity-Aware Sampling Distribution
    plt.figure(figsize=(11, 6), dpi=150)
    natural_pcts = [d["pct_frames"] for d in distribution_data]
    sampling_pcts = [d["effective_sampling_pct"] for d in distribution_data]
    w = 0.35

    plt.bar(x - w/2, natural_pcts, w, label=f"Natural IO Distribution (N={total_frames:,})", color="#0284c7", alpha=0.85, edgecolor="black")
    plt.bar(x + w/2, sampling_pcts, w, label="Intensity-Aware Sampling (α=0.5 Sqrt-Inverse)", color="#10b981", alpha=0.85, edgecolor="black")

    for i in range(len(bin_labels)):
        if natural_pcts[i] > 0 or sampling_pcts[i] > 0:
            mult = distribution_data[i]["sampling_multiplier"]
            mult_str = f"{mult:.1f}x" if mult > 0 else "0x"
            plt.text(x[i] + w/2, sampling_pcts[i] + 1.0, mult_str, ha="center", fontsize=8.5, fontweight="bold", color="#047857")

    plt.xlabel("Intensity Bins", fontsize=11, fontweight="bold")
    plt.ylabel("Probability / Representation (%)", fontsize=11, fontweight="bold")
    plt.xticks(x, bin_labels, rotation=25, ha="right")
    plt.title("Indian Ocean Training Distribution: Natural vs. Intensity-Aware Sampling\nControlled Boosting of Upper-End Bins (≥110 kt) without Extreme Tail Overfitting", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.ylim(0, max(max(natural_pcts), max(sampling_pcts)) * 1.20)
    plt.tight_layout()

    plot2_path = out_p / "io_natural_vs_balanced_distribution.png"
    plt.savefig(plot2_path)
    plt.close()

    print(f"\n[Saved JSON] {json_path}")
    print(f"[Saved Plot] {plot1_path}")
    print(f"[Saved Plot] {plot2_path}")

    return summary_json


def main():
    analyze_io_distribution()


if __name__ == "__main__":
    main()
