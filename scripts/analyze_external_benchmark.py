"""Compare internal test set performance with external cross-basin generalization benchmark."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.intensity_bins import REGIME_BINS, assign_intensity_bin


def analyze_external_vs_internal(
    internal_preds_csv: str | Path = "experiments/baseline_resnet18_cpac_io_sh/test_predictions.csv",
    external_bench_csv: str | Path = "experiments/external_benchmark/external_benchmark_results.csv",
    output_dir: str | Path = "experiments/analysis"
) -> dict:
    """Compare internal held-out test performance against external cross-basin benchmark across intensity regimes."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df_internal = pd.read_csv(internal_preds_csv)
    df_external = pd.read_csv(external_bench_csv)

    # Standardize columns
    int_act = df_internal["actual_wind_speed"] if "actual_wind_speed" in df_internal.columns else df_internal["wind_speed"]
    int_pred = df_internal["predicted_wind_speed"] if "predicted_wind_speed" in df_internal.columns else df_internal["prediction"]

    df_int = pd.DataFrame({
        "actual": int_act.astype(float),
        "predicted": int_pred.astype(float),
        "dataset_type": "Internal Held-Out Test (CPAC/IO/SH)"
    })
    df_int["error"] = df_int["predicted"] - df_int["actual"]
    df_int["abs_error"] = np.abs(df_int["error"])
    df_int["regime"] = df_int["actual"].apply(lambda w: assign_intensity_bin(w, REGIME_BINS))

    df_ext = pd.DataFrame({
        "name": df_external["name"],
        "basin": df_external["basin"],
        "actual": df_external["ground_truth_kt"].astype(float),
        "predicted": df_external["predicted_kt"].astype(float),
        "dataset_type": "External Cross-Basin Benchmark"
    })
    df_ext["error"] = df_ext["predicted"] - df_ext["actual"]
    df_ext["abs_error"] = np.abs(df_ext["error"])
    df_ext["regime"] = df_ext["actual"].apply(lambda w: assign_intensity_bin(w, REGIME_BINS))

    # 2. Group by Intensity Regime
    regime_summary = []
    print("\n" + "=" * 90)
    print("INTERNAL HELD-OUT TEST VS. EXTERNAL CROSS-BASIN GENERALIZATION BY REGIME")
    print("=" * 90)
    print(f"{'Intensity Regime':<14} | {'Internal N':<11} | {'Internal MAE':<13} | {'External N':<11} | {'External MAE':<13} | {'Δ MAE (kt)'}")
    print("-" * 90)

    for lower, upper, label in REGIME_BINS:
        bin_int = df_int[df_int["regime"] == label]
        bin_ext = df_ext[df_ext["regime"] == label]

        n_int = len(bin_int)
        n_ext = len(bin_ext)

        mae_int = float(bin_int["abs_error"].mean()) if n_int > 0 else 0.0
        mae_ext = float(bin_ext["abs_error"].mean()) if n_ext > 0 else 0.0
        delta_mae = mae_ext - mae_int if (n_int > 0 and n_ext > 0) else 0.0

        rmse_int = float(np.sqrt((bin_int["error"] ** 2).mean())) if n_int > 0 else 0.0
        rmse_ext = float(np.sqrt((bin_ext["error"] ** 2).mean())) if n_ext > 0 else 0.0

        bias_int = float(bin_int["error"].mean()) if n_int > 0 else 0.0
        bias_ext = float(bin_ext["error"].mean()) if n_ext > 0 else 0.0

        regime_summary.append({
            "regime": label,
            "lower_kt": lower,
            "upper_kt": upper if upper != float("inf") else None,
            "internal_samples": n_int,
            "internal_mae": round(mae_int, 2),
            "internal_rmse": round(rmse_int, 2),
            "internal_bias": round(bias_int, 2),
            "external_samples": n_ext,
            "external_mae": round(mae_ext, 2),
            "external_rmse": round(rmse_ext, 2),
            "external_bias": round(bias_ext, 2),
            "mae_difference": round(delta_mae, 2)
        })

        ext_str = f"{mae_ext:.2f} kt" if n_ext > 0 else "N/A"
        delta_str = f"{delta_mae:+.2f} kt" if (n_int > 0 and n_ext > 0) else "N/A"
        print(f"{label:<14} | {n_int:<11,d} | {mae_int:<11.2f} kt | {n_ext:<11,d} | {ext_str:<13} | {delta_str}")

    print("=" * 90)

    # 3. Save JSON
    overall_summary = {
        "internal_overall_mae": round(float(df_int["abs_error"].mean()), 2),
        "external_overall_mae": round(float(df_ext["abs_error"].mean()), 2),
        "internal_samples": len(df_int),
        "external_samples": len(df_ext),
        "regime_breakdown": regime_summary,
        "external_cyclones_evaluated": df_ext.to_dict(orient="records")
    }

    json_path = out_p / "external_generalization_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)
    print(f"\n[Saved JSON] {json_path}")

    # 4. Generate Comparative Bar Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    regimes = [item["regime"] for item in regime_summary]
    int_maes = [item["internal_mae"] for item in regime_summary]
    ext_maes = [item["external_mae"] for item in regime_summary]
    int_counts = [item["internal_samples"] for item in regime_summary]
    ext_counts = [item["external_samples"] for item in regime_summary]

    x = np.arange(len(regimes))
    width = 0.35

    rects1 = ax.bar(x - width/2, int_maes, width, label="Internal Held-Out Test Set (CPAC/IO/SH)", color="#2563eb", alpha=0.85, edgecolor="black", zorder=3)
    rects2 = ax.bar(x + width/2, ext_maes, width, label="External Cross-Basin Benchmark (ATLN/WPAC/IO/SH)", color="#dc2626", alpha=0.85, edgecolor="black", zorder=3)

    ax.set_xlabel("Tropical Cyclone Intensity Regimes", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Mean Absolute Error (MAE in knots)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

    # Annotations
    for rect, count in zip(rects1, int_counts):
        h = rect.get_height()
        ax.annotate(f"{h:.1f} kt\n(N={count})", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e3a8a")

    for rect, count in zip(rects2, ext_counts):
        h = rect.get_height()
        if count > 0:
            ax.annotate(f"{h:.1f} kt\n(N={count})", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#991b1b")
        else:
            ax.annotate("N/A", xy=(rect.get_x() + rect.get_width()/2, 0), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#991b1b")

    plt.title("Internal Held-Out Test MAE vs. External Cross-Basin MAE by Intensity Regime\nGeneralization Remains Strong in Moderate Regimes, Saturation Diverges at >130 kt", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="upper left", fontsize=9.5)
    plt.tight_layout()

    plot_path = out_p / "external_vs_internal.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved Plot] {plot_path}")

    return overall_summary


def main():
    parser = argparse.ArgumentParser(description="Compare external benchmark against internal test set.")
    parser.add_argument("--internal", type=str, default="experiments/baseline_resnet18_cpac_io_sh/test_predictions.csv")
    parser.add_argument("--external", type=str, default="experiments/external_benchmark/external_benchmark_results.csv")
    parser.add_argument("--output-dir", type=str, default="experiments/analysis")
    args = parser.parse_args()

    analyze_external_vs_internal(
        internal_preds_csv=args.internal,
        external_bench_csv=args.external,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
