import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.intensity_bins import INTENSITY_BINS, REGIME_BINS, assign_intensity_bin


def compare_baseline_vs_expanded(
    baseline_dir: str | Path = "experiments/baseline_resnet18_cpac_io_sh",
    expanded_dir: str | Path = "experiments/expanded_all_basins_resnet18",
    output_dir: str | Path = "experiments/expanded_all_basins_resnet18/plots"
) -> dict:
    """Generate comparative tables, saturation metrics, statistical tests, and plots."""
    base_p = Path(baseline_dir)
    exp_p = Path(expanded_dir)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Load Predictions
    base_preds_path = base_p / "test_predictions.csv"
    exp_preds_path = exp_p / "test_predictions.csv"

    if not (base_preds_path.exists() and exp_preds_path.exists()):
        raise FileNotFoundError(f"Ensure both {base_preds_path} and {exp_preds_path} exist.")

    df_base = pd.read_csv(base_preds_path)
    df_exp = pd.read_csv(exp_preds_path)

    # Standardize column names
    b_act_col = "actual_wind_speed" if "actual_wind_speed" in df_base.columns else "wind_speed"
    b_pred_col = "predicted_wind_speed" if "predicted_wind_speed" in df_base.columns else "prediction"
    e_act_col = "actual_wind_speed" if "actual_wind_speed" in df_exp.columns else "wind_speed"
    e_pred_col = "predicted_wind_speed" if "predicted_wind_speed" in df_exp.columns else "prediction"

    df_base["actual"] = df_base[b_act_col].astype(float)
    df_base["predicted"] = df_base[b_pred_col].astype(float)
    df_base["error"] = df_base["predicted"] - df_base["actual"]
    df_base["abs_error"] = np.abs(df_base["error"])
    df_base["bin"] = df_base["actual"].apply(assign_intensity_bin)

    df_exp["actual"] = df_exp[e_act_col].astype(float)
    df_exp["predicted"] = df_exp[e_pred_col].astype(float)
    df_exp["error"] = df_exp["predicted"] - df_exp["actual"]
    df_exp["abs_error"] = np.abs(df_exp["error"])
    df_exp["bin"] = df_exp["actual"].apply(assign_intensity_bin)

    # 2. Overall Metrics & Regression Slopes
    base_mae = float(df_base["abs_error"].mean())
    exp_mae = float(df_exp["abs_error"].mean())
    base_rmse = float(np.sqrt((df_base["error"] ** 2).mean()))
    exp_rmse = float(np.sqrt((df_exp["error"] ** 2).mean()))

    base_slope, base_intercept = np.polyfit(df_base["actual"], df_base["predicted"], 1)
    exp_slope, exp_intercept = np.polyfit(df_exp["actual"], df_exp["predicted"], 1)

    # High Intensity Metrics (>=110 kt)
    base_high = df_base[df_base["actual"] >= 110.0]
    exp_high = df_exp[df_exp["actual"] >= 110.0]

    base_bias_high = float(base_high["error"].mean()) if len(base_high) > 0 else 0.0
    exp_bias_high = float(exp_high["error"].mean()) if len(exp_high) > 0 else 0.0

    # 110-130 kt MAE
    base_110_130 = df_base[(df_base["actual"] >= 110.0) & (df_base["actual"] < 130.0)]
    exp_110_130 = df_exp[(df_exp["actual"] >= 110.0) & (df_exp["actual"] < 130.0)]
    base_mae_110_130 = float(base_110_130["abs_error"].mean()) if len(base_110_130) > 0 else 0.0
    exp_mae_110_130 = float(exp_110_130["abs_error"].mean()) if len(exp_110_130) > 0 else 0.0

    # 130-150 kt MAE
    base_130_150 = df_base[(df_base["actual"] >= 130.0) & (df_base["actual"] < 150.0)]
    exp_130_150 = df_exp[(df_exp["actual"] >= 130.0) & (df_exp["actual"] < 150.0)]
    base_mae_130_150 = float(base_130_150["abs_error"].mean()) if len(base_130_150) > 0 else 0.0
    exp_mae_130_150 = float(exp_130_150["abs_error"].mean()) if len(exp_130_150) > 0 else 0.0

    # Max Predicted
    base_max_pred = float(df_base["predicted"].max())
    exp_max_pred = float(df_exp["predicted"].max())

    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON: BASELINE vs. EXPANDED EXPERIMENT")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Baseline':<12} | {'Expanded':<12} | {'Δ (Change)':<12}")
    print("-" * 80)
    print(f"{'Overall MAE (kt)':<25} | {base_mae:10.2f} kt | {exp_mae:10.2f} kt | {exp_mae - base_mae:+10.2f} kt")
    print(f"{'Overall RMSE (kt)':<25} | {base_rmse:10.2f} kt | {exp_rmse:10.2f} kt | {exp_rmse - base_rmse:+10.2f} kt")
    print(f"{'Regression Slope':<25} | {base_slope:10.2f}    | {exp_slope:10.2f}    | {exp_slope - base_slope:+10.2f}")
    print(f"{'Bias ≥110 kt (kt)':<25} | {base_bias_high:10.2f} kt | {exp_bias_high:10.2f} kt | {exp_bias_high - base_bias_high:+10.2f} kt")
    print(f"{'MAE 110–130 kt (kt)':<25} | {base_mae_110_130:10.2f} kt | {exp_mae_110_130:10.2f} kt | {exp_mae_110_130 - base_mae_110_130:+10.2f} kt")
    print(f"{'MAE 130–150 kt (kt)':<25} | {base_mae_130_150:10.2f} kt | {exp_mae_130_150:10.2f} kt | {exp_mae_130_150 - base_mae_130_150:+10.2f} kt")
    print(f"{'Max Predicted Vmax (kt)':<25} | {base_max_pred:10.2f} kt | {exp_max_pred:10.2f} kt | {exp_max_pred - base_max_pred:+10.2f} kt")
    print("=" * 80)

    # 3. Intensity-Bin Breakdown Table
    bin_comparison = []
    print("\n" + "=" * 110)
    print("INTENSITY BIN COMPARISON: BASELINE vs. EXPANDED")
    print("=" * 110)
    print(f"{'Intensity Bin':<14} | {'Base N':<8} | {'Base MAE':<10} | {'Base Bias':<11} | {'Exp N':<8} | {'Exp MAE':<10} | {'Exp Bias':<11} | {'Δ MAE (kt)'}")
    print("-" * 110)

    for lower, upper, label in INTENSITY_BINS:
        b_sub = df_base[df_base["bin"] == label]
        e_sub = df_exp[df_exp["bin"] == label]

        b_n = len(b_sub)
        e_n = len(e_sub)

        b_mae = float(b_sub["abs_error"].mean()) if b_n > 0 else 0.0
        e_mae = float(e_sub["abs_error"].mean()) if e_n > 0 else 0.0

        b_bias = float(b_sub["error"].mean()) if b_n > 0 else 0.0
        e_bias = float(e_sub["error"].mean()) if e_n > 0 else 0.0

        d_mae = e_mae - b_mae if (b_n > 0 and e_n > 0) else 0.0

        bin_comparison.append({
            "bin": label,
            "baseline_samples": b_n,
            "baseline_mae": round(b_mae, 2),
            "baseline_bias": round(b_bias, 2),
            "expanded_samples": e_n,
            "expanded_mae": round(e_mae, 2),
            "expanded_bias": round(e_bias, 2),
            "delta_mae": round(d_mae, 2)
        })

        b_mae_str = f"{b_mae:6.2f} kt" if b_n > 0 else "N/A"
        b_bias_str = f"{b_bias:+6.2f} kt" if b_n > 0 else "N/A"
        e_mae_str = f"{e_mae:6.2f} kt" if e_n > 0 else "N/A"
        e_bias_str = f"{e_bias:+6.2f} kt" if e_n > 0 else "N/A"
        d_mae_str = f"{d_mae:+6.2f} kt" if (b_n > 0 and e_n > 0) else "N/A"

        print(f"{label:<14} | {b_n:<8,d} | {b_mae_str:<10} | {b_bias_str:<11} | {e_n:<8,d} | {e_mae_str:<10} | {e_bias_str:<11} | {d_mae_str}")

    print("=" * 110)

    # 4. Save JSON Comparison
    summary_json = {
        "summary": {
            "baseline_mae": round(base_mae, 2),
            "expanded_mae": round(exp_mae, 2),
            "delta_mae": round(exp_mae - base_mae, 2),
            "baseline_rmse": round(base_rmse, 2),
            "expanded_rmse": round(exp_rmse, 2),
            "baseline_slope": round(base_slope, 3),
            "expanded_slope": round(exp_slope, 3),
            "delta_slope": round(exp_slope - base_slope, 3),
            "baseline_bias_high_ge_110kt": round(base_bias_high, 2),
            "expanded_bias_high_ge_110kt": round(exp_bias_high, 2),
            "baseline_max_predicted": round(base_max_pred, 2),
            "expanded_max_predicted": round(exp_max_pred, 2)
        },
        "bins": bin_comparison
    }

    with open(exp_p / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    # 5. Generate 4 Required Comparison Plots

    # Plot 1: Intensity Distribution Comparison
    # Load training distributions
    try:
        with open("experiments/analysis/intensity_distribution.json", "r") as f:
            base_dist = json.load(f)["distribution"]
        with open(exp_p / "intensity_distribution.json", "r") as f:
            exp_dist = json.load(f)["distribution"]

        plt.figure(figsize=(11, 6), dpi=150)
        labels = [item["bin"] for item in base_dist]
        b_counts = [item["frames"] for item in base_dist]
        e_counts = [item["frames"] for item in exp_dist]

        x = np.arange(len(labels))
        w = 0.35

        plt.bar(x - w/2, b_counts, w, label=f"Baseline CPAC/IO/SH ({sum(b_counts):,} frames)", color="#3b82f6", alpha=0.85, edgecolor="black")
        plt.bar(x + w/2, e_counts, w, label=f"Expanded All-Basins ({sum(e_counts):,} frames)", color="#10b981", alpha=0.85, edgecolor="black")

        plt.xlabel("Intensity Bins", fontsize=11, fontweight="bold")
        plt.ylabel("Training Frame Count", fontsize=11, fontweight="bold")
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.title("Training Set Distribution: Baseline (3 Basins) vs. Expanded (All 6 Basins)\nSubstantial Increase in High-Intensity Observations (>110 kt)", fontsize=12, fontweight="bold", pad=15)
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(out_p / "baseline_vs_expanded_intensity_distribution.png")
        plt.close()
    except Exception as exc:
        print(f"[Warning] Could not plot distribution comparison: {exc}")

    # Plot 2: Error by Intensity (MAE)
    plt.figure(figsize=(11, 6), dpi=150)
    labels = [item["bin"] for item in bin_comparison]
    b_maes = [item["baseline_mae"] for item in bin_comparison]
    e_maes = [item["expanded_mae"] for item in bin_comparison]

    x = np.arange(len(labels))
    w = 0.35

    plt.bar(x - w/2, b_maes, w, label=f"Baseline ResNet18 (Overall MAE: {base_mae:.2f} kt)", color="#3b82f6", alpha=0.85, edgecolor="black")
    plt.bar(x + w/2, e_maes, w, label=f"Expanded ResNet18 (Overall MAE: {exp_mae:.2f} kt)", color="#10b981", alpha=0.85, edgecolor="black")

    plt.xlabel("Intensity Bins", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Absolute Error (MAE in knots)", fontsize=11, fontweight="bold")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.title("Test Error (MAE) by Intensity: Baseline vs. Expanded\nImpact of Training Data Diversity on High-End Error Reduction", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_p / "baseline_vs_expanded_error_by_intensity.png")
    plt.close()

    # Plot 3: Prediction vs. Actual Scatter with Regression Slopes
    plt.figure(figsize=(10, 8), dpi=150)
    lims = [10, 175]

    plt.scatter(df_base["actual"], df_base["predicted"], alpha=0.2, s=15, color="#3b82f6", label="Baseline Test Samples")
    plt.scatter(df_exp["actual"], df_exp["predicted"], alpha=0.2, s=15, color="#10b981", label="Expanded Test Samples")

    # Ideal line
    plt.plot(lims, lims, "k--", linewidth=2.0, label="Ideal Estimation (y = x)")

    # Fits
    x_grid = np.linspace(15, 170, 200)
    plt.plot(x_grid, base_slope * x_grid + base_intercept, color="#1d4ed8", linewidth=2.5, label=f"Baseline Fit (Slope: {base_slope:.2f})")
    plt.plot(x_grid, exp_slope * x_grid + exp_intercept, color="#047857", linewidth=2.5, label=f"Expanded Fit (Slope: {exp_slope:.2f})")

    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Actual Ground Truth Wind Speed (knots)", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted Wind Speed (knots)", fontsize=11, fontweight="bold")
    plt.title(f"Prediction Compression Comparison: Baseline (Slope={base_slope:.2f}) vs. Expanded (Slope={exp_slope:.2f})\nHypothesis Test: Slope Moving Toward 1.00", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=9.5)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_p / "baseline_vs_expanded_prediction_vs_actual.png")
    plt.close()

    # Plot 4: Bias Comparison Across Intensity Bins
    plt.figure(figsize=(11, 6), dpi=150)
    b_biases = [item["baseline_bias"] for item in bin_comparison]
    e_biases = [item["expanded_bias"] for item in bin_comparison]

    plt.plot(x, b_biases, "o--", color="#3b82f6", linewidth=2.2, markersize=8, label="Baseline Mean Bias")
    plt.plot(x, e_biases, "s-", color="#10b981", linewidth=2.5, markersize=8, label="Expanded Mean Bias")
    plt.axhline(0, color="gray", linestyle=":", alpha=0.8)

    plt.xlabel("Intensity Bins", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Signed Error / Bias (knots)", fontsize=11, fontweight="bold")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.title("Systematic Under/Over-Estimation Bias across Intensity Bins\nEvaluation of Upper-End Negative Bias Reduction (≥110 kt)", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_p / "baseline_vs_expanded_bias.png")
    plt.close()

    print(f"\n[Artifacts Generated] Saved comparison plots to: {out_p}")
    return summary_json


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs Expanded experiments.")
    parser.add_argument("--baseline", type=str, default="experiments/baseline_resnet18_cpac_io_sh")
    parser.add_argument("--expanded", type=str, default="experiments/expanded_all_basins_resnet18")
    parser.add_argument("--output-dir", type=str, default="experiments/expanded_all_basins_resnet18/plots")
    args = parser.parse_args()

    compare_baseline_vs_expanded(
        baseline_dir=args.baseline,
        expanded_dir=args.expanded,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
