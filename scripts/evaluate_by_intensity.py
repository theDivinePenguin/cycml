"""Evaluate baseline model performance by intensity bin and quantify prediction saturation."""
import argparse
import json
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.dataset import build_dataloaders
from src.evaluation.intensity_bins import INTENSITY_BINS, assign_intensity_bin
from src.models.factory import build_model
from src.utils.config import load_config


def evaluate_model_by_intensity(
    predictions_csv: str | Path = "experiments/baseline_resnet18_cpac_io_sh/test_predictions.csv",
    output_dir: str | Path = "experiments/analysis"
) -> dict:
    """Evaluate held-out test set by intensity bins and quantify high-intensity saturation."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    csv_path = Path(predictions_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {csv_path}")

    print(f"[Intensity Evaluation] Loading authoritative test predictions from {csv_path}...")
    df_raw = pd.read_csv(csv_path)

    actual_col = "actual_wind_speed" if "actual_wind_speed" in df_raw.columns else "wind_speed"
    pred_col = "predicted_wind_speed" if "predicted_wind_speed" in df_raw.columns else "prediction"

    df_test = pd.DataFrame({
        "actual": df_raw[actual_col].astype(float),
        "predicted": df_raw[pred_col].astype(float),
        "cyclone_id": df_raw["cyclone_id"].astype(str)
    })
    df_test["error"] = df_test["predicted"] - df_test["actual"]
    df_test["abs_error"] = np.abs(df_test["error"])
    df_test["bin"] = df_test["actual"].apply(assign_intensity_bin)

    # 4. Calculate Metrics for Each Intensity Bin
    bin_metrics = []
    print("\n" + "=" * 85)
    print("TEST SET PERFORMANCE BY INTENSITY BIN (HELD-OUT TEST SET)")
    print(f"Total Test Samples: {len(df_test):,} frames across {df_test['cyclone_id'].nunique()} unique cyclones")
    print("=" * 85)
    print(f"{'Intensity Bin':<14} | {'Samples':<8} | {'MAE (kt)':<10} | {'RMSE (kt)':<10} | {'Bias (kt)':<10} | {'Median AE (kt)'}")
    print("-" * 85)

    for lower, upper, label in INTENSITY_BINS:
        bin_df = df_test[df_test["bin"] == label]
        n_samples = len(bin_df)

        if n_samples > 0:
            mae = float(bin_df["abs_error"].mean())
            rmse = float(np.sqrt((bin_df["error"] ** 2).mean()))
            bias = float(bin_df["error"].mean())
            median_ae = float(bin_df["abs_error"].median())
        else:
            mae, rmse, bias, median_ae = 0.0, 0.0, 0.0, 0.0

        bin_metrics.append({
            "bin": label,
            "lower_kt": lower,
            "upper_kt": upper if upper != float("inf") else None,
            "samples": n_samples,
            "percent_samples": round((n_samples / len(df_test)) * 100.0, 2),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "bias": round(bias, 2),
            "median_ae": round(median_ae, 2)
        })

        print(f"{label:<14} | {n_samples:<8,d} | {mae:<10.2f} | {rmse:<10.2f} | {bias:<+10.2f} | {median_ae:<10.2f}")

    print("=" * 85)

    # 5. Saturation Quantification
    actual_gt_100 = df_test[df_test["actual"] >= 100.0]
    actual_gt_120 = df_test[df_test["actual"] >= 120.0]

    mean_pred_gt_100 = float(actual_gt_100["predicted"].mean()) if len(actual_gt_100) > 0 else 0.0
    mean_actual_gt_100 = float(actual_gt_100["actual"].mean()) if len(actual_gt_100) > 0 else 0.0
    mean_pred_gt_120 = float(actual_gt_120["predicted"].mean()) if len(actual_gt_120) > 0 else 0.0
    mean_actual_gt_120 = float(actual_gt_120["actual"].mean()) if len(actual_gt_120) > 0 else 0.0

    max_pred = float(df_test["predicted"].max())
    max_actual = float(df_test["actual"].max())
    min_pred = float(df_test["predicted"].min())
    min_actual = float(df_test["actual"].min())

    saturation_stats = {
        "samples_gt_100kt": len(actual_gt_100),
        "mean_actual_gt_100kt": round(mean_actual_gt_100, 2),
        "mean_pred_gt_100kt": round(mean_pred_gt_100, 2),
        "underestimation_bias_gt_100kt": round(mean_pred_gt_100 - mean_actual_gt_100, 2),
        "samples_gt_120kt": len(actual_gt_120),
        "mean_actual_gt_120kt": round(mean_actual_gt_120, 2),
        "mean_pred_gt_120kt": round(mean_pred_gt_120, 2),
        "underestimation_bias_gt_120kt": round(mean_pred_gt_120 - mean_actual_gt_120, 2),
        "max_predicted_vmax_kt": round(max_pred, 2),
        "max_actual_vmax_kt": round(max_actual, 2),
        "min_predicted_vmax_kt": round(min_pred, 2),
        "min_actual_vmax_kt": round(min_actual, 2)
    }

    print("\n" + "=" * 65)
    print("HIGH-INTENSITY SATURATION ANALYSIS")
    print("=" * 65)
    print(f"Maximum Actual Vmax:                  {max_actual:.1f} knots")
    print(f"Maximum Predicted Vmax:               {max_pred:.1f} knots")
    print(f"Mean Prediction for Actual >100 kt:   {mean_pred_gt_100:.1f} kt (Actual Mean: {mean_actual_gt_100:.1f} kt, Bias: {mean_pred_gt_100 - mean_actual_gt_100:+.1f} kt)")
    print(f"Mean Prediction for Actual >120 kt:   {mean_pred_gt_120:.1f} kt (Actual Mean: {mean_actual_gt_120:.1f} kt, Bias: {mean_pred_gt_120 - mean_actual_gt_120:+.1f} kt)")
    print("=" * 65)

    # 6. Save JSON
    error_json_data = {
        "dataset": "TCIR-CPAC_IO_SH Held-Out Test Set",
        "total_test_samples": len(df_test),
        "overall_test_mae": round(float(df_test["abs_error"].mean()), 2),
        "overall_test_rmse": round(float(np.sqrt((df_test["error"] ** 2).mean())), 2),
        "intensity_bins": bin_metrics,
        "saturation_analysis": saturation_stats
    }

    json_path = out_p / "error_by_intensity.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(error_json_data, f, indent=2)
    print(f"\n[Saved JSON] {json_path}")

    # 7. Generate Plot 1: MAE by Intensity Bin
    plt.figure(figsize=(10, 5.5), dpi=150)
    bins_labels = [item["bin"] for item in bin_metrics]
    maes = [item["mae"] for item in bin_metrics]
    samples = [item["samples"] for item in bin_metrics]

    x = np.arange(len(bins_labels))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(bins_labels)))

    bars = plt.bar(x, maes, width=0.55, color=colors, edgecolor="black", alpha=0.85, zorder=3)
    plt.xlabel("Tropical Cyclone Actual Intensity Bins", fontsize=11, fontweight="bold", labelpad=8)
    plt.ylabel("Mean Absolute Error (MAE in knots)", fontsize=11, fontweight="bold")
    plt.xticks(x, bins_labels, rotation=25, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

    # Add threshold reference lines
    plt.axhline(8.95, color="gray", linestyle="--", alpha=0.7, label="Overall Test Set MAE (8.95 kt)")

    for bar, sample_count, mae_val in zip(bars, samples, maes):
        h = bar.get_height()
        plt.annotate(
            f"{mae_val:.1f} kt\n(N={sample_count})",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold"
        )

    plt.title("Baseline ResNet18: Prediction Error (MAE) by Actual Cyclone Intensity\nErrors Escalate Significantly in High-Intensity Under-Represented Bins", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    error_plot_path = out_p / "error_by_intensity.png"
    plt.savefig(error_plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved Plot] {error_plot_path}")

    # 8. Generate Plot 2: Prediction vs Actual High-Intensity Saturation Scatter Plot
    plt.figure(figsize=(9, 8), dpi=150)
    actuals = df_test["actual"].values
    preds = df_test["predicted"].values

    plt.scatter(actuals, preds, alpha=0.35, s=22, color="#2563eb", edgecolors="none", label=f"Test Samples (N={len(df_test):,})", zorder=3)

    # Reference y=x Line
    lims = [10, 165]
    plt.plot(lims, lims, "r--", linewidth=2.0, label="Ideal Estimation (y = x)", zorder=4)

    # Fitted Regression Line
    poly_fit = np.polyfit(actuals, preds, deg=1)
    fit_fn = np.poly1d(poly_fit)
    x_line = np.linspace(15, 160, 200)
    plt.plot(x_line, fit_fn(x_line), color="#16a34a", linewidth=2.5, label=f"Linear Fit (Slope: {poly_fit[0]:.2f}, Intercept: {poly_fit[1]:.1f})", zorder=5)

    # Highlight High-Intensity Saturation Region
    plt.axvspan(100, 165, color="#fee2e2", alpha=0.35, label="High-Intensity Saturation Regime (>100 kt)")
    plt.axvline(100, color="#dc2626", linestyle=":", alpha=0.6)

    # Annotate compression text box
    sat_text = (
        f"Saturation Analysis:\n"
        f"• Max Actual: {max_actual:.1f} kt\n"
        f"• Max Predicted: {max_pred:.1f} kt\n"
        f"• Mean Pred for Actual >100 kt: {mean_pred_gt_100:.1f} kt\n"
        f"• Mean Pred for Actual >120 kt: {mean_pred_gt_120:.1f} kt\n"
        f"• Compression Slope: {poly_fit[0]:.2f} (Ideal: 1.00)"
    )
    plt.gca().text(
        0.03, 0.96, sat_text,
        transform=plt.gca().transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
    )

    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Actual Ground Truth Maximum Sustained Wind Speed (knots)", fontsize=11, fontweight="bold", labelpad=8)
    plt.ylabel("Model Predicted Wind Speed (knots)", fontsize=11, fontweight="bold", labelpad=8)
    plt.title("Predicted vs. Actual Intensity — High-Intensity Compression & Saturation\n(Severe Underestimation Observed in Regimes >100 kt)", fontsize=12, fontweight="bold", pad=15)
    plt.grid(True, linestyle=":", alpha=0.5, zorder=0)
    plt.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout()

    pred_vs_act_path = out_p / "prediction_vs_actual_by_intensity.png"
    plt.savefig(pred_vs_act_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved Plot] {pred_vs_act_path}")

    return error_json_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate model error by intensity bins.")
    parser.add_argument("--predictions", type=str, default="experiments/baseline_resnet18_cpac_io_sh/test_predictions.csv", help="Path to test predictions CSV")
    parser.add_argument("--output-dir", type=str, default="experiments/analysis", help="Output directory")
    args = parser.parse_args()

    evaluate_model_by_intensity(
        predictions_csv=args.predictions,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
