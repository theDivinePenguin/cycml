"""Standalone evaluation script and helper functions for test set evaluation."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import calculate_metrics
from src.visualization.plots import (
    plot_error_distribution,
    plot_error_vs_intensity,
    plot_prediction_vs_actual,
    plot_sample_predictions
)


def evaluate_model_on_dataset(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True
) -> Dict[str, Any]:
    """Run evaluation on a dataset loader and collect predictions, targets, and metadata."""
    model.eval()
    all_preds = []
    all_gts = []
    all_metadata = []
    sample_images = []

    with torch.no_grad():
        for batch_idx, (images, targets, meta) in enumerate(data_loader):
            images_dev = images.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                outputs = model(images_dev)

            preds = outputs.cpu().numpy().flatten()
            gts = targets.numpy().flatten()

            all_preds.extend(preds)
            all_gts.extend(gts)

            # Store images for visualization (up to first 50)
            if len(sample_images) < 50:
                sample_images.extend(images.numpy())

            # Unpack metadata batch dict
            batch_len = len(gts)
            for i in range(batch_len):
                row_meta = {k: meta[k][i] if isinstance(meta[k], list) else meta[k][i].item() if isinstance(meta[k], torch.Tensor) else meta[k][i] for k in meta}
                all_metadata.append(row_meta)

    preds_arr = np.array(all_preds)
    gts_arr = np.array(all_gts)

    metrics = calculate_metrics(preds_arr, gts_arr)

    return {
        "metrics": metrics,
        "predictions": preds_arr,
        "targets": gts_arr,
        "metadata": all_metadata,
        "sample_images": np.array(sample_images) if sample_images else None
    }


def generate_evaluation_artifacts(
    eval_results: Dict[str, Any],
    output_dir: str | Path,
    experiment_name: str = "Baseline"
) -> Dict[str, Any]:
    """Generate all required evaluation metrics, plots, and summary CSVs."""
    out_p = Path(output_dir)
    plots_p = out_p / "plots"
    plots_p.mkdir(parents=True, exist_ok=True)

    metrics = eval_results["metrics"]
    preds = eval_results["predictions"]
    gts = eval_results["targets"]
    meta = eval_results["metadata"]

    # 1. Save test metrics JSON
    metrics_path = out_p / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 2. Save test predictions CSV
    df_preds = pd.DataFrame(meta)
    df_preds["actual_wind_speed"] = gts
    df_preds["predicted_wind_speed"] = preds
    df_preds["error"] = preds - gts
    df_preds["abs_error"] = np.abs(preds - gts)

    csv_path = out_p / "test_predictions.csv"
    df_preds.to_csv(csv_path, index=False)

    # 3. Per-cyclone aggregated metrics
    storm_metrics = df_preds.groupby("cyclone_id").agg(
        n_frames=("abs_error", "count"),
        storm_mae=("abs_error", "mean"),
        storm_rmse=("error", lambda x: float(np.sqrt(np.mean(x ** 2)))),
        actual_max_wind=("actual_wind_speed", "max"),
        pred_max_wind=("predicted_wind_speed", "max")
    ).reset_index()

    mean_cyclone_mae = float(storm_metrics["storm_mae"].mean())
    median_cyclone_mae = float(storm_metrics["storm_mae"].median())

    metrics["mean_cyclone_mae"] = mean_cyclone_mae
    metrics["median_cyclone_mae"] = median_cyclone_mae

    storm_csv_path = out_p / "test_per_cyclone_metrics.csv"
    storm_metrics.to_csv(storm_csv_path, index=False)

    # 4. Generate Plots
    plot_prediction_vs_actual(
        predictions=preds,
        targets=gts,
        metrics=metrics,
        save_path=plots_p / "predicted_vs_actual.png",
        title=f"{experiment_name}: Predicted vs Actual Wind Speed (Test Set)"
    )

    plot_error_distribution(
        predictions=preds,
        targets=gts,
        metrics=metrics,
        save_path=plots_p / "error_distribution.png",
        title=f"{experiment_name}: Test Error Distribution"
    )

    plot_error_vs_intensity(
        predictions=preds,
        targets=gts,
        save_path=plots_p / "error_vs_intensity.png",
        title=f"{experiment_name}: MAE across Intensity Categories"
    )

    if eval_results.get("sample_images") is not None and len(eval_results["sample_images"]) > 0:
        plot_sample_predictions(
            images=eval_results["sample_images"],
            predictions=preds[:len(eval_results["sample_images"])],
            targets=gts[:len(eval_results["sample_images"])],
            metadata_list=meta[:len(eval_results["sample_images"])],
            save_path=plots_p / "sample_predictions.png"
        )

    print(f"\n[Evaluation Summary]")
    print(f"  • Test MAE:         {metrics['mae']:.2f} knots")
    print(f"  • Test RMSE:        {metrics['rmse']:.2f} knots")
    print(f"  • Test R²:          {metrics['r2']:.4f}")
    print(f"  • Median Abs Error: {metrics['median_ae']:.2f} knots")
    print(f"  • Mean Bias:        {metrics['mean_bias']:+.2f} knots")
    print(f"  • Max Abs Error:    {metrics['max_ae']:.2f} knots")
    print(f"  • Saved artifacts to: {out_p}")

    return metrics
