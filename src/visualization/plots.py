"""Plotting utilities for evaluation metrics, error distributions, and sample predictions."""
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_prediction_vs_actual(
    predictions: np.ndarray,
    targets: np.ndarray,
    metrics: Dict[str, float],
    save_path: str | Path,
    title: str = "Cyclone Intensity: Predicted vs Actual Wind Speed"
) -> None:
    """Generate scatter plot of predicted vs actual wind speed with ideal y=x line."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8), dpi=150)
    plt.scatter(targets, predictions, alpha=0.35, edgecolors="none", s=25, c="#1f77b4")

    # Ideal y=x line
    min_val = min(targets.min(), predictions.min()) - 5
    max_val = max(targets.max(), predictions.max()) + 5
    plt.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", linewidth=2, label="Ideal (y = x)")

    plt.xlabel("Actual Maximum Sustained Wind Speed (knots)", fontsize=12)
    plt.ylabel("Predicted Maximum Sustained Wind Speed (knots)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle=":", alpha=0.6)

    # Metric box
    mae = metrics.get("mae", 0.0)
    rmse = metrics.get("rmse", 0.0)
    r2 = metrics.get("r2", 0.0)
    text_str = f"MAE:  {mae:.2f} kt\nRMSE: {rmse:.2f} kt\nR²:   {r2:.3f}\nN:    {len(targets):,}"
    plt.gca().text(
        0.05, 0.95, text_str,
        transform=plt.gca().transAxes,
        fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
    )

    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    print(f"[Visualization] Saved scatter plot to: {p}")


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    metrics: Dict[str, float],
    save_path: str | Path,
    title: str = "Prediction Error Distribution (Residuals)"
) -> None:
    """Generate histogram and KDE of prediction errors (Predicted - Ground Truth)."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    errors = predictions - targets

    plt.figure(figsize=(9, 6), dpi=150)
    sns.histplot(errors, kde=True, bins=40, color="#2ca02c", edgecolor="black", alpha=0.6)
    plt.axvline(0, color="crimson", linestyle="--", linewidth=2, label="Zero Error")
    plt.axvline(float(np.mean(errors)), color="darkorange", linestyle="-", linewidth=2, label=f"Mean Bias ({np.mean(errors):.2f} kt)")

    plt.xlabel("Prediction Error: (Predicted - Actual) in knots", fontsize=12)
    plt.ylabel("Number of Satellite Frames", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.grid(True, linestyle=":", alpha=0.6)

    median_ae = metrics.get("median_ae", float(np.median(np.abs(errors))))
    text_str = f"Mean Bias:  {np.mean(errors):+.2f} kt\nStd Dev:    {np.std(errors):.2f} kt\nMedian AE:  {median_ae:.2f} kt"
    plt.gca().text(
        0.75, 0.95, text_str,
        transform=plt.gca().transAxes,
        fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
    )

    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    print(f"[Visualization] Saved error distribution plot to: {p}")


def plot_error_vs_intensity(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str | Path,
    title: str = "Mean Absolute Error across Intensity Categories"
) -> None:
    """Plot MAE grouped by tropical cyclone intensity categories."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    abs_errors = np.abs(predictions - targets)

    # Standard WMO/IMD/Saffir-Simpson intensity bins (in knots)
    bins = [0, 34, 48, 64, 90, 200]
    bin_labels = [
        "Tropical Depression (<34 kt)",
        "Moderate TS (34-47 kt)",
        "Severe TS (48-63 kt)",
        "Category 1-2 Cyclone (64-89 kt)",
        "Major Cyclone (≥90 kt)"
    ]

    bin_indices = np.digitize(targets, bins) - 1
    category_maes = []
    category_counts = []
    valid_labels = []

    for i, label in enumerate(bin_labels):
        mask = (bin_indices == i)
        if mask.any():
            category_maes.append(float(np.mean(abs_errors[mask])))
            category_counts.append(int(mask.sum()))
            valid_labels.append(f"{label}\n(N={mask.sum():,})")

    plt.figure(figsize=(10, 6), dpi=150)
    bars = plt.bar(valid_labels, category_maes, color="#4c72b0", edgecolor="black", alpha=0.85)

    for bar, mae in zip(bars, category_maes):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{mae:.2f} kt",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.ylabel("Mean Absolute Error (knots)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.ylim(0, max(category_maes) * 1.25 if category_maes else 10)
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    print(f"[Visualization] Saved error vs intensity plot to: {p}")


def plot_sample_predictions(
    images: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    metadata_list: List[dict],
    save_path: str | Path,
    n_samples: int = 6
) -> None:
    """Generate visual cards showing satellite IR images with actual and predicted wind speeds."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    errors = np.abs(predictions - targets)
    sorted_indices = np.argsort(errors)

    # Select representative samples: 2 best, 2 median, 2 worst
    n_total = len(sorted_indices)
    selected_idx = [
        sorted_indices[0],
        sorted_indices[min(1, n_total - 1)],
        sorted_indices[n_total // 2 - 1],
        sorted_indices[n_total // 2],
        sorted_indices[max(0, n_total - 2)],
        sorted_indices[n_total - 1]
    ][:n_samples]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=150)
    axes = axes.flatten()

    for ax_idx, s_idx in enumerate(selected_idx):
        ax = axes[ax_idx]
        img = images[s_idx]  # Shape (H, W) or (1, H, W)
        if img.ndim == 3:
            img = img[0]

        pred = predictions[s_idx]
        act = targets[s_idx]
        err = pred - act
        meta = metadata_list[s_idx] if s_idx < len(metadata_list) else {}
        cyclone_id = meta.get("cyclone_id", "Unknown")

        im = ax.imshow(img, cmap="inferno")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        quality = "Best Prediction" if ax_idx < 2 else ("Median Error" if ax_idx < 4 else "Largest Error")
        ax.set_title(f"[{quality}]\nCyclone: {cyclone_id}\nActual: {act:.1f} kt | Pred: {pred:.1f} kt\nError: {err:+.1f} kt", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    print(f"[Visualization] Saved sample prediction cards to: {p}")
