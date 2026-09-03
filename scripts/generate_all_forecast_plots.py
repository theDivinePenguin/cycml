"""Generate all publication-quality forecasting figures, Giri/Madi lifecycle forecasts, and update reports."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy import stats
import torch

from src.data.sequence_dataset import TCIRSequenceDataset
from src.models.temporal_forecaster import TemporalGRUForecaster, TemporalTransformerForecaster
from scripts.build_forecast_sequences import build_sequences_for_df


def generate_all_figures():
    figures_dir = Path("experiments/forecasting/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("experiments/forecasting/results")

    with open(results_dir / "comprehensive_forecasting_results.json") as f:
        comp_res = json.load(f)

    metrics = comp_res["multi_horizon_metrics"]
    test_seq_df = pd.read_csv("data/metadata/forecast_test_sequences_k5.csv")
    tf_preds_df = pd.read_csv("experiments/forecasting/checkpoints/cnn_transformer_k5/test_predictions.csv")
    gru_preds_df = pd.read_csv("experiments/forecasting/checkpoints/cnn_gru_k5/test_predictions.csv")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Figure: Forecast Error vs Horizon (MAE & RMSE Curves)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    horizons = ["+6h", "+12h", "+24h"]
    h_steps = [6, 12, 24]

    colors = {
        "Oracle Persistence": "#64748B",
        "Current-CNN Hold-Forward": "#EF4444",
        "CNN + GRU (K=5)": "#0D9488",
        "CNN + Transformer (K=5)": "#1E3A8A",
        "CNN + Transformer (K=1)": "#F59E0B",
    }
    styles = {
        "Oracle Persistence": "--o",
        "Current-CNN Hold-Forward": ":s",
        "CNN + GRU (K=5)": "-^",
        "CNN + Transformer (K=5)": "-D",
        "CNN + Transformer (K=1)": "-.v",
    }

    for model_name, h_dict in metrics.items():
        maes = [h_dict[h]["mae"] for h in horizons]
        rmses = [h_dict[h]["rmse"] for h in horizons]
        c = colors.get(model_name, "#333333")
        st = styles.get(model_name, "-o")

        axes[0].plot(h_steps, maes, st, label=model_name, color=c, linewidth=2.0, markersize=7)
        axes[1].plot(h_steps, rmses, st, label=model_name, color=c, linewidth=2.0, markersize=7)

    axes[0].set_title("Forecast MAE vs Lead Time", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Forecast Horizon (Hours)")
    axes[0].set_ylabel("Mean Absolute Error (knots)")
    axes[0].set_xticks(h_steps)
    axes[0].legend(frameon=True, fontsize=8.5)

    axes[1].set_title("Forecast RMSE vs Lead Time", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Forecast Horizon (Hours)")
    axes[1].set_ylabel("Root Mean Square Error (knots)")
    axes[1].set_xticks(h_steps)
    axes[1].legend(frameon=True, fontsize=8.5)

    plt.suptitle("Multi-Horizon Tropical Cyclone Intensity Forecasting Benchmark", fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(figures_dir / "forecast_error_vs_horizon.png")
    plt.close()

    # 2. Figure: Temporal Context Ablation (1-Frame vs 5-Frame)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ablation_models = [m for m in ["CNN + Transformer (K=1)", "CNN + Transformer (K=5)"] if m in metrics]
    x_idx = np.arange(len(horizons))
    width = 0.35
    for i, m_name in enumerate(ablation_models):
        m_maes = [metrics[m_name][h]["mae"] for h in horizons]
        ax.bar(x_idx + i * width, m_maes, width=width, label=m_name, color=["#F59E0B", "#1E3A8A"][i], edgecolor="black")
        for j, val in enumerate(m_maes):
            ax.text(x_idx[j] + i * width, val + 0.15, f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_title("Temporal Context Length Impact: 1-Frame (Single Image) vs 5-Frames (12h History)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("MAE (knots)")
    ax.set_xticks(x_idx + width / 2)
    ax.set_xticklabels(horizons)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "temporal_context_ablation.png")
    plt.close()

    # 3. Figure: Scatter Plots & Linear Fits (+6h, +12h, +24h for Temporal Transformer)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    for idx, (h_name, col_act, col_pred) in enumerate([
        ("+6h", "actual_plus_6h", "pred_plus_6h"),
        ("+12h", "actual_plus_12h", "pred_plus_12h"),
        ("+24h", "actual_plus_24h", "pred_plus_24h"),
    ]):
        ax = axes[idx]
        actual = tf_preds_df[col_act].values
        pred = tf_preds_df[col_pred].values
        mae = float(np.mean(np.abs(pred - actual)))
        r2 = metrics["CNN + Transformer (K=5)"][h_name]["r2"]

        ax.scatter(actual, pred, alpha=0.15, s=10, color="#1E3A8A", edgecolors="none")
        ax.plot([0, 165], [0, 165], "r--", linewidth=1.5, label="1:1 Perfect Forecast")

        # Regression line
        slope, intercept, _, _, _ = stats.linregress(actual, pred)
        x_vals = np.linspace(0, 160, 100)
        ax.plot(x_vals, slope * x_vals + intercept, "g-", linewidth=1.5, label=f"Fit (Slope: {slope:.2f})")

        ax.set_title(f"CNN + Transformer: {h_name} Forecast\nMAE: {mae:.2f} kt | R²: {r2:.3f}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Actual Ground Truth Intensity (knots)")
        ax.set_ylabel("Predicted Future Intensity (knots)")
        ax.set_xlim(0, 165)
        ax.set_ylim(0, 165)
        ax.legend(loc="upper left", fontsize=8.5)

    plt.suptitle("Temporal Transformer Multi-Horizon Predictions vs Actual Ground Truth (N=8,279)", fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(figures_dir / "predicted_vs_actual_scatter_6h_12h_24h.png")
    plt.close()

    # 4. Figure: Confusion Matrices for Intensification / Weakening
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    class_names = ["Weakening\n(ΔV ≤ -10)", "Stable\n(|ΔV| < 10)", "Intensifying\n(ΔV ≥ +10)"]

    for idx, (h_name, col_act, col_pred) in enumerate([
        ("+6h", "actual_plus_6h", "pred_plus_6h"),
        ("+12h", "actual_plus_12h", "pred_plus_12h"),
        ("+24h", "actual_plus_24h", "pred_plus_24h"),
    ]):
        ax = axes[idx]
        act_delta = tf_preds_df[col_act].values - tf_preds_df["vmax_curr"].values
        pred_delta = tf_preds_df[col_pred].values - tf_preds_df["vmax_curr"].values

        def to_class(d):
            c = np.ones_like(d, dtype=int)
            c[d <= -10.0] = 0
            c[d >= 10.0] = 2
            return c

        cm = confusion_matrix(to_class(act_delta), to_class(pred_delta), labels=[0, 1, 2])
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f"{h_name} Forecast Intensification Matrix", fontweight="bold", fontsize=11)
        ax.set_xlabel("Predicted Intensity Trend")
        ax.set_ylabel("Actual Ground Truth Trend")

    plt.suptitle("Tropical Cyclone Intensity Evolution Classification (Threshold: ±10 kt)", fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(figures_dir / "intensification_confusion_matrices.png")
    plt.close()

    # 5. Figure: Error by Intensity Regime
    regimes_dict = comp_res.get("intensity_regimes", {})
    if regimes_dict:
        fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
        r_labels = list(regimes_dict.keys())
        x_r = np.arange(len(r_labels))
        main_models = [m for m in ["Oracle Persistence", "Current-CNN Hold-Forward", "CNN + GRU (K=5)", "CNN + Transformer (K=5)"] if m in metrics]
        w = 0.2
        for i, m_name in enumerate(main_models):
            r_maes = [regimes_dict[r]["mae_by_model"].get(m_name, {}).get("+24h", 0.0) for r in r_labels]
            ax.bar(x_r + i * w, r_maes, width=w, label=f"{m_name} (+24h)", color=["#64748B", "#EF4444", "#0D9488", "#1E3A8A"][i], edgecolor="black")

        ax.set_title("24-Hour Forecast MAE Stratified by Saffir-Simpson Intensity Regime", fontweight="bold")
        ax.set_xlabel("Current Intensity Regime")
        ax.set_ylabel("+24h Forecast MAE (knots)")
        ax.set_xticks(x_r + w * (len(main_models) - 1) / 2)
        ax.set_xticklabels(r_labels, rotation=15)
        ax.legend(frameon=True, fontsize=8.5)
        plt.tight_layout()
        plt.savefig(figures_dir / "error_by_intensity_regime.png")
        plt.close()

    # 6 & 7. Zero-Shot Indian Ocean Storm Forecasting (Giri 201004I & Madi 201306I)
    print("\nRunning Zero-Shot Lifecycle Forecasting on Giri (201004I) & Madi (201306I)...")
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

    for cid, s_name in [
        ("201003I", "Super Cyclone Phet (Test Set)"),
        ("200801I", "VSCS Nargis (Test Set)"),
        ("201004I", "Super Cyclone Giri (Val Split)"),
        ("201306I", "VSCS Madi (Train Split)"),
    ]:
        storm_df = all_df[all_df["cyclone_id"] == cid]
        storm_seq_df = build_sequences_for_df(storm_df, k_history=5, cadence_hours=3)
        if len(storm_seq_df) == 0:
            print(f"Skipping {s_name} - not enough consecutive sequence frames")
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

        # Plot 3 horizons for this storm
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
        t_steps = np.arange(len(storm_seq_df)) * 3.0  # hours from initial observation

        for h_idx, (h_name, col_name) in enumerate([("+6h", "vmax_plus_6h"), ("+12h", "vmax_plus_12h"), ("+24h", "vmax_plus_24h")]):
            ax = axes[h_idx]
            act = targets_np[:, h_idx]
            p_tf = tf_preds[:, h_idx]
            p_gru = gru_preds[:, h_idx]
            p_pers = v_curr_np

            mae_tf = float(np.mean(np.abs(p_tf - act)))
            mae_gru = float(np.mean(np.abs(p_gru - act)))
            mae_pers = float(np.mean(np.abs(p_pers - act)))

            ax.plot(t_steps, act, "k-o", linewidth=2.5, label=f"Actual {h_name} Intensity", markersize=6)
            ax.plot(t_steps, p_pers, "--s", color="#64748B", label=f"Persistence (MAE: {mae_pers:.1f} kt)", markersize=5)
            ax.plot(t_steps, p_gru, "-.^", color="#0D9488", label=f"CNN + GRU (MAE: {mae_gru:.1f} kt)", markersize=5)
            ax.plot(t_steps, p_tf, "-D", color="#1E3A8A", label=f"CNN + Transformer (MAE: {mae_tf:.1f} kt)", markersize=6, linewidth=2.0)

            ax.set_title(f"{s_name} — {h_name} Forecast", fontweight="bold", fontsize=11)
            ax.set_xlabel("Elapsed Time (Hours)")
            ax.set_ylabel("Intensity (knots)")
            ax.legend(frameon=True, fontsize=8)

        clean_fn = s_name.lower().split(" (")[0].replace(" ", "_")
        plt.suptitle(f"Multi-Horizon Lifecycle Forecasting: {s_name} ({cid})", fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(figures_dir / f"{clean_fn}_lifecycle_forecast.png")
        plt.close()
        storm_ds.close()

    print(f"\n[All Publication Figures Successfully Generated in {figures_dir}]")


if __name__ == "__main__":
    generate_all_figures()
