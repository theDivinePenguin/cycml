"""Generate 2 separate standalone publication-quality graphs for Super Cyclone Phet:
1. True real-world intensity evolution across lifecycle.
2. ML predicted intensity trajectories (+6h, +12h, +24h) by the CNN + Temporal Transformer.
Also generates a combined side-by-side comparison figure.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_phet_graphs():
    out_dir = Path("diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load full storm Best Track data for Phet (201003I)
    all_df = pd.read_csv("data/metadata/metadata_all_basins.csv")
    phet_full = all_df[all_df["cyclone_id"] == "201003I"].sort_values("timestamp").reset_index(drop=True)
    phet_full["dt"] = pd.to_datetime(phet_full["timestamp"].astype(str), format="%Y%m%d%H")
    phet_full["elapsed_hours"] = (phet_full["dt"] - phet_full["dt"].iloc[0]).dt.total_seconds() / 3600.0

    # 2. Load ML Predictions diagnostic data for Phet
    diag_df = pd.read_csv("diagnostics/lifecycle_forecast_raw.csv")
    phet_preds = diag_df[diag_df["cyclone_id"] == "201003I"].sort_values("sequence_index").reset_index(drop=True)
    phet_preds["origin_dt"] = pd.to_datetime(phet_preds["forecast_origin_timestamp_t"].astype(str), format="%Y%m%d%H")
    phet_preds["origin_elapsed_hours"] = (phet_preds["origin_dt"] - phet_full["dt"].iloc[0]).dt.total_seconds() / 3600.0

    # Verification target dates
    phet_preds["target_dt_6h"] = phet_preds["origin_dt"] + pd.Timedelta(hours=6)
    phet_preds["target_dt_12h"] = phet_preds["origin_dt"] + pd.Timedelta(hours=12)
    phet_preds["target_dt_24h"] = phet_preds["origin_dt"] + pd.Timedelta(hours=24)

    phet_preds["target_elapsed_6h"] = phet_preds["origin_elapsed_hours"] + 6.0
    phet_preds["target_elapsed_12h"] = phet_preds["origin_elapsed_hours"] + 12.0
    phet_preds["target_elapsed_24h"] = phet_preds["origin_elapsed_hours"] + 24.0

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # =========================================================================
    # GRAPH 1: TRUE REAL-WORLD INTENSITY LIFECYCLE
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=180)

    # Intensity categories background shading
    ax1.axhspan(0, 34, color="#E2E8F0", alpha=0.4, label="Tropical Depression (<34 kt)")
    ax1.axhspan(34, 63, color="#FEF08A", alpha=0.35, label="Tropical Storm (34-63 kt)")
    ax1.axhspan(64, 82, color="#FED7AA", alpha=0.35, label="Category 1 (64-82 kt)")
    ax1.axhspan(83, 95, color="#FDBA74", alpha=0.35, label="Category 2 (83-95 kt)")
    ax1.axhspan(96, 112, color="#FCA5A5", alpha=0.35, label="Category 3 Major (96-112 kt)")
    ax1.axhspan(113, 140, color="#F87171", alpha=0.4, label="Category 4/5 Super Cyclone (113+ kt)")

    # Plot True Intensity Curve
    ax1.plot(
        phet_full["elapsed_hours"],
        phet_full["wind_speed"],
        color="#0F172A",
        linewidth=3.0,
        marker="o",
        markersize=6,
        label="Actual Ground Truth Vmax (Best Track)",
        zorder=5
    )

    # Key Milestone Annotations
    # 1. Peak Intensity
    peak_idx = phet_full["wind_speed"].idxmax()
    peak_t = phet_full.loc[peak_idx, "elapsed_hours"]
    peak_v = phet_full.loc[peak_idx, "wind_speed"]
    peak_date = phet_full.loc[peak_idx, "dt"].strftime("%b %d, %HZ")
    ax1.annotate(
        f"Peak Category 4: {peak_v:.0f} kt\n({peak_date})",
        xy=(peak_t, peak_v),
        xytext=(peak_t - 20, peak_v + 7),
        arrowprops=dict(facecolor="#DC2626", shrink=0.08, width=1.5, headwidth=7),
        fontsize=10,
        fontweight="bold",
        color="#DC2626",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DC2626", alpha=0.9)
    )

    # 2. Explosive Rapid Intensification
    ax1.annotate(
        "Explosive Rapid Intensification\n(+65 kt in 24h: 60 kt -> 125 kt)",
        xy=(48, 85),
        xytext=(15, 95),
        arrowprops=dict(facecolor="#2563EB", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight="bold",
        color="#1E40AF",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2563EB", alpha=0.9)
    )

    # 3. Decay near Oman / Recurvature
    ax1.annotate(
        "Landfall Weakening (Oman Coast)\n& Recurvature toward Pakistan",
        xy=(102, 60),
        xytext=(85, 80),
        arrowprops=dict(facecolor="#475569", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight="bold",
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#475569", alpha=0.9)
    )

    ax1.set_title("Super Cyclone Phet (201003I) — True Ground Truth Intensity Variation", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Elapsed Time from Storm Genesis (Hours)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Maximum Sustained Wind Speed (knots)", fontsize=11, fontweight="bold")
    ax1.set_ylim(15, 145)
    ax1.set_xlim(0, phet_full["elapsed_hours"].max() + 3)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower left", frameon=True, fontsize=8.5, ncol=2)

    g1_path = out_dir / "phet_true_intensity_lifecycle.png"
    plt.tight_layout()
    plt.savefig(g1_path)
    plt.close()
    print(f"[Generated Graph 1: True Intensity] -> {g1_path}")

    # =========================================================================
    # GRAPH 2: ML PREDICTED INTENSITY COURSE BY OUR MODEL
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 6), dpi=180)

    # Intensity categories background shading
    ax2.axhspan(0, 34, color="#E2E8F0", alpha=0.35)
    ax2.axhspan(34, 63, color="#FEF08A", alpha=0.3)
    ax2.axhspan(64, 82, color="#FED7AA", alpha=0.3)
    ax2.axhspan(83, 95, color="#FDBA74", alpha=0.3)
    ax2.axhspan(96, 112, color="#FCA5A5", alpha=0.3)
    ax2.axhspan(113, 140, color="#F87171", alpha=0.35)

    # Plot the 3 Machine Learning Forecast Trajectories
    # Plotted against forecast origin time
    t_origin = phet_preds["origin_elapsed_hours"].values

    ax2.plot(
        t_origin,
        phet_preds["transformer_pred_plus_6h"],
        color="#2563EB",
        linewidth=2.2,
        marker="^",
        markersize=5,
        label="ML Forecast (+6h Horizon) [MAE: 14.1 kt]"
    )
    ax2.plot(
        t_origin,
        phet_preds["transformer_pred_plus_12h"],
        color="#7C3AED",
        linewidth=2.2,
        marker="s",
        markersize=5,
        label="ML Forecast (+12h Horizon) [MAE: 16.0 kt]"
    )
    ax2.plot(
        t_origin,
        phet_preds["transformer_pred_plus_24h"],
        color="#DC2626",
        linewidth=2.5,
        marker="D",
        markersize=6,
        label="ML Forecast (+24h Horizon) [MAE: 19.2 kt]"
    )

    # Add reference envelope for ML forecast range (min to max across horizons)
    pred_min = np.minimum(np.minimum(phet_preds["transformer_pred_plus_6h"], phet_preds["transformer_pred_plus_12h"]), phet_preds["transformer_pred_plus_24h"])
    pred_max = np.maximum(np.maximum(phet_preds["transformer_pred_plus_6h"], phet_preds["transformer_pred_plus_12h"]), phet_preds["transformer_pred_plus_24h"])
    ax2.fill_between(t_origin, pred_min, pred_max, color="#818CF8", alpha=0.2, label="ML Multi-Horizon Forecast Spread")

    ax2.set_title("Super Cyclone Phet (201003I) — Machine Learning Predicted Course (CNN + Temporal Transformer)", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Forecast Origin Time (Elapsed Hours from Storm Genesis)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Predicted Future Maximum Wind Speed (knots)", fontsize=11, fontweight="bold")
    ax2.set_ylim(15, 145)
    ax2.set_xlim(0, phet_full["elapsed_hours"].max() + 3)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, fontsize=9.5)

    g2_path = out_dir / "phet_ml_predicted_lifecycle.png"
    plt.tight_layout()
    plt.savefig(g2_path)
    plt.close()
    print(f"[Generated Graph 2: ML Predicted Intensity] -> {g2_path}")

    # =========================================================================
    # GRAPH 3: SIDE-BY-SIDE DUAL COMPARISON PANEL
    # =========================================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(18, 6), dpi=180)

    # Panel A: True Intensity
    ax_a = axes3[0]
    ax_a.plot(phet_full["elapsed_hours"], phet_full["wind_speed"], color="#0F172A", linewidth=2.8, marker="o", markersize=5, label="Actual Vmax (Truth)")
    ax_a.set_title("(A) True Real-World Intensity Evolution", fontsize=13, fontweight="bold")
    ax_a.set_xlabel("Elapsed Time (Hours)", fontsize=10.5, fontweight="bold")
    ax_a.set_ylabel("Intensity (knots)", fontsize=10.5, fontweight="bold")
    ax_a.set_ylim(15, 145)
    ax_a.grid(True, linestyle="--", alpha=0.6)
    ax_a.legend(loc="upper left", frameon=True)

    # Panel B: ML Predictions vs True Ground Truth
    ax_b = axes3[1]
    ax_b.plot(phet_full["elapsed_hours"], phet_full["wind_speed"], color="#0F172A", linewidth=2.0, linestyle="--", label="Actual Vmax (Truth Reference)", alpha=0.6)
    ax_b.plot(t_origin, phet_preds["transformer_pred_plus_6h"], color="#2563EB", linewidth=2.0, marker="^", markersize=4, label="ML +6h Forecast")
    ax_b.plot(t_origin, phet_preds["transformer_pred_plus_12h"], color="#7C3AED", linewidth=2.0, marker="s", markersize=4, label="ML +12h Forecast")
    ax_b.plot(t_origin, phet_preds["transformer_pred_plus_24h"], color="#DC2626", linewidth=2.2, marker="D", markersize=5, label="ML +24h Forecast")
    ax_b.set_title("(B) ML Predicted Intensity Course (CNN + Temporal Transformer)", fontsize=13, fontweight="bold")
    ax_b.set_xlabel("Forecast Origin Time (Hours)", fontsize=10.5, fontweight="bold")
    ax_b.set_ylabel("Predicted Intensity (knots)", fontsize=10.5, fontweight="bold")
    ax_b.set_ylim(15, 145)
    ax_b.grid(True, linestyle="--", alpha=0.6)
    ax_b.legend(loc="upper right", frameon=True)

    plt.suptitle("Super Cyclone Phet (201003I, Arabian Sea) — True vs ML Predicted Intensity", fontsize=15, fontweight="bold", y=0.98)
    g3_path = out_dir / "phet_true_vs_predicted_comparison.png"
    plt.tight_layout()
    plt.savefig(g3_path)
    plt.close()
    print(f"[Generated Graph 3: Side-by-Side Comparison] -> {g3_path}")

    # Copy to artifact directory for markdown display
    art_dir = Path("/home/raymondj/.gemini/antigravity-ide/brain/912d16dc-2348-490b-a9b9-056fd0e8d85e/figures")
    art_dir.mkdir(parents=True, exist_ok=True)
    for p in [g1_path, g2_path, g3_path]:
        import shutil
        shutil.copy2(p, art_dir / p.name)


if __name__ == "__main__":
    generate_phet_graphs()
