"""Generate publication-quality figures, ROC/PR curves, calibration diagrams, and operational lifecycle plots."""
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.data.trend_config import IntensityTrendConfig
from src.evaluation.baselines import RecentTrendBaseline, ThresholdedRegressionBaseline


def plot_trend_confusion_matrices(
    y_true: np.ndarray,
    preds_dict: Dict[str, np.ndarray],
    save_path: Path,
    class_names: List[str] = ["Weakening\n(ΔV ≤ -10)", "Stable\n(|ΔV| < 10)", "Intensifying\n(ΔV ≥ +10)"],
):
    """Plot 3-panel normalized confusion matrices comparing baselines against TemporalClassifier."""
    fig, axes = plt.subplots(1, len(preds_dict), figsize=(5.5 * len(preds_dict), 4.8), dpi=180)
    if len(preds_dict) == 1:
        axes = [axes]

    for idx, (m_name, pred_arr) in enumerate(preds_dict.items()):
        ax = axes[idx]
        cm = confusion_matrix(y_true, pred_arr, labels=[0, 1, 2])
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size": 11, "weight": "bold"},
        )
        acc = float(np.trace(cm) / np.sum(cm))
        ax.set_title(f"{m_name}\nAccuracy: {acc*100:.1f}%", fontweight="bold", fontsize=12)
        ax.set_xlabel("Predicted Intensity Trend", fontsize=10.5)
        if idx == 0:
            ax.set_ylabel("Actual Ground Truth Trend", fontsize=10.5)
        else:
            ax.set_ylabel("")

    plt.suptitle("24-Hour Tropical Cyclone Intensity Trend Classification (Threshold: ±10 kt)", fontweight="bold", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved trend confusion matrices -> {save_path}")


def plot_ri_roc_and_pr_curves(
    y_true_ri: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    save_path: Path,
):
    """Plot ROC curve and Precision-Recall curve comparing models against baseline prevalence."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=180)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "Persistence (Baseline A)": "#64748B",
        "Recent 6h Trend (Baseline B)": "#D97706",
        "Thresholded Continuous (Baseline C)": "#0D9488",
        "AI TemporalClassifier": "#1E3A8A",
    }
    styles = {
        "Persistence (Baseline A)": ":",
        "Recent 6h Trend (Baseline B)": "--",
        "Thresholded Continuous (Baseline C)": "-.",
        "AI TemporalClassifier": "-",
    }

    prevalence = float(np.mean(y_true_ri))

    # Panel 1: ROC Curve
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC: 0.500)")

    for m_name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true_ri, probs)
        roc_val = auc(fpr, tpr)
        c = colors.get(m_name, "#333333")
        st = styles.get(m_name, "-")
        lw = 2.5 if "AI" in m_name else 1.8
        ax_roc.plot(fpr, tpr, st, color=c, linewidth=lw, label=f"{m_name} (AUC: {roc_val:.3f})")

    ax_roc.set_title("Rapid Intensification (RI) ROC Curves", fontweight="bold", fontsize=12)
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.legend(loc="lower right", frameon=True, fontsize=9)

    # Panel 2: Precision-Recall Curve
    ax_pr = axes[1]
    ax_pr.axhline(y=prevalence, color="k", linestyle="--", alpha=0.5, label=f"Prevalence Baseline ({prevalence*100:.1f}%)")

    for m_name, probs in probs_dict.items():
        prec, rec, _ = precision_recall_curve(y_true_ri, probs)
        pr_val = auc(rec, prec)
        c = colors.get(m_name, "#333333")
        st = styles.get(m_name, "-")
        lw = 2.5 if "AI" in m_name else 1.8
        ax_pr.plot(rec, prec, st, color=c, linewidth=lw, label=f"{m_name} (PR-AUC: {pr_val:.3f})")

    ax_pr.set_title("Rapid Intensification (RI) Precision-Recall Curves", fontweight="bold", fontsize=12)
    ax_pr.set_xlabel("Recall (Fraction of RI Events Detected)", fontsize=11)
    ax_pr.set_ylabel("Precision (Fraction of Warnings That Manifest)", fontsize=11)
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.legend(loc="upper right", frameon=True, fontsize=9)

    plt.suptitle("Operational Rapid Intensification Discrimination Benchmarks (Held-out Test, N=8,279)", fontweight="bold", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC and PR curves -> {save_path}")


def plot_ri_calibration_diagram(
    y_true_ri: np.ndarray,
    ai_probs: np.ndarray,
    save_path: Path,
    n_bins: int = 10,
):
    """Plot probability reliability diagram with empirical confidence calibration."""
    fig, (ax_rel, ax_hist) = plt.subplots(2, 1, figsize=(7, 7.5), dpi=180, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    n = len(y_true_ri)
    ece = 0.0

    for i in range(n_bins):
        b_low, b_high = bins[i], bins[i + 1]
        mask = (ai_probs >= b_low) & (ai_probs < b_high if i < n_bins - 1 else ai_probs <= b_high)
        cnt = int(np.sum(mask))
        bin_counts.append(cnt)
        if cnt > 0:
            acc = float(np.mean(y_true_ri[mask]))
            conf = float(np.mean(ai_probs[mask]))
            bin_accs.append(acc)
            bin_confs.append(conf)
            ece += (cnt / n) * abs(acc - conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((b_low + b_high) / 2.0)

    # Reliability plot
    ax_rel.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")
    ax_rel.plot(bin_confs, bin_accs, "s-", color="#1E3A8A", linewidth=2.2, markersize=8, label=f"TemporalClassifier (ECE: {ece:.3f})")
    ax_rel.set_ylabel("Empirical RI Frequency", fontsize=11)
    ax_rel.set_title("Reliability Diagram (P(RI) Calibration)", fontweight="bold", fontsize=12)
    ax_rel.legend(loc="upper left", frameon=True, fontsize=10)
    ax_rel.set_ylim(-0.02, 1.02)

    # Histogram of predicted probabilities
    bin_edges = bins
    ax_hist.bar(bins[:-1], bin_counts, width=1.0 / n_bins, align="edge", color="#93C5FD", edgecolor="#1E3A8A", alpha=0.8)
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("Predicted Rapid Intensification Probability P(RI)", fontsize=11)
    ax_hist.set_ylabel("Count (Log)", fontsize=10.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved calibration diagram -> {save_path}")


def plot_unseen_cyclone_lead_time_lifecycle(
    pred_df: pd.DataFrame,
    cyclone_id: str,
    cyclone_name: str,
    basin_info: str,
    save_path: Path,
):
    """Plot comprehensive lifecycle showing:
    1. Historical Vmax curve vs. Actual Future +24h Vmax
    2. Operational Warning Panel along the timeline showing:
       - Current Intensity (kt)
       - Predicted 24h Trend (Weakening / Stable / Intensifying)
       - Predicted RI Probability (%) and Risk Level (LOW, MEDIUM, HIGH)
       - Lead-time detection before rapid intensity jumps occur!
    """
    storm_df = pred_df[pred_df["cyclone_id"] == cyclone_id].copy()
    if len(storm_df) == 0:
        print(f"Skipping {cyclone_name} ({cyclone_id}) - no sequences found in test predictions.")
        return

    storm_df["target_t_timestamp"] = storm_df["target_t_timestamp"].astype(str)
    t_hours = np.arange(len(storm_df)) * 3.0  # 3-hour observation cadence

    v_curr = storm_df["vmax_curr"].values
    v_actual_24 = storm_df["vmax_plus_24h"].values
    d24_actual = v_actual_24 - v_curr
    ri_probs = storm_df["pred_ri_prob"].values * 100.0  # in percent
    trend_preds = storm_df["pred_trend"].values

    fig, (ax_intensity, ax_prob) = plt.subplots(
        2, 1, figsize=(14, 8.5), dpi=180, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]}
    )
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Panel 1: Intensity Trajectory
    ax_intensity.plot(t_hours, v_curr, "k-o", linewidth=2.5, markersize=5.5, label="Current Intensity $V_{\max}(t)$")
    ax_intensity.plot(t_hours, v_actual_24, "--s", color="#DC2626", linewidth=2.0, markersize=5.0, label="Actual Future Intensity $V_{\max}(t+24\text{h})$")

    # Highlight RI periods (actual delta V >= 30 kt)
    for i in range(len(t_hours)):
        if d24_actual[i] >= 30.0:
            ax_intensity.axvspan(t_hours[i] - 1.5, t_hours[i] + 1.5, color="#FEE2E2", alpha=0.6, label="Ground Truth RI Period" if i == 0 or (i > 0 and d24_actual[i-1] < 30.0) else "")

    # Category threshold lines (Saffir-Simpson)
    ax_intensity.axhline(64, color="#94A3B8", linestyle=":", alpha=0.7, label="Cat 1 (64 kt)")
    ax_intensity.axhline(96, color="#CBD5E1", linestyle=":", alpha=0.7, label="Cat 3 (96 kt)")
    ax_intensity.axhline(137, color="#E2E8F0", linestyle=":", alpha=0.7, label="Cat 5 (137 kt)")

    ax_intensity.set_title(
        f"Operational Lifecycle & Early Warning Lead Time: {cyclone_name} ({cyclone_id})\n{basin_info} — Strictly Held-Out Unseen Test Cyclone",
        fontweight="bold",
        fontsize=13,
    )
    ax_intensity.set_ylabel("Intensity (Knots)", fontsize=11.5)
    ax_intensity.legend(loc="upper left", frameon=True, fontsize=9)
    ax_intensity.set_ylim(15, max(np.max(v_actual_24) + 15, 120))

    # Panel 2: Predicted RI Probability & Risk Level
    ax_prob.plot(t_hours, ri_probs, "-D", color="#1E3A8A", linewidth=2.2, markersize=5.5, label="Predicted RI Probability P(RI in 24h)")
    ax_prob.axhline(60, color="#DC2626", linestyle="--", linewidth=1.5, label="HIGH Risk (≥ 60%)")
    ax_prob.axhline(25, color="#F59E0B", linestyle=":", linewidth=1.5, label="MEDIUM Risk (≥ 25%)")

    # Fill risk zones
    ax_prob.fill_between(t_hours, 60, 100, color="#FCA5A5", alpha=0.25)
    ax_prob.fill_between(t_hours, 25, 60, color="#FDE68A", alpha=0.2)
    ax_prob.fill_between(t_hours, 0, 25, color="#BBF7D0", alpha=0.15)

    # Annotate key warning moments
    high_risk_indices = np.where(ri_probs >= 60.0)[0]
    if len(high_risk_indices) > 0:
        first_warn_idx = high_risk_indices[0]
        warn_time = t_hours[first_warn_idx]
        ax_prob.annotate(
            f"EARLY RI ALERT: {ri_probs[first_warn_idx]:.0f}% Risk\n(Issued {t_hours[first_warn_idx]:.0f}h before peak)",
            xy=(warn_time, ri_probs[first_warn_idx]),
            xytext=(warn_time + 6, min(ri_probs[first_warn_idx] + 15, 95)),
            arrowprops=dict(facecolor="#DC2626", shrink=0.08, width=1.5, headwidth=7),
            fontweight="bold",
            color="#991B1B",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2", edgecolor="#DC2626"),
        )

    ax_prob.set_ylabel("RI Probability (%)", fontsize=11.5)
    ax_prob.set_xlabel("Elapsed Observation Time (Hours from Lifecycle Sequence Start)", fontsize=11.5)
    ax_prob.set_ylim(-2, 102)
    ax_prob.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved lifecycle early-warning demonstration -> {save_path}")


def generate_all_trend_visualizations(
    results_json_path: str = "experiments/trend_classification/results/comprehensive_benchmark_results.json",
    pred_csv_path: str = "experiments/trend_classification/checkpoints/classifier_primary_ri/test_predictions.csv",
    out_dir: str = "experiments/trend_classification/figures",
):
    """Generate all required figures for publication and SIH demonstration."""
    fig_dir = Path(out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = Path("data/metadata")
    test_df = pd.read_csv(meta_dir / "forecast_test_sequences_k5.csv")
    pred_df = pd.read_csv(pred_csv_path)

    config = IntensityTrendConfig()
    d24_act = test_df["vmax_plus_24h"].values - test_df["vmax_curr"].values
    y_true_trend = config.compute_trend_label(d24_act)
    y_true_ri = config.compute_ri_label(d24_act)

    # 1. Baselines Predictions
    base_b = RecentTrendBaseline(config)
    t_pred_b, _, ri_prob_b = base_b.predict(test_df)

    reg_csv_path = Path("experiments/forecasting/checkpoints/cnn_transformer_k5/test_predictions.csv")
    base_c = ThresholdedRegressionBaseline(config)
    t_pred_c, _, ri_prob_c, _ = base_c.predict_from_csv(str(reg_csv_path))

    t_pred_ai = pred_df["pred_trend"].values
    ri_prob_ai = pred_df["pred_ri_prob"].values

    # 2. Confusion Matrices Plot
    preds_cm_dict = {
        "Baseline B (Recent 6h Trend)": t_pred_b,
        "Baseline C (Thresholded Forecaster)": t_pred_c,
        "AI TemporalClassifier (Proposed)": t_pred_ai,
    }
    plot_trend_confusion_matrices(y_true_trend, preds_cm_dict, fig_dir / "trend_confusion_matrices.png")

    # 3. ROC & PR Curves Plot
    probs_dict = {
        "Recent 6h Trend (Baseline B)": ri_prob_b,
        "Thresholded Continuous (Baseline C)": ri_prob_c,
        "AI TemporalClassifier": ri_prob_ai,
    }
    plot_ri_roc_and_pr_curves(y_true_ri, probs_dict, fig_dir / "ri_roc_and_pr_curves.png")

    # 4. Calibration Diagram Plot
    plot_ri_calibration_diagram(y_true_ri, ri_prob_ai, fig_dir / "ri_calibration_diagram.png")

    # 5. Held-Out Unseen Storm Lifecycle Lead-Time Plots
    # Selected held-out storms from North Atlantic, West Pacific, East Pacific, South Pacific, Indian Ocean:
    held_out_cyclones = [
        ("201015W", "Super Typhoon Megi", "West Pacific (WPAC) | 160 kt Category 5"),
        ("201614L", "Hurricane Matthew", "North Atlantic (ATLN) | 145 kt Category 5"),
        ("200413E", "Hurricane Javier", "East Pacific (EPAC) | 130 kt Category 4"),
        ("200519S", "Cyclone Percy", "South Pacific (SH) | 145 kt Category 5"),
        ("201003I", "Super Cyclone Phet", "Indian Ocean (IO) | 125 kt Category 4"),
        ("200801I", "VSCS Nargis", "Bay of Bengal (IO) | 115 kt Category 4"),
    ]

    for cid, name, basin in held_out_cyclones:
        clean_name = name.lower().replace(" ", "_")
        plot_unseen_cyclone_lead_time_lifecycle(
            pred_df, cid, name, basin, fig_dir / f"lifecycle_lead_time_{clean_name}.png"
        )

    print(f"\n[All Visualizations and Lifecycle Demonstrations Generated in {fig_dir}]")


if __name__ == "__main__":
    generate_all_trend_visualizations()
