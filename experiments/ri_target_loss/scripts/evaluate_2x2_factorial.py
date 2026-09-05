"""Unified Evaluation and Factorial Synthesis for 2x2 Controlled Experiment.

Conditions:
  - Model A: Ultra Control (No dynamics, Huber loss, 1/6/12 weights, lambda_reg=0.1)
  - Model B: + Dynamics (15-d env with [dv6, dv12, acc], Huber loss, 1/6/12 weights, lambda_reg=0.1)
  - Model C: Power-1.5 Loss (12-d env, unweighted Power-1.5 loss, lambda_reg=0.5)
  - Model D: Both Dynamics + Power-1.5 Loss (15-d env, unweighted Power-1.5 loss, lambda_reg=0.5)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/home/raymondj/Projects/cycml")
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "ri_target_loss" / "results"
PLOTS_ROOT = PROJECT_ROOT / "experiments" / "ri_target_loss" / "plots"
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

CONDITIONS = {
    "A (Control: Huber)": {
        "test_csv": RESULTS_ROOT / "exp2_delta_1_6_12" / "test_predictions.csv",
        "train_ext_csv": PROJECT_ROOT / "experiments" / "ri_stress_test" / "results" / "phase10_train_extreme_fits.csv",
        "color": "#4A90E2",
    },
    "B (+ Dynamics: Huber)": {
        "test_csv": RESULTS_ROOT / "exp_2x2_B_dynamics" / "test_predictions.csv",
        "train_ext_csv": RESULTS_ROOT / "exp_2x2_B_dynamics" / "train_extremes_predictions.csv",
        "color": "#50E3C2",
    },
    "C (Power-1.5: No Dyn)": {
        "test_csv": RESULTS_ROOT / "exp_2x2_C_power_loss" / "test_predictions.csv",
        "train_ext_csv": RESULTS_ROOT / "exp_2x2_C_power_loss" / "train_extremes_predictions.csv",
        "color": "#F5A623",
    },
    "D (Both: Dyn + Power-1.5)": {
        "test_csv": RESULTS_ROOT / "exp_2x2_D_both" / "test_predictions.csv",
        "train_ext_csv": RESULTS_ROOT / "exp_2x2_D_both" / "train_extremes_predictions.csv",
        "color": "#D0021B",
    },
}

def compute_calibration_ece(probs, targets, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def run_evaluation():
    print("="*80)
    print("RUNNING UNIFIED 2x2 FACTORIAL EVALUATION")
    print("="*80)

    holistic_records = []
    directional_records = []
    data_cache = {}

    for name, meta in CONDITIONS.items():
        if not meta["test_csv"].exists():
            print(f"Warning: {meta['test_csv']} does not exist yet! Skipping {name}.")
            continue

        df_test = pd.read_csv(meta["test_csv"])
        act_dv24 = (df_test["vmax_plus_24h"] - df_test["vmax_curr"]).values
        pred_dv24 = df_test["pred_delta_24h"].values
        ri_targets = (act_dv24 >= 30.0).astype(int)
        ri_probs = df_test["pred_ri_prob"].values

        # Load train extremes if available
        train_ext_slope = np.nan
        train_ext_max = np.nan
        train_ext_bias = np.nan
        train_ext_mae = np.nan
        df_tr_ext = None

        if meta["train_ext_csv"].exists():
            df_tr_ext = pd.read_csv(meta["train_ext_csv"])
            if "pred_delta_24h" in df_tr_ext.columns:
                tr_pred = df_tr_ext["pred_delta_24h"].values
                tr_act = (df_tr_ext["vmax_plus_24h"] - df_tr_ext["vmax_curr"]).values
            else:
                tr_pred = df_tr_ext["pred_dv24"].values
                tr_act = df_tr_ext["actual_dv24"].values
            
            slope_tr, _, _, _, _ = stats.linregress(tr_act, tr_pred)
            train_ext_slope = float(slope_tr)
            train_ext_max = float(np.max(tr_pred))
            train_ext_bias = float(np.mean(tr_pred - tr_act))
            train_ext_mae = float(np.mean(np.abs(tr_pred - tr_act)))

        data_cache[name] = {
            "df_test": df_test,
            "df_tr_ext": df_tr_ext,
            "act_dv24": act_dv24,
            "pred_dv24": pred_dv24,
            "ri_probs": ri_probs,
            "ri_targets": ri_targets,
        }

        # 1. Overall Test Metrics
        mae_24 = float(np.mean(np.abs(pred_dv24 - act_dv24)))
        p_act_6 = (df_test["vmax_plus_6h"] - df_test["vmax_curr"]).values if "vmax_plus_6h" in df_test else None
        p_act_12 = (df_test["vmax_plus_12h"] - df_test["vmax_curr"]).values if "vmax_plus_12h" in df_test else None
        mae_6 = float(np.mean(np.abs(df_test["pred_delta_6h"].values - p_act_6))) if p_act_6 is not None and "pred_delta_6h" in df_test else np.nan
        mae_12 = float(np.mean(np.abs(df_test["pred_delta_12h"].values - p_act_12))) if p_act_12 is not None and "pred_delta_12h" in df_test else np.nan

        slope_overall, _, r_val, _, _ = stats.linregress(act_dv24, pred_dv24)
        corr_overall = float(r_val)

        # 2. Test Tail Metrics (>= 30 kt)
        m_ri = act_dv24 >= 30.0
        tail_slope, _, r_tail, _, _ = stats.linregress(act_dv24[m_ri], pred_dv24[m_ri])
        tail_corr = float(r_tail)
        tail_bias = float(np.mean(pred_dv24[m_ri] - act_dv24[m_ri]))
        tail_mae = float(np.mean(np.abs(pred_dv24[m_ri] - act_dv24[m_ri])))
        max_pred_test = float(np.max(pred_dv24))

        # 3. Test Extreme Tail Metrics (>= 45 kt)
        m_ext = act_dv24 >= 45.0
        ext_slope, _, _, _, _ = stats.linregress(act_dv24[m_ext], pred_dv24[m_ext])
        ext_bias = float(np.mean(pred_dv24[m_ext] - act_dv24[m_ext]))
        ext_mae = float(np.mean(np.abs(pred_dv24[m_ext] - act_dv24[m_ext])))

        # 4. Classification & Calibration
        prec, rec, thresh = precision_recall_curve(ri_targets, ri_probs)
        pr_auc = float(auc(rec, prec))
        roc_auc = float(roc_auc_score(ri_targets, ri_probs))
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_idx = np.argmax(f1_scores)
        opt_thresh = float(thresh[min(best_idx, len(thresh) - 1)])
        bin_pred = (ri_probs >= opt_thresh).astype(int)
        opt_f1 = float(f1_score(ri_targets, bin_pred))
        opt_prec = float(precision_score(ri_targets, bin_pred, zero_division=0))
        opt_rec = float(recall_score(ri_targets, bin_pred, zero_division=0))

        brier = float(brier_score_loss(ri_targets, ri_probs))
        ece = compute_calibration_ece(ri_probs, ri_targets)

        # 5. Severe Reversals (actual <= -20 kt, pred >= +10 kt)
        severe_rev_count = int(((act_dv24 <= -20.0) & (pred_dv24 >= 10.0)).sum())
        severe_rev_rate = float(severe_rev_count / max(1, (act_dv24 <= -20.0).sum()))

        holistic_records.append({
            "Condition": name,
            "MAE +6h": mae_6,
            "MAE +12h": mae_12,
            "MAE +24h": mae_24,
            "Overall Slope": float(slope_overall),
            "Overall Corr": corr_overall,
            "Test Max Pred (kt)": max_pred_test,
            "Tail (>=30) Slope": float(tail_slope),
            "Tail (>=30) Corr": tail_corr,
            "Tail (>=30) Bias": tail_bias,
            "Tail (>=30) MAE": tail_mae,
            "Ext (>=45) Slope": float(ext_slope),
            "Ext (>=45) Bias": ext_bias,
            "Ext (>=45) MAE": ext_mae,
            "Train Ext Slope (>=45)": train_ext_slope,
            "Train Ext Max Pred (kt)": train_ext_max,
            "Train Ext Bias": train_ext_bias,
            "Train Ext MAE": train_ext_mae,
            "RI PR-AUC": pr_auc,
            "RI ROC-AUC": roc_auc,
            "RI Opt F1": opt_f1,
            "RI Opt Recall": opt_rec,
            "RI Opt Precision": opt_prec,
            "Brier Score": brier,
            "ECE": ece,
            "Severe Reversals (N)": severe_rev_count,
            "Severe Reversals (%)": severe_rev_rate * 100.0,
        })

        # Directional Statistics
        m_pos = act_dv24 > 0
        m_neg = act_dv24 < 0

        pos_slope, _, _, _, _ = stats.linregress(act_dv24[m_pos], pred_dv24[m_pos])
        neg_slope, _, _, _, _ = stats.linregress(act_dv24[m_neg], pred_dv24[m_neg])

        directional_records.append({
            "Condition": name,
            "Pos Mean Actual": float(np.mean(act_dv24[m_pos])),
            "Pos Mean Pred": float(np.mean(pred_dv24[m_pos])),
            "Pos MAE": float(np.mean(np.abs(pred_dv24[m_pos] - act_dv24[m_pos]))),
            "Pos Bias": float(np.mean(pred_dv24[m_pos] - act_dv24[m_pos])),
            "Pos Slope": float(pos_slope),
            "Neg Mean Actual": float(np.mean(act_dv24[m_neg])),
            "Neg Mean Pred": float(np.mean(pred_dv24[m_neg])),
            "Neg MAE": float(np.mean(np.abs(pred_dv24[m_neg] - act_dv24[m_neg]))),
            "Neg Bias": float(np.mean(pred_dv24[m_neg] - act_dv24[m_neg])),
            "Neg Slope": float(neg_slope),
            "Asymmetry Ratio (Pos/Neg)": float(pos_slope / neg_slope) if neg_slope != 0 else np.nan,
        })

    df_holistic = pd.DataFrame(holistic_records)
    df_directional = pd.DataFrame(directional_records)

    df_holistic.to_csv(RESULTS_ROOT / "factorial_2x2_holistic_metrics.csv", index=False)
    df_directional.to_csv(RESULTS_ROOT / "factorial_2x2_directional_asymmetry.csv", index=False)

    print("\n" + "="*80)
    print("HOLISTIC METRICS SUMMARY TABLE:")
    print("="*80)
    print(df_holistic[["Condition", "MAE +24h", "Overall Slope", "Test Max Pred (kt)", "Tail (>=30) Slope", "Tail (>=30) Bias", "Train Ext Slope (>=45)", "Train Ext Max Pred (kt)", "RI PR-AUC", "Severe Reversals (%)"]].to_string(index=False))

    print("\n" + "="*80)
    print("DIRECTIONAL ASYMMETRY SUMMARY TABLE:")
    print("="*80)
    print(df_directional[["Condition", "Pos Slope", "Pos Bias", "Neg Slope", "Neg Bias", "Asymmetry Ratio (Pos/Neg)"]].to_string(index=False))

    # Factorial Main Effects & Interaction Calculation (if all 4 present)
    if len(df_holistic) == 4:
        print("\n" + "="*80)
        print("FACTORIAL EFFECTS ESTIMATION (2x2 DESIGN):")
        print("="*80)
        # Rows: 0=A, 1=B, 2=C, 3=D
        a_row = df_holistic.iloc[0]
        b_row = df_holistic.iloc[1]
        c_row = df_holistic.iloc[2]
        d_row = df_holistic.iloc[3]

        effects = []
        for metric in ["MAE +24h", "Overall Slope", "Test Max Pred (kt)", "Tail (>=30) Slope", "Tail (>=30) Bias", "Train Ext Slope (>=45)", "Train Ext Max Pred (kt)", "RI PR-AUC"]:
            va = a_row[metric]
            vb = b_row[metric]
            vc = c_row[metric]
            vd = d_row[metric]

            eff_dyn = vb - va
            eff_loss = vc - va
            eff_comb = vd - va
            interaction = (vd - vb) - (vc - va)

            effects.append({
                "Metric": metric,
                "Model A (Control)": va,
                "Model B (+Dyn)": vb,
                "Model C (PowerLoss)": vc,
                "Model D (Both)": vd,
                "Dynamics Effect (B - A)": eff_dyn,
                "Loss Effect (C - A)": eff_loss,
                "Combined (D - A)": eff_comb,
                "Interaction ((D-B)-(C-A))": interaction,
            })

        df_effects = pd.DataFrame(effects)
        df_effects.to_csv(RESULTS_ROOT / "factorial_2x2_effects.csv", index=False)
        print(df_effects.to_string(index=False))

    # -------------------------------------------------------------------------
    # GENERATE PUBLICATION-GRADE PLOTS
    # -------------------------------------------------------------------------
    print("\nGenerating comparative figures...")

    # Plot 1: 4-Panel Scatter of Predicted vs Actual dV24 on Test Set
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (name, meta) in enumerate(CONDITIONS.items()):
        if name not in data_cache:
            continue
        ax = axes[idx]
        act = data_cache[name]["act_dv24"]
        pred = data_cache[name]["pred_dv24"]
        color = meta["color"]

        # Background scatter
        ax.scatter(act, pred, alpha=0.18, color=color, s=12, edgecolors="none", label="Test Sequences (N=7,901)")

        # Identity line
        ax.plot([-70, 90], [-70, 90], "k--", alpha=0.5, label="1:1 Perfect Line")

        # Full regression line
        slope_full, intercept_full, _, _, _ = stats.linregress(act, pred)
        x_grid = np.linspace(-65, 85, 100)
        ax.plot(x_grid, slope_full * x_grid + intercept_full, color="#2C3E50", lw=2.2, label=f"Overall Slope: {slope_full:.3f}")

        # Tail regression line (>= 30 kt)
        m_tail = act >= 30.0
        slope_t, int_t, _, _, _ = stats.linregress(act[m_tail], pred[m_tail])
        x_tail = np.linspace(30, 85, 50)
        ax.plot(x_tail, slope_t * x_tail + int_t, color="#D9534F", lw=2.5, ls="-", label=f"Tail Slope (≥30): {slope_t:.3f}")

        # Highlight maximum prediction
        max_p = np.max(pred)
        ax.axhline(max_p, color="#E74C3C", ls=":", alpha=0.85, label=f"Max Pred: {max_p:.1f} kt")

        ax.set_title(name, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlim(-70, 90)
        ax.set_ylim(-70, 90)
        ax.set_xlabel("Actual ΔV (+24h) [kt]", fontsize=12)
        ax.set_ylabel("Predicted ΔV (+24h) [kt]", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    plt.suptitle("2x2 Factorial Controlled Experiment: Predicted vs Actual ΔV24 Calibration", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(PLOTS_ROOT / "2x2_factorial_predicted_vs_actual_scatter.png", dpi=300)
    plt.close()
    print(f"Saved {PLOTS_ROOT / '2x2_factorial_predicted_vs_actual_scatter.png'}")

    # Plot 2: Directional Asymmetry Curve Across Models
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bins = [-60, -30, -15, 0, 15, 30, 45, 90]
    bin_labels = ["<-30", "[-30,-15)", "[-15,0)", "[0,15)", "[15,30)", "[30,45)", "≥45"]

    for name, meta in CONDITIONS.items():
        if name not in data_cache:
            continue
        act = data_cache[name]["act_dv24"]
        pred = data_cache[name]["pred_dv24"]
        color = meta["color"]

        bin_means_pred = []
        bin_biases = []
        for i in range(len(bins) - 1):
            m = (act >= bins[i]) & (act < bins[i+1])
            if m.sum() > 0:
                bin_means_pred.append(np.mean(pred[m]))
                bin_biases.append(np.mean(pred[m] - act[m]))
            else:
                bin_means_pred.append(np.nan)
                bin_biases.append(np.nan)

        ax1.plot(range(len(bin_labels)), bin_means_pred, marker="o", lw=2, color=color, label=name)
        ax2.plot(range(len(bin_labels)), bin_biases, marker="s", lw=2, color=color, label=name)

    # Reference on ax1: bin center actuals
    bin_centers = [-45, -22.5, -7.5, 7.5, 22.5, 37.5, 55]
    ax1.plot(range(len(bin_labels)), bin_centers, "k--", alpha=0.5, label="Ground Truth Ideal")
    ax1.set_xticks(range(len(bin_labels)))
    ax1.set_xticklabels(bin_labels, rotation=30)
    ax1.set_ylabel("Mean Predicted ΔV (+24h) [kt]", fontsize=12)
    ax1.set_title("Conditional Expectation E[Pred | Actual Bin]", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=10)

    ax2.axhline(0, color="k", ls="--", alpha=0.5)
    ax2.set_xticks(range(len(bin_labels)))
    ax2.set_xticklabels(bin_labels, rotation=30)
    ax2.set_ylabel("Mean Bias (Pred - Actual) [kt]", fontsize=12)
    ax2.set_title("Stratified Forecast Bias Across Intensity Regime", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=10)

    plt.suptitle("Directional Asymmetry: Strengthening vs Weakening Response Across 4 Conditions", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_ROOT / "2x2_directional_asymmetry_curves.png", dpi=300)
    plt.close()
    print(f"Saved {PLOTS_ROOT / '2x2_directional_asymmetry_curves.png'}")

    # Plot 3: Training Extreme Tail Comparison (N=738)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, meta in CONDITIONS.items():
        if name not in data_cache or data_cache[name]["df_tr_ext"] is None:
            continue
        df_tr = data_cache[name]["df_tr_ext"]
        if "pred_delta_24h" in df_tr.columns:
            tr_pred = df_tr["pred_delta_24h"].values
            tr_act = (df_tr["vmax_plus_24h"] - df_tr["vmax_curr"]).values
        else:
            tr_pred = df_tr["pred_dv24"].values
            tr_act = df_tr["actual_dv24"].values
        color = meta["color"]
        slope_tr, intercept_tr, _, _, _ = stats.linregress(tr_act, tr_pred)
        x_tr = np.linspace(45, 105, 100)
        ax.scatter(tr_act, tr_pred, alpha=0.3, color=color, s=16)
        ax.plot(x_tr, slope_tr * x_tr + intercept_tr, color=color, lw=2.5, label=f"{name} (Slope: {slope_tr:.3f}, Max: {np.max(tr_pred):.1f} kt)")

    ax.plot([45, 105], [45, 105], "k--", alpha=0.5, label="1:1 Perfect Ceiling-Free Line")
    ax.set_title("Training Set Extreme Tail Fit (Actual ΔV ≥ +45 kt, N=738 Sequences)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual Ground Truth ΔV (+24h) [kt]", fontsize=12)
    ax.set_ylabel("Predicted ΔV (+24h) [kt]", fontsize=12)
    ax.set_xlim(42, 110)
    ax.set_ylim(20, 110)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(PLOTS_ROOT / "2x2_train_extreme_tail_comparison.png", dpi=300)
    plt.close()
    print(f"Saved {PLOTS_ROOT / '2x2_train_extreme_tail_comparison.png'}")

    print("\nEvaluation and plot generation successfully finished!")

if __name__ == "__main__":
    run_evaluation()
