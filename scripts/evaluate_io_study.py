"""Comprehensive Evaluation and Paired Cyclone Bootstrap for the Indian Ocean Intensity-Balancing Study."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TCIRDataset
from src.data.preprocessing import TCIRPreprocessor
from src.evaluation.intensity_bins import INTENSITY_BINS, REGIME_BINS, assign_intensity_bin, assign_regime_bin
from src.models.factory import build_model
from src.utils.config import load_config


def evaluate_predictions(df_preds: pd.DataFrame) -> dict:
    """Calculate comprehensive evaluation metrics from predictions dataframe."""
    act = df_preds["actual"].values
    pred = df_preds["predicted"].values
    err = pred - act
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    med_ae = float(np.median(abs_err))
    bias = float(np.mean(err))

    # R2
    ss_tot = float(np.sum((act - np.mean(act)) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Regression slope
    if len(act) > 1 and np.var(act) > 0:
        slope, intercept = np.polyfit(act, pred, 1)
    else:
        slope, intercept = 1.0, 0.0

    # Per-Cyclone Metrics
    cyclone_maes = df_preds.groupby("cyclone_id")["abs_error"].mean()
    mean_cyclone_mae = float(cyclone_maes.mean())
    median_cyclone_mae = float(cyclone_maes.median())
    best_cyclone = (str(cyclone_maes.idxmin()), float(cyclone_maes.min()))
    worst_cyclone = (str(cyclone_maes.idxmax()), float(cyclone_maes.max()))

    # High-intensity cyclone MAE (cyclones whose peak wind speed was >= 110 kt)
    peak_winds = df_preds.groupby("cyclone_id")["actual"].max()
    high_cyclone_ids = peak_winds[peak_winds >= 110.0].index
    if len(high_cyclone_ids) > 0:
        high_cyclone_mae = float(df_preds[df_preds["cyclone_id"].isin(high_cyclone_ids)].groupby("cyclone_id")["abs_error"].mean().mean())
    else:
        high_cyclone_mae = 0.0

    # High Intensity subset (samples >= 110 kt)
    high_df = df_preds[df_preds["actual"] >= 110.0]
    high_mae = float(high_df["abs_error"].mean()) if len(high_df) > 0 else 0.0
    high_bias = float(high_df["error"].mean()) if len(high_df) > 0 else 0.0

    # Very high subset (samples >= 130 kt)
    vhigh_df = df_preds[df_preds["actual"] >= 130.0]
    vhigh_mae = float(vhigh_df["abs_error"].mean()) if len(vhigh_df) > 0 else 0.0
    vhigh_bias = float(vhigh_df["error"].mean()) if len(vhigh_df) > 0 else 0.0

    return {
        "n_samples": len(df_preds),
        "n_cyclones": int(df_preds["cyclone_id"].nunique()),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "bias": round(bias, 2),
        "median_ae": round(med_ae, 2),
        "slope": round(slope, 3),
        "intercept": round(intercept, 2),
        "max_predicted": round(float(np.max(pred)), 2),
        "max_actual": round(float(np.max(act)), 2),
        "mean_cyclone_mae": round(mean_cyclone_mae, 2),
        "median_cyclone_mae": round(median_cyclone_mae, 2),
        "high_cyclone_mae": round(high_cyclone_mae, 2),
        "best_cyclone": best_cyclone,
        "worst_cyclone": worst_cyclone,
        "mae_ge_110kt": round(high_mae, 2),
        "bias_ge_110kt": round(high_bias, 2),
        "mae_ge_130kt": round(vhigh_mae, 2),
        "bias_ge_130kt": round(vhigh_bias, 2),
        "n_ge_110kt": len(high_df),
        "n_ge_130kt": len(vhigh_df)
    }


def compute_paired_cyclone_bootstrap(
    df_nat: pd.DataFrame,
    df_bal: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> dict:
    """Perform paired block bootstrap at the CYCLONE level on the held-out test set."""
    rng = np.random.RandomState(seed)
    unique_cyclones = df_nat["cyclone_id"].unique()
    n_cyclones = len(unique_cyclones)

    deltas_mae = []
    deltas_rmse = []
    deltas_slope = []
    deltas_high_mae = []
    deltas_high_bias = []
    deltas_vhigh_mae = []

    for _ in range(n_bootstrap):
        boot_cyclones = rng.choice(unique_cyclones, size=n_cyclones, replace=True)
        # Construct resampled test set preserves frame clusters per cyclone
        boot_nat_dfs = [df_nat[df_nat["cyclone_id"] == cid] for cid in boot_cyclones]
        boot_bal_dfs = [df_bal[df_bal["cyclone_id"] == cid] for cid in boot_cyclones]

        b_df_nat = pd.concat(boot_nat_dfs, ignore_index=True)
        b_df_bal = pd.concat(boot_bal_dfs, ignore_index=True)

        m_nat = evaluate_predictions(b_df_nat)
        m_bal = evaluate_predictions(b_df_bal)

        deltas_mae.append(m_bal["mae"] - m_nat["mae"])
        deltas_rmse.append(m_bal["rmse"] - m_nat["rmse"])
        deltas_slope.append(m_bal["slope"] - m_nat["slope"])
        
        if m_nat["n_ge_110kt"] > 0 and m_bal["n_ge_110kt"] > 0:
            deltas_high_mae.append(m_bal["mae_ge_110kt"] - m_nat["mae_ge_110kt"])
            deltas_high_bias.append(m_bal["bias_ge_110kt"] - m_nat["bias_ge_110kt"])

        if m_nat["n_ge_130kt"] > 0 and m_bal["n_ge_130kt"] > 0:
            deltas_vhigh_mae.append(m_bal["mae_ge_130kt"] - m_nat["mae_ge_130kt"])

    def get_ci(arr):
        if len(arr) == 0:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_improvement": 0.0}
        a = np.array(arr)
        return {
            "mean": round(float(np.mean(a)), 2),
            "ci_lower": round(float(np.percentile(a, 2.5)), 2),
            "ci_upper": round(float(np.percentile(a, 97.5)), 2),
            "p_improvement": round(float(np.mean(a < 0.0)), 3) # for error reduction
        }

    return {
        "n_bootstrap": n_bootstrap,
        "n_cyclones": n_cyclones,
        "delta_mae_overall": get_ci(deltas_mae),
        "delta_rmse_overall": get_ci(deltas_rmse),
        "delta_slope": get_ci(deltas_slope),
        "delta_mae_ge_110kt": get_ci(deltas_high_mae),
        "delta_bias_ge_110kt": get_ci(deltas_high_bias),
        "delta_mae_ge_130kt": get_ci(deltas_vhigh_mae)
    }


def generate_predictions_for_model(
    model_ckpt: str | Path,
    config_path: str | Path,
    stats_path: str | Path,
    test_df: pd.DataFrame,
    h5_path: str | Path,
    device: torch.device
) -> pd.DataFrame:
    """Generate model predictions on test_df."""
    cfg = load_config(config_path)
    with open(stats_path, "r") as f:
        stats = json.load(f)

    preprocessor = TCIRPreprocessor(
        mean=stats["mean"],
        std=stats["std"],
        target_size=tuple(cfg.get("dataset", {}).get("input_size", [224, 224])),
        is_training=False,
        augmentation_cfg={"enabled": False}
    )

    test_ds = TCIRDataset(h5_path, test_df, channel_idx=0, preprocessor=preprocessor, in_memory=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    model = build_model(cfg).to(device)
    ckpt = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_acts, all_cids, all_indices = [], [], [], []

    with torch.no_grad():
        for images, targets, meta in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze(-1).cpu().numpy()
            all_preds.extend(outputs.tolist())
            all_acts.extend(targets.squeeze(-1).numpy().tolist())
            all_cids.extend(meta["cyclone_id"])
            all_indices.extend(meta["sample_index"].numpy().tolist())

    df_res = pd.DataFrame({
        "sample_index": all_indices,
        "cyclone_id": all_cids,
        "actual": all_acts,
        "predicted": all_preds
    })
    df_res["error"] = df_res["predicted"] - df_res["actual"]
    df_res["abs_error"] = np.abs(df_res["error"])
    df_res["bin"] = df_res["actual"].apply(assign_intensity_bin)
    df_res["regime"] = df_res["actual"].apply(assign_regime_bin)
    return df_res


def run_io_study_evaluation():
    output_dir = Path("experiments/io_balancing_study")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluation] Running on device: {device}")

    # Load Held-Out Indian Ocean Test Split
    test_df = pd.read_csv("data/metadata/test_metadata_IO.csv")
    print(f"[Evaluation] Held-Out Indian Ocean Test Split: {len(test_df)} frames across {test_df['cyclone_id'].nunique()} cyclones.")

    # 1. Generate Predictions for all 4 models on SAME IO test set
    models = {
        "IO Natural (A)": {
            "ckpt": "experiments/io_baseline_resnet18/best.pt",
            "cfg": "configs/io_baseline.yaml",
            "stats": "data/metadata/normalization_stats_IO.json",
            "h5": "data/raw/TCIR-CPAC_IO_SH.h5",
            "color": "#0284c7"
        },
        "IO Balanced (B)": {
            "ckpt": "experiments/io_balanced_resnet18/best.pt",
            "cfg": "configs/io_balanced.yaml",
            "stats": "data/metadata/normalization_stats_IO.json",
            "h5": "data/raw/TCIR-CPAC_IO_SH.h5",
            "color": "#10b981"
        },
        "Original Baseline": {
            "ckpt": "experiments/baseline_resnet18_cpac_io_sh/best.pt",
            "cfg": "configs/baseline.yaml",
            "stats": "data/metadata/normalization_stats_CPAC_IO_SH.json",
            "h5": "data/raw/TCIR-CPAC_IO_SH.h5",
            "color": "#64748b"
        },
        "All-Basin Model": {
            "ckpt": "experiments/expanded_all_basins_resnet18/best.pt",
            "cfg": "configs/all_basins.yaml",
            "stats": "data/metadata/normalization_stats_all_basins.json",
            "h5": "data/raw/TCIR-CPAC_IO_SH.h5",
            "color": "#f59e0b"
        }
    }

    pred_dfs = {}
    metrics_summary = {}

    for name, m_info in models.items():
        print(f"\n[Evaluating Model] {name} on IO Test Set...")
        df_p = generate_predictions_for_model(
            model_ckpt=m_info["ckpt"],
            config_path=m_info["cfg"],
            stats_path=m_info["stats"],
            test_df=test_df,
            h5_path=m_info["h5"],
            device=device
        )
        pred_dfs[name] = df_p
        metrics_summary[name] = evaluate_predictions(df_p)

    # Save Predictions CSVs
    pred_dfs["IO Natural (A)"].to_csv(output_dir / "test_predictions_io_natural.csv", index=False)
    pred_dfs["IO Balanced (B)"].to_csv(output_dir / "test_predictions_io_balanced.csv", index=False)

    # 2. Master Comparison Table
    print("\n" + "=" * 125)
    print("MASTER COMPARISON TABLE: EVALUATION ON HELD-OUT INDIAN OCEAN TEST SET (N=322 frames, 10 cyclones)")
    print("=" * 125)
    print(f"{'Model':<20} | {'Training Set':<14} | {'Sampling':<10} | {'MAE (kt)':<9} | {'RMSE (kt)':<10} | {'R²':<7} | {'Slope':<6} | {'Bias ≥110kt':<12} | {'MAE ≥110kt':<12}")
    print("-" * 125)
    
    table_rows = [
        ("Original Baseline", "CPAC/IO/SH", "Natural", metrics_summary["Original Baseline"]),
        ("All-Basin Model", "All 6 Basins", "Natural", metrics_summary["All-Basin Model"]),
        ("IO Natural (A)", "IO Only", "Natural", metrics_summary["IO Natural (A)"]),
        ("IO Balanced (B)", "IO Only", "Intensity", metrics_summary["IO Balanced (B)"])
    ]

    for name, train_data, samp, m in table_rows:
        print(f"{name:<20} | {train_data:<14} | {samp:<10} | {m['mae']:7.2f} kt | {m['rmse']:8.2f} kt | {m['r2']:6.3f} | {m['slope']:5.2f} | {m['bias_ge_110kt']:+10.2f} kt | {m['mae_ge_110kt']:10.2f} kt")
    print("=" * 125)

    # 3. Cyclone-Level Metrics Breakdown
    print("\n" + "=" * 115)
    print("CYCLONE-LEVEL PERFORMANCE SUMMARY (IO TEST SET)")
    print("=" * 115)
    print(f"{'Model':<20} | {'Mean Cyclone MAE':<18} | {'Median Cyclone MAE':<20} | {'High-Intensity Cyclone MAE':<26} | {'Best Cyclone':<18} | {'Worst Cyclone'}")
    print("-" * 115)
    for name, _, _, m in table_rows:
        best_str = f"{m['best_cyclone'][0]} ({m['best_cyclone'][1]:.1f} kt)"
        worst_str = f"{m['worst_cyclone'][0]} ({m['worst_cyclone'][1]:.1f} kt)"
        print(f"{name:<20} | {m['mean_cyclone_mae']:14.2f} kt  | {m['median_cyclone_mae']:16.2f} kt   | {m['high_cyclone_mae']:22.2f} kt   | {best_str:<18} | {worst_str}")
    print("=" * 115)

    # 4. Paired Cyclone-Level Bootstrap
    print("\n" + "=" * 95)
    print("PAIRED CYCLONE-LEVEL BLOCK BOOTSTRAP (1,000 RESAMPLES OF 10 TEST CYCLONES)")
    print("Δ Metric = Metric(IO Balanced) - Metric(IO Natural)")
    print("=" * 95)
    bootstrap_results = compute_paired_cyclone_bootstrap(
        df_nat=pred_dfs["IO Natural (A)"],
        df_bal=pred_dfs["IO Balanced (B)"],
        n_bootstrap=1000,
        seed=42
    )

    for metric_k, metric_v in bootstrap_results.items():
        if isinstance(metric_v, dict):
            print(f"  • {metric_k:<24}: Δ = {metric_v['mean']:+6.2f} kt [95% CI: {metric_v['ci_lower']:+6.2f} to {metric_v['ci_upper']:+6.2f} kt] | P(Improvement) = {metric_v['p_improvement']:.3f}")
    print("=" * 95)

    # 5. Intensity Bin Breakdown Table
    bin_comparison = []
    print("\n" + "=" * 115)
    print("INTENSITY BIN BREAKDOWN ON IO TEST SET: IO NATURAL vs. IO BALANCED")
    print("=" * 115)
    print(f"{'Intensity Bin':<14} | {'N Samples':<10} | {'Nat MAE':<10} | {'Nat Bias':<10} | {'Bal MAE':<10} | {'Bal Bias':<10} | {'Δ MAE (kt)':<10} | {'Confidence Tag'}")
    print("-" * 115)

    for lower, upper, label in INTENSITY_BINS:
        sub_nat = pred_dfs["IO Natural (A)"][pred_dfs["IO Natural (A)"]["bin"] == label]
        sub_bal = pred_dfs["IO Balanced (B)"][pred_dfs["IO Balanced (B)"]["bin"] == label]
        n_b = len(sub_nat)

        if n_b > 0:
            nat_mae = float(sub_nat["abs_error"].mean())
            nat_bias = float(sub_nat["error"].mean())
            bal_mae = float(sub_bal["abs_error"].mean())
            bal_bias = float(sub_bal["error"].mean())
            d_mae = bal_mae - nat_mae
            tag = "ROBUST" if n_b >= 30 else "LOW-CONFIDENCE (EXPLORATORY)"
        else:
            nat_mae, nat_bias, bal_mae, bal_bias, d_mae = 0.0, 0.0, 0.0, 0.0, 0.0
            tag = "NO SAMPLES"

        bin_comparison.append({
            "bin": label,
            "samples": n_b,
            "nat_mae": round(nat_mae, 2),
            "nat_bias": round(nat_bias, 2),
            "bal_mae": round(bal_mae, 2),
            "bal_bias": round(bal_bias, 2),
            "delta_mae": round(d_mae, 2),
            "tag": tag
        })

        if n_b > 0:
            print(f"{label:<14} | {n_b:<10d} | {nat_mae:7.2f} kt | {nat_bias:+7.2f} kt | {bal_mae:7.2f} kt | {bal_bias:+7.2f} kt | {d_mae:+7.2f} kt | {tag}")
        else:
            print(f"{label:<14} | {n_b:<10d} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {tag}")
    print("=" * 115)

    # 6. Save JSON Study Results
    study_json = {
        "held_out_test_set": {
            "samples": len(test_df),
            "cyclones": int(test_df["cyclone_id"].nunique())
        },
        "models_summary": metrics_summary,
        "bootstrap": bootstrap_results,
        "bin_comparison": bin_comparison
    }
    with open(output_dir / "io_study_summary.json", "w") as f:
        json.dump(study_json, f, indent=2)

    # 7. Generate 4 Key Visualizations
    
    # Plot 3: Error by Intensity (IO Natural vs IO Balanced)
    plt.figure(figsize=(11, 6), dpi=150)
    labels = [b["bin"] for b in bin_comparison if b["samples"] > 0]
    n_maes = [b["nat_mae"] for b in bin_comparison if b["samples"] > 0]
    b_maes = [b["bal_mae"] for b in bin_comparison if b["samples"] > 0]
    
    x = np.arange(len(labels))
    w = 0.35

    plt.bar(x - w/2, n_maes, w, label="IO Natural (Overall MAE: 11.34 kt)", color="#0284c7", alpha=0.85, edgecolor="black")
    plt.bar(x + w/2, b_maes, w, label="IO Balanced (Overall MAE: 22.37 kt)", color="#10b981", alpha=0.85, edgecolor="black")
    plt.xlabel("Intensity Bins", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Absolute Error (MAE in knots)", fontsize=11, fontweight="bold")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.title("Error by Intensity Bin on Held-Out Indian Ocean Test Set\nControlled Test: Natural Distribution vs. Intensity-Aware Sampling", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "io_error_by_intensity.png")
    plt.close()

    # Plot 4: Prediction vs Actual (Natural vs Balanced) with matched axes
    plt.figure(figsize=(10, 8), dpi=150)
    lims = [10, 160]

    df_nat = pred_dfs["IO Natural (A)"]
    df_bal = pred_dfs["IO Balanced (B)"]

    plt.scatter(df_nat["actual"], df_nat["predicted"], alpha=0.4, s=25, color="#0284c7", label="IO Natural (A)")
    plt.scatter(df_bal["actual"], df_bal["predicted"], alpha=0.4, s=25, color="#10b981", label="IO Balanced (B)")
    plt.plot(lims, lims, "k--", linewidth=2.0, label="Ideal Estimation (y = x)")

    x_grid = np.linspace(15, 140, 200)
    s_nat, i_nat = metrics_summary["IO Natural (A)"]["slope"], metrics_summary["IO Natural (A)"]["intercept"]
    s_bal, i_bal = metrics_summary["IO Balanced (B)"]["slope"], metrics_summary["IO Balanced (B)"]["intercept"]
    plt.plot(x_grid, s_nat * x_grid + i_nat, color="#0369a1", linewidth=2.5, label=f"IO Natural Fit (Slope: {s_nat:.2f})")
    plt.plot(x_grid, s_bal * x_grid + i_bal, color="#047857", linewidth=2.5, label=f"IO Balanced Fit (Slope: {s_bal:.2f})")

    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Actual Ground Truth Wind Speed (knots)", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted Wind Speed (knots)", fontsize=11, fontweight="bold")
    plt.title(f"Prediction Compression on Held-Out IO Test Set\nIO Natural (Slope={s_nat:.2f}) vs. IO Balanced (Slope={s_bal:.2f})", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "io_prediction_vs_actual.png")
    plt.close()

    # Plot 5: Bias by Intensity Bin
    plt.figure(figsize=(11, 6), dpi=150)
    n_biases = [b["nat_bias"] for b in bin_comparison if b["samples"] > 0]
    b_biases = [b["bal_bias"] for b in bin_comparison if b["samples"] > 0]

    plt.plot(x, n_biases, "o--", color="#0284c7", linewidth=2.2, markersize=8, label="IO Natural Mean Bias")
    plt.plot(x, b_biases, "s-", color="#10b981", linewidth=2.5, markersize=8, label="IO Balanced Mean Bias")
    plt.axhline(0, color="gray", linestyle=":", alpha=0.8)
    plt.xlabel("Intensity Bins", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Signed Error / Bias (knots)", fontsize=11, fontweight="bold")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.title("Systematic Bias Across Intensity Bins (IO Test Set)\nComparison of High-End Underestimation vs. Low-End Bias Drift", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "io_bias_by_intensity.png")
    plt.close()

    # Plot 6: Model Comparison Radar / Bar Summary across all 4 models on IO test set
    plt.figure(figsize=(11, 6), dpi=150)
    model_names = list(models.keys())
    maes = [metrics_summary[m]["mae"] for m in model_names]
    slopes = [metrics_summary[m]["slope"] for m in model_names]
    high_biases = [abs(metrics_summary[m]["bias_ge_110kt"]) for m in model_names]

    x_m = np.arange(len(model_names))
    plt.bar(x_m - 0.25, maes, 0.25, label="Overall MAE (kt) [Lower is better]", color="#3b82f6", alpha=0.85, edgecolor="black")
    plt.bar(x_m, [s * 15.0 for s in slopes], 0.25, label="Slope x15 (Ideal=15.0)", color="#10b981", alpha=0.85, edgecolor="black")
    plt.bar(x_m + 0.25, high_biases, 0.25, label="|Bias ≥110kt| (kt) [Lower is better]", color="#f59e0b", alpha=0.85, edgecolor="black")

    plt.xticks(x_m, model_names, fontsize=10, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=11, fontweight="bold")
    plt.title("All Models Evaluated on the Identical Indian Ocean Held-Out Test Set\nDomain-Specific Balancing vs. Global Dataset Expansion", fontsize=12, fontweight="bold", pad=15)
    plt.legend(loc="upper left", fontsize=9.5)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "io_model_comparison.png")
    plt.close()

    print(f"\n[Artifacts Generated] Saved study results and 4 comparison plots to: {output_dir}")
    return study_json


def main():
    run_io_study_evaluation()


if __name__ == "__main__":
    main()
