"""Comprehensive evaluation, baseline comparison, and stratified analysis for Cyclone Intensity Trend & Rapid Intensification."""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve

from src.data.trend_config import IntensityTrendConfig
from src.evaluation.baselines import PersistenceBaseline, RecentTrendBaseline, ThresholdedRegressionBaseline
from src.evaluation.classification_metrics import (
    compute_ri_metrics,
    compute_stratified_evaluation,
    compute_trend_metrics,
)


def cyclone_block_bootstrap_classification(
    test_df: pd.DataFrame,
    trend_preds: np.ndarray,
    ri_probs: np.ndarray,
    ri_threshold: float = 0.5,
    n_bootstraps: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """Perform 1,000-iteration cyclone-grouped block bootstrap to compute 95% confidence intervals."""
    cyclone_ids = test_df["cyclone_id"].values
    unique_cids = np.unique(cyclone_ids)
    n_cyclones = len(unique_cids)
    cid_to_indices = {cid: np.where(cyclone_ids == cid)[0] for cid in unique_cids}

    d24 = test_df["vmax_plus_24h"].values - test_df["vmax_curr"].values
    config = IntensityTrendConfig()
    y_trend = config.compute_trend_label(d24)
    y_ri = config.compute_ri_label(d24)

    acc_samples = []
    macro_f1_samples = []
    pr_auc_samples = []
    roc_auc_samples = []
    ri_f1_samples = []

    np.random.seed(42)
    for _ in range(n_bootstraps):
        sampled_cids = np.random.choice(unique_cids, size=n_cyclones, replace=True)
        idx_pool = np.concatenate([cid_to_indices[cid] for cid in sampled_cids])

        sub_y_trend = y_trend[idx_pool]
        sub_pred_trend = trend_preds[idx_pool]
        sub_y_ri = y_ri[idx_pool]
        sub_prob_ri = ri_probs[idx_pool]

        t_m = compute_trend_metrics(sub_y_trend, sub_pred_trend)
        r_m = compute_ri_metrics(sub_y_ri, sub_prob_ri, threshold=ri_threshold)

        acc_samples.append(t_m["accuracy"])
        macro_f1_samples.append(t_m["macro_f1"])
        pr_auc_samples.append(r_m["pr_auc"])
        roc_auc_samples.append(r_m["roc_auc"])
        ri_f1_samples.append(r_m[f"f1_at_{ri_threshold:.2f}"])

    return {
        "trend_accuracy": {
            "mean": round(float(np.mean(acc_samples)), 4),
            "ci95_low": round(float(np.percentile(acc_samples, 2.5)), 4),
            "ci95_high": round(float(np.percentile(acc_samples, 97.5)), 4),
        },
        "trend_macro_f1": {
            "mean": round(float(np.mean(macro_f1_samples)), 4),
            "ci95_low": round(float(np.percentile(macro_f1_samples, 2.5)), 4),
            "ci95_high": round(float(np.percentile(macro_f1_samples, 97.5)), 4),
        },
        "ri_pr_auc": {
            "mean": round(float(np.mean(pr_auc_samples)), 4),
            "ci95_low": round(float(np.percentile(pr_auc_samples, 2.5)), 4),
            "ci95_high": round(float(np.percentile(pr_auc_samples, 97.5)), 4),
        },
        "ri_roc_auc": {
            "mean": round(float(np.mean(roc_auc_samples)), 4),
            "ci95_low": round(float(np.percentile(roc_auc_samples, 2.5)), 4),
            "ci95_high": round(float(np.percentile(roc_auc_samples, 97.5)), 4),
        },
        "ri_f1": {
            "mean": round(float(np.mean(ri_f1_samples)), 4),
            "ci95_low": round(float(np.percentile(ri_f1_samples, 2.5)), 4),
            "ci95_high": round(float(np.percentile(ri_f1_samples, 97.5)), 4),
        },
    }


def run_comprehensive_evaluation(
    model_pred_csv: str = "experiments/trend_classification/checkpoints/classifier_primary_ri/test_predictions.csv",
    out_dir: str = "experiments/trend_classification/results",
):
    """Run evaluation across all baselines and trained model on held-out test set."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    meta_dir = Path("data/metadata")
    test_df = pd.read_csv(meta_dir / "forecast_test_sequences_k5.csv")
    config = IntensityTrendConfig()

    d24_test = test_df["vmax_plus_24h"].values - test_df["vmax_curr"].values
    y_trend = config.compute_trend_label(d24_test)
    y_ri = config.compute_ri_label(d24_test)

    results = {
        "dataset_summary": {
            "n_test_sequences": len(test_df),
            "n_unseen_cyclones": int(test_df["cyclone_id"].nunique()),
            "ri_prevalence": round(float(np.mean(y_ri)), 4),
            "class_distribution": {
                "WEAKENING": int(np.sum(y_trend == 0)),
                "STABLE": int(np.sum(y_trend == 1)),
                "INTENSIFYING": int(np.sum(y_trend == 2)),
                "RI_EVENTS": int(np.sum(y_ri == 1)),
            },
        },
        "models": {},
    }

    # 1. Baseline A: Persistence
    print("Evaluating Baseline A: Persistence Trend...")
    base_a = PersistenceBaseline(config)
    t_pred_a, t_prob_a, ri_prob_a = base_a.predict(test_df)
    results["models"]["Baseline A (Persistence)"] = {
        "trend_metrics": compute_trend_metrics(y_trend, t_pred_a),
        "ri_metrics": compute_ri_metrics(y_ri, ri_prob_a, threshold=0.5),
        "bootstrap_ci": cyclone_block_bootstrap_classification(test_df, t_pred_a, ri_prob_a, ri_threshold=0.5),
    }

    # 2. Baseline B: Recent 6h Trend Extrapolation
    print("Evaluating Baseline B: Recent Trend Extrapolation...")
    base_b = RecentTrendBaseline(config)
    t_pred_b, t_prob_b, ri_prob_b = base_b.predict(test_df)
    results["models"]["Baseline B (Recent 6h Trend)"] = {
        "trend_metrics": compute_trend_metrics(y_trend, t_pred_b),
        "ri_metrics": compute_ri_metrics(y_ri, ri_prob_b, threshold=0.5),
        "bootstrap_ci": cyclone_block_bootstrap_classification(test_df, t_pred_b, ri_prob_b, ri_threshold=0.5),
    }

    # 3. Baseline C: Thresholded Continuous Regression
    reg_csv_path = Path("experiments/forecasting/checkpoints/cnn_transformer_k5/test_predictions.csv")
    if reg_csv_path.exists():
        print("Evaluating Baseline C: Thresholded Continuous Regression...")
        base_c = ThresholdedRegressionBaseline(config)
        t_pred_c, t_prob_c, ri_prob_c, _ = base_c.predict_from_csv(str(reg_csv_path))
        results["models"]["Baseline C (Thresholded Continuous Forecaster)"] = {
            "trend_metrics": compute_trend_metrics(y_trend, t_pred_c),
            "ri_metrics": compute_ri_metrics(y_ri, ri_prob_c, threshold=0.5),
            "bootstrap_ci": cyclone_block_bootstrap_classification(test_df, t_pred_c, ri_prob_c, ri_threshold=0.5),
        }

    # 4. Primary AI Model: TemporalClassifier
    model_pred_path = Path(model_pred_csv)
    if model_pred_path.exists():
        print(f"Evaluating AI Classifier from {model_pred_csv}...")
        pred_df = pd.read_csv(model_pred_path)
        t_pred_ai = pred_df["pred_trend"].values
        ri_prob_ai = pred_df["pred_ri_prob"].values

        # Load optimal threshold from test_metrics.json if present
        ckpt_metrics_path = model_pred_path.parent / "test_metrics.json"
        opt_thresh = 0.5
        if ckpt_metrics_path.exists():
            with open(ckpt_metrics_path) as f:
                cm = json.load(f)
                opt_thresh = cm.get("val_opt_ri_threshold", 0.5)

        ai_trend_metrics = compute_trend_metrics(y_trend, t_pred_ai)
        ai_ri_metrics = compute_ri_metrics(y_ri, ri_prob_ai, threshold=opt_thresh)
        ai_stratified = compute_stratified_evaluation(test_df, t_pred_ai, ri_prob_ai, ri_threshold=opt_thresh)
        ai_bootstrap = cyclone_block_bootstrap_classification(test_df, t_pred_ai, ri_prob_ai, ri_threshold=opt_thresh)

        results["models"]["TemporalClassifier (Multi-Task)"] = {
            "optimal_threshold": opt_thresh,
            "trend_metrics": ai_trend_metrics,
            "ri_metrics": ai_ri_metrics,
            "stratified": ai_stratified,
            "bootstrap_ci": ai_bootstrap,
        }

    # Save complete JSON report
    out_file = out_path / "comprehensive_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comprehensive benchmark report -> {out_file}")

    # Print summary comparison table
    print("\n" + "=" * 95)
    print("CYCLONE EVOLUTION & RAPID INTENSIFICATION BENCHMARK LADDER (HELD-OUT TEST SET, N=8,279)")
    print("=" * 95)
    header = f"{'Model Architecture':<40} | {'Trend Acc':<10} | {'Macro F1':<10} | {'RI ROC-AUC':<11} | {'RI PR-AUC':<10} | {'RI F1':<10}"
    print(header)
    print("-" * 95)

    for m_name, m_data in results["models"].items():
        tm = m_data["trend_metrics"]
        rm = m_data["ri_metrics"]
        f1_key = [k for k in rm.keys() if k.startswith("f1_at_")]
        f1_val = rm[f1_key[0]] if f1_key else rm.get("optimal_f1", 0.0)
        row = (
            f"{m_name:<40} | "
            f"{tm['accuracy']*100:5.2f}%    | "
            f"{tm['macro_f1']:7.4f}   | "
            f"{rm['roc_auc']:8.4f}    | "
            f"{rm['pr_auc']:7.4f}   | "
            f"{f1_val:7.4f}"
        )
        print(row)
    print("=" * 95)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-csv", type=str, default="experiments/trend_classification/checkpoints/classifier_primary_ri/test_predictions.csv")
    parser.add_argument("--out-dir", type=str, default="experiments/trend_classification/results")
    args = parser.parse_args()

    run_comprehensive_evaluation(args.pred_csv, args.out_dir)
