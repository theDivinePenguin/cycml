"""Systematic comparison of RI loss pos_weight multipliers to analyze the Precision vs. Recall trade-off."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch

from scripts.train_trend_classifier import run_training


def run_pos_weight_ablation(epochs: int = 4, batch_size: int = 16):
    """Run comparison across 4 key pos_weight configurations:
    1. Unweighted (1.0x BCE)
    2. 0.5x calculated weight
    3. 1.0x calculated weight (w_pos = N_neg / N_pos ~ 13.8)
    4. 2.0x calculated weight (emphasizing high recall)
    """
    experiments = [
        {"name": "pos_weight_1x_unweighted", "mult": 0.0, "label": "1× Unweighted (1.0)"},
        {"name": "pos_weight_0_5x_calc", "mult": 0.5, "label": "0.5× Calculated (6.9×)"},
        {"name": "pos_weight_1_0x_calc", "mult": 1.0, "label": "1.0× Calculated (13.8×)"},
        {"name": "pos_weight_2_0x_calc", "mult": 2.0, "label": "2.0× Calculated (27.6×)"},
    ]

    out_dir = Path("experiments/trend_classification/pos_weight_study")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []

    for exp in experiments:
        exp_name = exp["name"]
        mult = exp["mult"]
        label = exp["label"]

        ckpt_dir = Path("experiments/trend_classification/checkpoints") / exp_name
        metrics_file = ckpt_dir / "test_metrics.json"

        if metrics_file.exists():
            print(f"\n[{exp_name}] Already completed. Loading existing metrics...")
            with open(metrics_file) as f:
                metrics = json.load(f)
        else:
            print(f"\n=======================================================")
            print(f"RUNNING POS-WEIGHT EXPERIMENT: {label} ({exp_name})")
            print(f"=======================================================")
            metrics = run_training(
                save_dir_name=exp_name,
                epochs=epochs,
                batch_size=batch_size,
                pos_weight_mult=mult,
                cooldown_seconds=15,
            )

        opt_th = metrics.get("val_opt_ri_threshold", 0.5)
        summary_records.append({
            "experiment": exp_name,
            "weight_configuration": label,
            "effective_pos_weight": metrics.get("eff_pos_weight", 1.0),
            "opt_decision_threshold": opt_th,
            "ri_pr_auc": metrics["ri_pr_auc"],
            "ri_roc_auc": metrics["ri_roc_auc"],
            "ri_precision": metrics.get("ri_precision", metrics.get(f"precision_at_{opt_th:.2f}", 0.0)),
            "ri_recall": metrics.get("ri_recall", metrics.get(f"recall_at_{opt_th:.2f}", 0.0)),
            "ri_f1": metrics.get("ri_f1", metrics.get(f"f1_at_{opt_th:.2f}", 0.0)),
            "ri_brier_score": metrics.get("ri_brier", 0.0),
            "trend_accuracy": metrics.get("trend_accuracy", 0.0),
            "trend_macro_f1": metrics.get("trend_macro_f1", 0.0),
        })

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(out_dir / "pos_weight_ablation_summary.csv", index=False)
    with open(out_dir / "pos_weight_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    print("\n" + "=" * 105)
    print("POS-WEIGHT ABLATION: RECALL VS FALSE ALARM TRADE-OFF (HELD-OUT TEST SET)")
    print("=" * 105)
    print(summary_df[["weight_configuration", "effective_pos_weight", "ri_precision", "ri_recall", "ri_f1", "ri_pr_auc", "ri_roc_auc"]].to_string(index=False))
    print("=" * 105)
    return summary_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    run_pos_weight_ablation(epochs=args.epochs, batch_size=args.batch_size)
