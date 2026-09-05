"""Diagnostic failure-case analysis for tropical cyclone intensity forecasting and RI classification."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def analyze_failures(
    manifest_csv: str,
    predictions_csv: str,
    output_dir: str = "reports/failures",
    top_n: int = 20,
):
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    df_meta = pd.read_csv(manifest_csv)
    df_pred = pd.read_csv(predictions_csv)

    # Join metadata if not already merged
    if "vmax_curr" not in df_pred.columns:
        df_merged = pd.concat([df_meta.reset_index(drop=True), df_pred.reset_index(drop=True)], axis=1)
    else:
        df_merged = df_pred

    # Check forecast error columns
    horizons = ["6h", "12h", "24h"]
    for h in horizons:
        actual_col = f"actual_plus_{h}" if f"actual_plus_{h}" in df_merged.columns else f"vmax_plus_{h}"
        pred_col = f"pred_plus_{h}" if f"pred_plus_{h}" in df_merged.columns else f"predicted_{h}"

        if actual_col in df_merged.columns and pred_col in df_merged.columns:
            df_merged[f"error_{h}"] = df_merged[pred_col] - df_merged[actual_col]
            df_merged[f"abs_error_{h}"] = np.abs(df_merged[f"error_{h}"])

    # 1. Largest absolute forecast errors (+24h)
    if "abs_error_24h" in df_merged.columns:
        top_24h_errors = df_merged.sort_values(by="abs_error_24h", ascending=False).head(top_n)
        top_24h_path = out_p / f"largest_errors_24h.csv"
        top_24h_errors.to_csv(top_24h_path, index=False)
        print(f"[Saved] Top {top_n} largest +24h forecast errors -> {top_24h_path}")

    # 2. Missed RI Events & False RI Alarms
    if "ri_probability" in df_merged.columns or "ri_prob" in df_merged.columns:
        prob_col = "ri_probability" if "ri_probability" in df_merged.columns else "ri_prob"
        v_curr = df_merged["vmax_curr"]
        actual_24 = df_merged.get("actual_plus_24h", df_merged.get("vmax_plus_24h"))

        if actual_24 is not None:
            actual_delta_24 = actual_24 - v_curr
            actual_ri = actual_delta_24 >= 30.0

            # Missed RI (Actual RI = True, Predicted P(RI) is lowest)
            missed_ri = df_merged[actual_ri].sort_values(by=prob_col, ascending=True).head(top_n)
            missed_path = out_p / "missed_ri_events.csv"
            missed_ri.to_csv(missed_path, index=False)
            print(f"[Saved] Top {top_n} Missed RI events -> {missed_path}")

            # False RI (Actual RI = False, Predicted P(RI) is highest)
            false_ri = df_merged[~actual_ri].sort_values(by=prob_col, ascending=False).head(top_n)
            false_path = out_p / "false_ri_alarms.csv"
            false_ri.to_csv(false_path, index=False)
            print(f"[Saved] Top {top_n} False RI alarms -> {false_path}")

    print("\n[Done] Diagnostic failure analysis complete.")


def main():
    parser = argparse.ArgumentParser(description="Analyze forecast failures and extreme errors.")
    parser.add_argument("--manifest", type=str, required=True, help="Sequence manifest CSV")
    parser.add_argument("--predictions", type=str, required=True, help="Model predictions CSV")
    parser.add_argument("--output-dir", type=str, default="reports/failures", help="Output directory")
    args = parser.parse_args()

    analyze_failures(manifest_csv=args.manifest, predictions_csv=args.predictions, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
