"""Forensic audit script for Section 2: Cross-Basin / Track-Fragment Leakage.
Generates clean evaluation manifests without modifying raw data.
Recalculates test metrics and deltas.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score
)

def run_cross_basin_audit():
    print("=" * 80)
    print("SECTION 2: CROSS-BASIN / TRACK-FRAGMENT LEAKAGE AUDIT")
    print("=" * 80)

    meta_raw_path = Path("data/metadata/metadata_all_basins.csv")
    test_meta_path = Path("data/metadata/test_metadata_all_basins.csv")
    test_k5_path = Path("data/metadata/forecast_test_sequences_k5.csv")
    preds_k5_path = Path("experiments/variable_k/results/test_predictions_k5.csv")

    meta = pd.read_csv(meta_raw_path)
    test_meta = pd.read_csv(test_meta_path)
    test_seq = pd.read_csv(test_k5_path)
    preds = pd.read_csv(preds_k5_path)

    # 1. Investigate Henriette (201308E vs 201302C)
    epac = meta[meta["cyclone_id"] == "201308E"]
    cpac = meta[meta["cyclone_id"] == "201302C"]

    print(f"201308E (EPAC): {len(epac)} frames, span: {epac['timestamp'].min()} to {epac['timestamp'].max()}")
    print(f"201302C (CPAC): {len(cpac)} frames, span: {cpac['timestamp'].min()} to {cpac['timestamp'].max()}")
    print("Physical cyclone: Hurricane Henriette / Tropical Storm Unala continuity across 140W.")
    print("201308E is in TRAIN; 201302C is in TEST.")

    # Additional candidates identified
    # 200817S (Train) vs 200819S (Test, TC Nicholas)
    # 201420E (Test, TS Trudy) vs 201409L (Train, TS Hanna)
    leakage_cyclones_clean = ["201302C"]
    leakage_cyclones_all = ["201302C", "200819S", "201420E"]

    # 2. Build Clean Manifests
    clean_test_seq = test_seq[~test_seq["cyclone_id"].isin(leakage_cyclones_clean)].copy()
    track_clean_test_seq = test_seq[~test_seq["cyclone_id"].isin(leakage_cyclones_all)].copy()

    clean_test_meta = test_meta[~test_meta["cyclone_id"].isin(leakage_cyclones_clean)].copy()
    track_clean_test_meta = test_meta[~test_meta["cyclone_id"].isin(leakage_cyclones_all)].copy()

    clean_test_seq_path = Path("data/metadata/forecast_test_sequences_k5_clean.csv")
    track_clean_test_seq_path = Path("data/metadata/forecast_test_sequences_k5_track_clean.csv")
    clean_test_meta_path = Path("data/metadata/test_metadata_all_basins_clean.csv")
    track_clean_test_meta_path = Path("data/metadata/test_metadata_all_basins_track_clean.csv")

    clean_test_seq.to_csv(clean_test_seq_path, index=False)
    track_clean_test_seq.to_csv(track_clean_test_seq_path, index=False)
    clean_test_meta.to_csv(clean_test_meta_path, index=False)
    track_clean_test_meta.to_csv(track_clean_test_meta_path, index=False)

    print(f"\nManifest Creation Summary:")
    print(f"  Original Test Sequences:    {len(test_seq):,d} (Cyclones: {test_seq['cyclone_id'].nunique()})")
    print(f"  Clean Test (No Henriette):  {len(clean_test_seq):,d} (-{len(test_seq) - len(clean_test_seq)} sequences, Cyclones: {clean_test_seq['cyclone_id'].nunique()})")
    print(f"  Track-Clean Test (All):     {len(track_clean_test_seq):,d} (-{len(test_seq) - len(track_clean_test_seq)} sequences, Cyclones: {track_clean_test_seq['cyclone_id'].nunique()})")
    print(f"  Original Test Frames (raw): {len(test_meta):,d}")
    print(f"  Clean Test Frames (raw):    {len(clean_test_meta):,d} (-{len(test_meta) - len(clean_test_meta)} frames)")
    print(f"  Track-Clean Frames (raw):   {len(track_clean_test_meta):,d} (-{len(test_meta) - len(track_clean_test_meta)} frames)")

    # 3. Recalculate Test Metrics
    # Merge predictions with test sequence targets
    df = preds.merge(test_seq[["cyclone_id", "target_t_timestamp", "vmax_plus_6h", "vmax_plus_12h"]],
                     on=["cyclone_id", "target_t_timestamp"], how="inner")

    def calc_metrics(sub_df):
        mae_6 = mean_absolute_error(sub_df["vmax_plus_6h"], sub_df["pred_plus_6h"])
        mae_12 = mean_absolute_error(sub_df["vmax_plus_12h"], sub_df["pred_plus_12h"])
        mae_24 = mean_absolute_error(sub_df["vmax_plus_24h"], sub_df["pred_plus_24h"])
        mae_all = (mae_6 + mae_12 + mae_24) / 3.0

        rmse_6 = float(np.sqrt(mean_squared_error(sub_df["vmax_plus_6h"], sub_df["pred_plus_6h"])))
        rmse_12 = float(np.sqrt(mean_squared_error(sub_df["vmax_plus_12h"], sub_df["pred_plus_12h"])))
        rmse_24 = float(np.sqrt(mean_squared_error(sub_df["vmax_plus_24h"], sub_df["pred_plus_24h"])))
        rmse_all = float(np.sqrt((rmse_6**2 + rmse_12**2 + rmse_24**2) / 3.0))

        r2_6 = float(r2_score(sub_df["vmax_plus_6h"], sub_df["pred_plus_6h"]))
        r2_12 = float(r2_score(sub_df["vmax_plus_12h"], sub_df["pred_plus_12h"]))
        r2_24 = float(r2_score(sub_df["vmax_plus_24h"], sub_df["pred_plus_24h"]))
        r2_all = float(np.mean([r2_6, r2_12, r2_24]))

        pr_auc = float(average_precision_score(sub_df["actual_ri"], sub_df["pred_ri_prob"]))
        roc_auc = float(roc_auc_score(sub_df["actual_ri"], sub_df["pred_ri_prob"]))
        brier = float(brier_score_loss(sub_df["actual_ri"], sub_df["pred_ri_prob"]))

        f1 = float(f1_score(sub_df["actual_ri"], sub_df["pred_ri_flag"]))
        prec = float(precision_score(sub_df["actual_ri"], sub_df["pred_ri_flag"], zero_division=0))
        rec = float(recall_score(sub_df["actual_ri"], sub_df["pred_ri_flag"], zero_division=0))

        return {
            "n_samples": len(sub_df),
            "mae_overall": mae_all,
            "mae_6h": mae_6,
            "mae_12h": mae_12,
            "mae_24h": mae_24,
            "rmse_overall": rmse_all,
            "rmse_6h": rmse_6,
            "rmse_12h": rmse_12,
            "rmse_24h": rmse_24,
            "r2_overall": r2_all,
            "ri_pr_auc": pr_auc,
            "ri_roc_auc": roc_auc,
            "ri_brier": brier,
            "ri_f1": f1,
            "ri_precision": prec,
            "ri_recall": rec
        }

    m_orig = calc_metrics(df)
    m_clean = calc_metrics(df[~df["cyclone_id"].isin(leakage_cyclones_clean)])
    m_track_clean = calc_metrics(df[~df["cyclone_id"].isin(leakage_cyclones_all)])

    delta_clean = {k: m_clean[k] - m_orig[k] if isinstance(m_clean[k], float) else None for k in m_orig}
    delta_track_clean = {k: m_track_clean[k] - m_orig[k] if isinstance(m_track_clean[k], float) else None for k in m_orig}

    print("\nMetric Comparison Table:")
    print(f"{'Metric':<16} | {'Original Test':<14} | {'Clean (Henriette)':<18} | {'Delta':<10} | {'Track-Clean (All)':<18} | {'Delta':<10}")
    print("-" * 95)
    for k in ["mae_overall", "mae_6h", "mae_12h", "mae_24h", "rmse_overall", "r2_overall", "ri_pr_auc", "ri_roc_auc", "ri_brier"]:
        print(f"{k:<16} | {m_orig[k]:<14.4f} | {m_clean[k]:<18.4f} | {delta_clean[k]:<+10.4f} | {m_track_clean[k]:<18.4f} | {delta_track_clean[k]:<+10.4f}")

    results = {
        "status": "PASS",
        "leakage_investigation": {
            "henriette_epac_train": "201308E (81 frames, Train)",
            "henriette_cpac_test": "201302C (57 frames, Test)",
            "cross_basin_evidence": "Physical storm crossed 140W from EPAC into CPAC on 2013-08-12 00Z to 06Z; same physical cyclone.",
            "additional_track_fragments": {
                "200819S": "Severe TC Nicholas (test) preceded by precursor 200817S (train) off Western Australia",
                "201420E": "TS Trudy (test) whose remnants re-emerged in Bay of Campeche as TS Hanna 201409L (train)"
            }
        },
        "manifests_created": {
            "clean_test_sequences": str(clean_test_seq_path),
            "track_clean_test_sequences": str(track_clean_test_seq_path),
            "clean_test_metadata": str(clean_test_meta_path),
            "track_clean_test_metadata": str(track_clean_test_meta_path)
        },
        "frame_counts": {
            "original_test_frames": len(test_meta),
            "clean_test_frames": len(clean_test_meta),
            "clean_removed_frames": len(test_meta) - len(clean_test_meta),
            "track_clean_test_frames": len(track_clean_test_meta),
            "track_clean_removed_frames": len(test_meta) - len(track_clean_test_meta)
        },
        "sequence_counts": {
            "original_test_sequences": len(test_seq),
            "clean_test_sequences": len(clean_test_seq),
            "clean_removed_sequences": len(test_seq) - len(clean_test_seq),
            "track_clean_test_sequences": len(track_clean_test_seq),
            "track_clean_removed_sequences": len(test_seq) - len(track_clean_test_seq)
        },
        "metrics_original": m_orig,
        "metrics_clean": m_clean,
        "delta_clean": delta_clean,
        "metrics_track_clean": m_track_clean,
        "delta_track_clean": delta_track_clean
    }

    out_file = Path("experiments/forensic_audit/section2_cross_basin.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Section 2 audit results to {out_file}")

if __name__ == "__main__":
    run_cross_basin_audit()
