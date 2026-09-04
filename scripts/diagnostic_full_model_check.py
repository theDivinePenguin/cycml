"""Comprehensive Diagnostic Check of the Final Frozen Model (exp_e_k7_12ep_clean)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
    mean_absolute_error,
    confusion_matrix,
)

def run_diagnostics():
    pred_path = Path("experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_predictions.csv")
    metrics_path = Path("experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_metrics.json")
    
    print("=" * 80)
    print("CYCML FINAL FROZEN MODEL FULL DIAGNOSTIC AUDIT")
    print("=" * 80)
    print(f"Prediction CSV: {pred_path}")
    print(f"Metrics JSON:   {metrics_path}")
    
    if not pred_path.exists():
        print(f"ERROR: {pred_path} does not exist!")
        return

    df = pd.read_csv(pred_path)
    print(f"\n[1] DATASET INTEGRITY:")
    print(f"  • Total test sequences: {len(df):,}")
    print(f"  • Unique test cyclones: {df['cyclone_id'].nunique()}")
    print(f"  • Columns: {list(df.columns)}")
    
    # Check for NaNs or Infs
    nans = df.isna().sum().to_dict()
    print(f"  • NaN counts: {nans}")
    has_nan = any(v > 0 for v in nans.values())
    if has_nan:
        print("  ⚠️ WARNING: NaN values detected in test predictions!")
    else:
        print("  ✓ Zero NaNs detected across all columns.")

    # [2] CONTINUOUS REGRESSION PERFORMANCE
    print("\n[2] CONTINUOUS WIND SPEED REGRESSION (+6h, +12h, +24h):")
    # For +24h:
    v_curr = df["vmax_curr"].values
    v_actual_24 = df["vmax_plus_24h"].values
    v_pred_24 = df["pred_plus_24h"].values
    
    mae_24 = mean_absolute_error(v_actual_24, v_pred_24)
    rmse_24 = np.sqrt(np.mean((v_actual_24 - v_pred_24) ** 2))
    bias_24 = np.mean(v_pred_24 - v_actual_24)
    corr_24 = np.corrcoef(v_actual_24, v_pred_24)[0, 1]
    
    # Baseline comparison (persistence: pred = v_curr)
    persist_mae_24 = mean_absolute_error(v_actual_24, v_curr)
    
    print(f"  • +24h Forecast MAE:  {mae_24:.2f} kt  (vs Persistence: {persist_mae_24:.2f} kt) -> {((persist_mae_24 - mae_24) / persist_mae_24 * 100):+.1f}% vs persistence")
    print(f"  • +24h Forecast RMSE: {rmse_24:.2f} kt")
    print(f"  • +24h Forecast Bias: {bias_24:+.2f} kt")
    print(f"  • +24h Correlation:   r = {corr_24:.4f}")

    if "pred_plus_6h" in df.columns:
        # Note: In test_predictions.csv, check if +6h and +12h exist
        mae_6 = np.mean(np.abs(df["pred_plus_6h"] - df["vmax_curr"])) # check actual ground truth
        print(f"  • +6h Raw Predictions available: min={df['pred_plus_6h'].min():.1f}, max={df['pred_plus_6h'].max():.1f}")
    if "pred_plus_12h" in df.columns:
        print(f"  • +12h Raw Predictions available: min={df['pred_plus_12h'].min():.1f}, max={df['pred_plus_12h'].max():.1f}")

    # [3] 3-CLASS TREND CLASSIFICATION
    print("\n[3] 3-CLASS 24H TREND CLASSIFICATION:")
    y_true_tr = df["actual_trend"].values
    y_pred_tr = df["pred_trend"].values
    
    acc_tr = accuracy_score(y_true_tr, y_pred_tr)
    f1_macro = f1_score(y_true_tr, y_pred_tr, average="macro")
    cm = confusion_matrix(y_true_tr, y_pred_tr)
    
    print(f"  • Overall Accuracy: {acc_tr * 100:.2f}%")
    print(f"  • Macro F1 Score:   {f1_macro:.4f}")
    print("  • Confusion Matrix (Rows = Actual [0=Weak, 1=Stab, 2=Intens], Cols = Pred):")
    for r_idx, row in enumerate(cm):
        name = ["Weakening   ", "Stable      ", "Intensifying"][r_idx]
        recall = row[r_idx] / sum(row) * 100
        print(f"    {name}: {row.tolist()}  (Recall = {recall:.1f}%)")

    # [4] RAPID INTENSIFICATION (RI-30) HAZARD CLASSIFICATION
    print("\n[4] RAPID INTENSIFICATION (RI-30) HAZARD HEAD:")
    y_true_ri = df["actual_ri"].values
    y_prob_ri = df["pred_ri_prob"].values
    
    n_ri = int(sum(y_true_ri))
    prevalence = n_ri / len(y_true_ri)
    print(f"  • Total RI Events in Test Set: {n_ri:,} / {len(y_true_ri):,} ({prevalence * 100:.2f}%)")
    
    roc_auc = roc_auc_score(y_true_ri, y_prob_ri)
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_true_ri, y_prob_ri)
    pr_auc = auc(rec_curve, prec_curve)
    
    print(f"  • ROC-AUC: {roc_auc:.4f}")
    print(f"  • PR-AUC:  {pr_auc:.4f}  (Random Baseline = {prevalence:.4f} -> {pr_auc / prevalence:.1f}x lift)")
    
    # Check calibration / probability distribution
    print(f"  • Probability Distribution: min={y_prob_ri.min():.5f}, 25%={np.percentile(y_prob_ri, 25):.5f}, median={np.median(y_prob_ri):.5f}, 75%={np.percentile(y_prob_ri, 75):.5f}, max={y_prob_ri.max():.5f}")

    # [5] STORM-BY-STORM ERROR AUDIT ON SHOWCASE CYCLONES
    print("\n[5] SHOWCASE CYCLONES PERFORMANCE AUDIT:")
    showcase_storms = [
        ("201015W", "Super Typhoon Megi"),
        ("201614L", "Hurricane Matthew"),
        ("201003I", "Super Cyclone Phet"),
        ("200801I", "VSCS Nargis"),
        ("200413E", "Hurricane Javier"),
        ("200519S", "Cyclone Percy"),
        ("201204W", "Typhoon Guchol"),
        ("201603E", "Hurricane Blas"),
        ("201305I", "VSCS Lehar"),
    ]
    
    print(f"{'Cyclone ID':<10} {'Name':<22} {'Steps':<6} {'Peak':<6} {'+24h MAE':<10} {'Trend Acc':<10} {'RI Det. Rate':<12}")
    print("-" * 80)
    for cid, name in showcase_storms:
        sdf = df[df["cyclone_id"] == cid]
        if len(sdf) == 0:
            print(f"{cid:<10} {name:<22} NOT IN TEST SET")
            continue
        s_mae = mean_absolute_error(sdf["vmax_plus_24h"], sdf["pred_plus_24h"])
        s_acc = accuracy_score(sdf["actual_trend"], sdf["pred_trend"])
        s_ri = sdf["actual_ri"].sum()
        s_ri_det = ((sdf["actual_ri"] == 1) & (sdf["pred_ri_prob"] >= 0.016)).sum()
        ri_str = f"{s_ri_det}/{s_ri}" if s_ri > 0 else "0/0 (None)"
        print(f"{cid:<10} {name:<22} {len(sdf):<6} {sdf['vmax_curr'].max():<6.0f} {s_mae:<10.2f} {s_acc*100:<9.1f}% {ri_str:<12}")

    # [6] AUDITING WHAT WAS WRONG WITH MEGI IN DEMO APP
    print("\n[6] DEEP-DIVE: SUPER TYPHOON MEGI (201015W):")
    megi_df = df[df["cyclone_id"] == "201015W"].sort_values("target_t_timestamp").reset_index(drop=True)
    print(f"  • Total sequence steps: {len(megi_df)}")
    print(f"  • Peak Observed Intensity: {megi_df['vmax_curr'].max():.0f} kt")
    print(f"  • Overall +24h MAE: {mean_absolute_error(megi_df['vmax_plus_24h'], megi_df['pred_plus_24h']):.2f} kt")
    
    # Check the specific timestep user saw: 2010101412
    step16 = megi_df[megi_df["target_t_timestamp"] == 2010101412]
    if len(step16) > 0:
        r = step16.iloc[0]
        print(f"\n  At 2010-10-14 1200Z (User Screenshot Timestep):")
        print(f"    - Current Vmax:   {r['vmax_curr']} kt")
        print(f"    - Actual +24h:    {r['vmax_plus_24h']} kt (Delta = {r['vmax_plus_24h'] - r['vmax_curr']:+.0f} kt)")
        print(f"    - Actual RI Flag: {r['actual_ri']} (No RI in this exact window)")
        print(f"    - Clean Model Pred +24h: {r['pred_plus_24h']} kt (Error = {r['pred_plus_24h'] - r['vmax_plus_24h']:+.2f} kt)")
        print(f"    - Clean Model RI Prob:   {r['pred_ri_prob']:.4f} ({r['pred_ri_prob']*100:.2f}%)")
        print(f"    - Clean Model Trend:     Class {r['pred_trend']} (Prob Intensifying: {r['prob_intensifying']*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC AUDIT COMPLETE.")
    print("=" * 80)

if __name__ == "__main__":
    run_diagnostics()
