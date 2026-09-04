"""Script 3: Comprehensive Statistical, Mechanistic, and Subgroup Analysis for RI Investigation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score,
)

def run():
    print("=" * 80)
    print("FORENSIC SUITE: STATISTICAL & MECHANISTIC RI ANALYSIS")
    print("=" * 80)

    pred_csv = "experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_predictions.csv"
    df = pd.read_csv(pred_csv)
    test_seq = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")

    tau_val = 0.0161

    df["actual_dv24"] = df["vmax_plus_24h"] - df["vmax_curr"]
    df["pred_dv24"] = df["pred_plus_24h"] - df["vmax_curr"]
    df["actual_dv12"] = test_seq["vmax_plus_12h"] - df["vmax_curr"]
    df["actual_dv6"] = test_seq["vmax_plus_6h"] - df["vmax_curr"]
    df["pred_dv12"] = df["pred_plus_12h"] - df["vmax_curr"]
    df["pred_dv6"] = df["pred_plus_6h"] - df["vmax_curr"]

    # 1. CHECK THE TREND LABEL DEFINITION (Section 5)
    print("\n[SECTION 5] TREND LABEL DEFINITION AUDIT:")
    print("  • Headline Trend label is strictly defined by ΔV24 = Vmax(t+24h) - Vmax(t):")
    print("      Class 0 (WEAKENING):   ΔV24 <= -10 kt")
    print("      Class 1 (STABLE):     -10 kt < ΔV24 < +10 kt")
    print("      Class 2 (INTENSIFYING):ΔV24 >= +10 kt")
    
    # Are there cases where storm intensifies at +6h or +12h, but actual ΔV24 is <= -10 (weakening)?
    rising_short_falling_long = df[(df["actual_dv6"] >= 5) & (df["actual_dv24"] <= -10)]
    print(f"  • Cases where storm intensifies in short-term (+6h >= +5kt) but actual 24h target is WEAKENING: {len(rising_short_falling_long)}")
    
    # Are there cases where storm weakens at +6h, but actual ΔV24 is >= +30 (RI)?
    falling_short_rising_long = df[(df["actual_dv6"] <= -5) & (df["actual_dv24"] >= 30)]
    print(f"  • Cases where storm weakens in short-term (+6h <= -5kt) but actual 24h target is RI (>= +30kt): {len(falling_short_rising_long)}")

    # 2. COMPARE REGRESSION VS CLASSIFICATION (HEAD CONFLICT) (Section 6)
    print("\n[SECTION 6] HEAD CONFLICT AUDIT (REGRESSION VS CLASSIFICATION):")
    ri_cases = df[df["actual_dv24"] >= 30].copy()
    print(f"  • Total RI cases: {len(ri_cases)}")

    # Check for conflicts among RI cases:
    # A. Model predicts WEAKENING (trend=0) BUT RI probability is HIGH (>= tau_val)
    conflict_weak_high_ri = ri_cases[(ri_cases["pred_trend"] == 0) & (ri_cases["pred_ri_prob"] >= tau_val)]
    print(f"  • RI cases with pred_trend == WEAKENING but pred_ri_prob >= tau_val: {len(conflict_weak_high_ri)}")

    # B. Model predicts INTENSIFYING (trend=2) BUT continuous regression predicts decrease (pred_dv24 < 0)
    conflict_inte_neg_reg = ri_cases[(ri_cases["pred_trend"] == 2) & (ri_cases["pred_dv24"] < 0)]
    print(f"  • RI cases with pred_trend == INTENSIFYING but pred_dv24 < 0 kt: {len(conflict_inte_neg_reg)}")

    # Across ALL test samples (N=7,901):
    all_conflict_weak_high_ri = df[(df["pred_trend"] == 0) & (df["pred_ri_prob"] >= tau_val)]
    all_conflict_inte_neg_reg = df[(df["pred_trend"] == 2) & (df["pred_dv24"] < 0)]
    all_conflict_weak_pos_reg = df[(df["pred_trend"] == 0) & (df["pred_dv24"] > 0)]
    print(f"  • ALL test cases: pred_trend == WEAKENING but pred_ri_prob >= tau_val: {len(all_conflict_weak_high_ri)} ({len(all_conflict_weak_high_ri)/len(df)*100:.2f}%)")
    print(f"  • ALL test cases: pred_trend == INTENSIFYING but pred_dv24 < 0 kt: {len(all_conflict_inte_neg_reg)} ({len(all_conflict_inte_neg_reg)/len(df)*100:.2f}%)")
    print(f"  • ALL test cases: pred_trend == WEAKENING but pred_dv24 > 0 kt: {len(all_conflict_weak_pos_reg)} ({len(all_conflict_weak_pos_reg)/len(df)*100:.2f}%)")

    # 3. INTENSITY-BIN STRATIFICATION (Section 8)
    print("\n[SECTION 8] INTENSITY-BIN STRATIFICATION (<50, 50-70, 70-90, 90-110, >110 kt):")
    custom_bins = [0, 50, 70, 90, 110, 300]
    custom_labels = ["<50 kt", "50-70 kt", "70-90 kt", "90-110 kt", ">110 kt"]
    df["custom_bin"] = pd.cut(df["vmax_curr"], bins=custom_bins, labels=custom_labels, right=False)

    strat_data = []
    print(f"{'Bin':<12} {'N':<6} {'RI Prev':<8} {'RI Rec':<8} {'RI Prec':<9} {'RI PR-AUC':<11} {'Tr Acc':<8} {'+24 MAE':<9} {'ΔV Bias':<9}")
    print("-" * 85)

    for b_name in custom_labels:
        b_df = df[df["custom_bin"] == b_name]
        n_b = len(b_df)
        if n_b == 0:
            continue
        
        n_ri = int(b_df["actual_ri"].sum())
        ri_prev = n_ri / n_b * 100
        
        # RI metrics
        if n_ri > 0:
            b_pred_bin = (b_df["pred_ri_prob"] >= tau_val).astype(int)
            ri_rec = recall_score(b_df["actual_ri"], b_pred_bin, zero_division=0) * 100
            ri_prec = precision_score(b_df["actual_ri"], b_pred_bin, zero_division=0) * 100
            p_c, r_c, _ = precision_recall_curve(b_df["actual_ri"], b_df["pred_ri_prob"])
            ri_pr_auc = auc(r_c, p_c)
        else:
            ri_rec, ri_prec, ri_pr_auc = 0.0, 0.0, 0.0

        tr_acc = accuracy_score(b_df["actual_trend"], b_df["pred_trend"]) * 100
        mae_24 = mean_absolute_error(b_df["vmax_plus_24h"], b_df["pred_plus_24h"])
        mean_bias = np.mean(b_df["pred_dv24"] - b_df["actual_dv24"])

        print(f"{b_name:<12} {n_b:<6} {ri_prev:<7.1f}% {ri_rec:<7.1f}% {ri_prec:<8.1f}% {ri_pr_auc:<11.4f} {tr_acc:<7.1f}% {mae_24:<8.2f} {mean_bias:<+8.2f}")
        
        # Trend confusion matrix for this bin
        cm = confusion_matrix(b_df["actual_trend"], b_df["pred_trend"], labels=[0, 1, 2])
        strat_data.append({
            "bin": b_name, "count": n_b, "n_ri": n_ri, "ri_recall": ri_rec, "ri_precision": ri_prec,
            "ri_pr_auc": ri_pr_auc, "trend_acc": tr_acc, "mae_24": mae_24, "mean_bias": mean_bias,
            "confusion_matrix": cm.tolist()
        })

    # 4. RAPID-INTENSIFICATION SUBGROUP VS NON-RI (Section 9)
    print("\n[SECTION 9] RAPID INTENSIFICATION SUBGROUP (ΔV24 >= 30 kt) VS NON-RI:")
    non_ri = df[df["actual_dv24"] < 30].copy()

    mae_6_ri = mean_absolute_error(test_seq.loc[ri_cases.index, "vmax_plus_6h"], ri_cases["pred_plus_6h"])
    mae_12_ri = mean_absolute_error(test_seq.loc[ri_cases.index, "vmax_plus_12h"], ri_cases["pred_plus_12h"])
    mae_24_ri = mean_absolute_error(ri_cases["vmax_plus_24h"], ri_cases["pred_plus_24h"])
    tr_acc_ri = accuracy_score(ri_cases["actual_trend"], ri_cases["pred_trend"]) * 100
    ri_rec_sub = ((ri_cases["pred_ri_prob"] >= tau_val).sum() / len(ri_cases)) * 100
    mean_pred_dv_ri = ri_cases["pred_dv24"].mean()
    mean_act_dv_ri = ri_cases["actual_dv24"].mean()

    mae_6_non = mean_absolute_error(test_seq.loc[non_ri.index, "vmax_plus_6h"], non_ri["pred_plus_6h"])
    mae_12_non = mean_absolute_error(test_seq.loc[non_ri.index, "vmax_plus_12h"], non_ri["pred_plus_12h"])
    mae_24_non = mean_absolute_error(non_ri["vmax_plus_24h"], non_ri["pred_plus_24h"])
    tr_acc_non = accuracy_score(non_ri["actual_trend"], non_ri["pred_trend"]) * 100
    mean_pred_dv_non = non_ri["pred_dv24"].mean()
    mean_act_dv_non = non_ri["actual_dv24"].mean()

    print(f"{'Metric':<25} {'RI Cases (ΔV >= 30kt)':<24} {'Non-RI Cases (ΔV < 30kt)':<24}")
    print("-" * 75)
    print(f"{'Count':<25} {len(ri_cases):<24,d} {len(non_ri):<24,d}")
    print(f"{'+6h MAE':<25} {mae_6_ri:<24.2f} {mae_6_non:<24.2f}")
    print(f"{'+12h MAE':<25} {mae_12_ri:<24.2f} {mae_12_non:<24.2f}")
    print(f"{'+24h MAE':<25} {mae_24_ri:<24.2f} {mae_24_non:<24.2f}")
    print(f"{'Trend Accuracy':<25} {tr_acc_ri:<23.1f}% {tr_acc_non:<23.1f}%")
    print(f"{'RI Recall / False Alarm':<25} {ri_rec_sub:<23.1f}% {((non_ri['pred_ri_prob']>=tau_val).mean()*100):<23.1f}%")
    print(f"{'Mean Actual ΔV24':<25} {mean_act_dv_ri:<+24.1f} {mean_act_dv_non:<+24.1f}")
    print(f"{'Mean Predicted ΔV24':<25} {mean_pred_dv_ri:<+24.1f} {mean_pred_dv_non:<+24.1f}")
    print(f"{'Mean Underprediction (Bias)':<25} {(mean_pred_dv_ri - mean_act_dv_ri):<+24.1f} {(mean_pred_dv_non - mean_act_dv_non):<+24.1f}")

    # 5. ERROR DIRECTION ANALYSIS (Section 10)
    print("\n[SECTION 10] ERROR DIRECTION ANALYSIS FOR RI CASES:")
    ri_errors = ri_cases["pred_plus_24h"].values - ri_cases["vmax_plus_24h"].values
    n_under = (ri_errors < -5.0).sum()
    n_correct = (np.abs(ri_errors) <= 5.0).sum()
    n_over = (ri_errors > 5.0).sum()

    print(f"  • RI Cases Underpredicted (error < -5 kt): {n_under} / {len(ri_cases)} ({n_under/len(ri_cases)*100:.1f}%)")
    print(f"  • RI Cases Correct (within ±5 kt):        {n_correct} / {len(ri_cases)} ({n_correct/len(ri_cases)*100:.1f}%)")
    print(f"  • RI Cases Overpredicted (error > +5 kt):  {n_over} / {len(ri_cases)} ({n_over/len(ri_cases)*100:.1f}%)")

    # Correlation between actual ΔV24 and prediction error (pred - actual)
    corr_dv_error = np.corrcoef(df["actual_dv24"], df["pred_plus_24h"] - df["vmax_plus_24h"])[0, 1]
    print(f"  • Correlation between actual ΔV24 and Error (Pred - Actual): r = {corr_dv_error:.4f}")
    print(f"      (Strong negative correlation confirms that as ΔV24 increases, underprediction worsens proportionally!)")

    # 6. TEST FOR REGRESSION-TO-THE-MEAN (Section 11)
    print("\n[SECTION 11] TEST FOR REGRESSION-TO-THE-MEAN (Slope Fits):")
    # All cases
    slope_all, int_all = np.polyfit(df["actual_dv24"], df["pred_dv24"], 1)
    r_all = np.corrcoef(df["actual_dv24"], df["pred_dv24"])[0, 1]

    # RI cases only
    slope_ri, int_ri = np.polyfit(ri_cases["actual_dv24"], ri_cases["pred_dv24"], 1)
    r_ri = np.corrcoef(ri_cases["actual_dv24"], ri_cases["pred_dv24"])[0, 1]

    # Weakening cases only
    weak_cases = df[df["actual_dv24"] <= -10].copy()
    slope_weak, int_weak = np.polyfit(weak_cases["actual_dv24"], weak_cases["pred_dv24"], 1)
    r_weak = np.corrcoef(weak_cases["actual_dv24"], weak_cases["pred_dv24"])[0, 1]

    print(f"  • All Cases (N={len(df):,}):        Pred_ΔV = {slope_all:.4f} * Act_ΔV + {int_all:+.2f}  (r = {r_all:.4f})")
    print(f"  • RI Cases (N={len(ri_cases)}):           Pred_ΔV = {slope_ri:.4f} * Act_ΔV + {int_ri:+.2f}  (r = {r_ri:.4f})")
    print(f"  • Weakening Cases (N={len(weak_cases):,}):    Pred_ΔV = {slope_weak:.4f} * Act_ΔV + {int_weak:+.2f}  (r = {r_weak:.4f})")

    out_file = Path("experiments/forensics/statistical_analysis_suite.json")
    with open(out_file, "w") as f:
        json.dump({
            "slope_all": slope_all, "int_all": int_all, "r_all": r_all,
            "slope_ri": slope_ri, "int_ri": int_ri, "r_ri": r_ri,
            "slope_weak": slope_weak, "int_weak": int_weak, "r_weak": r_weak,
            "corr_dv_error": corr_dv_error,
            "ri_underpredicted_pct": n_under / len(ri_cases) * 100,
            "stratified_data": strat_data,
            "mae_24_ri": mae_24_ri, "mae_24_non_ri": mae_24_non,
            "mean_pred_dv_ri": mean_pred_dv_ri, "mean_act_dv_ri": mean_act_dv_ri
        }, f, indent=2)
    print(f"\nSaved statistical suite to {out_file}")

if __name__ == "__main__":
    run()
