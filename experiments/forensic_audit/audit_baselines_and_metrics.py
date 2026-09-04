"""Forensic Audit Script: Baselines, Metrics, Intensity Stratification, and Change Forecasting."""
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
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)

def audit_performance():
    print("=" * 80)
    print("FORENSIC AUDIT 3: PERFORMANCE, BASELINES, & CHANGE FORECASTING")
    print("=" * 80)

    pred_csv = "experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_predictions.csv"
    df = pd.read_csv(pred_csv)
    
    # Load test sequences for history Vmax (to compute linear persistence)
    test_seq = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")
    train_seq = pd.read_csv("data/metadata/forecast_train_sequences_k7.csv")
    
    # Check alignment between df and test_seq
    assert len(df) == len(test_seq)
    
    # Ground truth
    v_curr = df["vmax_curr"].values
    v_24 = df["vmax_plus_24h"].values
    p_6 = df["pred_plus_6h"].values
    p_12 = df["pred_plus_12h"].values
    p_24 = df["pred_plus_24h"].values
    
    # 1. BASELINE COMPARISON
    print("\n[1] FORECAST PERFORMANCE VS BASELINES (+24h):")
    # Persistence: pred = v_curr
    mae_pers = mean_absolute_error(v_24, v_curr)
    rmse_pers = np.sqrt(mean_squared_error(v_24, v_curr))
    bias_pers = np.mean(v_curr - v_24)
    
    # Linear Persistence: extrapolate trend over past 6 hours: v(t) - v(t-6h)
    # in test_seq: history_vmax is list of 7 values [t-18, t-15, t-12, t-9, t-6, t-3, t]
    # index 4 is t-6h, index 6 is t
    d_6h_past = []
    for _, r in test_seq.iterrows():
        hv = json.loads(r["history_vmax"])
        d_6h_past.append(hv[6] - hv[4]) # 6h change
    d_6h_past = np.array(d_6h_past)
    
    # Extrapolate to +24h: v_curr + 4 * d_6h_past, clipped at min 15 kt
    pred_lin_pers_24 = np.clip(v_curr + 4.0 * d_6h_past, 15.0, 185.0)
    mae_lin_pers = mean_absolute_error(v_24, pred_lin_pers_24)
    rmse_lin_pers = np.sqrt(mean_squared_error(v_24, pred_lin_pers_24))
    
    # Mean Climatology Baseline: mean of training +24h intensities
    train_mean_24 = train_seq["vmax_plus_24h"].mean()
    pred_mean_24 = np.full_like(v_24, train_mean_24)
    mae_mean = mean_absolute_error(v_24, pred_mean_24)
    rmse_mean = np.sqrt(mean_squared_error(v_24, pred_mean_24))
    
    # Model +24h
    mae_model_24 = mean_absolute_error(v_24, p_24)
    rmse_model_24 = np.sqrt(mean_squared_error(v_24, p_24))
    bias_model_24 = np.mean(p_24 - v_24)
    corr_24 = np.corrcoef(v_24, p_24)[0, 1]
    
    print(f"  • Persistence Baseline:      MAE = {mae_pers:5.2f} kt | RMSE = {rmse_pers:5.2f} kt | Bias = {bias_pers:+5.2f} kt")
    print(f"  • Linear Persistence (+6h):  MAE = {mae_lin_pers:5.2f} kt | RMSE = {rmse_lin_pers:5.2f} kt")
    print(f"  • Mean Climatology Baseline: MAE = {mae_mean:5.2f} kt | RMSE = {rmse_mean:5.2f} kt (Train Mean = {train_mean_24:.1f} kt)")
    print(f"  • Model Forecast (+24h):     MAE = {mae_model_24:5.2f} kt | RMSE = {rmse_model_24:5.2f} kt | Bias = {bias_model_24:+5.2f} kt | r = {corr_24:.4f}")
    print(f"  • Skill over Persistence:    {((mae_pers - mae_model_24) / mae_pers * 100):+.1f}% MAE reduction")

    # +6h and +12h Baselines
    v_6 = test_seq["vmax_plus_6h"].values
    v_12 = test_seq["vmax_plus_12h"].values
    mae_pers_6 = mean_absolute_error(v_6, v_curr)
    mae_model_6 = mean_absolute_error(v_6, p_6)
    mae_pers_12 = mean_absolute_error(v_12, v_curr)
    mae_model_12 = mean_absolute_error(v_12, p_12)
    
    print(f"\n  • +6h Horizon:  Model MAE = {mae_model_6:4.2f} kt vs Persistence MAE = {mae_pers_6:4.2f} kt ({((mae_pers_6 - mae_model_6)/mae_pers_6*100):+.1f}%)")
    print(f"  • +12h Horizon: Model MAE = {mae_model_12:4.2f} kt vs Persistence MAE = {mae_pers_12:4.2f} kt ({((mae_pers_12 - mae_model_12)/mae_pers_12*100):+.1f}%)")
    print(f"  • +24h Horizon: Model MAE = {mae_model_24:4.2f} kt vs Persistence MAE = {mae_pers_24 if 'mae_pers_24' in locals() else mae_pers:4.2f} kt ({((mae_pers - mae_model_24)/mae_pers*100):+.1f}%)")

    # 2. CHANGE FORECAST ANALYSIS: delta V24
    print("\n[2] CHANGE FORECAST ANALYSIS (Delta V24 = V24 - Vcurr):")
    true_dv24 = v_24 - v_curr
    pred_dv24 = p_24 - v_curr
    
    dv_mae = mean_absolute_error(true_dv24, pred_dv24)
    dv_corr = np.corrcoef(true_dv24, pred_dv24)[0, 1]
    dv_bias = np.mean(pred_dv24 - true_dv24)
    
    # Linear regression slope: pred_dv24 = slope * true_dv24 + intercept
    slope, intercept = np.polyfit(true_dv24, pred_dv24, 1)
    
    print(f"  • True Delta V24: mean = {true_dv24.mean():+5.2f} kt, std = {true_dv24.std():5.2f} kt, min = {true_dv24.min():+5.1f} kt, max = {true_dv24.max():+5.1f} kt")
    print(f"  • Pred Delta V24: mean = {pred_dv24.mean():+5.2f} kt, std = {pred_dv24.std():5.2f} kt, min = {pred_dv24.min():+5.1f} kt, max = {pred_dv24.max():+5.1f} kt")
    print(f"  • Delta V24 Correlation: r = {dv_corr:.4f}")
    print(f"  • Delta V24 Regression Slope: {slope:.4f} (Ideal = 1.0; <1 indicates regression toward mean)")
    print(f"  • Delta V24 Bias: {dv_bias:+.2f} kt")
    
    # Performance subset: strengthening vs weakening vs stable vs RI
    is_weak = true_dv24 <= -10.0
    is_stab = (true_dv24 > -10.0) & (true_dv24 < 10.0)
    is_inte = true_dv24 >= 10.0
    is_ri = true_dv24 >= 30.0
    
    print(f"\n  Subset Analysis on Delta V24:")
    print(f"    - Weakening (N={is_weak.sum():,}):   True Mean ΔV = {true_dv24[is_weak].mean():+5.1f} kt | Pred Mean ΔV = {pred_dv24[is_weak].mean():+5.1f} kt | MAE = {mean_absolute_error(true_dv24[is_weak], pred_dv24[is_weak]):.2f} kt")
    print(f"    - Stable    (N={is_stab.sum():,}):   True Mean ΔV = {true_dv24[is_stab].mean():+5.1f} kt | Pred Mean ΔV = {pred_dv24[is_stab].mean():+5.1f} kt | MAE = {mean_absolute_error(true_dv24[is_stab], pred_dv24[is_stab]):.2f} kt")
    print(f"    - Intensifying (N={is_inte.sum():,}): True Mean ΔV = {true_dv24[is_inte].mean():+5.1f} kt | Pred Mean ΔV = {pred_dv24[is_inte].mean():+5.1f} kt | MAE = {mean_absolute_error(true_dv24[is_inte], pred_dv24[is_inte]):.2f} kt")
    print(f"    - RI (>=+30kt) (N={is_ri.sum():,}):    True Mean ΔV = {true_dv24[is_ri].mean():+5.1f} kt | Pred Mean ΔV = {pred_dv24[is_ri].mean():+5.1f} kt | MAE = {mean_absolute_error(true_dv24[is_ri], pred_dv24[is_ri]):.2f} kt")

    # 3. INTENSITY-STRATIFIED PERFORMANCE
    print("\n[3] INTENSITY-STRATIFIED PERFORMANCE (+24h):")
    # Saffir-Simpson bins on v_curr:
    # <34: TD, 34-63: TS, 64-82: Cat1, 83-95: Cat2, 96-112: Cat3, 113-136: Cat4, >=137: Cat5
    bins = [0, 34, 64, 83, 96, 113, 137, 300]
    bin_labels = ["<34 (TD)", "34-63 (TS)", "64-82 (Cat 1)", "83-95 (Cat 2)", "96-112 (Cat 3)", "113-136 (Cat 4)", ">=137 (Cat 5)"]
    df["intensity_bin"] = pd.cut(v_curr, bins=bins, labels=bin_labels, right=False)
    
    print(f"{'Bin':<16} {'N':<6} {'Obs Mean':<9} {'Pred Mean':<10} {'Bias (kt)':<10} {'+6h MAE':<9} {'+12h MAE':<10} {'+24h MAE':<10} {'Persist MAE':<12}")
    print("-" * 95)
    
    strat_results = []
    for b_name in bin_labels:
        b_df = df[df["intensity_bin"] == b_name]
        if len(b_df) == 0:
            continue
        b_v24 = b_df["vmax_plus_24h"].values
        b_p24 = b_df["pred_plus_24h"].values
        b_p12 = b_df["pred_plus_12h"].values
        b_p6 = b_df["pred_plus_6h"].values
        b_vc = b_df["vmax_curr"].values
        b_v6 = test_seq.loc[b_df.index, "vmax_plus_6h"].values
        b_v12 = test_seq.loc[b_df.index, "vmax_plus_12h"].values
        
        b_mae6 = mean_absolute_error(b_v6, b_p6)
        b_mae12 = mean_absolute_error(b_v12, b_p12)
        b_mae24 = mean_absolute_error(b_v24, b_p24)
        b_pers24 = mean_absolute_error(b_v24, b_vc)
        b_bias = np.mean(b_p24 - b_v24)
        
        print(f"{b_name:<16} {len(b_df):<6} {b_v24.mean():<9.1f} {b_p24.mean():<10.1f} {b_bias:<+10.2f} {b_mae6:<9.2f} {b_mae12:<10.2f} {b_mae24:<10.2f} {b_pers24:<12.2f}")
        strat_results.append({
            "bin": b_name,
            "count": len(b_df),
            "obs_mean": float(b_v24.mean()),
            "pred_mean": float(b_p24.mean()),
            "bias": float(b_bias),
            "mae_6h": float(b_mae6),
            "mae_12h": float(b_mae12),
            "mae_24h": float(b_mae24),
            "persist_mae_24h": float(b_pers24)
        })

    # 4. SATURATION & DISTRIBUTION CHECKS
    print("\n[4] PREDICTION DISTRIBUTION & SATURATION ANALYSIS:")
    print(f"  • True +24h:  min={v_24.min():.1f}, 25%={np.percentile(v_24, 25):.1f}, med={np.median(v_24):.1f}, 75%={np.percentile(v_24, 75):.1f}, 99%={np.percentile(v_24, 99):.1f}, max={v_24.max():.1f}")
    print(f"  • Pred +24h:  min={p_24.min():.1f}, 25%={np.percentile(p_24, 25):.1f}, med={np.median(p_24):.1f}, 75%={np.percentile(p_24, 75):.1f}, 99%={np.percentile(p_24, 99):.1f}, max={p_24.max():.1f}")
    print(f"  • True Peak Intensity in Test Set: {v_24.max():.1f} kt")
    print(f"  • Model Max Prediction:            {p_24.max():.1f} kt")

    # 5. RI HEAD CORRELATION WITH CURRENT INTENSITY
    print("\n[5] RAPID INTENSIFICATION (RI) HEAD FORENSICS:")
    ri_probs = df["pred_ri_prob"].values
    ri_true = df["actual_ri"].values
    
    corr_ri_vcurr = np.corrcoef(ri_probs, v_curr)[0, 1]
    print(f"  • Correlation between Predicted RI Prob and Current Vmax: r = {corr_ri_vcurr:.4f}")
    print(f"  • RI Event Count: {int(ri_true.sum())} / {len(ri_true)} ({ri_true.mean()*100:.2f}%)")
    
    roc_auc = roc_auc_score(ri_true, ri_probs)
    p_curve, r_curve, ths = precision_recall_curve(ri_true, ri_probs)
    pr_auc = auc(r_curve, p_curve)
    
    print(f"  • ROC-AUC: {roc_auc:.4f}")
    print(f"  • PR-AUC:  {pr_auc:.4f} (Prevalence = {ri_true.mean():.4f})")
    
    # Stratified RI Recall by intensity
    print("\n  RI Detection Rate by Initial Intensity:")
    tau_val = 0.0161
    for b_name in bin_labels:
        b_df = df[df["intensity_bin"] == b_name]
        n_ri_b = b_df["actual_ri"].sum()
        if n_ri_b > 0:
            det = ((b_df["actual_ri"] == 1) & (b_df["pred_ri_prob"] >= tau_val)).sum()
            prec = det / max(1, (b_df["pred_ri_prob"] >= tau_val).sum())
            print(f"    - {b_name:<16}: Recall = {det}/{n_ri_b} ({det/n_ri_b*100:5.1f}%) | Detected Alarms: {(b_df['pred_ri_prob'] >= tau_val).sum():<4}")

    # 6. TREND CLASSIFIER CONFUSION MATRIX
    print("\n[6] 3-CLASS TREND CLASSIFIER DETAILED METRICS:")
    tr_true = df["actual_trend"].values
    tr_pred = df["pred_trend"].values
    
    cm = confusion_matrix(tr_true, tr_pred)
    acc = accuracy_score(tr_true, tr_pred)
    f1_m = f1_score(tr_true, tr_pred, average="macro")
    
    print(f"  • Overall Accuracy: {acc*100:.2f}% | Macro F1: {f1_m:.4f}")
    print("  • Detailed Per-Class Breakdown:")
    names = ["WEAKENING", "STABLE", "INTENSIFYING"]
    for i, name in enumerate(names):
        rec = cm[i, i] / sum(cm[i, :]) * 100
        prec = cm[i, i] / sum(cm[:, i]) * 100
        f1_c = 2 * (prec * rec) / (prec + rec) / 100
        print(f"    - {name:<13}: Precision = {prec:5.1f}% | Recall = {rec:5.1f}% | F1 = {f1_c:.4f} | Support = {sum(cm[i, :]):,}")

    out_file = Path("experiments/forensic_audit/performance_audit.json")
    with open(out_file, "w") as f:
        json.dump({
            "mae_persistence_24": float(mae_pers),
            "rmse_persistence_24": float(rmse_pers),
            "mae_linear_persistence_24": float(mae_lin_pers),
            "mae_mean_climatology_24": float(mae_mean),
            "mae_model_24": float(mae_model_24),
            "rmse_model_24": float(rmse_model_24),
            "bias_model_24": float(bias_model_24),
            "corr_24": float(corr_24),
            "dv24_corr": float(dv_corr),
            "dv24_slope": float(slope),
            "dv24_bias": float(dv_bias),
            "ri_roc_auc": float(roc_auc),
            "ri_pr_auc": float(pr_auc),
            "corr_ri_vcurr": float(corr_ri_vcurr),
            "trend_accuracy": float(acc),
            "trend_macro_f1": float(f1_m),
            "stratified_results": strat_results
        }, f, indent=2)
    print(f"\nAudit saved to {out_file}")

if __name__ == "__main__":
    audit_performance()
