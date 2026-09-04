"""Forensic Audit Script: Storm-Level vs Sequence-Level Metrics, Redundancy, and Bootstrap Confidence Intervals."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, f1_score, precision_recall_curve, auc

def audit_cyclone_level():
    print("=" * 80)
    print("FORENSIC AUDIT 5: STORM-LEVEL EVALUATION, REDUNDANCY & BOOTSTRAP CIs")
    print("=" * 80)

    pred_csv = "experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_predictions.csv"
    df = pd.read_csv(pred_csv)

    tau = 0.0161
    
    # 1. SEQUENCE REDUNDANCY ANALYSIS
    print("\n[1] EFFECTIVE REDUNDANCY & SEQUENCE OVERLAP WITHIN TEST STORMS:")
    cids = df["cyclone_id"].unique()
    n_storms = len(cids)
    print(f"  • Total test cyclones: {n_storms}")
    print(f"  • Total test sequences: {len(df):,}")
    seq_counts = df.groupby("cyclone_id").size()
    print(f"  • Sequences per cyclone: min={seq_counts.min()}, median={seq_counts.median():.0f}, mean={seq_counts.mean():.1f}, max={seq_counts.max()}")
    
    top5 = seq_counts.sort_values(ascending=False).head(5)
    print("  • Top 5 cyclones by sample contribution:")
    for cid, cnt in top5.items():
        print(f"      {cid}: {cnt} sequences ({cnt/len(df)*100:.1f}%)")
    print(f"  • Top 10% of cyclones contribute: {seq_counts.sort_values(ascending=False).iloc[:int(n_storms*0.1)].sum() / len(df) * 100:.1f}% of all test sequences.")
    print(f"  • Consecutive 3-hourly sequences share 6 of 7 input frames (85.7% visual overlap).")

    # 2. SEQUENCE-LEVEL VS CYCLONE-LEVEL AGGREGATION
    print("\n[2] SEQUENCE-LEVEL VS CYCLONE-LEVEL AGGREGATED METRICS:")
    seq_mae24 = mean_absolute_error(df["vmax_plus_24h"], df["pred_plus_24h"])
    seq_acc_tr = (df["actual_trend"] == df["pred_trend"]).mean()
    seq_f1_tr = f1_score(df["actual_trend"], df["pred_trend"], average="macro")
    
    storm_metrics = []
    storm_dict = {}
    for cid, s_df in df.groupby("cyclone_id"):
        storm_dict[cid] = {
            "v_true": s_df["vmax_plus_24h"].values,
            "v_pred": s_df["pred_plus_24h"].values,
            "tr_true": s_df["actual_trend"].values,
            "tr_pred": s_df["pred_trend"].values,
            "ri_true": s_df["actual_ri"].values,
            "ri_prob": s_df["pred_ri_prob"].values,
        }
        s_mae24 = mean_absolute_error(s_df["vmax_plus_24h"], s_df["pred_plus_24h"])
        s_acc_tr = (s_df["actual_trend"] == s_df["pred_trend"]).mean()
        s_ri_events = int(s_df["actual_ri"].sum())
        s_ri_detected = int(((s_df["actual_ri"] == 1) & (s_df["pred_ri_prob"] >= tau)).sum())
        
        storm_metrics.append({
            "cyclone_id": cid,
            "n_samples": len(s_df),
            "mae_24": float(s_mae24),
            "trend_acc": float(s_acc_tr),
            "ri_events": s_ri_events,
            "ri_detected": s_ri_detected
        })
        
    storm_df = pd.DataFrame(storm_metrics)
    macro_storm_mae24 = storm_df["mae_24"].mean()
    macro_storm_mae24_std = storm_df["mae_24"].std()
    macro_storm_acc_tr = storm_df["trend_acc"].mean()

    print(f"  • Sequence-Level +24h MAE:        {seq_mae24:.2f} kt")
    print(f"  • Cyclone-Macro-Averaged +24h MAE: {macro_storm_mae24:.2f} ± {macro_storm_mae24_std:.2f} kt (unweighted across {n_storms} storms)")
    print(f"  • Difference (Weighting Effect):   {abs(seq_mae24 - macro_storm_mae24):.2f} kt")
    print(f"  • Sequence-Level Trend Accuracy:   {seq_acc_tr*100:.2f}%")
    print(f"  • Cyclone-Macro Trend Accuracy:    {macro_storm_acc_tr*100:.2f}%")

    # 3. FAST VECTORIZED PER-CYCLONE BOOTSTRAP (1,000 Resamples)
    print("\n[3] PER-CYCLONE CLUSTER BOOTSTRAP 95% CONFIDENCE INTERVALS (N=1,000 resamples):")
    rng = np.random.RandomState(42)
    boot_mae24 = []
    boot_f1_tr = []
    boot_pr_auc = []
    
    unique_cids = list(cids)
    n_c = len(unique_cids)
    
    for _ in range(1000):
        resampled_cids = rng.choice(unique_cids, size=n_c, replace=True)
        b_v_true = np.concatenate([storm_dict[c]["v_true"] for c in resampled_cids])
        b_v_pred = np.concatenate([storm_dict[c]["v_pred"] for c in resampled_cids])
        b_tr_true = np.concatenate([storm_dict[c]["tr_true"] for c in resampled_cids])
        b_tr_pred = np.concatenate([storm_dict[c]["tr_pred"] for c in resampled_cids])
        b_ri_true = np.concatenate([storm_dict[c]["ri_true"] for c in resampled_cids])
        b_ri_prob = np.concatenate([storm_dict[c]["ri_prob"] for c in resampled_cids])
        
        b_mae = mean_absolute_error(b_v_true, b_v_pred)
        b_f1 = f1_score(b_tr_true, b_tr_pred, average="macro")
        
        if b_ri_true.sum() > 0:
            prec, rec, _ = precision_recall_curve(b_ri_true, b_ri_prob)
            b_pr = auc(rec, prec)
            boot_pr_auc.append(b_pr)
            
        boot_mae24.append(b_mae)
        boot_f1_tr.append(b_f1)

    ci_mae24 = (np.percentile(boot_mae24, 2.5), np.percentile(boot_mae24, 97.5))
    ci_f1_tr = (np.percentile(boot_f1_tr, 2.5), np.percentile(boot_f1_tr, 97.5))
    ci_pr_auc = (np.percentile(boot_pr_auc, 2.5), np.percentile(boot_pr_auc, 97.5))

    print(f"  • +24h Forecast MAE:  {seq_mae24:.2f} kt  [95% CI: {ci_mae24[0]:.2f} - {ci_mae24[1]:.2f} kt]")
    print(f"  • Trend Macro F1:     {seq_f1_tr:.4f}     [95% CI: {ci_f1_tr[0]:.4f} - {ci_f1_tr[1]:.4f}]")
    print(f"  • RI PR-AUC:          0.4071     [95% CI: {ci_pr_auc[0]:.4f} - {ci_pr_auc[1]:.4f}]")

    out_file = Path("experiments/forensic_audit/cyclone_level_audit.json")
    with open(out_file, "w") as f:
        json.dump({
            "n_test_cyclones": n_storms,
            "seq_mae24": float(seq_mae24),
            "macro_storm_mae24": float(macro_storm_mae24),
            "macro_storm_mae24_std": float(macro_storm_mae24_std),
            "seq_acc_tr": float(seq_acc_tr),
            "macro_storm_acc_tr": float(macro_storm_acc_tr),
            "ci_mae24": [float(x) for x in ci_mae24],
            "ci_f1_tr": [float(x) for x in ci_f1_tr],
            "ci_pr_auc": [float(x) for x in ci_pr_auc]
        }, f, indent=2)
    print(f"\nAudit saved to {out_file}")

if __name__ == "__main__":
    audit_cyclone_level()
