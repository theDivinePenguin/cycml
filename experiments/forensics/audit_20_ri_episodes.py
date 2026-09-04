"""Script 1: Forensic Analysis of 20+ Real RI Episodes, Turning Points, and Temporal Lag."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def run():
    print("=" * 80)
    print("FORENSIC INVESTIGATION: 20+ REAL RI EPISODES, TURNING POINTS & LAG")
    print("=" * 80)

    pred_csv = "experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_predictions.csv"
    df = pd.read_csv(pred_csv)
    test_seq = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")

    # Add delta columns
    df["actual_dv24"] = df["vmax_plus_24h"] - df["vmax_curr"]
    df["pred_dv24"] = df["pred_plus_24h"] - df["vmax_curr"]
    df["actual_dv12"] = test_seq["vmax_plus_12h"] - df["vmax_curr"]
    df["actual_dv6"] = test_seq["vmax_plus_6h"] - df["vmax_curr"]

    # History intensities
    past_6 = []
    past_12 = []
    for _, r in test_seq.iterrows():
        hv = json.loads(r["history_vmax"])
        past_6.append(hv[6] - hv[4])
        past_12.append(hv[6] - hv[2])
    df["past_6h_dv"] = past_6
    df["past_12h_dv"] = past_12

    # Identify cyclones with strong RI (max dv24 >= 35 kt)
    cyclone_max_dv = df.groupby("cyclone_id")["actual_dv24"].max()
    top_ri_cyclones = cyclone_max_dv[cyclone_max_dv >= 35].sort_values(ascending=False).index.tolist()
    print(f"Total cyclones in test set with max Delta V24 >= 35 kt: {len(top_ri_cyclones)}")

    # Select at least 20 storms
    selected_cids = top_ri_cyclones[:25]
    print(f"Selected {len(selected_cids)} top RI cyclones for episode analysis.\n")

    episodes = {}
    lag_records = []
    tau_val = 0.0161 # canonical validation threshold

    for cid in selected_cids:
        sdf = df[df["cyclone_id"] == cid].sort_values("target_t_timestamp").reset_index(drop=True)
        # Find RI window: timesteps where actual_dv24 >= 30
        ri_indices = sdf.index[sdf["actual_dv24"] >= 30].tolist()
        if not ri_indices:
            continue
        
        # Take a window from 4 steps before first RI to 4 steps after last RI
        start_idx = max(0, ri_indices[0] - 4)
        end_idx = min(len(sdf), ri_indices[-1] + 5)
        window_df = sdf.iloc[start_idx:end_idx].copy()

        # Build chronological table
        table_rows = []
        for _, r in window_df.iterrows():
            table_rows.append({
                "timestamp": int(r["target_t_timestamp"]),
                "v_curr": float(r["vmax_curr"]),
                "act_plus_6": float(r["vmax_curr"] + r["actual_dv6"]),
                "act_plus_12": float(r["vmax_curr"] + r["actual_dv12"]),
                "act_plus_24": float(r["vmax_plus_24h"]),
                "pred_plus_6": float(r["pred_plus_6h"]),
                "pred_plus_12": float(r["pred_plus_12h"]),
                "pred_plus_24": float(r["pred_plus_24h"]),
                "pred_trend": int(r["pred_trend"]),
                "trend_name": ["WEAKENING", "STABLE", "INTENSIFYING"][int(r["pred_trend"])],
                "ri_prob": float(r["pred_ri_prob"]),
                "ri_flag": int(r["pred_ri_prob"] >= tau_val),
                "actual_dv24": float(r["actual_dv24"]),
                "pred_dv24": float(r["pred_dv24"]),
                "past_6h_dv": float(r["past_6h_dv"]),
                "past_12h_dv": float(r["past_12h_dv"]),
            })

        # Identify T0, T1, T2:
        # T0: last timestamp before actual_dv24 >= 30
        # T1: first timestamp where actual_dv24 >= 30
        # T2: timestamp where actual_dv24 reaches peak during RI episode
        t0_row = sdf.iloc[ri_indices[0] - 1] if ri_indices[0] > 0 else sdf.iloc[0]
        t1_row = sdf.iloc[ri_indices[0]]
        
        # Peak of dv24 in this episode
        peak_ri_idx = ri_indices[np.argmax(sdf.loc[ri_indices, "actual_dv24"].values)]
        t2_row = sdf.iloc[peak_ri_idx]

        # Calculate lag:
        # actual_onset: timestamp of first step where actual_dv24 >= 30 (or actual intensity starts climbing rapidly)
        # model_onset_trend: first timestamp where model predicts INTENSIFYING (class 2)
        # model_onset_ri: first timestamp where model predicts RI (ri_prob >= tau_val)
        actual_onset_idx = ri_indices[0]
        actual_onset_ts = int(sdf.iloc[actual_onset_idx]["target_t_timestamp"])

        # Model RI alarm onset in window [start_idx, end_idx]
        model_ri_alarms = [i for i in range(start_idx, end_idx) if sdf.iloc[i]["pred_ri_prob"] >= tau_val]
        # Model Intensifying onset
        model_inte_steps = [i for i in range(start_idx, end_idx) if sdf.iloc[i]["pred_trend"] == 2]

        # Lag in steps (each step = 3 hours)
        # If model alarms after actual onset: lag > 0 (delayed)
        # If model alarms before actual onset: lag < 0 (early warning)
        # If model never alarms: lag = None (complete miss)
        ri_lag_hours = None
        trend_lag_hours = None

        if model_ri_alarms:
            # Find first alarm around actual onset
            first_alarm_idx = model_ri_alarms[0]
            ri_lag_hours = (first_alarm_idx - actual_onset_idx) * 3
        
        if model_inte_steps:
            first_inte_idx = model_inte_steps[0]
            trend_lag_hours = (first_inte_idx - actual_onset_idx) * 3

        # Check if Point B occurs in this storm:
        # Point B: storm is entering RI (actual_dv24 >= 30 or past_6h >= 5), yet model predicts WEAKENING (class 0)
        point_b_steps = []
        for i in ri_indices:
            if sdf.iloc[i]["pred_trend"] == 0:
                point_b_steps.append(int(sdf.iloc[i]["target_t_timestamp"]))

        episodes[cid] = {
            "cyclone_id": cid,
            "max_dv24": float(sdf.loc[ri_indices, "actual_dv24"].max()),
            "ri_step_count": len(ri_indices),
            "actual_onset_ts": actual_onset_ts,
            "t0": {
                "ts": int(t0_row["target_t_timestamp"]),
                "v_curr": float(t0_row["vmax_curr"]),
                "act_dv24": float(t0_row["actual_dv24"]),
                "pred_dv24": float(t0_row["pred_dv24"]),
                "pred_trend": int(t0_row["pred_trend"]),
                "ri_prob": float(t0_row["pred_ri_prob"]),
            },
            "t1": {
                "ts": int(t1_row["target_t_timestamp"]),
                "v_curr": float(t1_row["vmax_curr"]),
                "act_dv24": float(t1_row["actual_dv24"]),
                "pred_dv24": float(t1_row["pred_dv24"]),
                "pred_trend": int(t1_row["pred_trend"]),
                "ri_prob": float(t1_row["pred_ri_prob"]),
            },
            "t2": {
                "ts": int(t2_row["target_t_timestamp"]),
                "v_curr": float(t2_row["vmax_curr"]),
                "act_dv24": float(t2_row["actual_dv24"]),
                "pred_dv24": float(t2_row["pred_dv24"]),
                "pred_trend": int(t2_row["pred_trend"]),
                "ri_prob": float(t2_row["pred_ri_prob"]),
            },
            "ri_lag_hours": ri_lag_hours,
            "trend_lag_hours": trend_lag_hours,
            "point_b_occurred": len(point_b_steps) > 0,
            "point_b_timestamps": point_b_steps,
            "chronological_table": table_rows
        }

        lag_records.append({
            "cyclone_id": cid,
            "max_dv24": float(sdf.loc[ri_indices, "actual_dv24"].max()),
            "ri_lag_hours": ri_lag_hours,
            "trend_lag_hours": trend_lag_hours,
            "point_b_count": len(point_b_steps),
            "ri_detected": (sdf.loc[ri_indices, "pred_ri_prob"] >= tau_val).sum() > 0,
            "ri_detected_fraction": float((sdf.loc[ri_indices, "pred_ri_prob"] >= tau_val).mean())
        })

    lag_df = pd.DataFrame(lag_records)
    print("\n[LAG SUMMARY ACROSS TOP RI STORMS]:")
    print(f"{'Cyclone ID':<12} {'Max ΔV24':<10} {'RI Lag (h)':<12} {'Trend Lag (h)':<14} {'Point B Steps':<14} {'RI Recall':<10}")
    print("-" * 80)
    for _, r in lag_df.iterrows():
        ri_lag_str = f"{r['ri_lag_hours']:+0.0f}h" if pd.notna(r['ri_lag_hours']) else "NEVER"
        tr_lag_str = f"{r['trend_lag_hours']:+0.0f}h" if pd.notna(r['trend_lag_hours']) else "NEVER"
        print(f"{r['cyclone_id']:<12} {r['max_dv24']:<10.0f} {ri_lag_str:<12} {tr_lag_str:<14} {int(r['point_b_count']):<14} {r['ri_detected_fraction']*100:5.1f}%")

    # Quantiles of lag for storms where RI was detected
    valid_ri_lags = lag_df["ri_lag_hours"].dropna().values
    valid_tr_lags = lag_df["trend_lag_hours"].dropna().values

    print("\n[LAG DISTRIBUTION (HOURS)]:")
    print(f"  • RI Alarm Lag:    Mean = {valid_ri_lags.mean():+5.1f}h | Median = {np.median(valid_ri_lags):+5.1f}h | 25% = {np.percentile(valid_ri_lags, 25):+5.1f}h | 75% = {np.percentile(valid_ri_lags, 75):+5.1f}h")
    print(f"  • Trend Intensifying Lag: Mean = {valid_tr_lags.mean():+5.1f}h | Median = {np.median(valid_tr_lags):+5.1f}h | 25% = {np.percentile(valid_tr_lags, 25):+5.1f}h | 75% = {np.percentile(valid_tr_lags, 75):+5.1f}h")
    print(f"  • Proportion of top RI storms where Point B occurred: {(lag_df['point_b_count'] > 0).mean()*100:.1f}% ({sum(lag_df['point_b_count'] > 0)} / {len(lag_df)})")

    out_file = Path("experiments/forensics/ri_episodes_and_lag.json")
    with open(out_file, "w") as f:
        json.dump(episodes, f, indent=2)
    print(f"\nSaved detailed episodes and turning point tables to {out_file}")

if __name__ == "__main__":
    run()
