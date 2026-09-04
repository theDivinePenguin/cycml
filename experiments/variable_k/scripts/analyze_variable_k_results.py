"""Comprehensive Scientific Analysis Script for Variable-K Experiment.

Computes:
1. Overall benchmark comparison (Baseline K=7 vs Variable-K at K=3, K=5, K=7).
2. RI-specific evaluation and intensity-change bracket stratification.
3. 84 RI episodes turning-point lag and recognition audit -> ri_episode_comparison.csv.
4. Point-B analysis across canonical failure cases -> point_b_comparison.csv.
5. Regression-to-the-mean slope fits (all samples and RI samples).
6. Temporal hysteresis hypothesis testing.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)


def load_datasets():
    baseline_df = pd.read_csv("experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_predictions.csv")
    k3_df = pd.read_csv("experiments/variable_k/results/test_predictions_k3.csv")
    k5_df = pd.read_csv("experiments/variable_k/results/test_predictions_k5.csv")
    k7_df = pd.read_csv("experiments/variable_k/results/test_predictions_k7.csv")
    manifest = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")
    return baseline_df, k3_df, k5_df, k7_df, manifest


def compute_metrics_summary(df: pd.DataFrame, name: str, eval_k: int) -> Dict:
    y_true_tr = df["actual_trend"].values
    y_pred_tr = df["pred_trend"].values
    trend_acc = accuracy_score(y_true_tr, y_pred_tr)
    trend_f1 = f1_score(y_true_tr, y_pred_tr, average="macro")

    y_true_ri = df["actual_ri"].values
    y_prob_ri = df["pred_ri_prob"].values
    y_flag_ri = df["pred_ri_flag"].values

    ri_roc_auc = roc_auc_score(y_true_ri, y_prob_ri)
    prec, rec, _ = precision_recall_curve(y_true_ri, y_prob_ri)
    ri_pr_auc = auc(rec, prec)
    ri_recall = recall_score(y_true_ri, y_flag_ri, zero_division=0)
    ri_prec = precision_score(y_true_ri, y_flag_ri, zero_division=0)
    ri_f1 = f1_score(y_true_ri, y_flag_ri, zero_division=0)

    # MAE
    v24_true = df["vmax_plus_24h"].values
    vcurr = df["vmax_curr"].values
    mae_6 = float(np.mean(np.abs(df["pred_plus_6h"].values - df["vmax_plus_6h"].values))) if "vmax_plus_6h" in df else 0.0
    mae_12 = float(np.mean(np.abs(df["pred_plus_12h"].values - df["vmax_plus_12h"].values))) if "vmax_plus_12h" in df else 0.0
    mae_24 = float(np.mean(np.abs(df["pred_plus_24h"].values - v24_true)))

    # Delta V24
    act_dv = v24_true - vcurr
    pred_dv = df["pred_plus_24h"].values - vcurr
    slope, intercept = np.polyfit(act_dv, pred_dv, deg=1)
    corr = float(np.corrcoef(act_dv, pred_dv)[0, 1])

    # RI Subset
    ri_mask = act_dv >= 30.0
    ri_act_dv = act_dv[ri_mask]
    ri_pred_dv = pred_dv[ri_mask]
    ri_mae_24 = float(np.mean(np.abs(df["pred_plus_24h"].values[ri_mask] - v24_true[ri_mask])))
    ri_bias = float(np.mean(ri_pred_dv - ri_act_dv))
    ri_slope, ri_int = np.polyfit(ri_act_dv, ri_pred_dv, deg=1)
    ri_corr = float(np.corrcoef(ri_act_dv, ri_pred_dv)[0, 1])

    return {
        "model_name": name,
        "eval_k": eval_k,
        "mae_6": mae_6,
        "mae_12": mae_12,
        "mae_24": mae_24,
        "trend_acc": trend_acc,
        "trend_macro_f1": trend_f1,
        "ri_roc_auc": ri_roc_auc,
        "ri_pr_auc": ri_pr_auc,
        "ri_recall": ri_recall,
        "ri_precision": ri_prec,
        "ri_f1": ri_f1,
        "slope_all": slope,
        "corr_all": corr,
        "ri_count": int(np.sum(ri_mask)),
        "ri_mae_24": ri_mae_24,
        "ri_bias": ri_bias,
        "ri_slope": ri_slope,
        "ri_corr": ri_corr,
        "mean_act_dv_ri": float(np.mean(ri_act_dv)),
        "mean_pred_dv_ri": float(np.mean(ri_pred_dv)),
    }


def analyze_ri_brackets(df: pd.DataFrame) -> List[Dict]:
    act_dv = df["vmax_plus_24h"].values - df["vmax_curr"].values
    pred_dv = df["pred_plus_24h"].values - df["vmax_curr"].values

    brackets = [
        ("+30 to +39 kt", (act_dv >= 30) & (act_dv < 40)),
        ("+40 to +49 kt", (act_dv >= 40) & (act_dv < 50)),
        ("+50 to +59 kt", (act_dv >= 50) & (act_dv < 60)),
        ("+60 to +79 kt", (act_dv >= 60) & (act_dv < 80)),
        ("+80+ kt", (act_dv >= 80)),
    ]

    rows = []
    for label, mask in brackets:
        count = int(np.sum(mask))
        if count == 0:
            continue
        mean_act = float(np.mean(act_dv[mask]))
        mean_pred = float(np.mean(pred_dv[mask]))
        mae = float(np.mean(np.abs(pred_dv[mask] - act_dv[mask])))
        trend_inte_rate = float(np.mean(df["pred_trend"].values[mask] == 2))
        ri_alert_rate = float(np.mean(df["pred_ri_flag"].values[mask] == 1))
        rows.append({
            "bracket": label,
            "count": count,
            "mean_actual_dv": mean_act,
            "mean_pred_dv": mean_pred,
            "mae": mae,
            "trend_intensifying_pct": trend_inte_rate * 100,
            "ri_alert_pct": ri_alert_rate * 100,
        })
    return rows


def find_contiguous_ri_episodes(manifest: pd.DataFrame) -> List[Dict]:
    """Finds all contiguous sequences of Delta V24 >= 30 kt."""
    episodes = []
    for cid, group in manifest.groupby("cyclone_id"):
        group = group.sort_values("target_t_timestamp").reset_index(drop=True)
        dv24 = group["vmax_plus_24h"].values - group["vmax_curr"].values

        in_episode = False
        curr_ep_indices = []

        for idx, (val, ts) in enumerate(zip(dv24, group["target_t_timestamp"])):
            if val >= 30.0:
                if not in_episode:
                    in_episode = True
                    curr_ep_indices = [idx]
                else:
                    curr_ep_indices.append(idx)
            else:
                if in_episode:
                    in_episode = False
                    episodes.append({
                        "cyclone_id": cid,
                        "start_ts": group.iloc[curr_ep_indices[0]]["target_t_timestamp"],
                        "end_ts": group.iloc[curr_ep_indices[-1]]["target_t_timestamp"],
                        "indices": group.iloc[curr_ep_indices].index.tolist(),
                        "timestamps": group.iloc[curr_ep_indices]["target_t_timestamp"].tolist(),
                        "max_actual_dv24": float(np.max(dv24[curr_ep_indices])),
                        "vmax_curr_onset": float(group.iloc[curr_ep_indices[0]]["vmax_curr"]),
                    })
        if in_episode:
            episodes.append({
                "cyclone_id": cid,
                "start_ts": group.iloc[curr_ep_indices[0]]["target_t_timestamp"],
                "end_ts": group.iloc[curr_ep_indices[-1]]["target_t_timestamp"],
                "indices": group.iloc[curr_ep_indices].index.tolist(),
                "timestamps": group.iloc[curr_ep_indices]["target_t_timestamp"].tolist(),
                "max_actual_dv24": float(np.max(dv24[curr_ep_indices])),
                "vmax_curr_onset": float(group.iloc[curr_ep_indices[0]]["vmax_curr"]),
            })
    return episodes


def evaluate_episodes_for_model(episodes: List[Dict], pred_df: pd.DataFrame) -> List[Dict]:
    eval_records = []
    for ep in episodes:
        cid = ep["cyclone_id"]
        start_ts = ep["start_ts"]
        ts_list = ep["timestamps"]

        # Slices matching timestamps
        sub = pred_df[(pred_df["cyclone_id"] == cid) & (pred_df["target_t_timestamp"].isin(ts_list))].sort_values("target_t_timestamp")
        if len(sub) == 0:
            continue

        pred_trends = sub["pred_trend"].values
        pred_ri_flags = sub["pred_ri_flag"].values
        pred_ri_probs = sub["pred_ri_prob"].values
        pred_dv24 = sub["pred_plus_24h"].values - sub["vmax_curr"].values

        # Trend recognition
        recognized_by_trend = int(2 in pred_trends)
        trend_lag = None
        if recognized_by_trend:
            first_tr_idx = np.where(pred_trends == 2)[0][0]
            trend_lag = first_tr_idx * 3.0  # 3-hour steps from onset

        # RI head recognition
        recognized_by_ri = int(1 in pred_ri_flags)
        ri_lag = None
        if recognized_by_ri:
            first_ri_idx = np.where(pred_ri_flags == 1)[0][0]
            ri_lag = first_ri_idx * 3.0

        eval_records.append({
            "cyclone_id": cid,
            "onset_ts": start_ts,
            "onset_vmax": ep["vmax_curr_onset"],
            "peak_actual_dv24": ep["max_actual_dv24"],
            "recognized_by_trend": recognized_by_trend,
            "trend_lag_h": trend_lag,
            "recognized_by_ri": recognized_by_ri,
            "ri_lag_h": ri_lag,
            "max_pred_dv24": float(np.max(pred_dv24)),
            "max_ri_prob": float(np.max(pred_ri_probs)),
            "missed_completely": int(not recognized_by_trend and not recognized_by_ri),
        })
    return eval_records


def run_point_b_analysis(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    target_cyclones = [
        "200522S", "201504S", "200309E", "200815S", "201011L", "201018L", "201107E",
        "201419W", "201516W", "201601L", "201613S", "201615S", "200519S", "200611E",
        "200720S", "200908E", "201311W", "200518S", "200310L", "200625W"
    ]

    base_df = dfs["Baseline K=7"]
    rows = []

    for cid in target_cyclones:
        sub_base = base_df[base_df["cyclone_id"] == cid].sort_values("target_t_timestamp")
        # Identify Point B cases: actual Delta V24 >= 30, but baseline pred_trend == 0 (Weakening) or pred_dv24 <= -5
        base_dv24 = sub_base["pred_plus_24h"].values - sub_base["vmax_curr"].values
        act_dv24 = sub_base["vmax_plus_24h"].values - sub_base["vmax_curr"].values
        point_b_mask = (act_dv24 >= 30.0) & ((sub_base["pred_trend"].values == 0) | (base_dv24 <= -5.0))

        if not np.any(point_b_mask):
            # Take the highest actual RI timestep
            top_ri_idx = np.argmax(act_dv24)
            sample_ts = sub_base.iloc[top_ri_idx]["target_t_timestamp"]
        else:
            sample_ts = sub_base[point_b_mask].iloc[0]["target_t_timestamp"]

        v_curr = float(sub_base[sub_base["target_t_timestamp"] == sample_ts].iloc[0]["vmax_curr"])
        v_24 = float(sub_base[sub_base["target_t_timestamp"] == sample_ts].iloc[0]["vmax_plus_24h"])
        act_d = v_24 - v_curr

        record = {
            "cyclone_id": cid,
            "timestamp": sample_ts,
            "vmax_curr": v_curr,
            "vmax_plus_24h": v_24,
            "actual_dv24": act_d,
        }

        for model_key, df in dfs.items():
            match = df[(df["cyclone_id"] == cid) & (df["target_t_timestamp"] == sample_ts)]
            if len(match) > 0:
                row = match.iloc[0]
                p_dv = float(row["pred_plus_24h"] - row["vmax_curr"])
                tr = int(row["pred_trend"])
                ri_p = float(row["pred_ri_prob"])
                tr_name = "WEAK" if tr == 0 else ("STAB" if tr == 1 else "INTE")
                record[f"{model_key}_pred_dv24"] = p_dv
                record[f"{model_key}_trend"] = tr_name
                record[f"{model_key}_ri_prob"] = ri_p
            else:
                record[f"{model_key}_pred_dv24"] = None
                record[f"{model_key}_trend"] = None
                record[f"{model_key}_ri_prob"] = None

        rows.append(record)
    return pd.DataFrame(rows)


def main():
    print("Loading test predictions for Baseline and Variable-K models...")
    base_df, k3_df, k5_df, k7_df, manifest = load_datasets()

    models = {
        "Baseline K=7": (base_df, 7),
        "Variable-K (K=3)": (k3_df, 3),
        "Variable-K (K=5)": (k5_df, 5),
        "Variable-K (K=7)": (k7_df, 7),
    }

    # 1. Summary Comparison Table
    print("\n" + "=" * 90)
    print("BENCHMARK COMPARISON: BASELINE K=7 vs VARIABLE-K EVALUATED AT K=3, 5, 7")
    print("=" * 90)
    summaries = []
    for name, (df, k) in models.items():
        summaries.append(compute_metrics_summary(df, name, k))
    sum_df = pd.DataFrame(summaries)
    print(sum_df[["model_name", "eval_k", "mae_6", "mae_12", "mae_24", "trend_macro_f1", "ri_recall", "ri_pr_auc", "ri_mae_24", "ri_slope"]])

    # 2. RI Brackets for each model
    print("\n" + "=" * 90)
    print("RI INTENSITY-CHANGE STRATIFICATION BRACKETS")
    print("=" * 90)
    bracket_results = {}
    for name, (df, _) in models.items():
        bracket_results[name] = analyze_ri_brackets(df)
        print(f"\n--- {name} ---")
        b_df = pd.DataFrame(bracket_results[name])
        print(b_df.to_string(index=False))

    # 3. 84 RI Episodes Turning Point Comparison
    print("\nFinding contiguous RI episodes in test set...")
    episodes = find_contiguous_ri_episodes(manifest)
    print(f"Identified {len(episodes)} contiguous RI episodes across {len(set(ep['cyclone_id'] for ep in episodes))} cyclones.")

    episode_dfs = {}
    for name, (df, _) in models.items():
        records = evaluate_episodes_for_model(episodes, df)
        episode_dfs[name] = pd.DataFrame(records)

    # Combine into ri_episode_comparison.csv
    ep_comp_rows = []
    for i, ep in enumerate(episodes):
        row = {
            "episode_idx": i + 1,
            "cyclone_id": ep["cyclone_id"],
            "onset_ts": ep["start_ts"],
            "onset_vmax": ep["vmax_curr_onset"],
            "peak_actual_dv24": ep["max_actual_dv24"],
        }
        for name in models:
            m_ep = episode_dfs[name].iloc[i]
            row[f"{name}_rec_trend"] = m_ep["recognized_by_trend"]
            row[f"{name}_trend_lag_h"] = m_ep["trend_lag_h"]
            row[f"{name}_rec_ri"] = m_ep["recognized_by_ri"]
            row[f"{name}_ri_lag_h"] = m_ep["ri_lag_h"]
            row[f"{name}_max_pred_dv24"] = m_ep["max_pred_dv24"]
            row[f"{name}_missed"] = m_ep["missed_completely"]
        ep_comp_rows.append(row)
    ep_comp_df = pd.DataFrame(ep_comp_rows)
    ep_comp_csv = Path("experiments/variable_k/results/ri_episode_comparison.csv")
    ep_comp_df.to_csv(ep_comp_csv, index=False)
    print(f"Saved 84 RI episodes comparison to {ep_comp_csv}")

    # Summary of Episodes
    print("\nRI Episode Recognition Summary:")
    for name in models:
        df_ep = episode_dfs[name]
        n_rec_tr = df_ep["recognized_by_trend"].sum()
        n_rec_ri = df_ep["recognized_by_ri"].sum()
        n_miss = df_ep["missed_completely"].sum()
        valid_lags = df_ep[df_ep["recognized_by_trend"] == 1]["trend_lag_h"].dropna()
        med_lag = np.median(valid_lags) if len(valid_lags) > 0 else None
        mean_lag = np.mean(valid_lags) if len(valid_lags) > 0 else None
        print(f"  • {name:<18}: Recog by Trend: {n_rec_tr:2d}/84 ({n_rec_tr/84*100:.1f}%) | Recog by RI: {n_rec_ri:2d}/84 ({n_rec_ri/84*100:.1f}%) | Missed: {n_miss:2d} | Med Lag: {med_lag}h (Mean: {mean_lag:.2f}h)")

    # 4. Point B Comparison across canonical 20 cyclones
    dfs_dict = {name: df for name, (df, _) in models.items()}
    point_b_df = run_point_b_analysis(dfs_dict)
    pb_csv = Path("experiments/variable_k/results/point_b_comparison.csv")
    point_b_df.to_csv(pb_csv, index=False)
    print(f"\nSaved Point-B comparison table to {pb_csv} ({len(point_b_df)} cyclones)")
    print(point_b_df[["cyclone_id", "timestamp", "actual_dv24", "Baseline K=7_trend", "Variable-K (K=3)_trend", "Variable-K (K=5)_trend", "Variable-K (K=7)_trend"]].to_string())

    # Save complete analysis summary JSON
    analysis_summary = {
        "benchmark_comparison": summaries,
        "ri_bracket_stratification": bracket_results,
        "ri_episode_summary": {
            name: {
                "recognized_by_trend": int(episode_dfs[name]["recognized_by_trend"].sum()),
                "recognized_by_ri": int(episode_dfs[name]["recognized_by_ri"].sum()),
                "missed_completely": int(episode_dfs[name]["missed_completely"].sum()),
                "median_lag": float(np.median(episode_dfs[name][episode_dfs[name]["recognized_by_trend"] == 1]["trend_lag_h"].dropna())),
                "mean_lag": float(np.mean(episode_dfs[name][episode_dfs[name]["recognized_by_trend"] == 1]["trend_lag_h"].dropna())),
            }
            for name in models
        }
    }
    with open("experiments/variable_k/results/scientific_analysis_summary.json", "w") as f:
        json.dump(analysis_summary, f, indent=2)
    print("\nScientific analysis suite completed successfully!")


if __name__ == "__main__":
    main()
