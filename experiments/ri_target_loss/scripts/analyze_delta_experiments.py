"""Comprehensive Scientific Analysis Script for RI Target Loss and Delta Experiments.

Compares:
- Baseline Clean K=7
- Experiment 1A: Absolute + Delta Head
- Experiment 1B: Delta-Only Head
- Experiment 2: RI-Aware Weighted Delta Losses (if present)

Evaluates:
1. Overall metrics (+6, +12, +24 MAE, RMSE, R²)
2. RI-specific metrics (PR-AUC, ROC-AUC, Recall, Precision, F1, RI MAE, RI Bias, RI Slope, RI Corr)
3. All-sample intensity change slope and bias
4. Turning-point diagnostics on canonical failure cases (200522S, 201504S, 201018L, 201516W, 201601L, etc.)
5. 84 contiguous RI episodes recognition audit
6. Tradeoff analysis: RI amplitude accuracy vs non-RI forecast error
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def load_manifest():
    return pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")


def evaluate_dataframe_predictions(df: pd.DataFrame, name: str, pred_v24_col: str, pred_dv24_col: str) -> Dict:
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

    # MAE and RMSE
    v24_true = df["vmax_plus_24h"].values
    vcurr = df["vmax_curr"].values
    v24_pred = df[pred_v24_col].values

    mae_24 = float(np.mean(np.abs(v24_pred - v24_true)))
    rmse_24 = float(np.sqrt(np.mean((v24_pred - v24_true) ** 2)))
    r2_24 = float(r2_score(v24_true, v24_pred))

    # All-samples Delta V24
    act_dv = v24_true - vcurr
    pred_dv = df[pred_dv24_col].values
    slope_all, int_all = np.polyfit(act_dv, pred_dv, deg=1)
    corr_all = float(np.corrcoef(act_dv, pred_dv)[0, 1])

    # RI Subset (act_dv >= 30)
    ri_mask = act_dv >= 30.0
    ri_act_dv = act_dv[ri_mask]
    ri_pred_dv = pred_dv[ri_mask]
    ri_mae_24 = float(np.mean(np.abs(v24_pred[ri_mask] - v24_true[ri_mask])))
    ri_bias = float(np.mean(ri_pred_dv - ri_act_dv))
    ri_slope, ri_int = np.polyfit(ri_act_dv, ri_pred_dv, deg=1)
    ri_corr = float(np.corrcoef(ri_act_dv, ri_pred_dv)[0, 1])

    # Non-RI Subset (act_dv < 30)
    non_ri_mask = ~ri_mask
    non_ri_mae_24 = float(np.mean(np.abs(v24_pred[non_ri_mask] - v24_true[non_ri_mask])))

    return {
        "model_name": name,
        "trend_acc": float(trend_acc),
        "trend_macro_f1": float(trend_f1),
        "ri_roc_auc": float(ri_roc_auc),
        "ri_pr_auc": float(ri_pr_auc),
        "ri_recall": float(ri_recall),
        "ri_precision": float(ri_prec),
        "ri_f1": float(ri_f1),
        "overall_mae_24": mae_24,
        "overall_rmse_24": rmse_24,
        "overall_r2_24": r2_24,
        "non_ri_mae_24": non_ri_mae_24,
        "ri_count": int(np.sum(ri_mask)),
        "ri_mae_24": ri_mae_24,
        "ri_bias": ri_bias,
        "ri_slope": float(ri_slope),
        "ri_corr": float(ri_corr),
        "mean_act_dv_ri": float(np.mean(ri_act_dv)),
        "mean_pred_dv_ri": float(np.mean(ri_pred_dv)),
        "slope_all": float(slope_all),
        "corr_all": float(corr_all),
        "overall_bias": float(np.mean(pred_dv - act_dv)),
    }


def find_contiguous_ri_episodes(manifest: pd.DataFrame) -> List[Dict]:
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
                        "timestamps": group.iloc[curr_ep_indices]["target_t_timestamp"].tolist(),
                        "max_actual_dv24": float(np.max(dv24[curr_ep_indices])),
                        "vmax_curr_onset": float(group.iloc[curr_ep_indices[0]]["vmax_curr"]),
                    })
        if in_episode:
            episodes.append({
                "cyclone_id": cid,
                "start_ts": group.iloc[curr_ep_indices[0]]["target_t_timestamp"],
                "end_ts": group.iloc[curr_ep_indices[-1]]["target_t_timestamp"],
                "timestamps": group.iloc[curr_ep_indices]["target_t_timestamp"].tolist(),
                "max_actual_dv24": float(np.max(dv24[curr_ep_indices])),
                "vmax_curr_onset": float(group.iloc[curr_ep_indices[0]]["vmax_curr"]),
            })
    return episodes


def evaluate_episodes_for_df(episodes: List[Dict], pred_df: pd.DataFrame, pred_dv_col: str) -> List[Dict]:
    eval_records = []
    for ep in episodes:
        cid = ep["cyclone_id"]
        ts_list = ep["timestamps"]
        sub = pred_df[(pred_df["cyclone_id"] == cid) & (pred_df["target_t_timestamp"].isin(ts_list))].sort_values("target_t_timestamp")
        if len(sub) == 0:
            continue

        pred_trends = sub["pred_trend"].values
        pred_ri_flags = sub["pred_ri_flag"].values
        pred_ri_probs = sub["pred_ri_prob"].values
        pred_dv24 = sub[pred_dv_col].values

        recognized_by_trend = int(2 in pred_trends)
        trend_lag = None
        if recognized_by_trend:
            first_tr_idx = np.where(pred_trends == 2)[0][0]
            trend_lag = first_tr_idx * 3.0

        recognized_by_ri = int(1 in pred_ri_flags)
        ri_lag = None
        if recognized_by_ri:
            first_ri_idx = np.where(pred_ri_flags == 1)[0][0]
            ri_lag = first_ri_idx * 3.0

        eval_records.append({
            "cyclone_id": cid,
            "onset_ts": ep["start_ts"],
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


def run_point_b_diagnostics(models_dict: Dict[str, Tuple[pd.DataFrame, str, str]]) -> pd.DataFrame:
    target_cyclones = [
        "200522S", "201504S", "200309E", "200815S", "201011L", "201018L", "201107E",
        "201419W", "201516W", "201601L", "201613S", "201615S", "200519S", "200611E",
        "200720S", "200908E", "201311W", "200518S", "200310L", "200625W"
    ]

    base_df, _, base_dv_col = models_dict["Baseline Clean K=7"]
    rows = []

    for cid in target_cyclones:
        sub_base = base_df[base_df["cyclone_id"] == cid].sort_values("target_t_timestamp")
        act_dv24 = sub_base["vmax_plus_24h"].values - sub_base["vmax_curr"].values
        base_dv24 = sub_base[base_dv_col].values
        point_b_mask = (act_dv24 >= 30.0) & ((sub_base["pred_trend"].values == 0) | (base_dv24 <= -5.0))

        if not np.any(point_b_mask):
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
            "actual_plus_24h": v_24,
            "actual_dv24": act_d,
        }

        for model_key, (df, p_v24_col, p_dv_col) in models_dict.items():
            match = df[(df["cyclone_id"] == cid) & (df["target_t_timestamp"] == sample_ts)]
            if len(match) > 0:
                row = match.iloc[0]
                p_dv = float(row[p_dv_col])
                p_v = float(row[p_v24_col])
                tr = int(row["pred_trend"])
                ri_p = float(row["pred_ri_prob"])
                tr_name = "WEAK" if tr == 0 else ("STAB" if tr == 1 else "INTE")
                record[f"{model_key}_pred_dv24"] = p_dv
                record[f"{model_key}_pred_v24"] = p_v
                record[f"{model_key}_trend"] = tr_name
                record[f"{model_key}_ri_prob"] = ri_p
            else:
                record[f"{model_key}_pred_dv24"] = None
                record[f"{model_key}_pred_v24"] = None
                record[f"{model_key}_trend"] = None
                record[f"{model_key}_ri_prob"] = None

        rows.append(record)
    return pd.DataFrame(rows)


def main():
    manifest = load_manifest()

    # Load available predictions
    models = {}

    def clean_df(d):
        if "target_t_timestamp" in d.columns:
            d["target_t_timestamp"] = d["target_t_timestamp"].astype(str).str.extract(r"(\d+)")[0].astype(np.int64)
        return d

    # 1. Baseline
    base_csv = Path("experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_predictions.csv")
    if base_csv.exists():
        base_df = clean_df(pd.read_csv(base_csv))
        base_df["pred_dv24"] = base_df["pred_plus_24h"] - base_df["vmax_curr"]
        models["Baseline Clean K=7"] = (base_df, "pred_plus_24h", "pred_dv24")

    # 2. Exp 1A: Absolute + Delta
    exp1a_csv = Path("experiments/ri_target_loss/results/exp1_abs_delta/test_predictions.csv")
    if exp1a_csv.exists():
        exp1a_df = clean_df(pd.read_csv(exp1a_csv))
        models["Exp 1A: Abs+Delta (Recon)"] = (exp1a_df, "recon_plus_24h", "pred_delta_24h")
        if "direct_plus_24h" in exp1a_df:
            exp1a_df["direct_dv24"] = exp1a_df["direct_plus_24h"] - exp1a_df["vmax_curr"]
            models["Exp 1A: Abs+Delta (Direct)"] = (exp1a_df, "direct_plus_24h", "direct_dv24")

    # 3. Exp 1B: Delta-Only
    exp1b_csv = Path("experiments/ri_target_loss/results/exp1_delta_only/test_predictions.csv")
    if exp1b_csv.exists():
        exp1b_df = clean_df(pd.read_csv(exp1b_csv))
        models["Exp 1B: Delta-Only (Recon)"] = (exp1b_df, "recon_plus_24h", "pred_delta_24h")

    # 4. Exp 2 Profiles
    for prof in ["moderate", "strong", "very_strong"]:
        for prefix in ["exp2_delta_", "exp2_weighted_"]:
            prof_csv = Path(f"experiments/ri_target_loss/results/{prefix}{prof}/test_predictions.csv")
            if prof_csv.exists():
                prof_df = clean_df(pd.read_csv(prof_csv))
                models[f"Exp 2: Weighted ({prof})"] = (prof_df, "recon_plus_24h", "pred_delta_24h")
                break

    print(f"Loaded {len(models)} models for comparative analysis.")

    # 1. Summary Metrics Table
    summaries = []
    for name, (df, p_v_col, p_dv_col) in models.items():
        res = evaluate_dataframe_predictions(df, name, p_v_col, p_dv_col)
        summaries.append(res)
    sum_df = pd.DataFrame(summaries)
    print("\n" + "=" * 105)
    print("COMPARATIVE EVALUATION SUMMARY ACROSS EXPERIMENTAL VARIANTS")
    print("=" * 105)
    print(sum_df[["model_name", "overall_mae_24", "non_ri_mae_24", "ri_mae_24", "ri_recall", "ri_pr_auc", "ri_bias", "ri_slope", "slope_all"]].to_string(index=False))

    # Save summary table
    out_dir = Path("experiments/ri_target_loss/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    sum_df.to_csv(out_dir / "comparative_evaluation_summary.csv", index=False)

    # 2. 84 RI Episodes Audit
    episodes = find_contiguous_ri_episodes(manifest)
    episode_summaries = {}
    for name, (df, _, p_dv_col) in models.items():
        ep_records = evaluate_episodes_for_df(episodes, df, p_dv_col)
        ep_df = pd.DataFrame(ep_records)
        n_rec_tr = int(ep_df["recognized_by_trend"].sum())
        n_rec_ri = int(ep_df["recognized_by_ri"].sum())
        n_miss = int(ep_df["missed_completely"].sum())
        valid_lags = ep_df[ep_df["recognized_by_trend"] == 1]["trend_lag_h"].dropna()
        med_lag = float(np.median(valid_lags)) if len(valid_lags) > 0 else 0.0
        mean_lag = float(np.mean(valid_lags)) if len(valid_lags) > 0 else 0.0
        episode_summaries[name] = {
            "rec_trend": n_rec_tr,
            "rec_ri": n_rec_ri,
            "missed": n_miss,
            "med_lag": med_lag,
            "mean_lag": mean_lag,
        }

    print("\n" + "=" * 90)
    print("84 CONTIGUOUS RI EPISODES RECOGNITION AUDIT")
    print("=" * 90)
    for name, s in episode_summaries.items():
        print(f"  • {name:<30}: Trend Rec: {s['rec_trend']}/84 ({s['rec_trend']/84*100:.1f}%) | RI Rec: {s['rec_ri']}/84 ({s['rec_ri']/84*100:.1f}%) | Missed: {s['missed']:2d} | Med Lag: {s['med_lag']:.1f}h")

    # 3. Point B Diagnostics
    if "Baseline Clean K=7" in models:
        pb_df = run_point_b_diagnostics(models)
        pb_df.to_csv(out_dir / "point_b_comparison.csv", index=False)
        print("\nSaved Point-B comparison table to", out_dir / "point_b_comparison.csv")


if __name__ == "__main__":
    main()
