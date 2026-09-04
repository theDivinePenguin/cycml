"""Export multi-model cyclone forecasts for the interactive test clone with dynamic model switching."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.trend_config import IntensityTrendConfig


def clean_ts(val):
    m = re.search(r'(\d+)', str(val))
    return int(m.group(1)) if m else int(val)


MODELS_META = [
    {
        "id": "exp2_ultra",
        "name": "Exp 2: Ultra (1 / 6 / 12 Weights)",
        "badge": "Peak RI PR-AUC (0.419)",
        "tag": "Recommended",
        "lead_mae": "+6h: 3.47 kt · +12h: 6.18 kt · +24h: 10.84 kt",
        "ri_mae": "24.02 kt (-2.7 kt vs Baseline)",
        "ri_precision": "36.62% · Recall 51.93%",
        "slope": "Overall Slope: 0.629 · RI Slope: 0.083",
        "pred_csv": "experiments/ri_target_loss/results/exp2_delta_1_6_12/test_predictions.csv",
        "is_delta": True,
    },
    {
        "id": "exp2_strong",
        "name": "Exp 2: Strong (1 / 3 / 6 Weights)",
        "badge": "Highest RI Slope (0.088)",
        "tag": "High Responsiveness",
        "lead_mae": "+6h: 3.55 kt · +12h: 6.36 kt · +24h: 10.97 kt",
        "ri_mae": "27.55 kt",
        "ri_precision": "38.91% · Recall 48.80%",
        "slope": "Overall Slope: 0.590 · RI Slope: 0.088",
        "pred_csv": "experiments/ri_target_loss/results/exp2_delta_strong/test_predictions.csv",
        "is_delta": True,
    },
    {
        "id": "exp2_moderate",
        "name": "Exp 2: Moderate (1 / 2 / 4 Weights)",
        "badge": "Best 3-Horizon MAE (6.73 kt)",
        "tag": "Balanced",
        "lead_mae": "+6h: 3.46 kt · +12h: 6.13 kt · +24h: 10.59 kt",
        "ri_mae": "26.97 kt",
        "ri_precision": "37.59% · Recall 49.36%",
        "slope": "Overall Slope: 0.582 · RI Slope: 0.068",
        "pred_csv": "experiments/ri_target_loss/results/exp2_delta_moderate/test_predictions.csv",
        "is_delta": True,
    },
    {
        "id": "exp1_delta",
        "name": "Exp 1B: Delta-Only (Unweighted 1/1/1)",
        "badge": "Highest Precision (40.9%)",
        "tag": "Unweighted",
        "lead_mae": "+6h: 3.50 kt · +12h: 6.23 kt · +24h: 10.75 kt",
        "ri_mae": "28.60 kt",
        "ri_precision": "40.88% · Recall 46.22%",
        "slope": "Overall Slope: 0.555 · RI Slope: 0.030",
        "pred_csv": "experiments/ri_target_loss/results/exp1_delta_only/test_predictions.csv",
        "is_delta": True,
    },
    {
        "id": "baseline",
        "name": "Baseline Clean K=7 (Absolute Formulation)",
        "badge": "Production Benchmark",
        "tag": "Baseline",
        "lead_mae": "+6h: 4.98 kt · +12h: 6.99 kt · +24h: 10.75 kt",
        "ri_mae": "26.68 kt",
        "ri_precision": "30.10% · Recall 51.20%",
        "slope": "Overall Slope: 0.580 · RI Slope: 0.080",
        "pred_csv": "experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_predictions.csv",
        "is_delta": False,
    },
]


def export_all_models_data(
    out_json_paths=("frontend_test_clone/public/storm_data.json", "frontend_test_clone/src/data/storm_data.json"),
):
    config = IntensityTrendConfig()
    meta_dir = Path("data/metadata")
    test_df = pd.read_csv(meta_dir / "forecast_test_sequences_k7.csv")
    val_df = pd.read_csv(meta_dir / "forecast_val_sequences_k7.csv")

    env_test_df = pd.read_csv(meta_dir / "environmental_cache_k7_test.csv") if (meta_dir / "environmental_cache_k7_test.csv").exists() else None
    env_val_df = pd.read_csv(meta_dir / "environmental_cache_k7_val.csv") if (meta_dir / "environmental_cache_k7_val.csv").exists() else None

    # Load all prediction CSVs
    model_dfs = {}
    for m in MODELS_META:
        p = Path(m["pred_csv"])
        if p.exists():
            df = pd.read_csv(p)
            if "target_t_timestamp" in df.columns:
                df["clean_ts"] = df["target_t_timestamp"].apply(clean_ts)
            model_dfs[m["id"]] = df
            print(f"Loaded {len(df)} predictions for model {m['id']} from {p}")
        else:
            print(f"Warning: {p} does not exist yet, skipping.")

    # 14 Showcase storms
    storms_catalog = [
        {"id": "201015W", "name": "Super Typhoon Megi", "basin": "West Pacific (WPAC)", "peak_intensity": 160, "category": "Category 5 Super Typhoon", "split": "Held-Out Test Set", "source_df": test_df, "description": "Explosive Category 5 Super Typhoon that underwent extreme rapid intensification (+95 kt over 24h) in the Philippine Sea."},
        {"id": "201614L", "name": "Hurricane Matthew", "basin": "North Atlantic (ATLN)", "peak_intensity": 145, "category": "Category 5 Major Hurricane", "split": "Held-Out Test Set", "source_df": test_df, "description": "Unprecedented low-latitude Atlantic Category 5 hurricane that rapidly intensified by 70 kt in 24 hours."},
        {"id": "201003I", "name": "Super Cyclone Phet", "basin": "Arabian Sea / North Indian Ocean (IO)", "peak_intensity": 125, "category": "Category 4 Super Cyclonic Storm", "split": "Held-Out Test Set", "source_df": test_df, "description": "Historic Arabian Sea Super Cyclone exhibiting rapid core consolidation prior to landfall in Oman."},
        {"id": "200801I", "name": "VSCS Nargis", "basin": "Bay of Bengal / North Indian Ocean (IO)", "peak_intensity": 115, "category": "Category 4 Very Severe Cyclonic Storm", "split": "Held-Out Test Set", "source_df": test_df, "description": "Devastating North Indian Ocean cyclone that intensified rapidly before making landfall in the Ayeyarwady Delta."},
        {"id": "200413E", "name": "Hurricane Javier", "basin": "East Pacific (EPAC)", "peak_intensity": 130, "category": "Category 4 Major Hurricane", "split": "Held-Out Test Set", "source_df": test_df, "description": "Classic East Pacific rapid intensifier achieving 130 kt maximum sustained winds over warm equatorial SSTs."},
        {"id": "200519S", "name": "Cyclone Percy", "basin": "South Pacific / Southern Hemisphere (SH)", "peak_intensity": 145, "category": "Category 5 Tropical Cyclone", "split": "Held-Out Test Set", "source_df": test_df, "description": "High-end Category 5 Southern Hemisphere cyclone exhibiting symmetric annular eye eyewall thermodynamics."},
        {"id": "201004I", "name": "Super Cyclone Giri", "basin": "Bay of Bengal (IO)", "peak_intensity": 135, "category": "Category 4 Super Cyclonic Storm", "split": "Validation Split", "source_df": val_df, "description": "Extremely rapid intensifier (+70 kt in 24h) striking Myanmar as an intense Category 4 system."},
        {"id": "201419W", "name": "Super Typhoon Vongfong", "basin": "West Pacific (WPAC)", "peak_intensity": 155, "category": "Category 5 Super Typhoon", "split": "Held-Out Test Set", "source_df": test_df, "description": "Most intense tropical cyclone of 2014 globally, peaking at 155 kt with a distinct 40 km pinhole eye."},
        {"id": "200419W", "name": "Super Typhoon Chaba", "basin": "West Pacific (WPAC)", "peak_intensity": 155, "category": "Category 5 Super Typhoon", "split": "Held-Out Test Set", "source_df": test_df, "description": "Massive Category 5 system tracked across 101 continuous observations with explosive convective eyewall deepening."},
        {"id": "201011L", "name": "Hurricane Igor", "basin": "North Atlantic (ATLN)", "peak_intensity": 135, "category": "Category 4 Major Hurricane", "split": "Held-Out Test Set", "source_df": test_df, "description": "Colossal Cape Verde hurricane that intensified rapidly across exceptionally warm subtropical Atlantic waters."},
        {"id": "201404S", "name": "Cyclone Bruce", "basin": "South Indian Ocean (SH)", "peak_intensity": 140, "category": "Category 5 Tropical Cyclone", "split": "Held-Out Test Set", "source_df": test_df, "description": "Classic annular Category 5 Southern Indian Ocean cyclone maintaining a perfectly circular eyewall against dry air."},
        {"id": "200522S", "name": "Cyclone Ingrid", "basin": "Northern Australia / Coral Sea (SH)", "peak_intensity": 135, "category": "Category 5 Severe Tropical Cyclone", "split": "Held-Out Test Set", "source_df": test_df, "description": "Historic severe tropical cyclone that struck three Australian coastal territories at intense category strength."},
        {"id": "201104W", "name": "Super Typhoon Songda", "basin": "West Pacific (WPAC)", "peak_intensity": 140, "category": "Category 5 Super Typhoon", "split": "Held-Out Test Set", "source_df": test_df, "description": "Early-season May super typhoon exhibiting violent rapid deepening (+50 kt in 24h) east of the Philippines."},
        {"id": "201305I", "name": "VSCS Lehar", "basin": "Bay of Bengal / North Indian Ocean (IO)", "peak_intensity": 75, "category": "Category 1 Very Severe Cyclonic Storm", "split": "Held-Out Test Set", "source_df": test_df, "description": "Prominent North Indian Ocean system tracked continuously across 59 steps from the Andaman Sea toward Andhra Pradesh."},
    ]

    storms_by_model = {}

    for model_meta in MODELS_META:
        m_id = model_meta["id"]
        pred_df = model_dfs.get(m_id)
        is_delta = model_meta["is_delta"]

        model_storm_map = {}

        for storm_meta in storms_catalog:
            cid = storm_meta["id"]
            source_df = storm_meta["source_df"]
            sub_df = source_df[source_df["cyclone_id"] == cid].copy().sort_values("target_t_timestamp").reset_index(drop=True)

            if len(sub_df) == 0:
                continue

            timesteps = []
            for step_idx, row in sub_df.iterrows():
                ts = str(row["target_t_timestamp"])
                v_curr = float(row["vmax_curr"])
                v_plus_24 = float(row["vmax_plus_24h"])
                actual_delta_24 = v_plus_24 - v_curr
                actual_trend_name = config.get_trend_name(config.compute_trend_label(actual_delta_24))
                actual_ri_flag = 1 if actual_delta_24 >= config.ri_threshold_kt else 0

                pred_row = None
                if pred_df is not None:
                    p_matches = pred_df[(pred_df["cyclone_id"] == cid) & (pred_df["clean_ts"] == int(ts))]
                    if len(p_matches) > 0:
                        pred_row = p_matches.iloc[0]

                if pred_row is not None:
                    ri_prob = float(pred_row["pred_ri_prob"])
                    pred_trend_idx = int(pred_row["pred_trend"])
                    pred_trend_name = config.get_trend_name(pred_trend_idx)
                    prob_weak = float(pred_row["prob_weakening"])
                    prob_stab = float(pred_row["prob_stable"])
                    prob_inte = float(pred_row["prob_intensifying"])

                    if is_delta and "recon_plus_6h" in pred_row:
                        p_plus_6 = round(float(pred_row["recon_plus_6h"]), 1)
                        p_plus_12 = round(float(pred_row["recon_plus_12h"]), 1)
                        p_plus_24 = round(float(pred_row["recon_plus_24h"]), 1)
                    else:
                        p_plus_6 = round(float(pred_row["pred_plus_6h"]), 1)
                        p_plus_12 = round(float(pred_row["pred_plus_12h"]), 1)
                        p_plus_24 = round(float(pred_row["pred_plus_24h"]), 1)
                else:
                    # Fallback to persistence
                    hist_v = json.loads(row["history_vmax"]) if isinstance(row["history_vmax"], str) else row["history_vmax"]
                    d_6h = hist_v[4] - hist_v[2]
                    extrap = d_6h * 4.0
                    ri_prob = float(1.0 / (1.0 + np.exp(-(extrap - 30.0) / 10.0)))
                    pred_trend_idx = config.compute_trend_label(extrap)
                    pred_trend_name = config.get_trend_name(pred_trend_idx)
                    prob_weak = 0.8 if pred_trend_idx == 0 else 0.1
                    prob_stab = 0.8 if pred_trend_idx == 1 else 0.1
                    prob_inte = 0.8 if pred_trend_idx == 2 else 0.1
                    p_plus_6 = round(v_curr + d_6h, 1)
                    p_plus_12 = round(v_curr + d_6h * 2, 1)
                    p_plus_24 = round(v_curr + extrap, 1)

                risk_level = config.get_ri_risk_level(ri_prob)

                if v_curr < 34:
                    cat_name = "Tropical Depression (TD)"
                elif v_curr < 64:
                    cat_name = "Tropical Storm (TS)"
                elif v_curr < 83:
                    cat_name = "Category 1 Hurricane / Cyclone"
                elif v_curr < 96:
                    cat_name = "Category 2 Hurricane / Cyclone"
                elif v_curr < 113:
                    cat_name = "Category 3 Major Hurricane"
                elif v_curr < 137:
                    cat_name = "Category 4 Major Hurricane"
                else:
                    cat_name = "Category 5 Super Typhoon / Hurricane"

                env_df = env_test_df if storm_meta["split"] == "Held-Out Test Set" else env_val_df
                env_row = None
                if env_df is not None:
                    e_matches = env_df[(env_df["cyclone_id"] == cid) & (env_df["timestamp"] == int(ts))]
                    if len(e_matches) > 0:
                        env_row = e_matches.iloc[0]

                sst_val = round(float(env_row["sst"]), 1) if env_row is not None and pd.notnull(env_row.get("sst")) else 28.5
                ohc_val = round(float(env_row["cohc"]), 1) if env_row is not None and pd.notnull(env_row.get("cohc")) else 45.0
                shear_val = round(float(env_row["shrd"]), 1) if env_row is not None and pd.notnull(env_row.get("shrd")) else 12.0
                rh_val = round(float(env_row["rhmd"]), 1) if env_row is not None and pd.notnull(env_row.get("rhmd")) else 65.0
                mslp_val = round(float(env_row["mslp"]), 1) if env_row is not None and pd.notnull(env_row.get("mslp")) else 990.0

                hist_ts = json.loads(row["history_timestamps"]) if isinstance(row["history_timestamps"], str) else row["history_timestamps"]
                hist_v = json.loads(row["history_vmax"]) if isinstance(row["history_vmax"], str) else row["history_vmax"]

                timesteps.append({
                    "step_index": step_idx,
                    "timestamp": ts,
                    "elapsed_hours": step_idx * 3.0,
                    "vmax_curr": v_curr,
                    "vmax_plus_24h": v_plus_24,
                    "actual_delta_24": round(actual_delta_24, 1),
                    "actual_trend": actual_trend_name,
                    "actual_ri": actual_ri_flag,
                    "category": cat_name,
                    "predicted_trend": pred_trend_name,
                    "predicted_trend_probs": {
                        "WEAKENING": round(prob_weak, 3),
                        "STABLE": round(prob_stab, 3),
                        "INTENSIFYING": round(prob_inte, 3),
                    },
                    "ri_probability": round(ri_prob * 100.0, 1),
                    "risk_level": risk_level,
                    "predicted_plus_6h": p_plus_6,
                    "predicted_plus_12h": p_plus_12,
                    "predicted_plus_24h": p_plus_24,
                    "latitude": float(row.get("latitude", 0.0)),
                    "longitude": float(row.get("longitude", 0.0)),
                    "environmental": {
                        "sst": sst_val,
                        "ohc": ohc_val,
                        "shear": shear_val,
                        "rh": rh_val,
                        "mslp": mslp_val,
                    },
                    "history_frames": [
                        {"offset": "-18h", "timestamp": str(hist_ts[0]), "vmax": float(hist_v[0])},
                        {"offset": "-15h", "timestamp": str(hist_ts[1]), "vmax": float(hist_v[1])},
                        {"offset": "-12h", "timestamp": str(hist_ts[2]), "vmax": float(hist_v[2])},
                        {"offset": "-9h",  "timestamp": str(hist_ts[3]), "vmax": float(hist_v[3])},
                        {"offset": "-6h",  "timestamp": str(hist_ts[4]), "vmax": float(hist_v[4])},
                        {"offset": "-3h",  "timestamp": str(hist_ts[5]), "vmax": float(hist_v[5])},
                        {"offset": "NOW",  "timestamp": str(hist_ts[6]), "vmax": float(hist_v[6])},
                    ],
                })

            model_storm_map[cid] = {
                "id": cid,
                "name": storm_meta["name"],
                "basin": storm_meta["basin"],
                "peak_intensity": storm_meta["peak_intensity"],
                "category": storm_meta["category"],
                "split": storm_meta["split"],
                "description": storm_meta["description"],
                "n_timesteps": len(timesteps),
                "timesteps": timesteps,
            }

        storms_by_model[m_id] = model_storm_map

    export_payload = {
        "models": [m for m in MODELS_META if m["id"] in model_dfs],
        "storms": storms_by_model,
    }

    for p in out_json_paths:
        path = Path(p)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)
        print(f"Exported multi-model dataset to {path}")

    return export_payload


if __name__ == "__main__":
    export_all_models_data()
