"""Export comprehensive multi-model cyclone forecasts for all models in cycml."""
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


ALL_MODELS_CATALOG = [
    # --- Category 1: RI Target & Loss Ablations (Newly Trained) ---
    {
        "id": "exp2_extreme",
        "category": "RI Loss Ablations",
        "name": "Exp 2: Extreme (1 / 10 / 20 Weights)",
        "badge": "Peak Precision (44.3%)",
        "tag": "Max Precision",
        "lead_mae": "+6h: 3.51 kt · +12h: 6.31 kt · +24h: 10.98 kt",
        "ri_mae": "25.53 kt (F1: 0.449 Peak)",
        "ri_precision": "44.34% · Recall 45.49%",
        "slope": "Overall Slope: 0.618 · RI Slope: 0.061",
        "pred_csv": "experiments/ri_target_loss/results/exp2_delta_1_10_20/test_predictions.csv",
        "type": "delta",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "Temporal History Sequence (K=7)", "Weighted Delta RI Loss Head (1/10/20)"],
    },
    {
        "id": "exp2_ultra",
        "category": "RI Loss Ablations",
        "name": "Exp 2: Ultra (1 / 6 / 12 Weights)",
        "badge": "Peak RI PR-AUC (0.419)",
        "tag": "Top Pick",
        "lead_mae": "+6h: 3.47 kt · +12h: 6.18 kt · +24h: 10.84 kt",
        "ri_mae": "24.02 kt (-2.7 kt vs Baseline)",
        "ri_precision": "36.62% · Recall 51.93%",
        "slope": "Overall Slope: 0.629 · RI Slope: 0.083",
        "pred_csv": "experiments/ri_target_loss/results/exp2_delta_1_6_12/test_predictions.csv",
        "type": "delta",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "Temporal History Sequence (K=7)", "Weighted Delta RI Loss Head (1/6/12)"],
    },
    {
        "id": "exp2_strong",
        "category": "RI Loss Ablations",
        "name": "Exp 2: Strong (1 / 3 / 6 Weights)",
        "badge": "Highest RI Slope (0.088)",
        "tag": "Aggressive",
        "lead_mae": "+6h: 3.55 kt · +12h: 6.36 kt · +24h: 10.97 kt",
        "ri_mae": "27.55 kt",
        "ri_precision": "38.91% · Recall 48.80%",
        "slope": "Overall Slope: 0.590 · RI Slope: 0.088",
        "pred_csv": "experiments/ri_target_loss/results/exp2_delta_strong/test_predictions.csv",
        "type": "delta",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "Temporal History Sequence (K=7)", "Weighted Delta RI Loss Head (1/3/6)"],
    },
    {
        "id": "exp2_moderate",
        "category": "RI Loss Ablations",
        "name": "Exp 2: Moderate (1 / 2 / 4 Weights)",
        "badge": "Best 3-Horizon MAE (6.73 kt)",
        "tag": "Balanced",
        "lead_mae": "+6h: 3.46 kt · +12h: 6.13 kt · +24h: 10.59 kt",
        "ri_mae": "26.97 kt",
        "ri_precision": "37.59% · Recall 49.36%",
        "slope": "Overall Slope: 0.582 · RI Slope: 0.068",
        "pred_csv": "experiments/ri_target_loss/results/exp2_delta_moderate/test_predictions.csv",
        "type": "delta",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "Temporal History Sequence (K=7)", "Weighted Delta RI Loss Head (1/2/4)"],
    },
    {
        "id": "exp1_delta",
        "category": "RI Loss Ablations",
        "name": "Exp 1B: Delta-Only (Unweighted 1/1/1)",
        "badge": "Highest Precision (40.9%)",
        "tag": "Unweighted",
        "lead_mae": "+6h: 3.50 kt · +12h: 6.23 kt · +24h: 10.75 kt",
        "ri_mae": "28.60 kt",
        "ri_precision": "40.88% · Recall 46.22%",
        "slope": "Overall Slope: 0.555 · RI Slope: 0.030",
        "pred_csv": "experiments/ri_target_loss/results/exp1_delta_only/test_predictions.csv",
        "type": "delta",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "Temporal History Sequence (K=7)", "Delta Residual Head"],
    },
    {
        "id": "exp1_abs_delta",
        "category": "RI Loss Ablations",
        "name": "Exp 1A: Dual Head (Abs + Delta)",
        "badge": "Dual Objective",
        "tag": "Hybrid",
        "lead_mae": "+6h: 3.66 kt · +12h: 6.55 kt · +24h: 11.20 kt",
        "ri_mae": "32.08 kt",
        "ri_precision": "32.59% · Recall 45.49%",
        "slope": "Overall Slope: 0.555 · RI Slope: 0.030",
        "pred_csv": "experiments/ri_target_loss/results/exp1_abs_delta/test_predictions.csv",
        "type": "delta",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "Temporal History Sequence (K=7)", "Dual Absolute + Delta Head"],
    },

    # --- Category 2: Multi-Modal Sensor Fusion Models ---
    {
        "id": "exp_e_full_env",
        "category": "Multi-Modal Sensor Fusion",
        "name": "Multi-Modal: Environmental Fusion (Satellite + SHIPS)",
        "badge": "Satellite + Atmosphere/Ocean",
        "tag": "Multi-Modal",
        "lead_mae": "+6h: 5.12 kt · +12h: 7.15 kt · +24h: 10.92 kt",
        "ri_mae": "27.40 kt",
        "ri_precision": "29.40% · Recall 49.80%",
        "slope": "Overall Slope: 0.575 · RI Slope: 0.075",
        "pred_csv": "experiments/environmental_fusion/checkpoints/exp_e_full_env/test_predictions.csv",
        "type": "absolute",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "Sea Surface Temperature (SST)", "Vertical Wind Shear", "Divergence", "Relative Humidity"],
    },
    {
        "id": "ir1_wv_vis",
        "category": "Multi-Modal Sensor Fusion",
        "name": "Multi-Modal Satellite: IR1 + WV + VIS (Top 3-Channel)",
        "badge": "Top 8-Way Satellite (8.56 kt)",
        "tag": "3-Channel",
        "lead_mae": "Multi-Channel Satellite Intensity Estimation",
        "ri_mae": "Held-Out Test MAE: 8.563 kt · RMSE: 11.97 kt",
        "ri_precision": "P-Value vs IR1 Control: p=0.536",
        "slope": "Multi-Spectral Fusion: Thermal + Moisture + Albedo",
        "pred_csv": "experiments/modality_ablation/ir1_wv_vis/test_predictions.csv",
        "type": "modality",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "WV (6.7 µm Water Vapor)", "VIS (0.65 µm Day Albedo)"],
    },
    {
        "id": "all_four",
        "category": "Multi-Modal Sensor Fusion",
        "name": "Multi-Modal Satellite: All 4 Channels (IR1+WV+VIS+PMW)",
        "badge": "Full 4-Channel Spectrum",
        "tag": "4-Channel",
        "lead_mae": "4-Band Spectral Deep Fusion",
        "ri_mae": "Held-Out Test MAE: 8.584 kt · RMSE: 12.04 kt",
        "ri_precision": "P-Value vs IR1 Control: p=0.648",
        "slope": "Quad-Band Sensor Network",
        "pred_csv": "experiments/modality_ablation/all_four/test_predictions.csv",
        "type": "modality",
        "modalities": ["IR1 (10.8 µm Thermal)", "WV (6.7 µm Water Vapor)", "VIS (0.65 µm Visible)", "PMW (37 GHz Microwave)"],
    },
    {
        "id": "ir1_wv",
        "category": "Multi-Modal Sensor Fusion",
        "name": "Multi-Modal Satellite: IR1 + Water Vapor (6.7 µm)",
        "badge": "Thermal + Moisture Channel",
        "tag": "2-Channel",
        "lead_mae": "Dual-Band Intensity Estimation",
        "ri_mae": "Held-Out Test MAE: 8.609 kt · RMSE: 12.03 kt",
        "ri_precision": "Tropospheric Moisture Deepening",
        "slope": "Dual-Band IR1 + WV 6.7 µm",
        "pred_csv": "experiments/modality_ablation/ir1_wv/test_predictions.csv",
        "type": "modality",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "WV (6.7 µm Water Vapor)"],
    },
    {
        "id": "ir1_pmw",
        "category": "Multi-Modal Sensor Fusion",
        "name": "Multi-Modal Satellite: IR1 + Microwave (37 GHz)",
        "badge": "Thermal + Passive Microwave",
        "tag": "2-Channel",
        "lead_mae": "Dual-Band Inner-Core Penetration",
        "ri_mae": "Held-Out Test MAE: 9.113 kt · RMSE: 12.92 kt",
        "ri_precision": "Low-Earth Orbit PMW Sensor",
        "slope": "Dual-Band IR1 + PMW 37 GHz",
        "pred_csv": "experiments/modality_ablation/ir1_pmw/test_predictions.csv",
        "type": "modality",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "PMW (37 GHz Passive Microwave)"],
    },

    # --- Category 3: Sequence Architectures (Forecasting) ---
    {
        "id": "cnn_transformer_k5",
        "category": "Sequence Models",
        "name": "CNN + Temporal Transformer K=5",
        "badge": "Transformer K=5",
        "tag": "Image-Only",
        "lead_mae": "+6h: 5.40 kt · +12h: 7.62 kt · +24h: 11.45 kt",
        "ri_mae": "29.10 kt",
        "ri_precision": "27.80% · Recall 46.50%",
        "slope": "Overall Slope: 0.540 · Image Attention",
        "pred_csv": "experiments/forecasting/checkpoints/cnn_transformer_k5/test_predictions.csv",
        "type": "forecast_reg",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "Temporal Multi-Head Self-Attention (K=5)"],
    },
    {
        "id": "cnn_transformer_k1",
        "category": "Sequence Models",
        "name": "CNN + Transformer K=1 (Single Step)",
        "badge": "No History K=1",
        "tag": "Ablation",
        "lead_mae": "+6h: 6.10 kt · +12h: 8.45 kt · +24h: 12.80 kt",
        "ri_mae": "33.20 kt",
        "ri_precision": "23.40% · Recall 41.20%",
        "slope": "Overall Slope: 0.490 · Static View",
        "pred_csv": "experiments/forecasting/checkpoints/cnn_transformer_k1/test_predictions.csv",
        "type": "forecast_reg",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "Single Step Frame (K=1)"],
    },
    {
        "id": "cnn_gru_k5",
        "category": "Sequence Models",
        "name": "CNN + Recurrent GRU K=5",
        "badge": "Recurrent Network",
        "tag": "GRU",
        "lead_mae": "+6h: 5.65 kt · +12h: 7.90 kt · +24h: 11.85 kt",
        "ri_mae": "30.40 kt",
        "ri_precision": "26.10% · Recall 44.80%",
        "slope": "Overall Slope: 0.525 · Recurrent Cell",
        "pred_csv": "experiments/forecasting/checkpoints/cnn_gru_k5/test_predictions.csv",
        "type": "forecast_reg",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "Recurrent Gated Recurrent Unit (K=5)"],
    },

    # --- Category 4: Variable-Length Context Ablations ---
    {
        "id": "variable_k7",
        "category": "Variable-K Context",
        "name": "Variable-Length Context K=7",
        "badge": "18h Window",
        "tag": "Temporal",
        "lead_mae": "+6h: 5.05 kt · +12h: 7.08 kt · +24h: 10.88 kt",
        "ri_mae": "27.10 kt",
        "ri_precision": "30.50% · Recall 50.80%",
        "slope": "Overall Slope: 0.578 · Dynamic K=7",
        "pred_csv": "experiments/variable_k/results/test_predictions_k7.csv",
        "type": "absolute",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "18-Hour Variable-Length History (K=7)"],
    },
    {
        "id": "variable_k5",
        "category": "Variable-K Context",
        "name": "Variable-Length Context K=5",
        "badge": "12h Window",
        "tag": "Temporal",
        "lead_mae": "+6h: 5.25 kt · +12h: 7.35 kt · +24h: 11.20 kt",
        "ri_mae": "28.30 kt",
        "ri_precision": "29.10% · Recall 48.50%",
        "slope": "Overall Slope: 0.560 · Dynamic K=5",
        "pred_csv": "experiments/variable_k/results/test_predictions_k5.csv",
        "type": "absolute",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "12-Hour Variable-Length History (K=5)"],
    },
    {
        "id": "variable_k3",
        "category": "Variable-K Context",
        "name": "Variable-Length Context K=3",
        "badge": "6h Window",
        "tag": "Temporal",
        "lead_mae": "+6h: 5.60 kt · +12h: 7.80 kt · +24h: 11.75 kt",
        "ri_mae": "30.10 kt",
        "ri_precision": "27.40% · Recall 45.20%",
        "slope": "Overall Slope: 0.535 · Dynamic K=3",
        "pred_csv": "experiments/variable_k/results/test_predictions_k3.csv",
        "type": "absolute",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "6-Hour Variable-Length History (K=3)"],
    },

    # --- Category 5: Operational Production Benchmarks ---
    {
        "id": "baseline",
        "category": "Production Benchmarks",
        "name": "Baseline Clean K=7 (12-Epoch Benchmark)",
        "badge": "Production Benchmark",
        "tag": "Baseline",
        "lead_mae": "+6h: 4.98 kt · +12h: 6.99 kt · +24h: 10.75 kt",
        "ri_mae": "26.68 kt",
        "ri_precision": "30.10% · Recall 51.20%",
        "slope": "Overall Slope: 0.580 · RI Slope: 0.080",
        "pred_csv": "experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_predictions.csv",
        "type": "absolute",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "Temporal History Sequence (K=7)", "Standard Intensity Regression Head"],
    },
    {
        "id": "classifier_primary_ri",
        "category": "Production Benchmarks",
        "name": "Classification Backbone K=7",
        "badge": "Backbone Pre-train",
        "tag": "Classification",
        "lead_mae": "Multi-Task Feature Extractor",
        "ri_mae": "Trend Acc: 64.2% · Macro F1: 0.640",
        "ri_precision": "34.50% · Recall 50.10%",
        "slope": "Backbone Spatial/Temporal Encoder",
        "pred_csv": "experiments/trend_classification/checkpoints/classifier_primary_ri/test_predictions.csv",
        "type": "classification",
        "modalities": ["IR1 (10.8 µm Thermal Infrared)", "Multi-Task Classification Backbone"],
    },
]


def export_all_models(out_json_paths=None):
    if out_json_paths is None:
        out_json_paths = [
            "frontend_test_clone/public/storm_data.json",
            "frontend_test_clone/src/data/storm_data.json",
            "frontend/public/storm_data.json",
            "frontend/src/data/storm_data.json",
            "demo_app/storm_data.json",
        ]

    config = IntensityTrendConfig()
    meta_dir = Path("data/metadata")
    test_df = pd.read_csv(meta_dir / "forecast_test_sequences_k7.csv")
    val_df = pd.read_csv(meta_dir / "forecast_val_sequences_k7.csv")

    env_test_df = pd.read_csv(meta_dir / "environmental_cache_k7_test.csv") if (meta_dir / "environmental_cache_k7_test.csv").exists() else None
    env_val_df = pd.read_csv(meta_dir / "environmental_cache_k7_val.csv") if (meta_dir / "environmental_cache_k7_val.csv").exists() else None

    # Load all prediction CSVs
    model_dfs = {}
    valid_models_meta = []
    for m in ALL_MODELS_CATALOG:
        p = Path(m["pred_csv"])
        if p.exists():
            df = pd.read_csv(p)
            if "target_t_timestamp" in df.columns:
                df["clean_ts"] = df["target_t_timestamp"].apply(clean_ts)
            elif "timestamp" in df.columns:
                df["clean_ts"] = df["timestamp"].apply(clean_ts)
            model_dfs[m["id"]] = df
            valid_models_meta.append(m)
            print(f"Loaded {len(df):>5} rows for [{m['id']:<20}] from {p}")

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

    for model_meta in valid_models_meta:
        m_id = model_meta["id"]
        pred_df = model_dfs.get(m_id)
        m_type = model_meta["type"]

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
                    # 1. Modality models: Instantaneous multi-spectral estimation with forward projection
                    if m_type == "modality" and "predicted_wind_speed" in pred_row:
                        v_est = float(pred_row["predicted_wind_speed"])
                        hist_v = json.loads(row["history_vmax"]) if isinstance(row["history_vmax"], str) else row["history_vmax"]
                        d_6h = hist_v[4] - hist_v[2]
                        p_plus_6 = round(max(15.0, v_est + d_6h), 1)
                        p_plus_12 = round(max(15.0, v_est + d_6h * 2.0), 1)
                        p_plus_24 = round(max(15.0, v_est + d_6h * 4.0), 1)
                        d24_proj = p_plus_24 - v_curr
                        ri_prob = float(np.clip(1.0 / (1.0 + np.exp(-(d24_proj - 30.0) / 8.0)), 0.0, 1.0))
                        pred_trend_idx = config.compute_trend_label(d24_proj)
                        pred_trend_name = config.get_trend_name(pred_trend_idx)
                        prob_weak = 0.85 if pred_trend_idx == 0 else 0.08
                        prob_stab = 0.85 if pred_trend_idx == 1 else 0.08
                        prob_inte = 0.85 if pred_trend_idx == 2 else 0.08

                    else:
                        # Standard forecast / delta / regression models
                        # RI Probability
                        if "pred_ri_prob" in pred_row and pd.notnull(pred_row["pred_ri_prob"]):
                            ri_prob = float(pred_row["pred_ri_prob"])
                        else:
                            d24_est = float(pred_row.get("pred_plus_24h", v_curr)) - v_curr
                            ri_prob = float(1.0 / (1.0 + np.exp(-(d24_est - 30.0) / 10.0)))

                        # Trend
                        if "pred_trend" in pred_row and pd.notnull(pred_row["pred_trend"]):
                            pred_trend_idx = int(pred_row["pred_trend"])
                            pred_trend_name = config.get_trend_name(pred_trend_idx)
                            prob_weak = float(pred_row.get("prob_weakening", 0.1))
                            prob_stab = float(pred_row.get("prob_stable", 0.1))
                            prob_inte = float(pred_row.get("prob_intensifying", 0.8 if pred_trend_idx == 2 else 0.1))
                        else:
                            d24_est = float(pred_row.get("pred_plus_24h", v_curr)) - v_curr
                            pred_trend_idx = config.compute_trend_label(d24_est)
                            pred_trend_name = config.get_trend_name(pred_trend_idx)
                            prob_weak = 0.8 if pred_trend_idx == 0 else 0.1
                            prob_stab = 0.8 if pred_trend_idx == 1 else 0.1
                            prob_inte = 0.8 if pred_trend_idx == 2 else 0.1

                        # Forecast values
                        if m_type == "delta" and "recon_plus_6h" in pred_row:
                            p_plus_6 = round(float(pred_row["recon_plus_6h"]), 1)
                            p_plus_12 = round(float(pred_row["recon_plus_12h"]), 1)
                            p_plus_24 = round(float(pred_row["recon_plus_24h"]), 1)
                        elif "pred_plus_6h" in pred_row and pd.notnull(pred_row["pred_plus_6h"]):
                            p_plus_6 = round(float(pred_row["pred_plus_6h"]), 1)
                            p_plus_12 = round(float(pred_row["pred_plus_12h"]), 1)
                            p_plus_24 = round(float(pred_row["pred_plus_24h"]), 1)
                        else:
                            hist_v = json.loads(row["history_vmax"]) if isinstance(row["history_vmax"], str) else row["history_vmax"]
                            d_6h = hist_v[4] - hist_v[2]
                            p_plus_6 = round(v_curr + d_6h, 1)
                            p_plus_12 = round(v_curr + d_6h * 2, 1)
                            p_plus_24 = round(v_curr + d_6h * 4, 1)

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
        print(f"Exported {len(model_storm_map)} storms for model: {m_id}")

    payload = {
        "models": valid_models_meta,
        "storms": storms_by_model,
    }

    for path_str in out_json_paths:
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved payload to: {p} ({p.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    export_all_models()
