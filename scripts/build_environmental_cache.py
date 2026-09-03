"""
Build comprehensive environmental feature cache for TCIR 5-frame sequence datasets.
Extracts:
  - RSST: Sea Surface Temperature (deg C)
  - COHC: Climatological Ocean Heat Content relative to 26C (kJ/cm^2)
  - SHRD: 850-200 hPa Deep-Layer Vertical Wind Shear (kt)
  - RHMD: 700-500 hPa Mid-Level Relative Humidity (%)
  - VMAX: Current Observed Intensity (kt)
  - MSLP: Central Minimum Sea Level Pressure (hPa)
  - environment_age_hours: 0 for contemporaneous synoptic matches, 3 for causal forward-fill

Strict Scientific Guarantees:
  - Zero look-ahead: only TIME 0 predictors at or before time t are used.
  - Strict train-only normalization: mean/std computed exclusively on train split.
  - DTL excluded per user instruction.
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path


def tcir_to_atcf(cid: str) -> str:
    """Map TCIR cyclone_id (e.g. 200413E, 201015W, 201614L) to ATCF ID (e.g. EP132004, WP152010, AL142016)."""
    year = cid[:4]
    num = cid[4:6]
    basin = cid[6]
    b_map = {'I': 'IO', 'E': 'EP', 'L': 'AL', 'C': 'CP', 'W': 'WP', 'S': 'SH'}
    return f"{b_map.get(basin, basin)}{num}{year}"


def parse_ships_file(filepath: str, basin_code: str) -> dict:
    """Parse a fixed-width CIRA SHIPS ASCII predictor file.
    Returns: dict with key (atcf_id, timestamp) -> dict of TIME 0 variables.
    """
    records = {}
    if not os.path.exists(filepath):
        print(f"Warning: file not found {filepath}")
        return records

    with open(filepath, 'r', errors='ignore') as f:
        current_head = None
        current_data = {}
        for line in f:
            if 'HEAD' in line:
                if current_head is not None:
                    records[current_head['key']] = current_data
                parts = line.split()
                if len(parts) >= 8:
                    yymmdd, hh, vmax, lat, lon, mslp, atcf_id = (
                        parts[1], parts[2], float(parts[3]), float(parts[4]),
                        float(parts[5]), float(parts[6]), parts[7]
                    )
                    yy = int(yymmdd[:2])
                    year = 2000 + yy if yy < 50 else 1900 + yy
                    ts = int(f"{year}{yymmdd[2:]}{hh}")
                    lon_deg = -lon if basin_code in ['EP', 'CP', 'AL'] else lon
                    current_head = {
                        'key': (atcf_id, ts),
                        'atcf_id': atcf_id,
                        'timestamp': ts,
                        'lat': lat,
                        'lon': lon_deg,
                        'vmax': vmax,
                        'mslp': mslp
                    }
                    current_data = dict(current_head)
            elif current_head is not None:
                tag = line[115:121].strip() if len(line) >= 121 else ''
                tokens = line[:115].split()
                if tag and len(tokens) >= 3:
                    try:
                        val = float(tokens[2])  # Index 2 is TIME 0
                        current_data[tag] = val
                    except ValueError:
                        pass
        if current_head is not None:
            records[current_head['key']] = current_data
    return records


def load_all_ships_databases(ships_dir: str = "data/ships") -> dict:
    """Load and parse all 6 global SHIPS database files."""
    files = {
        'IO': os.path.join(ships_dir, 'lsdiagi_1990_2021_5day.txt'),
        'EP': os.path.join(ships_dir, 'lsdiage_1982_2022_sat_ts_5day.txt'),
        'CP': os.path.join(ships_dir, 'lsdiagc_1982_2022_sat_ts_5day.txt'),
        'AL': os.path.join(ships_dir, 'lsdiaga_1982_2022_sat_ts_5day.txt'),
        'WP': os.path.join(ships_dir, 'lsdiagw_1990_2021_5day.txt'),
        'SH': os.path.join(ships_dir, 'lsdiags_1998_2021_5day.txt'),
    }
    all_records = {}
    for basin, fpath in files.items():
        print(f"Parsing SHIPS basin {basin} from {fpath}...")
        parsed = parse_ships_file(fpath, basin)
        print(f"  -> Loaded {len(parsed):,} observations for {basin}.")
        all_records.update(parsed)
    print(f"Total global SHIPS observation records loaded: {len(all_records):,}\n")
    return all_records


def build_cache_for_split(
    seq_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    ships_records: dict,
    split_name: str
) -> pd.DataFrame:
    """Match each sequence in seq_df to SHIPS environmental predictors at time t."""
    # Build fast pressure lookup from metadata_all_basins
    press_lookup = meta_df.set_index(['cyclone_id', 'timestamp'])['pressure'].to_dict()

    extracted_rows = []
    matched_count = 0

    for idx, row in seq_df.iterrows():
        cid = str(row['cyclone_id'])
        ts = int(row['target_t_timestamp'])
        v_curr = float(row['vmax_curr'])
        atcf_id = tcir_to_atcf(cid)

        # Lookup TCIR MSLP
        tcir_mslp = press_lookup.get((cid, ts), np.nan)

        matched_rec = None
        env_age_hours = 0  # 0 for contemporaneous match, 3 for causal forward-fill

        # 1. Exact match at synoptic hour
        if (atcf_id, ts) in ships_records:
            matched_rec = ships_records[(atcf_id, ts)]
            env_age_hours = 0
        else:
            # 2. Causal forward-fill from t - 3h
            hh = ts % 100
            if hh in [3, 9, 15, 21]:
                prev_ts = ts - 3
                if (atcf_id, prev_ts) in ships_records:
                    matched_rec = ships_records[(atcf_id, prev_ts)]
                    env_age_hours = 3

        if matched_rec is not None:
            matched_count += 1
            has_env = 1

            rsst_raw = matched_rec.get('RSST', 9999.0)
            cohc_raw = matched_rec.get('COHC', 9999.0)
            shrd_raw = matched_rec.get('SHRD', 9999.0)
            rhmd_raw = matched_rec.get('RHMD', 9999.0)
            s_mslp = matched_rec.get('MSLP', 9999.0)

            # Conversions with explicit missing checks (9999 or <= 0)
            sst = rsst_raw / 10.0 if (0 < rsst_raw < 900) else np.nan
            cohc = cohc_raw if (0 <= cohc_raw < 900) else np.nan
            shrd = shrd_raw / 10.0 if (0 <= shrd_raw < 900) else np.nan
            rhmd = rhmd_raw if (0 <= rhmd_raw <= 100) else np.nan

            # Prefer TCIR verified MSLP, fallback to SHIPS MSLP if valid
            mslp = tcir_mslp if not np.isnan(tcir_mslp) else (s_mslp if 800 <= s_mslp <= 1050 else np.nan)
        else:
            has_env = 0
            env_age_hours = -1
            sst = np.nan
            cohc = np.nan
            shrd = np.nan
            rhmd = np.nan
            mslp = tcir_mslp

        extracted_rows.append({
            'sequence_idx': idx,
            'cyclone_id': cid,
            'timestamp': ts,
            'vmax': v_curr,
            'mslp': mslp,
            'sst': sst,
            'cohc': cohc,
            'shrd': shrd,
            'rhmd': rhmd,
            'environment_age_hours': env_age_hours,
            'has_env_data': has_env,
            'missing_sst': 1 if np.isnan(sst) else 0,
            'missing_cohc': 1 if np.isnan(cohc) else 0,
            'missing_shrd': 1 if np.isnan(shrd) else 0,
            'missing_rhmd': 1 if np.isnan(rhmd) else 0,
            'missing_mslp': 1 if np.isnan(mslp) else 0,
        })

    cache_df = pd.DataFrame(extracted_rows)
    pct = (matched_count / len(seq_df) * 100) if len(seq_df) > 0 else 0
    print(f"[{split_name.upper()}] Matched {matched_count:,} / {len(seq_df):,} sequences ({pct:.1f}%).")
    return cache_df


def main():
    parser = argparse.ArgumentParser(description="Build environmental feature cache")
    parser.add_argument("--k-history", type=int, default=5, help="Length of history sequence (5 or 7)")
    parser.add_argument("--ships-dir", type=str, default="data/ships")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata")
    parser.add_argument("--output-dir", type=str, default="data/metadata")
    parser.add_argument("--exp-dir", type=str, default="experiments/environmental_fusion")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.exp_dir, exist_ok=True)

    k = args.k_history
    # 1. Load sequence manifests
    print(f"Loading sequence manifests for K={k}...")
    train_df = pd.read_csv(os.path.join(args.metadata_dir, f"forecast_train_sequences_k{k}.csv"))
    val_df = pd.read_csv(os.path.join(args.metadata_dir, f"forecast_val_sequences_k{k}.csv"))
    test_df = pd.read_csv(os.path.join(args.metadata_dir, f"forecast_test_sequences_k{k}.csv"))
    meta_df = pd.read_csv(os.path.join(args.metadata_dir, "metadata_all_basins.csv"))

    # 2. Parse SHIPS records
    ships_records = load_all_ships_databases(args.ships_dir)

    # 3. Build cache for each split
    print(f"Building cache for train, val, and test splits (K={k})...")
    train_cache = build_cache_for_split(train_df, meta_df, ships_records, "train")
    val_cache = build_cache_for_split(val_df, meta_df, ships_records, "val")
    test_cache = build_cache_for_split(test_df, meta_df, ships_records, "test")

    # 4. Strict Train-Only Normalization Statistics
    print("\nComputing strict train-set-only normalization parameters...")
    feature_cols = ['vmax', 'mslp', 'sst', 'cohc', 'shrd', 'rhmd']
    norm_stats = {}

    for col in feature_cols:
        train_vals = train_cache[col].dropna()
        mean_val = float(train_vals.mean())
        std_val = float(train_vals.std()) if train_vals.std() > 1e-6 else 1.0
        norm_stats[col] = {
            'mean': mean_val,
            'std': std_val,
            'missing_fill': 0.0,
            'count': int(len(train_vals)),
            'missing_pct': float(100.0 * (1.0 - len(train_vals) / len(train_cache)))
        }
        print(f"  • {col:5s}: mean = {mean_val:8.2f}, std = {std_val:8.2f}, missing = {norm_stats[col]['missing_pct']:5.1f}%")

    stats_path = os.path.join(args.exp_dir, f"norm_stats_k{k}.json")
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved strict train normalization statistics to {stats_path}")

    # 5. Build standardized Float32 PyTorch tensors for zero-overhead DataLoader ingestion
    print("\nConstructing Float32 standardized PyTorch tensors with missingness indicators...")
    splits = [('train', train_cache), ('val', val_cache), ('test', test_cache)]
    tensor_dict = {}

    for name, c_df in splits:
        feat_list = []
        mask_list = []

        for col in feature_cols:
            raw = c_df[col].values.astype(np.float32)
            mask = np.isnan(raw).astype(np.float32)
            # Impute missing values with training mean
            raw[np.isnan(raw)] = norm_stats[col]['mean']
            # Standardize
            normed = (raw - norm_stats[col]['mean']) / norm_stats[col]['std']
            feat_list.append(normed.astype(np.float32))
            mask_list.append(mask)

        # Environmental vector: [vmax, mslp, sst, cohc, shrd, rhmd] + 6 missingness masks
        feats_matrix = np.column_stack(feat_list)  # (N, 6)
        masks_matrix = np.column_stack(mask_list)  # (N, 6)
        combined = np.hstack([feats_matrix, masks_matrix])  # (N, 12)

        tensor_dict[name] = torch.tensor(combined, dtype=torch.float32)
        print(f"[{name.upper()}] Tensor shape: {tensor_dict[name].shape}")

        # Save CSV for inspection
        csv_path = os.path.join(args.output_dir, f"environmental_cache_k{k}_{name}.csv")
        c_df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

    tensor_dict['feature_names'] = feature_cols
    tensor_dict['feature_dim'] = len(feature_cols)
    tensor_dict['total_dim'] = len(feature_cols) * 2  # features + missing masks

    # Save PyTorch binary tensor cache for fast DataLoader access
    pt_path = os.path.join(args.output_dir, f"environmental_features_k{k}.pt")
    torch.save(tensor_dict, pt_path)
    print(f"\nSuccessfully built and saved full environmental cache to {pt_path}")


if __name__ == "__main__":
    main()
