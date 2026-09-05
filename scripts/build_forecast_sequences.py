"""Build and save temporal forecasting sequence manifests with strict cyclone isolation and zero leakage.

Physical time representations:
  - Cadence: 3 hours
  - Horizons: +6h, +12h, +24h (all 3 required for future evaluation)
  - K=1:  0 hours context ([t])
  - K=3:  6 hours context ([t-6h, t-3h, t])
  - K=5: 12 hours context ([t-12h, t-9h, t-6h, t-3h, t])
  - K=7: 18 hours context ([t-18h, ..., t])
  - K=9: 24 hours context ([t-24h, ..., t])
  - K=11: 30 hours context ([t-30h, ..., t])
  - K=13: 36 hours context ([t-36h, ..., t])
"""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
import pandas as pd


def parse_tcir_timestamp(ts_val):
    """Parse TCIR timestamp string/int (YYYYMMDDHH) into datetime object."""
    s = str(int(ts_val)).strip()
    if len(s) == 10:
        return datetime.strptime(s, "%Y%m%d%H")
    elif len(s) == 8:
        return datetime.strptime(s + "00", "%Y%m%d%H")
    else:
        raise ValueError(f"Unrecognized timestamp format: {ts_val}")


def build_sequences_for_df(df: pd.DataFrame, k_history: int = 5, cadence_hours: int = 3) -> pd.DataFrame:
    """Construct valid forecasting sequences requiring:
    - History: t - (k-1)*cadence, ..., t - cadence, t (all present)
    - Future targets: t + 6h, t + 12h, t + 24h (all present)
    """
    df = df.copy()
    df["dt"] = df["timestamp"].apply(parse_tcir_timestamp)
    df = df.sort_values(by=["cyclone_id", "dt"]).reset_index(drop=True)

    # Build per-cyclone lookup: cyclone_id -> {dt: row_dict}
    cyclone_map = {}
    for idx, row in df.iterrows():
        cid = row["cyclone_id"]
        if cid not in cyclone_map:
            cyclone_map[cid] = {}
        cyclone_map[cid][row["dt"]] = {
            "row_idx": idx,
            "wind_speed": row["wind_speed"],
            "timestamp": row["timestamp"],
            "h5_file": row["h5_file"],
            "h5_row_index": row["h5_row_index"],
            "dt": row["dt"],
            "latitude": row.get("latitude", 0.0),
            "longitude": row.get("longitude", 0.0),
            "source_dataset": row.get("source_dataset", "")
        }

    sequences = []

    for cid, dt_dict in cyclone_map.items():
        for t_dt, curr_info in dt_dict.items():
            # Check future targets: +6h, +12h, +24h
            t6 = t_dt + timedelta(hours=6)
            t12 = t_dt + timedelta(hours=12)
            t24 = t_dt + timedelta(hours=24)

            if t6 not in dt_dict or t12 not in dt_dict or t24 not in dt_dict:
                continue

            # Check historical sequence: [t - (k_history-1)*cadence, ..., t]
            history_dts = [t_dt - timedelta(hours=cadence_hours * step) for step in range(k_history - 1, -1, -1)]
            if not all(h_dt in dt_dict for h_dt in history_dts):
                continue

            hist_rows = [dt_dict[h_dt] for h_dt in history_dts]
            target_6 = dt_dict[t6]
            target_12 = dt_dict[t12]
            target_24 = dt_dict[t24]

            seq_entry = {
                "cyclone_id": cid,
                "target_t_timestamp": curr_info["timestamp"],
                "target_t_dt": t_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "vmax_curr": curr_info["wind_speed"],
                "vmax_plus_6h": target_6["wind_speed"],
                "vmax_plus_12h": target_12["wind_speed"],
                "vmax_plus_24h": target_24["wind_speed"],
                # Row indices in the split DataFrame for history frames
                "history_row_indices": json.dumps([h["row_idx"] for h in hist_rows]),
                "history_h5_files": json.dumps([h["h5_file"] for h in hist_rows]),
                "history_h5_rows": json.dumps([h["h5_row_index"] for h in hist_rows]),
                "history_timestamps": json.dumps([h["timestamp"] for h in hist_rows]),
                "history_vmax": json.dumps([h["wind_speed"] for h in hist_rows]),
                # Target row indices
                "target_6h_row_idx": target_6["row_idx"],
                "target_12h_row_idx": target_12["row_idx"],
                "target_24h_row_idx": target_24["row_idx"],
                # Metadata
                "latitude": curr_info["latitude"],
                "longitude": curr_info["longitude"],
                "source_dataset": curr_info["source_dataset"]
            }
            sequences.append(seq_entry)

    seq_df = pd.DataFrame(sequences)
    return seq_df


def build_and_save_manifests(k_history: int, meta_dir: Path, cadence_hours: int = 3, force: bool = False):
    train_path = meta_dir / f"forecast_train_sequences_k{k_history}.csv"
    val_path = meta_dir / f"forecast_val_sequences_k{k_history}.csv"
    test_path = meta_dir / f"forecast_test_sequences_k{k_history}.csv"

    if train_path.exists() and val_path.exists() and test_path.exists() and not force:
        print(f"[K={k_history}] Manifests already exist. Skipping. (Pass --force to rebuild)")
        return

    span_hours = (k_history - 1) * cadence_hours
    print(f"\nBuilding Sequences for K={k_history} ({span_hours}h historical span, Cadence={cadence_hours}h)...")

    train_df = pd.read_csv(meta_dir / "train_metadata_all_basins.csv")
    val_df = pd.read_csv(meta_dir / "val_metadata_all_basins.csv")
    test_df = pd.read_csv(meta_dir / "test_metadata_all_basins.csv")

    train_seq = build_sequences_for_df(train_df, k_history=k_history, cadence_hours=cadence_hours)
    val_seq = build_sequences_for_df(val_df, k_history=k_history, cadence_hours=cadence_hours)
    test_seq = build_sequences_for_df(test_df, k_history=k_history, cadence_hours=cadence_hours)

    print(f"  • Train Sequences: {len(train_seq):,} from {train_seq['cyclone_id'].nunique()} cyclones")
    print(f"  • Val Sequences:   {len(val_seq):,} from {val_seq['cyclone_id'].nunique()} cyclones")
    print(f"  • Test Sequences:  {len(test_seq):,} from {test_seq['cyclone_id'].nunique()} cyclones")

    # Anti-leakage split verification
    train_cids = set(train_seq["cyclone_id"].unique())
    val_cids = set(val_seq["cyclone_id"].unique())
    test_cids = set(test_seq["cyclone_id"].unique())

    assert len(train_cids & val_cids) == 0, f"[K={k_history}] Leakage detected between Train and Val!"
    assert len(train_cids & test_cids) == 0, f"[K={k_history}] Leakage detected between Train and Test!"
    assert len(val_cids & test_cids) == 0, f"[K={k_history}] Leakage detected between Val and Test!"

    train_seq.to_csv(train_path, index=False)
    val_seq.to_csv(val_path, index=False)
    test_seq.to_csv(test_path, index=False)
    print(f"  [Saved K={k_history}] -> {train_path.name}, {val_path.name}, {test_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Build temporal sequence manifests for arbitrary K.")
    parser.add_argument("--k-list", type=int, nargs="+", default=[1, 3, 5, 7, 9, 11, 13], help="List of K history lengths")
    parser.add_argument("--cadence-hours", type=int, default=3, help="Cadence interval in hours (default: 3)")
    parser.add_argument("--metadata-dir", type=str, default="data/metadata", help="Metadata directory")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if files exist")
    args = parser.parse_args()

    meta_dir = Path(args.metadata_dir)
    print("=" * 80)
    print(f"BUILDING TEMPORAL FORECASTING MANIFESTS: K in {args.k_list}")
    print("Zero-Duplication Protocol: Only index pointers into raw HDF5 are saved.")
    print("=" * 80)

    for k in args.k_list:
        build_and_save_manifests(k_history=k, meta_dir=meta_dir, cadence_hours=args.cadence_hours, force=args.force)

    print("\n[Done] All requested sequence manifests verified and ready.")


if __name__ == "__main__":
    main()
