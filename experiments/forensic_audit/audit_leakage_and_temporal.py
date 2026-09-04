"""Forensic Audit Script: Leakage, Temporal Alignment, and Target Construction."""
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def run_audit():
    print("=" * 80)
    print("FORENSIC AUDIT 1: SPLIT LEAKAGE & TEMPORAL INTEGRITY")
    print("=" * 80)

    # Load sequence manifests
    train_seq = pd.read_csv("data/metadata/forecast_train_sequences_k7.csv")
    val_seq = pd.read_csv("data/metadata/forecast_val_sequences_k7.csv")
    test_seq = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")
    meta_df = pd.read_csv("data/metadata/metadata_all_basins.csv")
    train_df = pd.read_csv("data/metadata/train_metadata_all_basins.csv")
    val_df = pd.read_csv("data/metadata/val_metadata_all_basins.csv")
    test_df = pd.read_csv("data/metadata/test_metadata_all_basins.csv")

    # 1. Cyclone Set Intersections
    train_cids = set(train_seq["cyclone_id"].unique())
    val_cids = set(val_seq["cyclone_id"].unique())
    test_cids = set(test_seq["cyclone_id"].unique())

    inter_tr_va = train_cids & val_cids
    inter_tr_te = train_cids & test_cids
    inter_va_te = val_cids & test_cids

    print("\n[1] CYCLONE ID DISJOINTNESS:")
    print(f"  • Train cyclones: {len(train_cids)}")
    print(f"  • Val cyclones:   {len(val_cids)}")
    print(f"  • Test cyclones:  {len(test_cids)}")
    print(f"  • Train ∩ Val:  {len(inter_tr_va)}")
    print(f"  • Train ∩ Test: {len(inter_tr_te)}")
    print(f"  • Val ∩ Test:   {len(inter_va_te)}")

    # 2. Physical HDF5 Frame Disjointness
    print("\n[2] PHYSICAL HDF5 FRAME DISJOINTNESS AUDIT:")
    train_h5_frames = set(zip(train_df["h5_file"], train_df["h5_row_index"]))
    val_h5_frames = set(zip(val_df["h5_file"], val_df["h5_row_index"]))
    test_h5_frames = set(zip(test_df["h5_file"], test_df["h5_row_index"]))

    print(f"  • Train frame count: {len(train_h5_frames):,}")
    print(f"  • Val frame count:   {len(val_h5_frames):,}")
    print(f"  • Test frame count:  {len(test_h5_frames):,}")
    print(f"  • Train ∩ Val frames:  {len(train_h5_frames & val_h5_frames)}")
    print(f"  • Train ∩ Test frames: {len(train_h5_frames & test_h5_frames)}")
    print(f"  • Val ∩ Test frames:   {len(val_h5_frames & test_h5_frames)}")

    # Check whether any frame referenced in test_seq or val_seq resides in train_h5_frames
    test_seq_leak_frames = 0
    for _, r in test_seq.iterrows():
        h_files = json.loads(r["history_h5_files"])
        h_rows = json.loads(r["history_h5_rows"])
        for hf, hr in zip(h_files, h_rows):
            if (hf, hr) in train_h5_frames:
                test_seq_leak_frames += 1

    val_seq_leak_frames = 0
    for _, r in val_seq.iterrows():
        h_files = json.loads(r["history_h5_files"])
        h_rows = json.loads(r["history_h5_rows"])
        for hf, hr in zip(h_files, h_rows):
            if (hf, hr) in train_h5_frames:
                val_seq_leak_frames += 1

    print(f"  • Test sequence frames in train frames: {test_seq_leak_frames}")
    print(f"  • Val sequence frames in train frames:  {val_seq_leak_frames}")

    # 3. Temporal Sequence Alignment Verification for 20 Random Test Samples
    print("\n[3] 20 RANDOM TEST SAMPLES TEMPORAL ALIGNMENT CHECK:")
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(test_seq), size=20, replace=False)
    
    temporal_errors = 0
    recomp_errors = 0

    meta_lookup = meta_df.set_index(["cyclone_id", "timestamp"])["wind_speed"].to_dict()

    print(f"{'Idx':<4} {'Cyclone':<8} {'t-18':<10} {'t-9':<10} {'t_curr':<10} {'t+6':<10} {'t+12':<10} {'t+24':<10} {'v_curr':<6} {'v+24':<6} {'Aligned?':<8}")
    print("-" * 95)

    def to_dt(ts):
        s = str(int(ts))
        return datetime.strptime(s, "%Y%m%d%H")

    alignment_records = []
    for s_idx in sample_indices:
        r = test_seq.iloc[s_idx]
        cid = r["cyclone_id"]
        t_curr = int(r["target_t_timestamp"])
        v_curr = r["vmax_curr"]
        v6 = r["vmax_plus_6h"]
        v12 = r["vmax_plus_12h"]
        v24 = r["vmax_plus_24h"]

        hist_ts = json.loads(r["history_timestamps"])
        hist_v = json.loads(r["history_vmax"])

        is_strictly_ascending = all(hist_ts[i] < hist_ts[i+1] for i in range(len(hist_ts)-1))
        last_is_curr = (hist_ts[-1] == t_curr)
        
        dts = [to_dt(x) for x in hist_ts]
        cadences = [(dts[i+1] - dts[i]).total_seconds() / 3600 for i in range(len(dts)-1)]
        is_3h_cadence = all(c == 3.0 for c in cadences)
        total_history_span = (dts[-1] - dts[0]).total_seconds() / 3600

        dt_t = to_dt(t_curr)
        dt_6 = dt_t + timedelta(hours=6)
        dt_12 = dt_t + timedelta(hours=12)
        dt_24 = dt_t + timedelta(hours=24)

        ts_6 = int(dt_6.strftime("%Y%m%d%H"))
        ts_12 = int(dt_12.strftime("%Y%m%d%H"))
        ts_24 = int(dt_24.strftime("%Y%m%d%H"))

        raw_v_curr = meta_lookup.get((cid, t_curr))
        raw_v6 = meta_lookup.get((cid, ts_6))
        raw_v12 = meta_lookup.get((cid, ts_12))
        raw_v24 = meta_lookup.get((cid, ts_24))

        match_v = (raw_v_curr == v_curr and raw_v6 == v6 and raw_v12 == v12 and raw_v24 == v24)
        if not match_v:
            recomp_errors += 1

        aligned = (is_strictly_ascending and last_is_curr and is_3h_cadence and total_history_span == 18.0 and match_v)
        if not aligned:
            temporal_errors += 1

        print(f"{s_idx:<4} {cid:<8} {hist_ts[0]:<10} {hist_ts[3]:<10} {t_curr:<10} {ts_6:<10} {ts_12:<10} {ts_24:<10} {v_curr:<6.0f} {v24:<6.0f} {'YES' if aligned else 'FAIL':<8}")

        alignment_records.append({
            "sample_idx": int(s_idx),
            "cyclone_id": cid,
            "t_curr": t_curr,
            "v_curr": float(v_curr),
            "hist_ts": [int(x) for x in hist_ts],
            "v_plus_6": float(v6),
            "v_plus_12": float(v12),
            "v_plus_24": float(v24),
            "cadence_hours": cadences,
            "history_span_hours": total_history_span,
            "aligned": bool(aligned)
        })

    # Comprehensive verification over ALL 7,901 test sequences
    print("\n[4] EXHAUSTIVE CHECK OVER ALL 7,901 TEST SEQUENCES:")
    all_aligned = True
    bad_count = 0
    target_offset_errors = 0
    for idx, r in test_seq.iterrows():
        hist_ts = json.loads(r["history_timestamps"])
        t_curr = int(r["target_t_timestamp"])
        if hist_ts[-1] != t_curr or len(hist_ts) != 7:
            bad_count += 1
            all_aligned = False
        
        cid = r["cyclone_id"]
        dt_t = to_dt(t_curr)
        ts_6 = int((dt_t + timedelta(hours=6)).strftime("%Y%m%d%H"))
        ts_12 = int((dt_t + timedelta(hours=12)).strftime("%Y%m%d%H"))
        ts_24 = int((dt_t + timedelta(hours=24)).strftime("%Y%m%d%H"))
        
        if (meta_lookup.get((cid, ts_6)) != r["vmax_plus_6h"] or
            meta_lookup.get((cid, ts_12)) != r["vmax_plus_12h"] or
            meta_lookup.get((cid, ts_24)) != r["vmax_plus_24h"]):
            target_offset_errors += 1

    print(f"  • Sequences with last history frame != t_curr: {bad_count} / {len(test_seq)}")
    print(f"  • Sequences with target mismatch vs raw metadata: {target_offset_errors} / {len(test_seq)}")
    print(f"  • Full temporal alignment invariant holds: {'YES' if (bad_count == 0 and target_offset_errors == 0) else 'NO'}")

    out_file = Path("experiments/forensic_audit/temporal_leakage_audit.json")
    with open(out_file, "w") as f:
        json.dump({
            "train_cids": len(train_cids),
            "val_cids": len(val_cids),
            "test_cids": len(test_cids),
            "overlap_tr_va": len(inter_tr_va),
            "overlap_tr_te": len(inter_tr_te),
            "overlap_va_te": len(inter_va_te),
            "test_h5_frames_in_train": len(test_h5_frames & train_h5_frames),
            "test_seq_leak_frames": test_seq_leak_frames,
            "val_seq_leak_frames": val_seq_leak_frames,
            "temporal_errors_in_sample": temporal_errors,
            "recomp_errors_in_sample": recomp_errors,
            "exhaustive_bad_count": bad_count,
            "exhaustive_target_offset_errors": target_offset_errors,
            "sample_records": alignment_records
        }, f, indent=2)
    print(f"\nAudit saved to {out_file}")

if __name__ == "__main__":
    run_audit()
