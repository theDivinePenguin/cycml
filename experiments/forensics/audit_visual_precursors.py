"""Script 6: Quantitative Precursor Analysis on Input Frames [t-18h ... t] for RI Failure Cases."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

def run():
    print("=" * 80)
    print("FORENSIC INVESTIGATION: QUANTITATIVE VISUAL PRECURSORS IN INPUT FRAMES")
    print("=" * 80)

    test_seq = pd.read_csv("data/metadata/forecast_test_sequences_k7.csv")
    
    # Target failure episodes
    cases = [
        ("200413E", 2004091218, "Hurricane Javier (Pre-RI Failure)"),
        ("200413E", 2004091300, "Hurricane Javier (Active RI Failure)"),
        ("200522S", 2005031018, "Cyclone Ingrid (Land Emergence / Pre-RI Failure)"),
        ("200522S", 2005031106, "Cyclone Ingrid (Active RI Failure)"),
        ("201516W", 2015082215, "Typhoon Dujuan (Pre-RI Failure)"),
        ("201015W", 2010101412, "Typhoon Megi (Correct Non-RI Detection)"),
        ("200519S", 2005020300, "Cyclone Percy (Successful RI Detection)"),
        ("201204W", 2012061412, "Typhoon Guchol (Successful RI Detection)"),
    ]

    h5_cache = {}
    def get_h5(path):
        if path not in h5_cache:
            h5_cache[path] = h5py.File(path, "r")
        return h5_cache[path]

    results = []
    print(f"{'Storm':<18} {'TS (t)':<11} {'V(t)':<6} {'V(+24)':<7} {'IR1 Min(t)':<11} {'Δ MinTb(18h)':<13} {'CDO Pixels(t)':<14} {'Δ CDO(18h)':<12} {'Visual Signal'}")
    print("-" * 110)

    for cid, ts, desc in cases:
        match = test_seq[(test_seq["cyclone_id"] == cid) & (test_seq["target_t_timestamp"] == ts)]
        if len(match) == 0:
            continue
        row = match.iloc[0]

        h_files = json.loads(row["history_h5_files"])
        h_rows = json.loads(row["history_h5_rows"])
        h_ts = json.loads(row["history_timestamps"])
        h_v = json.loads(row["history_vmax"])

        # Extract 7 frames
        ir1_frames = []
        wv_frames = []
        for hf_p, hr_idx in zip(h_files, h_rows):
            hf = get_h5(hf_p)
            mat = hf["matrix"][hr_idx] # (201, 201, 4)
            ir1 = mat[:, :, 0]
            wv = mat[:, :, 1]
            ir1_frames.append(ir1)
            wv_frames.append(wv)

        ir1_arr = np.array(ir1_frames) # (7, 201, 201)
        wv_arr = np.array(wv_frames)

        # Quantify metrics across the 7 frames:
        # Center core: center 60x60 pixels (radius ~300 km)
        cy, cx = 100, 100
        core_ir1 = ir1_arr[:, cy-30:cy+30, cx-30:cx+30]

        min_tb = np.array([np.nanmin(core_ir1[i]) for i in range(7)])
        mean_tb = np.array([np.nanmean(core_ir1[i]) for i in range(7)])
        # Central Dense Overcast (CDO) area: number of pixels with Tb < 210 K in center
        cdo_pixels = np.array([np.sum(core_ir1[i] < 210.0) for i in range(7)])
        # Deep convective burst: pixels with Tb < 195 K
        burst_pixels = np.array([np.sum(core_ir1[i] < 195.0) for i in range(7)])

        # Trend over 18 hours (frame 6 - frame 0)
        delta_min_tb = min_tb[-1] - min_tb[0]
        delta_cdo = cdo_pixels[-1] - cdo_pixels[0]
        delta_burst = burst_pixels[-1] - burst_pixels[0]

        # Classification of visual precursor signal
        # Cooling cloud tops (delta_min_tb < -10) and expanding CDO (delta_cdo > +200) indicates strong visual intensification
        if delta_min_tb < -10.0 and delta_cdo > 200:
            vis_signal = "STRONG CONVECTIVE EXPANSION"
        elif delta_min_tb > 5.0 and delta_cdo < -100:
            vis_signal = "CONVECTIVE COLLAPSE / DECAY"
        elif cdo_pixels[-1] > 1500 and burst_pixels[-1] > 200:
            vis_signal = "ESTABLISHED MATURE CORE"
        else:
            vis_signal = "STABLE / WEAK PULSING"

        v_curr = float(row["vmax_curr"])
        v_24 = float(row["vmax_plus_24h"])

        print(f"{cid:<18} {ts:<11} {v_curr:<6.0f} {v_24:<7.0f} {min_tb[-1]:<11.1f} {delta_min_tb:<+13.1f} {cdo_pixels[-1]:<14d} {delta_cdo:<+12d} {vis_signal}")

        results.append({
            "cyclone_id": cid,
            "timestamp": ts,
            "description": desc,
            "v_curr": v_curr,
            "v_plus_24": v_24,
            "min_tb_series": [float(x) for x in min_tb],
            "mean_tb_series": [float(x) for x in mean_tb],
            "cdo_pixels_series": [int(x) for x in cdo_pixels],
            "burst_pixels_series": [int(x) for x in burst_pixels],
            "delta_min_tb": float(delta_min_tb),
            "delta_cdo": int(delta_cdo),
            "delta_burst": int(delta_burst),
            "visual_signal": vis_signal
        })

    for hf in h5_cache.values():
        hf.close()

    out_file = Path("experiments/forensics/visual_precursors_audit.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved visual precursors audit to {out_file}")

if __name__ == "__main__":
    run()
