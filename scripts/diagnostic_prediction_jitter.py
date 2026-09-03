"""Scientific Diagnostic: Consecutive-step change distribution & Raw vs EMA smoothing audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def compute_ema(series: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Compute causal exponential moving average."""
    if len(series) == 0:
        return series
    out = np.zeros_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def run_jitter_diagnostic(
    pred_csv_path: str = "experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_predictions.csv",
    output_fig_path: str = "figures/diagnostic_raw_vs_ema_jitter.png",
    alpha: float = 0.35,
):
    print("=" * 80)
    print(f"RUNNING SCIENTIFIC PREDICTION JITTER & EMA AUDIT (alpha={alpha})")
    print("=" * 80)

    df = pd.read_csv(pred_csv_path)
    print(f"Loaded {len(df):,} predictions across {df['cyclone_id'].nunique()} unique cyclones.")

    # Sort strictly by storm and timestamp
    df = df.sort_values(["cyclone_id", "target_t_timestamp"]).reset_index(drop=True)

    all_raw_deltas = []
    all_ema_deltas = []
    all_true_deltas = []

    raw_errors = []
    ema_errors = []

    df["pred_plus_24h_ema"] = np.nan

    for cid, group in df.groupby("cyclone_id"):
        if len(group) < 2:
            continue
        g_idx = group.index

        v_raw = group["pred_plus_24h"].values
        v_true = group["vmax_plus_24h"].values
        v_ema = compute_ema(v_raw, alpha=alpha)

        df.loc[g_idx, "pred_plus_24h_ema"] = v_ema

        # Calculate consecutive step changes: |V(t) - V(t-1)|
        d_raw = np.abs(np.diff(v_raw))
        d_ema = np.abs(np.diff(v_ema))
        d_true = np.abs(np.diff(v_true))

        all_raw_deltas.extend(d_raw)
        all_ema_deltas.extend(d_ema)
        all_true_deltas.extend(d_true)

        raw_errors.extend(np.abs(v_raw - v_true))
        ema_errors.extend(np.abs(v_ema - v_true))

    all_raw_deltas = np.array(all_raw_deltas)
    all_ema_deltas = np.array(all_ema_deltas)
    all_true_deltas = np.array(all_true_deltas)

    print("\n--- CONSECUTIVE 3-HOUR STEP CHANGE DISTRIBUTION (|V_t - V_{t-1}|) ---")
    print(f"{'Metric':<25} | {'Ground Truth':<15} | {'Raw Model':<15} | {'EMA (alpha=' + str(alpha) + ')':<15}")
    print("-" * 75)
    print(f"{'Mean Step Change (kt)':<25} | {np.mean(all_true_deltas):<15.2f} | {np.mean(all_raw_deltas):<15.2f} | {np.mean(all_ema_deltas):<15.2f}")
    print(f"{'Median Step Change (kt)':<25} | {np.median(all_true_deltas):<15.2f} | {np.median(all_raw_deltas):<15.2f} | {np.median(all_ema_deltas):<15.2f}")
    print(f"{'75th Percentile (kt)':<25} | {np.percentile(all_true_deltas, 75):<15.2f} | {np.percentile(all_raw_deltas, 75):<15.2f} | {np.percentile(all_ema_deltas, 75):<15.2f}")
    print(f"{'90th Percentile (kt)':<25} | {np.percentile(all_true_deltas, 90):<15.2f} | {np.percentile(all_raw_deltas, 90):<15.2f} | {np.percentile(all_ema_deltas, 90):<15.2f}")
    print(f"{'99th Percentile (kt)':<25} | {np.percentile(all_true_deltas, 99):<15.2f} | {np.percentile(all_raw_deltas, 99):<15.2f} | {np.percentile(all_ema_deltas, 99):<15.2f}")
    print(f"{'Max Step Jump (kt)':<25} | {np.max(all_true_deltas):<15.2f} | {np.max(all_raw_deltas):<15.2f} | {np.max(all_ema_deltas):<15.2f}")
    print(f"{'Jumps > 15 kt (%)':<25} | {(all_true_deltas > 15).mean()*100:<15.2f}% | {(all_raw_deltas > 15).mean()*100:<15.2f}% | {(all_ema_deltas > 15).mean()*100:<15.2f}%")
    print(f"{'Jumps > 25 kt (%)':<25} | {(all_true_deltas > 25).mean()*100:<15.2f}% | {(all_raw_deltas > 25).mean()*100:<15.2f}% | {(all_ema_deltas > 25).mean()*100:<15.2f}%")

    print("\n--- FORECAST ERROR IMPACT (+24h MAE) ---")
    print(f"Raw Model +24h MAE:         {np.mean(raw_errors):.2f} kt")
    print(f"EMA-Smoothed +24h MAE:      {np.mean(ema_errors):.2f} kt")
    print(f"Error Delta:                {np.mean(ema_errors) - np.mean(raw_errors):+.2f} kt")

    # Generate Publication Diagnostic Figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    showcase = [
        ("201003I", "Super Cyclone Phet (IO)", axes[0, 0]),
        ("201015W", "Super Typhoon Megi (WPAC)", axes[0, 1]),
        ("201614L", "Hurricane Matthew (ATLN)", axes[1, 0]),
    ]

    for cid, sname, ax in showcase:
        s_df = df[df["cyclone_id"] == cid]
        if len(s_df) == 0:
            continue
        hours = np.arange(len(s_df)) * 3.0
        v_true = s_df["vmax_plus_24h"].values
        v_raw = s_df["pred_plus_24h"].values
        v_ema = s_df["pred_plus_24h_ema"].values

        ax.plot(hours, v_true, color="#0F172A", lw=2.6, label="Ground Truth Vmax(t+24h)")
        ax.plot(hours, v_raw, color="#EF4444", lw=1.2, ls="--", alpha=0.75, marker=".", ms=4, label="Raw Model Prediction")
        ax.plot(hours, v_ema, color="#0284C7", lw=2.4, label=f"EMA-Smoothed (α={alpha})")

        ax.set_title(f"{sname} — +24h Forecast Lifecycle", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Elapsed Storm Lifecycle (Hours)", fontsize=10)
        ax.set_ylabel("Maximum Sustained Wind (Knots)", fontsize=10)
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.set_ylim(bottom=15)

    # Panel 4: Distribution of Consecutive Step Changes (Histogram)
    ax_hist = axes[1, 1]
    bins = np.linspace(0, 35, 36)
    ax_hist.hist(all_true_deltas, bins=bins, alpha=0.45, color="#0F172A", density=True, label="Ground Truth Changes")
    ax_hist.hist(all_raw_deltas, bins=bins, alpha=0.45, color="#EF4444", density=True, label="Raw Model Step Jumps")
    ax_hist.hist(all_ema_deltas, bins=bins, alpha=0.55, color="#0284C7", density=True, label=f"EMA (α={alpha}) Jumps")

    ax_hist.axvline(np.mean(all_true_deltas), color="#0F172A", ls=":", lw=1.5, label=f"True Mean ({np.mean(all_true_deltas):.1f} kt)")
    ax_hist.axvline(np.mean(all_raw_deltas), color="#EF4444", ls=":", lw=1.5, label=f"Raw Mean ({np.mean(all_raw_deltas):.1f} kt)")
    ax_hist.axvline(np.mean(all_ema_deltas), color="#0284C7", ls=":", lw=1.5, label=f"EMA Mean ({np.mean(all_ema_deltas):.1f} kt)")

    ax_hist.set_title("Consecutive 3-Hour Step Change Distribution |V_t - V_{t-1}|", fontsize=12, fontweight="bold", pad=8)
    ax_hist.set_xlabel("Absolute Step Change Between Adjacent Origins (Knots)", fontsize=10)
    ax_hist.set_ylabel("Probability Density", fontsize=10)
    ax_hist.legend(loc="upper right", frameon=True, fontsize=9)
    ax_hist.set_xlim(0, 35)

    plt.tight_layout()
    Path(output_fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig_path, dpi=300)
    plt.close()
    print(f"\nSaved diagnostic figure to {output_fig_path}")

    return {
        "mean_step_true": float(np.mean(all_true_deltas)),
        "mean_step_raw": float(np.mean(all_raw_deltas)),
        "mean_step_ema": float(np.mean(all_ema_deltas)),
        "p90_step_raw": float(np.percentile(all_raw_deltas, 90)),
        "p90_step_ema": float(np.percentile(all_ema_deltas, 90)),
        "raw_mae": float(np.mean(raw_errors)),
        "ema_mae": float(np.mean(ema_errors)),
    }


if __name__ == "__main__":
    run_jitter_diagnostic()
