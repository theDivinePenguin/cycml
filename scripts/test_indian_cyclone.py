"""Test single held-out Indian cyclone across all models with full lifecycle analysis and Grad-CAM."""
import argparse
import json
import random
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.dataset import TCIRDataset
from src.data.preprocessing import TCIRPreprocessor
from src.models.factory import build_model
from src.utils.config import load_config


class GradCAM:
    """Grad-CAM for single-channel ResNet18."""
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        output.backward(gradient=torch.ones_like(output))

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()


def test_indian_cyclone(
    cyclone_id: str = "201004I",
    h5_path: str = "data/raw/TCIR-CPAC_IO_SH.h5",
    test_meta_path: str = "data/metadata/test_metadata_IO.csv",
    output_dir: str = "experiments/indian_cyclone_test"
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_test = pd.read_csv(test_meta_path)
    
    if cyclone_id == "random" or cyclone_id is None:
        available_cids = df_test["cyclone_id"].unique().tolist()
        cyclone_id = random.choice(available_cids)
        print(f"[Random Selection] Selected Indian Ocean Test Cyclone: {cyclone_id}")

    storm_df = df_test[df_test["cyclone_id"] == cyclone_id].sort_values("sample_index").reset_index(drop=True)
    if len(storm_df) == 0:
        raise ValueError(f"Cyclone {cyclone_id} not found in {test_meta_path}!")

    n_frames = len(storm_df)
    min_v = storm_df["wind_speed"].min()
    max_v = storm_df["wind_speed"].max()
    year = storm_df["year"].iloc[0]

    storm_name = "Super Cyclone Giri (JTWC 04B / IMD BOB 04)" if cyclone_id == "201004I" else f"Indian Ocean Cyclone {cyclone_id}"

    print("=" * 95)
    print(f"TESTING INDIAN CYCLONE: {storm_name}")
    print(f"  • Cyclone ID:        {cyclone_id}")
    print(f"  • Year:              {year}")
    print(f"  • Total Frames:      {n_frames}")
    print(f"  • Intensity Range:   {min_v:.1f} kt to {max_v:.1f} kt (Peak Category: Cat {'4/5' if max_v >= 115 else '1-3' if max_v >= 64 else 'TS'})")
    print(f"  • Latitude Range:    {storm_df['latitude'].min():.1f}°N to {storm_df['latitude'].max():.1f}°N")
    print(f"  • Longitude Range:   {storm_df['longitude'].min():.1f}°E to {storm_df['longitude'].max():.1f}°E")
    print("=" * 95)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_config = {
        "All-Basin (IR1)": {
            "ckpt": "experiments/expanded_all_basins_resnet18/best.pt",
            "cfg": "configs/all_basins.yaml",
            "stats": "data/metadata/normalization_stats_all_basins.json",
            "color": "#f59e0b",
            "marker": "o",
            "channels": [0]
        },
        "Multi-Channel (4-Ch)": {
            "ckpt": "experiments/multichannel_resnet18/all_channels/best.pt",
            "cfg": "configs/multichannel_all_channels.yaml",
            "stats": "data/metadata/normalization_stats_multichannel.json",
            "color": "#dc2626",
            "marker": "D",
            "channels": [0, 1, 2, 3]
        },
        "Original Baseline": {
            "ckpt": "experiments/baseline_resnet18_cpac_io_sh/best.pt",
            "cfg": "configs/baseline.yaml",
            "stats": "data/metadata/normalization_stats_CPAC_IO_SH.json",
            "color": "#64748b",
            "marker": "s",
            "channels": [0]
        },
        "IO Natural (A)": {
            "ckpt": "experiments/io_baseline_resnet18/best.pt",
            "cfg": "configs/io_baseline.yaml",
            "stats": "data/metadata/normalization_stats_IO.json",
            "color": "#0284c7",
            "marker": "^",
            "channels": [0]
        },
        "IO Balanced (B)": {
            "ckpt": "experiments/io_balanced_resnet18/best.pt",
            "cfg": "configs/io_balanced.yaml",
            "stats": "data/metadata/normalization_stats_IO.json",
            "color": "#10b981",
            "marker": "d",
            "channels": [0]
        }
    }

    # Load Raw Images directly from HDF5
    with h5py.File(h5_path, "r") as h5:
        matrix_ds = h5["matrix"]
        sample_indices = storm_df["sample_index"].values
        raw_images_ir = [matrix_ds[idx, :, :, 0].astype(np.float32) for idx in sample_indices]
        raw_images_4ch = [matrix_ds[idx, :, :, [0, 1, 2, 3]].astype(np.float32) for idx in sample_indices]

    results_dict = {}

    for m_name, m_info in models_config.items():
        cfg = load_config(m_info["cfg"])
        with open(m_info["stats"]) as f:
            stats = json.load(f)

        ch_list = m_info.get("channels", [0])
        preprocessor = TCIRPreprocessor(
            mean=stats["mean"],
            std=stats["std"],
            target_size=(224, 224),
            channels=ch_list,
            is_training=False,
            augmentation_cfg={"enabled": False}
        )

        model = build_model(cfg).to(device)
        ckpt = torch.load(m_info["ckpt"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        raw_imgs = raw_images_4ch if len(ch_list) == 4 else raw_images_ir
        preds = []
        with torch.no_grad():
            for raw_img in raw_imgs:
                if len(ch_list) == 4:
                    img_t = torch.from_numpy(raw_img).permute(2, 0, 1).float()
                else:
                    img_t = torch.from_numpy(raw_img).unsqueeze(0).float()
                tensor = preprocessor(img_t).unsqueeze(0).to(device)
                pred = model(tensor).item()
                preds.append(pred)

        preds = np.array(preds)
        acts = storm_df["wind_speed"].values
        errs = preds - acts
        abs_errs = np.abs(errs)

        peak_idx = int(np.argmax(acts))
        peak_act = float(acts[peak_idx])
        peak_pred = float(preds[peak_idx])

        results_dict[m_name] = {
            "predictions": preds.tolist(),
            "mae": round(float(np.mean(abs_errs)), 2),
            "rmse": round(float(np.sqrt(np.mean(errs ** 2))), 2),
            "bias": round(float(np.mean(errs)), 2),
            "max_error": round(float(np.max(abs_errs)), 2),
            "peak_actual": peak_act,
            "peak_predicted": round(peak_pred, 2),
            "peak_error": round(peak_pred - peak_act, 2)
        }

    # Print Summary Table
    print("\n" + "=" * 105)
    print(f"PERFORMANCE ON HELD-OUT INDIAN CYCLONE: {cyclone_id} (N={n_frames} frames)")
    print("=" * 105)
    print(f"{'Model':<22} | {'MAE (kt)':<10} | {'RMSE (kt)':<10} | {'Mean Bias':<11} | {'Peak Act (kt)':<14} | {'Peak Pred (kt)':<15} | {'Peak Error (kt)'}")
    print("-" * 105)
    for m_name, res in results_dict.items():
        print(f"{m_name:<22} | {res['mae']:8.2f} kt | {res['rmse']:8.2f} kt | {res['bias']:+8.2f} kt | {res['peak_actual']:10.1f} kt   | {res['peak_predicted']:10.1f} kt    | {res['peak_error']:+10.1f} kt")
    print("=" * 105)

    # Frame by frame sample inspection
    print("\nSample Frame Progression across Lifecycle:")
    print(f"{'Frame #':<8} | {'Timestamp':<12} | {'Actual (kt)':<12} | {'All-Basin IR1':<14} | {'Multi-Ch (4Ch)':<14} | {'Baseline':<12} | {'IO Natural'}")
    print("-" * 95)
    step_indices = np.linspace(0, n_frames - 1, min(7, n_frames), dtype=int)
    for idx in step_indices:
        ts = str(storm_df["timestamp"].iloc[idx])
        act = storm_df["wind_speed"].iloc[idx]
        p_all = results_dict["All-Basin (IR1)"]["predictions"][idx]
        p_multi = results_dict["Multi-Channel (4-Ch)"]["predictions"][idx]
        p_base = results_dict["Original Baseline"]["predictions"][idx]
        p_nat = results_dict["IO Natural (A)"]["predictions"][idx]
        print(f"{idx+1:<8d} | {ts:<12} | {act:10.1f} kt | {p_all:11.1f} kt | {p_multi:11.1f} kt | {p_base:9.1f} kt | {p_nat:9.1f} kt")
    print("-" * 95)

    # 1. Plot Lifecycle Time-Series
    plt.figure(figsize=(12, 6), dpi=150)
    frame_numbers = np.arange(1, n_frames + 1)
    actuals = storm_df["wind_speed"].values

    plt.plot(frame_numbers, actuals, "k-", linewidth=3.0, label=f"Ground Truth Best-Track (Peak: {max_v:.0f} kt)")

    for m_name, m_info in models_config.items():
        preds = results_dict[m_name]["predictions"]
        mae = results_dict[m_name]["mae"]
        plt.plot(
            frame_numbers, preds,
            color=m_info["color"],
            linestyle="--",
            linewidth=2.0,
            marker=m_info["marker"],
            markersize=4.5,
            label=f"{m_name} (MAE: {mae:.1f} kt)"
        )

    plt.xlabel("Observation Frame Sequence (Lifecycle Timeline)", fontsize=11, fontweight="bold")
    plt.ylabel("Maximum Sustained Wind Speed $V_{\\max}$ (knots)", fontsize=11, fontweight="bold")
    plt.title(f"Intensity Estimation Across Full Lifecycle: {storm_name} ({year})\nHeld-Out Indian Ocean Cyclone (Zero Model Exposure in Training)", fontsize=12, fontweight="bold", pad=15)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    lifecycle_plot_path = plots_dir / f"cyclone_{cyclone_id}_lifecycle_comparison.png"
    plt.savefig(lifecycle_plot_path)
    plt.close()

    # 2. Grad-CAM at Peak Intensity Frame
    peak_idx = int(np.argmax(actuals))
    peak_raw_ir = raw_images_ir[peak_idx]
    peak_raw_4ch = raw_images_4ch[peak_idx]
    peak_act = actuals[peak_idx]

    # Generate Grad-CAM for All-Basin (IR1), Multi-Channel (4-Ch), and IO Natural models
    gradcam_cards = []
    for model_key in ["All-Basin (IR1)", "Multi-Channel (4-Ch)", "IO Natural (A)"]:
        m_info = models_config[model_key]
        cfg = load_config(m_info["cfg"])
        with open(m_info["stats"]) as f:
            stats = json.load(f)
        ch_list = m_info.get("channels", [0])
        preprocessor = TCIRPreprocessor(mean=stats["mean"], std=stats["std"], channels=ch_list, target_size=(224, 224), is_training=False)
        
        model = build_model(cfg).to(device)
        ckpt = torch.load(m_info["ckpt"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # ResNet18 target layer: layer4[-1]
        cam_generator = GradCAM(model, model.layer4[-1])
        if len(ch_list) == 4:
            img_t = torch.from_numpy(peak_raw_4ch).permute(2, 0, 1).float()
        else:
            img_t = torch.from_numpy(peak_raw_ir).unsqueeze(0).float()
        input_tensor = preprocessor(img_t).unsqueeze(0).to(device)
        cam = cam_generator.generate_heatmap(input_tensor)
        pred_val = model(input_tensor).item()
        cam_generator.remove_hooks()

        gradcam_cards.append({
            "name": model_key,
            "pred": pred_val,
            "cam": cam
        })

    # Plot Visual Inference Card with Grad-CAM (4 panels)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)

    # Raw IR
    im0 = axes[0].imshow(peak_raw_ir, cmap="gray", origin="upper")
    axes[0].set_title(f"Satellite IR1 Brightness Temp\nActual: {peak_act:.1f} kt ({storm_df['timestamp'].iloc[peak_idx]})", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Brightness Temp (K)")

    # All-Basin (IR1) Grad-CAM
    axes[1].imshow(peak_raw_ir, cmap="gray", origin="upper")
    im1 = axes[1].imshow(gradcam_cards[0]["cam"], cmap="jet", alpha=0.55, origin="upper")
    axes[1].set_title(f"All-Basin (IR1) Attention\nPredicted: {gradcam_cards[0]['pred']:.1f} kt (Err: {gradcam_cards[0]['pred']-peak_act:+.1f} kt)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Attention Weight")

    # Multi-Channel (4-Ch) Grad-CAM
    axes[2].imshow(peak_raw_ir, cmap="gray", origin="upper")
    im2 = axes[2].imshow(gradcam_cards[1]["cam"], cmap="jet", alpha=0.55, origin="upper")
    axes[2].set_title(f"Multi-Channel (4-Ch) Attention\nPredicted: {gradcam_cards[1]['pred']:.1f} kt (Err: {gradcam_cards[1]['pred']-peak_act:+.1f} kt)", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Attention Weight")

    # IO Natural Grad-CAM
    axes[3].imshow(peak_raw_ir, cmap="gray", origin="upper")
    im3 = axes[3].imshow(gradcam_cards[2]["cam"], cmap="jet", alpha=0.55, origin="upper")
    axes[3].set_title(f"IO Natural Attention\nPredicted: {gradcam_cards[2]['pred']:.1f} kt (Err: {gradcam_cards[2]['pred']-peak_act:+.1f} kt)", fontsize=11, fontweight="bold")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label="Attention Weight")

    plt.suptitle(f"Peak Intensity Analysis: {storm_name} (Category 4 Super Cyclone, Peak {peak_act:.0f} kt)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    gradcam_plot_path = plots_dir / f"cyclone_{cyclone_id}_peak_gradcam.png"
    plt.savefig(gradcam_plot_path)
    plt.close()

    # Save Results JSON
    summary_out = {
        "cyclone_id": cyclone_id,
        "storm_name": storm_name,
        "year": int(year),
        "frames": n_frames,
        "intensity_min_kt": float(min_v),
        "intensity_max_kt": float(max_v),
        "models": results_dict
    }
    with open(out_dir / f"cyclone_{cyclone_id}_test_results.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    print(f"\n[Saved Lifecycle Plot] {lifecycle_plot_path}")
    print(f"[Saved Grad-CAM Plot]  {gradcam_plot_path}")
    print(f"[Saved Results JSON]   {out_dir / f'cyclone_{cyclone_id}_test_results.json'}")

    return summary_out


def main():
    parser = argparse.ArgumentParser(description="Test Indian Ocean Cyclone")
    parser.add_argument("--cyclone-id", type=str, default="201004I", help="Cyclone ID or 'random'")
    args = parser.parse_args()
    test_indian_cyclone(cyclone_id=args.cyclone_id)


if __name__ == "__main__":
    main()
