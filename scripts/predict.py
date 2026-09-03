"""Inference script for estimating intensity on a single satellite image."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.data.preprocessing import TCIRPreprocessor
from src.models.factory import build_model
from src.utils.config import load_config


def wind_speed_to_category(wind_speed: float) -> str:
    """Convert wind speed in knots into standard meteorological intensity category."""
    if wind_speed < 34:
        return "Tropical Depression"
    elif 34 <= wind_speed < 48:
        return "Tropical Storm / Moderate Cyclonic Storm"
    elif 48 <= wind_speed < 64:
        return "Severe Cyclonic Storm"
    elif 64 <= wind_speed < 90:
        return "Very Severe Cyclonic Storm (Cat 1-2 equivalent)"
    elif 90 <= wind_speed < 120:
        return "Extremely Severe Cyclonic Storm (Cat 3-4 equivalent)"
    else:
        return "Super Cyclonic Storm (Cat 5 equivalent)"


def predict_single_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    stats_path: str | Path,
    config_path: str | Path,
    device: str = "auto"
) -> dict:
    """Run inference on an individual satellite image."""
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Input image not found: {img_p}")

    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")

    config = load_config(config_path)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]

    # Load image
    if img_p.suffix.lower() == ".npy":
        img_np = np.load(img_p)
        if img_np.ndim == 3 and img_np.shape[-1] == 4:
            img_np = img_np[:, :, 0]  # Extract IR1
    else:
        pil_img = Image.open(img_p).convert("L")
        img_np = np.array(pil_img, dtype=np.float32)

    tensor = torch.from_numpy(img_np).unsqueeze(0).float()  # (1, H, W)

    preprocessor = TCIRPreprocessor(
        mean=mean,
        std=std,
        target_size=tuple(config.get("dataset", {}).get("input_size", [224, 224])),
        is_training=False,
        augmentation_cfg={"enabled": False}
    )
    processed_tensor = preprocessor(tensor).unsqueeze(0).to(dev)  # (1, 1, 224, 224)

    model = build_model(config).to(dev)
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred_output = model(processed_tensor)
        predicted_knots = float(pred_output.item())

    category = wind_speed_to_category(predicted_knots)

    print("\n" + "=" * 50)
    print("CYCLONE INTENSITY ESTIMATION")
    print("=" * 50)
    print(f"Image:            {img_p.name}")
    print(f"Estimated Speed:  {predicted_knots:.1f} knots ({predicted_knots * 1.852:.1f} km/h)")
    print(f"Category:         {category}")
    print("=" * 50)

    return {
        "predicted_knots": predicted_knots,
        "predicted_kmh": predicted_knots * 1.852,
        "category": category
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate cyclone intensity from a single satellite image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (.png, .jpg, .npy)")
    parser.add_argument("--checkpoint", type=str, default="experiments/baseline_resnet18_cpac_io_sh/best.pt", help="Path to best.pt")
    parser.add_argument("--stats", type=str, default="data/metadata/normalization_stats_CPAC_IO_SH.json", help="Path to normalization_stats.json")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    args = parser.parse_args()

    predict_single_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        stats_path=args.stats,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
