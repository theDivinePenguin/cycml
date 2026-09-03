"""Standalone script to test any single cyclone satellite image on the trained model."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
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
        return "Tropical Storm"
    elif 48 <= wind_speed < 64:
        return "Severe Tropical Storm"
    elif 64 <= wind_speed < 83:
        return "Category 1 Hurricane"
    elif 83 <= wind_speed < 96:
        return "Category 2 Hurricane"
    elif 96 <= wind_speed < 113:
        return "Category 3 Major Hurricane"
    elif 113 <= wind_speed < 137:
        return "Category 4 Major Hurricane"
    else:
        return "Category 5 Major Hurricane"


def test_cyclone_image(
    image_path: str | Path,
    checkpoint_path: str | Path = "experiments/baseline_resnet18_cpac_io_sh/best.pt",
    stats_path: str | Path = "data/metadata/normalization_stats_CPAC_IO_SH.json",
    config_path: str | Path = "configs/baseline.yaml",
    storm_name: str = "Test Cyclone",
    ground_truth_kt: float | None = None,
    output_dir: str | Path = "experiments/single_tests"
) -> dict:
    """Run inference on a single cyclone image and generate a visualization card."""
    img_p = Path(image_path)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(config_path)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]

    # Load image (supports NPY, PNG, JPG)
    if img_p.suffix.lower() == ".npy":
        img_np = np.load(img_p)
        if img_np.ndim == 3 and img_np.shape[-1] == 4:
            img_np = img_np[:, :, 0]
    else:
        pil_img = Image.open(img_p).convert("L")
        img_raw = np.array(pil_img, dtype=np.float32)
        # If standard 0-255 image, calibrate to satellite IR brightness temperature (Kelvin ~ 170K-310K)
        if img_raw.max() <= 255.0 and img_raw.min() >= 0.0:
            img_np = 310.0 - (img_raw / 255.0) * (310.0 - 175.0)
        else:
            img_np = img_raw

    tensor = torch.from_numpy(img_np).unsqueeze(0).float()  # (1, H, W)

    preprocessor = TCIRPreprocessor(
        mean=mean,
        std=std,
        target_size=tuple(config.get("dataset", {}).get("input_size", [224, 224])),
        is_training=False,
        augmentation_cfg={"enabled": False}
    )
    processed = preprocessor(tensor).unsqueeze(0).to(dev)

    model = build_model(config).to(dev)
    checkpoint = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        predicted_knots = float(model(processed).item())

    category = wind_speed_to_category(predicted_knots)

    print("\n" + "=" * 55)
    print(f"CYCLONE INTENSITY TEST: {storm_name}")
    print("=" * 55)
    print(f"Source Image:     {img_p.name}")
    print(f"Predicted Speed:  {predicted_knots:.1f} knots ({predicted_knots * 1.852:.1f} km/h)")
    print(f"Category:         {category}")
    if ground_truth_kt is not None:
        err = predicted_knots - ground_truth_kt
        print(f"Ground Truth:     {ground_truth_kt:.1f} knots")
        print(f"Absolute Error:   {abs(err):.1f} knots ({err:+.1f} kt)")
    print("=" * 55)

    # Visualization Card
    plt.figure(figsize=(7, 7), dpi=150)
    im = plt.imshow(img_np, cmap="inferno")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Calibrated IR Brightness Temperature (K)")

    title_str = f"Cyclone: {storm_name}\nPredicted: {predicted_knots:.1f} kt ({category})"
    if ground_truth_kt is not None:
        title_str += f"\nActual: {ground_truth_kt:.1f} kt | Error: {predicted_knots - ground_truth_kt:+.1f} kt"

    plt.title(title_str, fontsize=12, fontweight="bold", pad=12)
    plt.axis("off")

    card_path = out_p / f"{img_p.stem}_inference_card.png"
    plt.savefig(card_path, bbox_inches="tight")
    plt.close()
    print(f"[Card Saved] {card_path}")

    return {
        "storm_name": storm_name,
        "predicted_knots": predicted_knots,
        "category": category,
        "card_path": str(card_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Test individual cyclone image on trained model.")
    parser.add_argument("--image", type=str, required=True, help="Path to cyclone image")
    parser.add_argument("--name", type=str, default="Satellite Cyclone", help="Cyclone name/label")
    parser.add_argument("--actual", type=float, default=None, help="Actual ground truth wind speed in knots (optional)")
    parser.add_argument("--checkpoint", type=str, default="experiments/baseline_resnet18_cpac_io_sh/best.pt", help="Path to checkpoint")
    parser.add_argument("--stats", type=str, default="data/metadata/normalization_stats_CPAC_IO_SH.json", help="Path to normalization stats")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    args = parser.parse_args()

    test_cyclone_image(
        image_path=args.image,
        storm_name=args.name,
        ground_truth_kt=args.actual,
        checkpoint_path=args.checkpoint,
        stats_path=args.stats,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
