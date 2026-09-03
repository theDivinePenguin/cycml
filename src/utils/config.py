"""Configuration loading and management utilities."""
from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary with parsed configuration parameters.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], save_path: str | Path) -> None:
    """Save configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary.
        save_path: Destination file path.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
