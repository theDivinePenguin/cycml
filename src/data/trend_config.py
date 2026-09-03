"""Configurable threshold definitions and labeling utilities for Intensity Trend & Rapid Intensification."""
from dataclasses import dataclass, field
from typing import Dict, Union
import numpy as np


@dataclass
class IntensityTrendConfig:
    """Configurable thresholds for 24-hour tropical cyclone intensity evolution and rapid intensification."""

    # 24h delta V thresholds (knots)
    weakening_threshold_kt: float = -10.0
    intensifying_threshold_kt: float = 10.0
    ri_threshold_kt: float = 30.0

    # Probability thresholds for operational RI risk categorisation
    # P(RI) < low_risk_cutoff -> LOW
    # low_risk_cutoff <= P(RI) < med_risk_cutoff -> MEDIUM
    # P(RI) >= med_risk_cutoff -> HIGH
    low_risk_cutoff: float = 0.25
    med_risk_cutoff: float = 0.60

    # Class names and indices
    trend_classes: Dict[int, str] = field(
        default_factory=lambda: {
            0: "WEAKENING",
            1: "STABLE",
            2: "INTENSIFYING",
        }
    )

    def compute_trend_label(self, delta_v_24: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Map 24h intensity change to trend class index:
        0: WEAKENING (delta_v_24 <= -10 kt)
        1: STABLE (-10 kt < delta_v_24 < 10 kt)
        2: INTENSIFYING (delta_v_24 >= 10 kt)
        """
        if isinstance(delta_v_24, (int, float)):
            if delta_v_24 <= self.weakening_threshold_kt:
                return 0
            elif delta_v_24 >= self.intensifying_threshold_kt:
                return 2
            else:
                return 1
        else:
            arr = np.asarray(delta_v_24)
            labels = np.ones(arr.shape, dtype=np.int64)
            labels[arr <= self.weakening_threshold_kt] = 0
            labels[arr >= self.intensifying_threshold_kt] = 2
            return labels

    def compute_ri_label(self, delta_v_24: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Map 24h intensity change to binary RI label:
        1: Rapid Intensification (delta_v_24 >= 30 kt)
        0: Non-RI (delta_v_24 < 30 kt)
        """
        if isinstance(delta_v_24, (int, float)):
            return 1 if delta_v_24 >= self.ri_threshold_kt else 0
        else:
            arr = np.asarray(delta_v_24)
            return (arr >= self.ri_threshold_kt).astype(np.int64)

    def get_trend_name(self, label: int) -> str:
        """Return human-readable string for trend class index."""
        return self.trend_classes.get(int(label), "UNKNOWN")

    def get_ri_risk_level(self, ri_prob: float) -> str:
        """Convert predicted RI probability to operational risk level."""
        if ri_prob >= self.med_risk_cutoff:
            return "HIGH"
        elif ri_prob >= self.low_risk_cutoff:
            return "MEDIUM"
        else:
            return "LOW"
