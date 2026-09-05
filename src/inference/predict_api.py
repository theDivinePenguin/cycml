"""Standardized Prediction API for Dashboard and Operational Real-Time Inference."""
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from src.data.preprocessing import TCIRPreprocessor


class StandardizedPredictor:
    """Wraps any trained CycML model checkpoint into a unified prediction interface.

    Standardized output schema:
      - current_intensity: float
      - predicted_6h: float
      - predicted_12h: float
      - predicted_24h: float
      - delta_6h: float
      - delta_12h: float
      - delta_24h: float
      - ri_probability: Optional[float]
      - trend: Optional[str]  ("WEAKENING", "STABLE", "INTENSIFYING")
      - q10: Optional[Dict[str, float]]
      - q50: Optional[Dict[str, float]]
      - q90: Optional[Dict[str, float]]
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        device: str = "auto",
        preprocessor: Optional[TCIRPreprocessor] = None,
    ):
        self.dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        self.model = model.to(self.dev)
        self.model.eval()
        self.model_type = model_type.lower()
        self.preprocessor = preprocessor

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model: nn.Module,
        model_type: str,
        device: str = "auto",
        preprocessor: Optional[TCIRPreprocessor] = None,
    ) -> "StandardizedPredictor":
        dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        ckpt = torch.load(str(checkpoint_path), map_location=dev)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        return cls(model, model_type=model_type, device=device, preprocessor=preprocessor)

    @torch.no_grad()
    def predict(
        self,
        sequence: Union[np.ndarray, torch.Tensor],
        v_curr: float,
        vis_masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
        x_env: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Run inference on a single sequence.

        Args:
            sequence: (K, C, H, W) or (1, K, C, H, W) tensor/array
            v_curr: Current observed intensity in knots
            vis_masks: Optional (K,) or (1, K) daytime flags
            x_env: Optional (12,) or (1, 12) environmental features
        """
        # Format sequence to 5D tensor: (1, K, C, H, W)
        if isinstance(sequence, np.ndarray):
            x = torch.from_numpy(sequence).float()
        else:
            x = sequence.float()

        if x.ndim == 4:
            x = x.unsqueeze(0)  # (1, K, C, H, W)
        x = x.to(self.dev)

        # Format vis_masks
        vm = None
        if vis_masks is not None:
            if isinstance(vis_masks, np.ndarray):
                vm = torch.from_numpy(vis_masks).float()
            else:
                vm = vis_masks.float()
            if vm.ndim == 1:
                vm = vm.unsqueeze(0)
            vm = vm.to(self.dev)

        # Format environmental features
        env_t = None
        if x_env is not None:
            if isinstance(x_env, np.ndarray):
                env_t = torch.from_numpy(x_env).float()
            else:
                env_t = x_env.float()
            if env_t.ndim == 1:
                env_t = env_t.unsqueeze(0)
            env_t = env_t.to(self.dev)

        v_curr_tensor = torch.tensor([v_curr], dtype=torch.float32, device=self.dev)

        result: Dict[str, Any] = {
            "current_intensity": round(float(v_curr), 1),
        }

        # Model type dispatch
        if self.model_type == "residual":
            v_hat, delta_hat = self.model(x, v_curr=v_curr_tensor, vis_masks=vm)
            v_vals = v_hat[0].cpu().numpy()
            d_vals = delta_hat[0].cpu().numpy()
            result.update({
                "predicted_6h": round(float(v_vals[0]), 1),
                "predicted_12h": round(float(v_vals[1]), 1),
                "predicted_24h": round(float(v_vals[2]), 1),
                "delta_6h": round(float(d_vals[0]), 1),
                "delta_12h": round(float(d_vals[1]), 1),
                "delta_24h": round(float(d_vals[2]), 1),
            })

        elif self.model_type == "ri_dedicated":
            ri_logit = self.model(x, vis_masks=vm, x_env=env_t)
            ri_prob = float(torch.sigmoid(ri_logit)[0, 0].cpu().item())
            result["ri_probability"] = round(ri_prob, 4)

        elif self.model_type in ["ri_multitask", "multitask"]:
            intensity_preds, ri_logits, trend_logits = self.model(x, vis_masks=vm, x_env=env_t)
            v_vals = intensity_preds[0].cpu().numpy()
            ri_prob = float(torch.sigmoid(ri_logits)[0, 0].cpu().item())
            trend_idx = int(torch.argmax(trend_logits[0]).cpu().item())
            trend_map = {0: "WEAKENING", 1: "STABLE", 2: "INTENSIFYING"}

            result.update({
                "predicted_6h": round(float(v_vals[0]), 1),
                "predicted_12h": round(float(v_vals[1]), 1),
                "predicted_24h": round(float(v_vals[2]), 1),
                "delta_6h": round(float(v_vals[0] - v_curr), 1),
                "delta_12h": round(float(v_vals[1] - v_curr), 1),
                "delta_24h": round(float(v_vals[2] - v_curr), 1),
                "ri_probability": round(ri_prob, 4),
                "trend": trend_map.get(trend_idx, "UNKNOWN"),
            })

        elif self.model_type == "probabilistic":
            q_out = self.model(x, vis_masks=vm)  # (1, 3, 3)
            q_np = q_out[0].cpu().numpy()  # (3 horizons, 3 quantiles)
            # Medians (q50) as primary forecasts
            result.update({
                "predicted_6h": round(float(q_np[0, 1]), 1),
                "predicted_12h": round(float(q_np[1, 1]), 1),
                "predicted_24h": round(float(q_np[2, 1]), 1),
                "delta_6h": round(float(q_np[0, 1] - v_curr), 1),
                "delta_12h": round(float(q_np[1, 1] - v_curr), 1),
                "delta_24h": round(float(q_np[2, 1] - v_curr), 1),
                "q10": {
                    "+6h": round(float(q_np[0, 0]), 1),
                    "+12h": round(float(q_np[1, 0]), 1),
                    "+24h": round(float(q_np[2, 0]), 1),
                },
                "q50": {
                    "+6h": round(float(q_np[0, 1]), 1),
                    "+12h": round(float(q_np[1, 1]), 1),
                    "+24h": round(float(q_np[2, 1]), 1),
                },
                "q90": {
                    "+6h": round(float(q_np[0, 2]), 1),
                    "+12h": round(float(q_np[1, 2]), 1),
                    "+24h": round(float(q_np[2, 2]), 1),
                },
            })

        else:
            # Standard intensity forecaster (GRU or Transformer)
            preds = self.model(x, vis_masks=vm)
            v_vals = preds[0].cpu().numpy()
            result.update({
                "predicted_6h": round(float(v_vals[0]), 1),
                "predicted_12h": round(float(v_vals[1]), 1),
                "predicted_24h": round(float(v_vals[2]), 1),
                "delta_6h": round(float(v_vals[0] - v_curr), 1),
                "delta_12h": round(float(v_vals[1] - v_curr), 1),
                "delta_24h": round(float(v_vals[2] - v_curr), 1),
            })

        return result
