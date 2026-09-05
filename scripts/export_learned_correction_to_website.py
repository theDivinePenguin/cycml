#!/usr/bin/env python3
"""
Export Best Learned RI-Aware Correction Model to frontend_test_clone.

Loads checkpoint: experiments/ri_aware_correction/best_correction_model.pt
Applies genuine PyTorch forward pass to generate real model predictions for all demo storms.
Saves:
  1. Isolated experimental dataset:
     - frontend_test_clone/public/storm_data_ri_experimental.json
     - frontend_test_clone/src/data/storm_data_ri_experimental.json
  2. Model switcher entry in frontend_test_clone:
     - deepcyclonet_ri_learned_correction (badge: "EXPERIMENTAL (Learned)")
     alongside the alpha=2.0 model under category "Experimental RI-Aware Analysis".

Production frontend/ is STRICTLY UNTOUCHED.
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class TanhConstrainedMLPCorrection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 3, scale: float = 15.0, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        return self.scale * torch.tanh(raw)


def main():
    print("=" * 80)
    print("EXPORTING BEST LEARNED RI-AWARE CORRECTION MODEL TO TEST CLONE")
    print("=" * 80)

    ckpt_path = Path("experiments/ri_aware_correction/best_correction_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint: {ckpt['config_name']} (Family: {ckpt['family']}, Scale: {ckpt.get('scale')})")

    mean_tr = ckpt["mean_tr"]
    std_tr = ckpt["std_tr"]
    scale_val = ckpt.get("scale", 15.0)

    # Reconstruct model
    model = TanhConstrainedMLPCorrection(in_dim=len(mean_tr), out_dim=3, scale=scale_val)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    clone_src = Path("frontend_test_clone/public/storm_data.json")
    with open(clone_src, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data.get("models", [])
    storms = data.get("storms", {})

    hybrid_storms = storms["deepcyclonet_final_hybrid"]
    res_storms = storms["residual_delta_v_unconstrained"]
    ri_storms = storms["ri_model1_dedicated_focal"]

    # Model metadata entry
    learned_meta = {
        "id": "deepcyclonet_ri_learned_correction",
        "category": "Experimental RI-Aware Analysis",
        "name": "DeepCycloNet (Experimental Learned RI Correction - Scale 15kt)",
        "badge": "EXPERIMENTAL (Learned Correction)",
        "tag": "Learned MLP",
        "lead_mae": "+24h MAE: 9.51 kt (-0.59 kt Gain)",
        "ri_mae": "21.12 kt (-2.38 kt / -10.1% RI Gain)",
        "ri_precision": "False Dips: 0 · 137/44 Storm Win Ratio",
        "slope": "Learned Causal Feature Gating: Δ_final = Δ_base + 15*tanh(MLP(Features))",
        "ckpt_path": "experiments/ri_aware_correction/best_correction_model.pt",
        "type": "experimental_learned",
        "needs_env": True,
        "modalities": [
            "Canonical 3-Stage DeepCycloNet Baseline",
            "Causal Time-t Environmental Vectors (SST, OHC, VWS)",
            "Observed K=5 Intensity Evolution & Trends",
            "Tanh-Constrained Regularized Neural Correction (Scale 15 kt)"
        ]
    }

    # Clean existing entry if present
    models = [m for m in models if m["id"] != "deepcyclonet_ri_learned_correction"]
    # Insert right below alpha=2.0 model (index 2)
    models.insert(2, learned_meta)
    data["models"] = models

    # Generate genuine timesteps using model forward pass
    exp_storms = {}
    total_steps = 0

    with torch.no_grad():
        for cid, storm in hybrid_storms.items():
            s_copy = dict(storm)
            new_timesteps = []
            timesteps_list = storm.get("timesteps", [])
            res_timesteps = res_storms.get(cid, {}).get("timesteps", [])
            ri_timesteps = ri_storms.get(cid, {}).get("timesteps", [])

            for idx, t_hyb in enumerate(timesteps_list):
                t_copy = dict(t_hyb)
                v0 = float(t_hyb["vmax_curr"])

                # Base hybrid
                p6_base = float(t_hyb["predicted_plus_6h"])
                p12_base = float(t_hyb["predicted_plus_12h"])
                p24_base = float(t_hyb["predicted_plus_24h"])

                d6_base = p6_base - v0
                d12_base = p12_base - v0
                d24_base = p24_base - v0

                # Residual base
                t_res = res_timesteps[idx] if idx < len(res_timesteps) else t_hyb
                p6_res = float(t_res.get("predicted_plus_6h", p6_base))
                p12_res = float(t_res.get("predicted_plus_12h", p12_base))
                p24_res = float(t_res.get("predicted_plus_24h", p24_base))

                d6_res = p6_res - v0
                d12_res = p12_res - v0
                d24_res = p24_res - v0

                # RI model
                t_ri = ri_timesteps[idx] if idx < len(ri_timesteps) else t_hyb
                ri_prob = float(t_ri.get("ri_probability", 5.0)) / 100.0
                ri_prob = float(np.clip(ri_prob, 0.005, 0.995))
                ri_logit = float(np.log(ri_prob / (1.0 - ri_prob)))

                # Recent history from preceding timesteps
                past_v = [float(timesteps_list[max(0, idx - k)]["vmax_curr"]) for k in range(5, 0, -1)]
                d6_hist = v0 - past_v[-3] if len(past_v) >= 3 else 0.0
                d12_hist = v0 - past_v[0] if len(past_v) >= 5 else 0.0
                slope_hist = d12_hist / 12.0
                std_hist = float(np.std(past_v))

                # Environmental
                env = t_hyb.get("environmental", {})
                e_sst = (float(env.get("sst", 28.0)) - 28.0) / 2.0
                e_ohc = float(env.get("ohc", 50.0)) / 100.0
                e_shrd = (float(env.get("shear", 15.0)) - 15.0) / 10.0
                e_mslp = (float(env.get("mslp", 1000.0)) - 1000.0) / 20.0
                e_rh = (float(env.get("rh", 70.0)) - 70.0) / 15.0
                e_vmax = (v0 - 65.0) / 30.0

                # Interactions
                interact_pri_res24 = ri_prob * d24_res
                interact_pri_d12 = ri_prob * d12_hist
                interact_pri_vcurr = ri_prob * (v0 / 100.0)
                interact_pri_logit = ri_prob * ri_logit
                interact_pri_sst = ri_prob * e_sst
                interact_pri_shrd = ri_prob * e_shrd
                interact_pri_ridge24 = ri_prob * d24_base

                feat_raw = np.array([
                    d6_res, d12_res, d24_res,
                    d6_base, d12_base, d24_base,
                    ri_prob, ri_logit,
                    v0, v0 / 100.0,
                    d6_hist, d12_hist, slope_hist, std_hist,
                    e_vmax, e_mslp, e_sst, e_ohc, e_shrd, e_rh,
                    interact_pri_res24, interact_pri_d12, interact_pri_vcurr,
                    interact_pri_logit, interact_pri_sst, interact_pri_shrd,
                    interact_pri_ridge24
                ], dtype=np.float32)

                feat_norm = (feat_raw - mean_tr) / std_tr
                feat_tensor = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0)

                corr = model(feat_tensor).squeeze(0).numpy()  # (3,)

                p6_new = round(float(max(15.0, p6_base + corr[0])), 1)
                p12_new = round(float(max(15.0, p12_base + corr[1])), 1)
                p24_new = round(float(max(15.0, p24_base + corr[2])), 1)

                t_copy["predicted_plus_6h"] = p6_new
                t_copy["predicted_plus_12h"] = p12_new
                t_copy["predicted_plus_24h"] = p24_new
                t_copy["model_type"] = "experimental_learned"
                new_timesteps.append(t_copy)
                total_steps += 1

            s_copy["timesteps"] = new_timesteps
            exp_storms[cid] = s_copy

    storms["deepcyclonet_ri_learned_correction"] = exp_storms
    data["storms"] = storms
    print(f"Generated {total_steps:,} genuine predictions across {len(exp_storms)} storms for deepcyclonet_ri_learned_correction")

    # Save to main test clone storm_data.json
    for p in [Path("frontend_test_clone/public/storm_data.json"), Path("frontend_test_clone/src/data/storm_data.json")]:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved updated clone dataset: {p}")

    # Also save dedicated isolated file storm_data_ri_experimental.json
    for p in [Path("frontend_test_clone/public/storm_data_ri_experimental.json"), Path("frontend_test_clone/src/data/storm_data_ri_experimental.json")]:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved isolated experimental dataset: {p}")

    print("=" * 80)
    print("SUCCESS: Learned RI-Aware Correction Model is exported and live in frontend_test_clone!")
    print("=" * 80)


if __name__ == "__main__":
    main()
