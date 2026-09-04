# Existing Pipeline Audit — Baseline K=7 Multi-Modal System

**Target Baseline Checkpoint**: `experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/best.pt`  
**Checkpoint SHA256**: `609841410eeafddfd20f53d4f0237b16c670e94acc60a6af0d22d65223eac56a`  
**Test Predictions CSV SHA256**: `1ddd212f305a248b17aa2785226a104cfe01814f0d534f5fcd1c118a69b48bea`  
**Test Manifest SHA256**: `2edb9c6511743a7feeefc359850703870195c98aa33838b5d9f32a61d31da77a`  
**Audit Date**: 2026-09-04

---

## 1. Exact Files Inspected

1. `src/models/environmental_temporal_classifier.py` — Multi-Modal Environmental Temporal Architecture.
2. `src/models/temporal_forecaster.py` — CNNFeatureEncoder (ResNet-18) and PositionalEncoding modules.
3. `src/models/temporal_classifier.py` — JointTrendRILoss multi-task loss definition.
4. `src/data/sequence_dataset.py` — TCIRSequenceDataset base frame loader with HDF5 SWMR caching.
5. `src/data/trend_dataset.py` — TCIRTrendDataset subclass adding trend labels, RI targets, and environmental vectors.
6. `src/data/trend_config.py` — IntensityTrendConfig threshold and label definitions.
7. `scripts/train_environmental_classifier.py` — Training loop, optimizer, learning rate schedule, validation checkpoints.
8. `scripts/evaluate_environmental_classifier.py` — Test set evaluation and prediction CSV export.
9. `data/metadata/normalization_stats_multichannel.json` — Precomputed channel-wise mean and std.
10. `experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/best.pt` — Frozen canonical benchmark weights.
11. `experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_metrics.json` — Benchmark performance numbers.
12. `experiments/environmental_fusion/checkpoints/exp_e_k7_12ep_clean/test_predictions.csv` — Baseline prediction records.

---

## 2. Existing Architecture Audit

The canonical system (`EnvironmentalTemporalClassifier`) is composed of:

### Visual Feature Extraction Branch
- **Spatial Encoder**: ResNet-18 initialized with ImageNet-1K pretrained weights (`IMAGENET1K_V1`). The classification layer `fc` is replaced with `nn.Identity()`, yielding a 512-dimensional spatial feature vector for each individual satellite frame.
- **Linear Feature Projection**: `nn.Linear(512, d_model)` with $d_{\text{model}} = 256$.
- **Explicit VIS Validity Gate**: `nn.Linear(1, d_model)` adds a learned daytime/nighttime embedding to each token:
  $$\mathbf{z}_t = \mathbf{W}_{\text{proj}} \mathbf{f}_t + \mathbf{W}_{\text{vis}} v_t$$
  where $v_t \in \{0.0, 1.0\}$ represents whether solar reflectance exceeds 10% of pixels.
- **Sinusoidal Positional Encoding**: Fixed sinusoidal table with `max_len = 32` (instantiated with `max_len = 10`). Adds positional embeddings to tokens:
  $$\mathbf{tokens} = \mathbf{tokens} + \mathbf{PE}_{:, :K, :}$$
- **Temporal Sequence Transformer**: 2-layer Transformer Encoder (`nn.TransformerEncoder`) with `batch_first=True`, `norm_first=True`, $n_{\text{heads}} = 8$, $d_{\text{ff}} = 1024$, and `GELU` activations with dropout $= 0.1$.
- **Temporal Summary Token**: Evaluated as the final token of the sequence (corresponding to time $t$):
  $$\mathbf{h}_{\text{vis}} = \mathbf{tokens}_{:, -1, :}$$

### Environmental Feature Branch
- **Input Dimension**: 12 features (6 continuous variables + 6 binary missingness indicators).
  - Continuous variables: $V_{\max}$, MSLP, SST, Ocean Heat Content (OHC), Vertical Wind Shear, Mid-Level Relative Humidity (RH).
  - Missingness indicators: $0.0$ if present, $1.0$ if imputed/missing.
- **Environmental MLP**:
  $$\mathbf{h}_{\text{env}} = \text{Linear}(12, 128) \to \text{LayerNorm} \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(128, 64) \to \text{LayerNorm} \to \text{GELU}$$

### Multi-Modal Fusion Layer
- Gated residual projection:
  $$\mathbf{h}_{\text{fused}} = \text{LayerNorm}\left(\mathbf{h}_{\text{vis}} + \mathbf{W}_2 \, \text{GELU}\left(\text{LayerNorm}\left(\mathbf{W}_1 [\mathbf{h}_{\text{vis}} \,\|\, \mathbf{h}_{\text{env}}] \right)\right)\right)$$
  yielding a 256-dimensional unified storm representation.

### Multi-Task Prediction Heads
1. **Rapid Intensification (RI) Head**:
   - `Linear(256, 128) -> GELU -> Dropout(0.1) -> Linear(128, 1)` $\to$ raw logit $z_{\text{ri}}$.
   - Predicts $P(\Delta V_{24} \ge +30\text{ kt})$.
2. **24-Hour Intensity Trend Head**:
   - `Linear(256, 128) -> GELU -> Dropout(0.1) -> Linear(128, 3)` $\to$ 3 logits $[z_{\text{weak}}, z_{\text{stab}}, z_{\text{inte}}]$.
   - Predicts dynamic category based on $\Delta V_{24}$: Class 0 ($\le -10$ kt), Class 1 ($[-10, +10]$ kt), Class 2 ($\ge +10$ kt).
3. **Multi-Horizon Numerical Regression Head**:
   - `Linear(256, 128) -> ReLU -> Dropout(0.1) -> Linear(128, 3)` $\to$ continuous intensities $[\hat{V}_{t+6\text{h}}, \hat{V}_{t+12\text{h}}, \hat{V}_{t+24\text{h}}]$.

---

## 3. Current K=7 Sequence Format

The primary dataset manifests are stored under `data/metadata/forecast_{train,val,test}_sequences_k7.csv`.
- Each row contains:
  - `cyclone_id`: Unique identifier (e.g. `200522S`).
  - `target_t_timestamp`: Observation timestamp $t$ (e.g. `2005031006`).
  - `history_timestamps`: JSON list of 7 timestamps: $[t-18\text{h}, t-15\text{h}, t-12\text{h}, t-9\text{h}, t-6\text{h}, t-3\text{h}, t]$.
  - `history_h5_files`: Paths to TCIR HDF5 files containing the 7 frames.
  - `history_h5_rows`: Matrix indices within each HDF5 file.
  - `vmax_curr`: Maximum sustained wind speed at time $t$.
  - `vmax_plus_6h`, `vmax_plus_12h`, `vmax_plus_24h`: Future ground-truth intensities.
- In every sequence, the current observation frame $t$ is strictly located at the **last index** (index 6).
- Subsequence slicing:
  - $K=3$: Selects frames at indices `[4, 5, 6]` $\implies [t-6\text{h}, t-3\text{h}, t]$.
  - $K=5$: Selects frames at indices `[2, 3, 4, 5, 6]` $\implies [t-12\text{h}, t-9\text{h}, t-6\text{h}, t-3\text{h}, t]$.
  - $K=7$: Selects all 7 frames `[0, 1, 2, 3, 4, 5, 6]` $\implies [t-18\text{h}, \dots, t]$.

---

## 4. Current Normalization

Precomputed training-derived stats from `data/metadata/normalization_stats_multichannel.json`:
- Channel 0 (**IR1**): $\mu = 247.9622\text{ K}$, $\sigma = 29.5694\text{ K}$
- Channel 1 (**WV**): $\mu = 237.4912\text{ K}$, $\sigma = 10.9702\text{ K}$
- Channel 2 (**VIS**): $\mu = 0.1065$, $\sigma = 0.1706$
- Applied per pixel as:
  $$x_{\text{norm}} = \frac{x - \mu}{\sigma + 10^{-7}}$$
- Daytime detection: frame is classified as daytime ($v_t = 1.0$) if $> 10\%$ of raw VIS pixels exceed $0.01$ reflectance; otherwise $v_t = 0.0$ and VIS pixels are masked to $0.0$.

---

## 5. Current Data Split

Strict cyclone-disjoint split by unique `cyclone_id`:
- **Training Set**: 36,343 sequences across 887 distinct tropical cyclones.
- **Validation Set**: 8,396 sequences across 211 distinct tropical cyclones.
- **Held-Out Test Set**: 7,901 sequences across 187 distinct tropical cyclones.
- **Leakage Safeguards**: No cyclone appears in more than one partition. Target timestamps and environmental features are strictly causal.

---

## 6. Current Multi-Task Loss Formulation

Defined in `JointTrendRILoss`:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{ri}} \mathcal{L}_{\text{ri}} + \lambda_{\text{trend}} \mathcal{L}_{\text{trend}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$$
with weights $\lambda_{\text{ri}} = 1.0$, $\lambda_{\text{trend}} = 1.0$, $\lambda_{\text{reg}} = 0.1$:
1. **$\mathcal{L}_{\text{ri}}$ (Binary Cross-Entropy with Positive Class Weighting)**:
   $$\mathcal{L}_{\text{ri}} = \text{BCEWithLogitsLoss}(z_{\text{ri}}, y_{\text{ri}}; w_{\text{pos}})$$
   where $w_{\text{pos}} = \frac{N_{\text{total}} - N_{\text{ri}}}{N_{\text{ri}}} \approx 13.8$ based on training set RI prevalence (6.87%).
2. **$\mathcal{L}_{\text{trend}}$ (Weighted Multi-Class Cross-Entropy)**:
   $$\mathcal{L}_{\text{trend}} = \text{CrossEntropyLoss}(\mathbf{z}_{\text{trend}}, y_{\text{trend}}; \mathbf{w}_{\text{class}})$$
   where class weights are inversely proportional to training frequency across [Weakening, Stable, Intensifying].
3. **$\mathcal{L}_{\text{reg}}$ (Smooth L1 Loss)**:
   $$\mathcal{L}_{\text{reg}} = \frac{1}{3} \sum_{h \in \{6, 12, 24\}} \text{SmoothL1Loss}(\hat{V}_{t+h}, V_{t+h}; \beta=1.0)$$

---

## 7. Optimizer, Scheduler & Checkpoint Selection

- **Optimizer**: `AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)`
- **Learning Rate Schedule**: `CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)`
- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- **Precision**: PyTorch Automatic Mixed Precision (`torch.amp.autocast('cuda')` with `GradScaler`)
- **Primary Checkpoint Criterion**: Best Validation RI PR-AUC (`best_ri_pr_auc.pt`, mirrored to `best.pt`).
- **Validation Decision Threshold Selection**: During validation evaluation, `find_optimal_threshold` scans the precision-recall curve on the validation set to select the optimal threshold $\tau_{\text{val}}$ that maximizes validation F1. In `exp_e_k7_12ep_clean`, $\tau_{\text{val}} = 0.0161$.

---

## 8. Architectural Compatibility with Arbitrary Sequence Lengths

### Findings from Direct Inspection & Forward Pass Verification
1. **CNN Spatial Encoder**: Processes each frame independently (`(B * K, C, H, W) -> (B * K, 512)`). Fully sequence-length agnostic.
2. **Linear Projections & VIS Gating**: Operate on token tensors of shape `(B, K, d_model)`. Fully sequence-length agnostic.
3. **Positional Encoding**: `PositionalEncoding` table has `max_len = 32`. Slicing `self.pe[:, :x.size(1), :]` dynamically accommodates any length $K \le 32$.
4. **Transformer Encoder**: Standard PyTorch `nn.TransformerEncoderLayer` with `batch_first=True` processes any sequence length $K \ge 1$ without structural modification.
5. **Token Pooling**: Slices `temporal_out[:, -1, :]`. Because the sequence is constructed so the current frame $t$ is always at the last index, the summary token always represents time $t$.
6. **Incompatibilities Discovered**:
   - **None in model architecture**. The model natively runs on $K=3, 5, 7$ without throwing errors.
   - **DataLoaders batching requirement**: When creating mini-batches in PyTorch, all samples within a single tensor batch must have identical tensor dimensions `(B, K, C, H, W)`. Therefore, variable-$K$ sampling must be applied at the mini-batch level (or via a dynamic collator that selects uniform $K \in \{3, 5, 7\}$ per batch), ensuring every sample in a given batch receives the same temporal context length without requiring artificial padding tokens.
