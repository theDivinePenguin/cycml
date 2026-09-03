# SIH Problem Statement 26070: Comprehensive Technical & Operational Project Report
## AI-Powered Tropical Cyclone Evolution & Rapid Intensification (RI) Prediction

---

## Executive Summary

This report documents the end-to-end research, engineering, validation, and operational deployment of an AI system addressing **Smart India Hackathon (SIH) Problem Statement 26070: "Prediction of Tropical Cyclone Patterns and Intensity"**.

Across global ocean basins, the single most hazardous meteorological event is **Tropical Cyclone Rapid Intensification (RI)**—defined by the World Meteorological Organization (WMO) and National Hurricane Center (NHC) as a maximum sustained wind speed ($V_{\max}$) increase of **$\ge 30\text{ knots}$ ($\approx 55\text{ km/h}$) within 24 hours**. Operational numerical weather prediction (NWP) models frequently suffer from structural phase lag and systematically under-forecast rapid intensification, resulting in delayed coastal evacuations and catastrophic loss of life and infrastructure.

### The Strategic Pivot: Why Continuous $V_{\max}$ MAE Was Replaced
Our initial investigations evaluated continuous multi-horizon numerical forecasting ($+6\text{h}/+12\text{h}/+24\text{h}$) using Spatio-Temporal Convolutional Neural Networks, GRUs, and Transformers. While the Temporal Transformer achieved a competitive $+24\text{h}$ Mean Absolute Error (MAE) of **11.56 kt**, continuous regression inherently penalizes high-variance outliers, forcing the network to hedge toward the climatological mean. Consequently, when continuous regression forecasts were thresholded to detect Rapid Intensification, the model **missed over 80% of explosive intensification events** ($F_1 = 0.2718$, recall $19.8\%$).

We executed a strategic pivot from numerical regression to an **operational Rapid Intensification and macro-dynamic evolution system**:
1. **Primary Objective (Headline Task)**: Predict whether Rapid Intensification ($\Delta V_{24} \ge +30\text{ kt}$) will occur over the next 24 hours ($P(\text{RI})$).
2. **Secondary Objective (Macro Dynamics)**: Classify the 24-hour intensity trend into three operational regimes: **Weakening** ($\le -10\text{ kt}$), **Stable** ($(-10, +10)\text{ kt}$), or **Intensifying** ($\ge +10\text{ kt}$).
3. **Supporting Auxiliary Objective**: Multi-horizon quantitative intensity guidance ($+6\text{h}, +12\text{h}, +24\text{h}$).

### Key Benchmark Achievements on Held-Out Unseen Cyclones ($N=8,279$)
* **5.4× Discrimination Above Climatological Baseline**: Achieved an RI Precision-Recall Area Under Curve (**PR-AUC**) of **`0.3690`** on the held-out test set compared to the random prevalence baseline of `0.0682`.
* **High RI Discrimination**: Achieved an RI Receiver Operating Characteristic (**ROC-AUC**) of **`0.8687`**.
* **+51.4% Relative $F_1$ Improvement**: Achieved an RI $F_1$ of **`0.4114`** compared to `0.2718` for the thresholded continuous forecaster, increasing detected RI events from **112 to 360 events** (a $3.2\times$ increase in detected life-threatening storms).
* **Category 1–2 Hurricane/Typhoon Explosive Detection**: Captured **75.86% of RI events** ($176 / 232$) in Category 1–2 storms with an RI PR-AUC of **`0.5163`**.
* **Zero-Leakage Generalization**: Evaluated on strictly unseen held-out cyclones across diverse basins, including **Super Typhoon Megi (100% RI recall)**, **Cyclone Percy (100% RI recall)**, **Hurricane Matthew (54.5% RI recall)**, and **Super Cyclone Phet (53.8% RI recall, 77.6% trend accuracy)**.

---

## 1. Dataset & Data Engineering

### 1.1 The Tropical Cyclone Image and Track Dataset (TCIR)
The project utilizes the benchmark **TCIR dataset**, comprising multi-source satellite observations for **885 global tropical cyclones** across four oceanic domains:
1. **WPAC**: Western North Pacific
2. **ATLN**: North Atlantic
3. **EPAC**: Eastern North Pacific
4. **CPAC / IO / SH**: Central Pacific, North & South Indian Ocean, Southern Hemisphere

Each observation timestamp contains multi-spectral and microwave imagery centered on the storm's vortex:
* **Infrared Channel 1 (IR1)**: 10.8 µm cloud-top brightness temperature (all-weather, 24/7 coverage).
* **Water Vapor (WV)**: 6.7 µm upper-tropospheric moisture and mid-level atmospheric dynamics.
* **Visible (VIS)**: 0.65 µm high-resolution cloud morphology and eye structure (daytime only).
* **Passive Microwave (PMW)**: Low-frequency rainband and deep eyewall precipitation structure.

### 1.2 Multi-Frame Sequence Construction
To capture temporal momentum, the input is structured as a **5-frame historical sequence** sampled at 3-hour intervals:
$$\mathcal{S}_t = \left[ \mathbf{X}_{t-12\text{h}}, \mathbf{X}_{t-9\text{h}}, \mathbf{X}_{t-6\text{h}}, \mathbf{X}_{t-3\text{h}}, \mathbf{X}_t \right], \quad \mathbf{X}_\tau \in \mathbb{R}^{3 \times 201 \times 201}$$
Active channels selected: **`[0: IR1, 1: WV, 2: VIS]`**.

### 1.3 Day/Night Visible Channel Gating
Because visible satellite imagery is completely dark or non-existent at night, raw missing values (`NaN`) or uncalibrated zero-radiances distort convolutional activations. We engineered an **explicit VIS validity vector**:
$$m_{\text{vis}}(\tau) = \begin{cases} 1.0 & \text{if VIS channel contains valid daytime radiances} \\ 0.0 & \text{if nighttime / missing radiances} \end{cases}$$
When $m_{\text{vis}} = 0$, missing pixels are imputed to channel mean, and the boolean mask is projected through a learned linear embedding layer and fused into the temporal token representations.

### 1.4 Strict Anti-Leakage Cyclone Splitting
Standard random sample splitting causes severe data leakage: consecutive 3-hour frames of the same cyclone share identical thermodynamic background environments. We implemented **cyclone-level grouped splitting**:

```text
Global Cyclone Manifests (885 Unique Storms, 46,376 Sequences)
┌──────────────────────────────────────────────────────────────┐
│  TRAIN SPLIT (60%) : 531 Cyclones, 38,097 5-Frame Sequences  │
│  VAL SPLIT   (20%) : 163 Cyclones,  8,773 5-Frame Sequences  │
│  TEST SPLIT  (20%) : 191 Cyclones,  8,279 5-Frame Sequences  │
└──────────────────────────────────────────────────────────────┘
```
**No storm appearing in the training split is ever present in validation or test.** Test evaluations are performed solely on unseen cyclones.

---

## 2. Phase 1–4 Retrospective: Continuous Forecasting & The MAE Trap

### 2.1 The Continuous Forecasting Experiments
We initially built and compared five continuous forecasting models predicting $V_{\max}$ at $+6\text{h}$, $+12\text{h}$, and $+24\text{h}$:
1. **Baseline A (Persistence)**: $V_{\max}(t + \tau) = V_{\max}(t)$.
2. **Baseline B (Recent Extrapolation)**: Linear continuation of the 6-hour observed trend: $V_{\max}(t + \tau) = V_{\max}(t) + \frac{\tau}{6}(V_{\max}(t) - V_{\max}(t-6\text{h}))$.
3. **CNN-LSTM Forecaster**: ResNet-18 spatial feature extractor coupled to a 2-layer Long Short-Term Memory network.
4. **CNN-GRU Forecaster**: ResNet-18 spatial encoder coupled to a 2-layer Gated Recurrent Unit.
5. **CNN-Transformer Forecaster**: ResNet-18 encoder coupled to a 2-layer sinusoidal-encoded Temporal Transformer ($d_{\text{model}} = 256$, 8 attention heads).

### 2.2 Continuous Benchmark Ladder Results
Evaluated on the held-out test set ($N=8,279$ sequences):

| Model Architecture | +6h MAE (kt) | +12h MAE (kt) | +24h MAE (kt) | Multi-Horizon Mean MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 5.12 | 8.84 | 14.72 | 9.56 |
| **Recent Trend Baseline** | 6.89 | 11.42 | 19.85 | 12.72 |
| **CNN-LSTM** | 4.82 | 7.91 | 12.44 | 8.39 |
| **CNN-GRU** | 4.76 | 7.84 | 12.21 | 8.27 |
| **CNN-Transformer** | **4.51** | **7.42** | **11.56** | **7.83** |

### 2.3 The Core Limitation: The Regression Hedging Penalty
While the CNN-Transformer reduced $+24\text{h}$ MAE to 11.56 kt, deep diagnostic evaluation revealed an operational flaw:
* Under continuous L1 or Smooth L1 loss, predicting a sudden jump of $+50\text{ kt}$ incurs a massive penalty if the storm only intensifies by $+20\text{ kt}$.
* As a consequence, continuous neural regressors hedge their outputs toward the conditional mean ($+5$ to $+10\text{ kt}$).
* When thresholding the continuous forecast to detect Rapid Intensification ($\Delta \hat{V}_{24} \ge +30\text{ kt}$), the model achieved:
  * **Precision**: $43.2\%$
  * **Recall**: **$19.8\%$ (Missed 453 out of 565 RI events)**
  * **$F_1$ Score**: **`0.2718`**
* **Conclusion**: Squeezing continuous MAE does not solve the operational forecasting problem. Operational forecasters do not fail because a forecast was off by 2 knots; they fail when they miss a Category 1 storm exploding into a Category 5 super typhoon overnight.

---

## 3. The Multi-Task Classification Architecture (`TemporalClassifier`)

### 3.1 Network Topology
To directly target operational utility, we engineered the unified multi-task `TemporalClassifier`:

```text
INPUT: 5 Multi-Channel Satellite Frames [t-12h, t-9h, t-6h, t-3h, t]
       Shape: (Batch, K=5, Channels=3, H=201, W=201)
                             │
                             ▼
              Shared ResNet-18 Spatial Encoder
         Weights shared across all 5 temporal timesteps
                             │
                             ▼ (Batch, K=5, 512)
             Linear Projection to d_model (256-d)
                             │
                             ▼
                 VIS Validity Mask Fusion
   Linear embedding of day/night flags added to visual features
                             │
                             ▼
             Sinusoidal Positional Encoding
                             │
                             ▼
                 Temporal Transformer Encoder
                 2 Layers, 8 Heads, d_ff=1024
                             │
                             ▼
            Final Timestep Feature Vector h_t (256-d)
           ──────────────────┼──────────────────
          │                  │                  │
          ▼                  ▼                  ▼
    Headline RI Head    Macro Trend Head    Auxiliary Vmax Head
      Linear(256, 128)    Linear(256, 128)    Linear(256, 128)
          GELU()              GELU()              ReLU()
      Linear(128, 1)      Linear(128, 3)      Linear(128, 3)
          Sigmoid             Softmax             Identity
          │                  │                  │
          ▼                  ▼                  ▼
     P(RI in 24h)     [Weak, Stab, Inte]    [+6h, +12h, +24h]
```

### 3.2 Joint Loss Formulation with Dynamic Positive Class Weighting
Rapid Intensification is an inherently imbalanced event: in the training split of 38,097 sequences, only **2,575 sequences (6.76%)** are true RI events ($N_{\text{neg}} / N_{\text{pos}} = 13.795$).

The network is trained with a composite multi-task objective:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{ri}} \mathcal{L}_{\text{BCE\_RI}} + \lambda_{\text{trend}} \mathcal{L}_{\text{CE\_Trend}} + \lambda_{\text{reg}} \mathcal{L}_{\text{Aux\_Reg}}$$

1. **Binary Cross-Entropy with Dynamic Class Weighting for RI**:
   $$\mathcal{L}_{\text{BCE\_RI}} = - \left[ w_{\text{pos}} \cdot y_{\text{ri}} \log \sigma(z_{\text{ri}}) + (1 - y_{\text{ri}}) \log(1 - \sigma(z_{\text{ri}})) \right]$$
   where $w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}} = 13.795$, preventing the network from trivializing the gradient toward non-RI.
2. **Class-Balanced Cross-Entropy for 3-Class Macro Trend**:
   Inverse-frequency class weights applied: $\mathbf{w}_{\text{trend}} = [1.218\text{ (Weak)}, 0.788\text{ (Stab)}, 1.099\text{ (Inte)}]$.
3. **Smooth L1 Loss for Auxiliary Numerical Guidance**:
   $$\mathcal{L}_{\text{Aux\_Reg}} = \frac{1}{3} \sum_{\tau \in \{6, 12, 24\}} \text{SmoothL1}(\hat{V}_{+\tau}, V_{+\tau})$$
   Loss weights configured: $\lambda_{\text{ri}} = 1.0$, $\lambda_{\text{trend}} = 1.0$, $\lambda_{\text{reg}} = 0.1$.

### 3.3 Warm-Start Transfer Learning
Rather than training from scratch, the shared ResNet-18 spatial encoder and Temporal Transformer backbone were **warm-started** from the pre-trained forecasting checkpoint (`cnn_transformer_k5/best.pt`), transferring 153 pre-trained tensor weights. Only the specialized classification heads and projection layers were randomly initialized.

---

## 4. Benchmark Ladder & Experimental Verification

### 4.1 Comparison Against Baselines on Held-Out Test Set ($N=8,279$)

| Model Architecture | Trend Acc (%) | Macro $F_1$ | RI ROC-AUC | RI PR-AUC | RI Recall | RI Precision | RI $F_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A (Persistence)** | 40.96% | 0.1937 | 0.5000 | 0.0682 | 0.0% | 0.0% | 0.0000 |
| **Baseline B (Recent 6h Trend)** | 57.49% | 0.5750 | 0.8146 | 0.2033 | 35.2% | 27.0% | 0.3057 |
| **Baseline C (Thresholded Continuous Forecaster)** | 60.97% | 0.6117 | 0.7899 | 0.3086 | 19.8% | 43.2% | 0.2718 |
| **TemporalClassifier (Multi-Task AI)** | **63.53%** | **0.6367** | **0.8687** | **0.3690** | **63.7%** | **30.4%** | **0.4114** |

### 4.2 Statistical Significance via 1,000-Iteration Cyclone Block Bootstrap
To ensure that performance improvements are statistically robust across random cyclone samplings and not driven by a few idiosyncratic storms, we executed a 1,000-iteration cyclone block bootstrap:

| Evaluation Metric | Bootstrap Mean | 95% Confidence Interval (Lower) | 95% Confidence Interval (Upper) |
| :--- | :---: | :---: | :---: |
| **Trend Accuracy** | 63.58% | **61.37%** | **65.91%** |
| **Trend Macro $F_1$** | 0.6364 | **0.6142** | **0.6603** |
| **RI ROC-AUC** | 0.8688 | **0.8414** | **0.8940** |
| **RI PR-AUC** | 0.3727 | **0.2885** | **0.4501** |
| **RI $F_1$ Score** | 0.4105 | **0.3492** | **0.4668** |

The 95% confidence interval for RI PR-AUC `[0.2885, 0.4501]` is strictly and dramatically higher than the random prevalence baseline of `0.0682`.

---

## 5. Operational Decision Threshold Sweep

In real-world disaster management, operational meteorological centers calibrate threshold sensitivity based on civil protection priorities. The table below outlines the trade-off across operating points:

| Operating Threshold ($\tau$) | Precision | Recall (Sensitivity) | $F_1$ Score | Detected RI (TP) | False Alarms (FP) | Missed RI (FN) | Operational Protocol |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.05** | 24.63% | **74.34%** | 0.3700 | **420 / 565** | 1,285 | 145 | **Maximum Sensitivity Advisory** |
| **0.10** | 28.55% | 66.90% | 0.4002 | 378 / 565 | 946 | 187 | Precautionary Watch |
| **0.141** | **30.38%** | **63.72%** | **0.4114** | **360 / 565** | **825** | **205** | **Optimal Validation $F_1$ Operating Point** |
| **0.20** | 32.75% | 59.82% | 0.4233 | 338 / 565 | 694 | 227 | Balanced Operational Stance |
| **0.30** | 34.87% | 53.63% | 0.4226 | 303 / 565 | 566 | 262 | Moderate Confidence Warning |
| **0.50** | 37.85% | 44.96% | 0.4110 | 254 / 565 | 417 | 311 | High Confidence Alert |
| **0.70** | **41.43%** | 38.05% | 0.3967 | 215 / 565 | **304** | 350 | **Strict High-Specificity Siren** |

* At $\tau = 0.05$, the system captures **nearly 3 out of every 4 RI events (74.34%)**, serving as an aggressive early-warning filter for emergency planners.
* At $\tau = 0.70$, false alarms drop by **over 76%** (from 1,285 down to 304), providing high-confidence confirmation.

---

## 6. Stratified Performance Across Intensity Regimes

### 6.1 Saffir-Simpson Hurricane Intensity Regimes
* **Category 1–2 Hurricane / Typhoon ($64 \le V_{\max} \le 95\text{ kt}$)** ($N=1,622$):
  * **RI Prevalence**: 14.3%
  * **RI PR-AUC**: **`0.5163`** (3.6× baseline)
  * **RI ROC-AUC**: **`0.8788`**
  * **RI Recall**: **`75.86%`** (Detected 176 out of 232 explosive deepening events)
  * **RI $F_1$**: **`0.5391`**
* **Tropical Depression & Tropical Storm ($< 64\text{ kt}$)** ($N=5,532$):
  * Trend Accuracy: **62.82%** (Macro $F_1$: 0.6102)
  * RI PR-AUC: **0.2942**, ROC-AUC: **0.8495**, Recall: **56.48%**
* **Category 3+ Major Hurricane / Super Typhoon ($\ge 96\text{ kt}$)** ($N=1,125$):
  * Trend Accuracy: **63.29%**
  * Weakening Event Detection Rate: **81.4%** as mature systems undergo eyewall replacement cycles or enter cold waters.

### 6.2 Macro Dynamic Event Types
* **Weakening Events ($N=2,380$)**: **79.03% classification accuracy** (Mean predicted $P(\text{RI}) = 2.4\%$).
* **Rapid Intensification Events ($N=565$)**: **78.05% trend accuracy** (Mean predicted $P(\text{RI}) = 45.9\%$, vs $6.8\%$ for non-RI events).

---

## 7. Proving Ground: Generalization Case Studies on Held-Out Cyclones

### 7.1 Cyclone Performance Summary

| Cyclone Name | Basin | Category | RI Events Captured | Max RI Prob | Trend Acc | Key Proving Ground Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Super Typhoon Megi** (`201015W`) | West Pacific | Cat 5 (160 kt) | **14 / 14 (100%)** | **99.6%** | **78.7%** | Predicted $>80\%$ RI risk **18 hours before** Megi jumped from 65 kt to 160 kt. |
| **Cyclone Percy** (`200519S`) | South Pacific | Cat 5 (145 kt) | **6 / 6 (100%)** | **99.5%** | **68.9%** | 100% recall during rapid intensification from Category 1 to Category 5. |
| **Hurricane Matthew** (`201614L`) | North Atlantic | Cat 5 (145 kt) | **6 / 11 (54.5%)** | **97.0%** | **65.4%** | Early warning at 65 kt prior to explosive deepening in the Caribbean. |
| **Super Cyclone Phet** (`201003I`) | Indian Ocean | Cat 4 (125 kt) | **7 / 13 (53.8%)** | **89.7%** | **77.6%** | High trend accuracy across Arabian Sea lifecycle prior to Oman landfall. |
| **VSCS Nargis** (`200801I`) | Bay of Bengal | Cat 4 (115 kt) | **2 / 15 (13.3%)** | **93.5%** | **63.6%** | Reached 93.5% RI confidence; overall macro trend accuracy 63.6%. |
| **Hurricane Javier** (`200413E`) | East Pacific | Cat 4 (130 kt) | 0 / 11 (0.0% @ 0.14) | **98.0%** | **49.2%** | High probabilities (max 98%), but peaked earlier in pre-deepening phase. |

---

## 8. Deep Diagnostic & Failure Analysis: The Hurricane Javier Case Study

### 8.1 The Phenomenon
During interactive testing of **Hurricane Javier (200413E)** at $t = +33\text{h} \rightarrow +45\text{h}$, the observed intensity rose from $63\text{ kt}$ to $125\text{ kt}$, yet the active forecast vector plunged downward from $65\text{ kt} \rightarrow 43\text{ kt}$.

### 8.2 Root Cause Analysis
1. **Unanchored Absolute Regression**: The auxiliary regression head predicts absolute $V_{\max}$ directly from visual embeddings without receiving current intensity $V_{\max}(t)$ as an input. The visual encoder estimated a $\approx 45\text{ kt}$ system. When the chart connected from the known Best Track point ($65\text{ kt}$) down to the model's output ($43\text{ kt}$), it created an artificial visual drop.
2. **Warm Cloud Tops & Vertical Wind Shear**: Raw HDF5 inspection revealed mean IR1 temperatures of **$273\text{ K}$ ($0^\circ\text{C}$)** and sheared, ragged convection. In the absence of a closed warm eye, pure computer vision classifies the system as weak.
3. **Missing Environmental Physics**: Javier rapidly intensified due to sub-surface oceanic heat ($> 29^\circ\text{C}$ SST) and dropping environmental shear. Satellite imagery alone cannot observe ocean heat content beneath the cloud canopy.

### 8.3 The Three Solutions
* **Solution 1 (Residual Delta Forecasting)**: Formulate the auxiliary numerical forecast as a **residual change relative to current intensity**: $\hat{V}(t + \tau) = V_{\max}(t) + \Delta \hat{V}(\tau)$.
* **Solution 2 (Cross-Head Consistency)**: Constrain numerical outputs to match the headline Trend and RI classifications.
* **Solution 3 (Environmental Feature Fusion)**: Concatenate satellite embeddings with SHIPS/ERA5 environmental predictors (SST, wind shear, moisture divergence).

---

## 9. Operational Web Application Deployment (`demo_app/`)

A standalone, meteorologically tailored command center web interface was deployed on port `8090`:
* **Headline RI Gauge**: Radial gauge displaying $P(\text{RI})$ dynamically shifting from green (LOW) to amber (MEDIUM) to glowing red (HIGH).
* **24-Hour Trend Badge**: Macro evolution classification with softmax probability distribution bars.
* **Observation Timeline Scrubber & Animation**: Playback control allowing simulated operational progression.
* **Proving Ground Canvas Chart**:
  * **Dual View Modes**: `⚡ Real-Time Forecast` (default strict 24h operational perspective) vs. `📊 Full Storm Audit` (retrospective macro review).
  * **Faint Full Predicted Outline**: Complete trajectory reference backdrop.
  * **Active Glowing Forecast Cone**: Illuminated $[t \rightarrow t+24\text{h}]$ forecast vector with exact horizon callout badges.
  * **Automated Operational Verdict**: Instant declaration of whether the AI alert preceded the observed intensity change.

---

## 10. Conclusion & Scientifically Defensible Operational Summary

1. **Defensible Scientific Stance**: We do not claim that machine learning *"eliminates phase lag"* or *"perfectly solves cyclone forecasting."* We demonstrate that **multi-temporal spatio-temporal attention over satellite sequences contains significant predictive signals for 24-hour intensity evolution and rapid intensification risk**, capturing up to $75\%$ of explosive intensification events in critical Category 1–2 systems.
2. **Operationally Meaningful Output**: Transitioning from a noisy continuous MAE to an **actionable probabilistic RI risk assessment** provides disaster response authorities with the critical lead time required for life-saving coastal evacuations.
