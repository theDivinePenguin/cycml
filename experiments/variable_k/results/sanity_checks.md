# Quality Control and Reproducibility Audit — Variable-K Experiment

**Audit Status**: ALL PASSED (16/16 criteria verified)

| # | Sanity Check Description | Status | Verification Details |
| :-: | :--- | :---: | :--- |
| 1 | Existing clean checkpoint best.pt is byte-for-byte identical | **PASS** | Existing clean checkpoint best.pt is byte-for-byte identical (SHA: `609841410eeafddf...`) |
| 2 | Existing baseline test_predictions.csv is byte-for-byte identical | **PASS** | Existing baseline test_predictions.csv is byte-for-byte identical (SHA: `1ddd212f305a248b...`) |
| 3 | Existing test manifest forecast_test_sequences_k7.csv is byte-for-byte identical | **PASS** | Existing test manifest forecast_test_sequences_k7.csv is byte-for-byte identical (SHA: `2edb9c6511743a7f...`) |
| 4 | All training, evaluation, and dataset code isolated under experiments/variable_k/ | **PASS** | All training, evaluation, and dataset code isolated under experiments/variable_k/ |
| 5 | All model outputs across K=3, 5, 7 test evaluations are finite numbers | **PASS** | All model outputs across K=3, 5, 7 test evaluations are finite numbers |
| 6 | Zero NaNs or Infs present in any generated prediction CSV | **PASS** | Zero NaNs or Infs present in any generated prediction CSV |
| 7 | Row-for-row alignment between predictions and test manifest (7,901 sequences) | **PASS** | Row-for-row alignment between predictions and test manifest (7,901 sequences) |
| 8 | Every sequence strictly terminates at current observation timestamp t | **PASS** | Every sequence strictly terminates at current observation timestamp t |
| 9 | K=3 slices exactly the last 3 frames [t-6h, t-3h, t] | **PASS** | K=3 slices exactly the last 3 frames [t-6h, t-3h, t] |
| 10 | K=5 slices exactly the last 5 frames [t-12h, t-9h, t-6h, t-3h, t] | **PASS** | K=5 slices exactly the last 5 frames [t-12h, t-9h, t-6h, t-3h, t] |
| 11 | K=7 uses all 7 frames [t-18h, t-15h, t-12h, t-9h, t-6h, t-3h, t] | **PASS** | K=7 uses all 7 frames [t-18h, t-15h, t-12h, t-9h, t-6h, t-3h, t] |
| 12 | Targets strictly evaluate Vmax at +6h, +12h, and +24h lead times | **PASS** | Targets strictly evaluate Vmax at +6h, +12h, and +24h lead times |
| 13 | Strict causality verified: all history timestamps <= target timestamp t | **PASS** | Strict causality verified: all history timestamps <= target timestamp t |
| 14 | Zero cyclone leakage: Train (867), Val (188), Test (187) | **PASS** | Zero cyclone leakage: Train (867), Val (188), Test (187) |
| 15 | Uses training set multichannel normalization stats without test leakage | **PASS** | Uses training set multichannel normalization stats without test leakage |
| 16 | Raw sigmoid RI probabilities and unclipped neural network regression outputs evaluated directly | **PASS** | Raw sigmoid RI probabilities and unclipped neural network regression outputs evaluated directly |

---

## Checkpoint & Manifest Integrity Hashes
- **Baseline Checkpoint `best.pt`**: `609841410eeafddfd20f53d4f0237b16c670e94acc60a6af0d22d65223eac56a`
- **Baseline `test_predictions.csv`**: `1ddd212f305a248b17aa2785226a104cfe01814f0d534f5fcd1c118a69b48bea`
- **Test Manifest `forecast_test_sequences_k7.csv`**: `2edb9c6511743a7feeefc359850703870195c98aa33838b5d9f32a61d31da77a`

All original files remain unmodified. The research branch was completely isolated under `experiments/variable_k/`.
