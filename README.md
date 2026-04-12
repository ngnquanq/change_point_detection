# Automatic Change-Point Detection via Deep Learning

A PyTorch reimplementation of the framework from:

> Fearnhead & Rigaill, *"Automatic change-point detection in time series via deep learning"*, JASA (2022).

The core idea: recast offline change-point detection as **supervised binary classification**. A neural network is trained to predict whether a short sequence of length *n* contains a change point (Y=1) or not (Y=0). Once trained, a sliding-window algorithm localizes change points in longer series.

---

## Results at a Glance

Trained on iid Gaussian noise (S1), n=100, N=10,000 sequences:

| Model | Detection Accuracy | Power (TPR) | Type-I Error (FPR) | Median Loc. Error |
|---|---|---|---|---|
| **Neural Network** | **0.81** | 0.68 | **0.07** | **21 positions** |
| CUSUM (baseline) | 0.54 | 0.94 | 0.86 | 33 positions |

CUSUM's high power comes at the cost of an 86% false positive rate — it flags almost every sequence as a change. The NN dramatically outperforms on specificity (FPR 0.07 vs 0.86) and overall accuracy, as confirmed by the ROC curve below.

---

## Visualizations

### Simulated Data — Three Noise Profiles

![Simulated sequences](models/mlp_s1/plots/fig1_simulated_sequences.png)

Three supported noise types:
- **S1** (i.i.d. Gaussian) — standard independent noise
- **S1'/S2** (AR(1), ρ=0.7) — autocorrelated noise; harder for CUSUM
- **S3** (Cauchy heavy-tail) — extreme spikes; requires robust preprocessing

### Training Curves

![Training curves](models/mlp_s1/plots/fig2_training_curves.png)

Loss drops sharply in the first 10 epochs. Early stopping fires at epoch 56 (patience=20), restoring the best checkpoint at epoch 35 (val acc = 80.6%).

### Neural Network vs CUSUM

![Performance comparison](models/mlp_s1/plots/fig3_performance_comparison.png)

Left: bar chart of key metrics. Center: localization error distribution — NN errors cluster near 0–25 positions while CUSUM spreads much wider. Right: ROC curve — the NN dominates CUSUM at every operating point.

### Algorithm 1 — Localization Demo

![Localization demo](models/mlp_s1/plots/fig4_localization_demo.png)

A 600-point series with 3 true change points (red dashed). The bottom panel shows per-window probability and rolling average L_hat. The model fires confidently near all 3 true change points.

---

## Architecture

### Model 1 — MLP (single mean changes)

```
Input x in R^n
    ↓
Linear(n → h) → ReLU → Linear(h → 1)   [raw logit]
```

Hidden layer width `h` has two variants:
- **Full**: `h = 2n − 2` — theoretically proven to replicate the CUSUM statistic
- **Pruned**: `h = 4 · ⌊log₂ n⌋` — near-identical accuracy with ~8× fewer parameters

For n=100: full → h=198 (~20K params), pruned → h=24 (~2.4K params).

### Model 2 — Residual CNN (complex/multiple changes)

```
Input x ∈ ℝ^(1×n)
    ↓
Conv1d(1→32, k=1)           ← input projection
    ↓
ResidualBlock × 7            (32→32 channels)
    ↓
ResidualBlock × 1            (32→64 channels)  ← channel expansion
    ↓
ResidualBlock × 13           (64→64 channels)
    ↓
AdaptiveAvgPool1d(1)         ← collapse temporal dim
    ↓
Linear(64→64) → ReLU → Linear(64→1)   [raw logit]
```

Each residual block: `Conv1d → BN → ReLU → Conv1d → BN + skip → ReLU`.
Same-padding (`pad = kernel_size // 2`) preserves sequence length throughout.
The model is length-agnostic via `AdaptiveAvgPool`.

---

## Data Pipeline

```
simulate_dataset(N, n, noise_type)
    ↓  X: (N, n),  y: (N,),  taus: (N,)
augment_reversed()           ← doubles N; reversed sequences share same label
    ↓  X: (2N, n)
apply_pretransform()         ← optional: append x², cross-products for variance detection
    ↓  X: (2N, n')
minmax_scale() / trimmed_scale()   ← per-sequence normalization to [0,1]
    ↓
make_dataloaders(flatten=True/False)   ← (n,) for MLP, (1,n) for CNN
    ↓
Trainer.train(train_loader, val_loader)
```

**Simulation**: N/2 sequences contain a change at τ ~ Uniform{1, …, n−2} with means μ_L, μ_R ~ Uniform(−2, 2). N/2 are stationary.

**Augmentation**: reversed sequences are added before the train/val split so each original and its reverse end up in the same fold.

**Preprocessing**: S3 (Cauchy) uses trimmed mean/std instead of min-max to handle extreme outliers.

---

## Training

| Hyperparameter | Value |
|---|---|
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | up to 200 |
| Batch size | 32 |
| Early stopping | patience = 20 epochs |

Models output raw logits. Sigmoid is applied post-hoc at inference only — `BCEWithLogitsLoss` fuses sigmoid + cross-entropy in log-space for numerical stability.

---

## Deployment — Algorithm 1 (Localization)

```
Given: long series T of length M, trained model, window size n

1. SLIDE    extract windows T[i : i+n] with step size s
2. SCORE    p_i = P(change | window_i),  L_i = 1{p_i ≥ 0.5}
3. SMOOTH   L̄_i = rolling average of L over a window of width w
4. SEGMENT  find maximal contiguous runs where L̄_i ≥ γ (default 0.5)
5. LOCALIZE τ̂ = argmax(L̄_i) within each segment
```

Windows straddling the true change point score high; windows entirely before or after score low. The peak of the probability signal localizes the change.

---

## Project Structure

```
change_point_detection/
├── configs/               YAML experiment configs
│   ├── mlp_s1.yaml        MLP on i.i.d. Gaussian noise
│   ├── mlp_s2.yaml        MLP on AR(1) noise
│   ├── mlp_s3.yaml        MLP on Cauchy noise
│   ├── rescnn_s1.yaml     ResidualCNN on Gaussian noise
│   └── rescnn_s2.yaml     ResidualCNN with x² and cross-product features
├── src/
│   ├── config.py          ExperimentConfig dataclasses + YAML I/O
│   ├── data/
│   │   ├── simulator.py   Sequence simulation (S1/S2/S3 noise profiles)
│   │   ├── transforms.py  Scaling, augmentation, pre-transforms
│   │   └── dataset.py     PyTorch Dataset + DataLoader factory
│   ├── models/
│   │   ├── mlp.py         MLPDetector (full / pruned variants)
│   │   └── rescnn.py      ResidualCNN (21 residual blocks)
│   ├── training/
│   │   └── trainer.py     Training loop, checkpointing, early stopping
│   ├── inference/
│   │   └── localizer.py   Algorithm 1: sliding-window localization
│   └── evaluation/
│       ├── metrics.py     Power, FPR, localization error, ROC
│       └── baselines.py   Two-sided CUSUM baseline
├── scripts/
│   ├── train.py           CLI: train a model from a config
│   ├── evaluate.py        CLI: evaluate NN vs CUSUM on a test set
│   ├── locate.py          CLI: run Algorithm 1 on a synthetic long series
│   └── visualize.py       CLI: generate all figures
└── tests/                 52 unit + integration tests (pytest)
```

---

## Quickstart

### Environment

```bash
conda activate insider-threat   # has torch 2.5.1, numpy 2.0.2, scipy 1.13.1
pip install matplotlib seaborn pyyaml pytest
```

### Train

```bash
# MLP on i.i.d. Gaussian noise (~1 min on CPU)
python scripts/train.py --config configs/mlp_s1.yaml --device cpu

# ResidualCNN (GPU recommended)
python scripts/train.py --config configs/rescnn_s1.yaml --device cuda
```

### Evaluate

```bash
python scripts/evaluate.py --experiment mlp_s1 --device cpu
```

### Localize change points in a long series

```bash
python scripts/locate.py --experiment mlp_s1 --series_length 500 --n_changes 2
```

### Generate visualizations

```bash
python scripts/visualize.py --experiment mlp_s1
# Saves 4 figures to models/mlp_s1/plots/
```

### Run tests

```bash
python -m pytest tests/ -v   # 52 tests
```

---

## Configuration Reference

All experiments are controlled by YAML files. Key fields:

```yaml
experiment_name: mlp_s1

simulation:
  n: 100           # sequence length (= window size for localization)
  N: 10000         # number of training sequences
  noise_type: S1   # S1 | S1_prime | S2 | S3
  rho: 0.0         # AR(1) coefficient (for S1_prime/S2)
  mu_range: [-2.0, 2.0]
  sigma: 1.0

model:
  architecture: mlp        # mlp | rescnn
  mlp_variant: pruned      # full | pruned
  use_squared: false        # append x² features
  use_cross_product: false  # append x_t·x_{t+1} features

training:
  epochs: 200
  batch_size: 32
  lr: 0.001
  patience: 20
  augment_reversed: true

localization:
  window_size: 100
  step_size: 1
  rolling_window: 10
  gamma: 0.5
```

---

## Reference

Paul Fearnhead and Guillem Rigaill (2022).
*Changepoint Detection in the Presence of Outliers.*
Journal of the American Statistical Association, 117(539), 1168–1179.
