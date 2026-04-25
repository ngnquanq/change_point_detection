# Automatic Change-Point Detection via Deep Learning

A PyTorch reimplementation of the framework from:

> Jie Li, Paul Fearnhead, Piotr Fryzlewicz, Tengyao Wang, *"Automatic change-point detection in time series via deep learning"*

The core idea: recast offline change-point detection as **supervised binary classification**. A neural network is trained to predict whether a short sequence of length *n* contains a change point (Y=1) or not (Y=0). Once trained, a sliding-window algorithm localizes change points in longer series.

> **Synthetic reproducibility note:** the canonical teacher-facing workflow now uses the fixed datasets in `data/paper_faithful/` plus a single entrypoint, `python scripts/reproduce_synthetic.py`. HASC is not part of that reproducibility path.

---

## Results at a Glance

### Experiment 1 — Synthetic Data (Single Change-Point)

Fresh results on the canonical `paper_faithful` test splits:

| Experiment | Test Acc | Power | FPR |
|---|---:|---:|---:|
| `mlp_s1` | 0.9190 | 0.8650 | 0.0270 |
| `rescnn_s1_paper` | 0.9255 | 0.8740 | 0.0230 |
| `mlp_s1prime` | 0.7290 | 0.6940 | 0.2360 |
| `rescnn_s1prime_paper` | 0.7435 | 0.6140 | 0.1270 |
| `mlp_s2` | 0.7540 | 0.6730 | 0.1650 |
| `rescnn_s2_paper` | 0.7525 | 0.6750 | 0.1700 |
| `mlp_s3` | 0.5970 | 0.7730 | 0.5790 |
| `rescnn_s3_paper` | 0.9475 | 0.9220 | 0.0270 |

Direct comparison against the authors' TensorFlow/Keras AutoCPD MLP on those same canonical splits:

| Experiment | Test Acc | Power | FPR |
|---|---:|---:|---:|
| `autocpd_s1_paper` | 0.9125 | 0.8710 | 0.0460 |
| `autocpd_s1prime_paper` | 0.7335 | 0.6960 | 0.2290 |
| `autocpd_s2_paper` | 0.7475 | 0.6840 | 0.1890 |
| `autocpd_s3_paper` | 0.5000 | 1.0000 | 1.0000 |

The shared CUSUM baseline on those same canonical test splits:

| Scenario | CUSUM Acc | CUSUM Power | CUSUM FPR |
|---|---:|---:|---:|
| `S1` | 0.8915 | 0.9750 | 0.1920 |
| `S1'` | 0.5005 | 1.0000 | 0.9990 |
| `S2` | 0.5000 | 1.0000 | 1.0000 |
| `S3` | 0.5110 | 1.0000 | 0.9780 |

These numbers are generated, not hand-maintained. The canonical teacher-facing summary lives at:

- `artifacts/synthetic/summary.md`
- `artifacts/synthetic/manifest.json`
- `comparison/results/AUTOCPD_PAPER_FAITHFUL_SUMMARY.md`

Those files are regenerated from the fixed synthetic datasets and the current trained artifacts. On these canonical splits, ResCNN is slightly ahead on `S1`, clearly better on `S1'`, tied on `S2`, and dramatically better on `S3`. The direct AutoCPD rerun is close to the PyTorch MLP on `S1`, `S1'`, and `S2`, but collapses on heavy-tailed `S3` under the original shallow-MLP + min-max setup.

---

### Experiment 3 — Real-world Application: HASC Dataset

The ResCNN model is applied to the **HASC (Human Activity Sensing Consortium)** dataset to demonstrate real-world generalization. The model is trained to classify 30 activity states (6 pure states + 24 transition states) using 3-channel accelerometer windows of length 700 (7 seconds at 100 Hz).

**Training setup:**
- Architecture: Deep Residual CNN (3-channel input, 30-class output)
- Training samples: 13,380 windows from persons `person101`–`person105`
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Early stopping: patience=20 epochs
- TensorBoard logs: `models/hasc`

**Final validation results (best model):**

| Metric | Value |
|---|---:|
| Val Loss | 0.1749 |
| Val Accuracy | **0.9621** |
| Val F1 (Macro) | 0.9042 |
| Val F1 (Weighted) | **0.9608** |

**Learning curves:**

![HASC learning curves](docs/report/hasc_learning_curve.png)

The model converges quickly and maintains consistently high validation accuracy (>90%) and F1-score throughout training with no signs of overfitting.

---

## Visualizations

### Canonical Dataset Overview

![Canonical train overview](data/paper_faithful/plots/paper_faithful_train_overview.png)

This overview is generated directly from the canonical `paper_faithful` train splits:
- **S1** (i.i.d. Gaussian) — standard independent noise
- **S1'/S2** (AR(1)-style dependence) — autocorrelated noise
- **S3** (Cauchy heavy-tail) — extreme spikes; uses the repo's robust preprocessing path

### Training Curves (Synthetic)

![Training curves](models/mlp_s1/plots/fig2_training_curves.png)

For the fresh canonical run, `mlp_s1` trained for 118 epochs and reached best validation accuracy `0.9700` at epoch 109. The training history figure is regenerated from `models/mlp_s1/history.json`.

### Canonical Model Comparison

![Canonical comparison](output/comparison/figure2_comparison.png)

This chart is generated from the same saved `eval_results.json` files that back the README tables. It compares `CUSUM`, the authors' `AutoCPD MLP`, the PyTorch `MLP`, and `ResCNN` on the canonical `paper_faithful_test` splits using the three metrics reported in this repo: detection accuracy, power, and false positive rate.

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

### Model 2 — Residual CNN (complex/multiple changes & HASC)

```
Input x ∈ ℝ^(C×n)     [C=1 for synthetic, C=3 for HASC]
    ↓
Conv1d(C→32, k=1, bias=False)      ← input projection
    ↓
ResidualBlock × 7                   (32→32 channels)
    ↓
ResidualBlock × 1                   (32→64 channels)  ← channel expansion at block 7
    ↓
ResidualBlock × 13                  (64→64 channels)
    ↓
AdaptiveAvgPool1d(1) → flatten      → (B, 64)
    ↓
      [Binary head]                         [Multi-class head — HASC]
Linear(64→50) → ReLU → Dropout(0.3)  Linear(64→50) → ReLU → Dropout(0.3)
Linear(50→40) → ReLU → Dropout(0.3)  Linear(50→40) → ReLU → Dropout(0.3)
Linear(40→30) → ReLU → Dropout(0.3)  Linear(40→30) → ReLU → Dropout(0.3)
Linear(30→20) → ReLU → Dropout(0.3)  Linear(30→K)            [K=30 logits]
Linear(20→10) → ReLU → Dropout(0.3)
Linear(10→1)                [raw logit]
```

**ResidualBlock** (`in_ch → out_ch`, kernel_size=8, pad=4):
```
x ──────────────────────── skip (Identity or Conv1d 1×1) ──────┐
│                                                                │
└─ Conv1d(k=8,p=4) → BN → ReLU → Conv1d(k=8,p=4) → BN ───────(+)→ ReLU
```
- Same-padding (`pad = kernel_size // 2 = 4`) preserves sequence length throughout.
- Channel expansion happens at block index `n_blocks // 3 = 7` (32 → 64).
- Skip uses 1×1 Conv when `in_ch ≠ out_ch`, otherwise `Identity`.
- The model is length-agnostic thanks to `AdaptiveAvgPool1d`.

---

## Data Pipeline

### Synthetic

```
simulate_dataset(N, n, noise_type)
    ↓  X: (N, n),  y: (N,),  taus: (N,)
augment_reversed()           ← doubles N; reversed sequences share same label
    ↓  X: (2N, n)
apply_pretransform()         ← optional: append x², cross-products for variance detection
    ↓  X: (2N, n')
minmax_scale() / trimmed_scale()   ← per-sequence normalization
    ↓
make_dataloaders(flatten=True/False)   ← (n,) for MLP, (1,n) for CNN
    ↓
Trainer.train(train_loader, val_loader)
```

For the canonical reproducibility path, the training and test inputs come from the fixed NPZ files under `data/paper_faithful/`:

- `s1_train.npz`, `s1_test.npz`
- `s1prime_train.npz`, `s1prime_test.npz`
- `s2_train.npz`, `s2_test.npz`
- `s3_train.npz`, `s3_test.npz`

Each file contains `X`, `y`, and `taus`. The matching `.hash.txt` files are used by `scripts/generate_reproducible_data.py --verify`.

### HASC (Real-world)

```
raw HASC CSV files (100 Hz, 3-axis accelerometer)
    ↓
scripts/split_hasc.py        ← segment by person, sliding windows (n=700, stride=50),
                                label pure/transition states → train.npz + val.npz + meta.json
    ↓
scripts/run_hasc.py          ← train ResidualCNN (3-channel, 30-class),
                                TensorBoard logging, early stopping, best model checkpoint
    ↓
models/hasc/best_model.pt
output/hasc_runs2/           ← TensorBoard event files
```

---

## Training

### Synthetic experiments

| Hyperparameter | Value |
|---|---|
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | up to 200 |
| Batch size | 32 |
| Early stopping | patience = 20 epochs |

Models output raw logits. Sigmoid is applied post-hoc at inference only — `BCEWithLogitsLoss` fuses sigmoid + cross-entropy in log-space for numerical stability.

### HASC experiment

| Hyperparameter | Value |
|---|---|
| Loss | CrossEntropyLoss |
| Optimizer | Adam (weight_decay=1e-4) |
| Learning rate | 0.001 |
| Scheduler | CosineAnnealingLR |
| Epochs | up to 100 |
| Batch size | 128 |
| Grad clip | 1.0 |
| Early stopping | patience = 20 epochs |

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
├── configs/               YAML experiment configs used by the canonical pipeline
│   ├── mlp_s1.yaml
│   ├── mlp_s1prime.yaml
│   ├── mlp_s2.yaml
│   ├── mlp_s3.yaml
│   ├── rescnn_s1_paper.yaml
│   ├── rescnn_s1prime_paper.yaml
│   ├── rescnn_s2_paper.yaml
│   ├── rescnn_s3_paper.yaml
│   └── rescnn_hasc.yaml   HASC experiment config (3-channel, 30-class)
├── data/
│   ├── paper_faithful/    Canonical reproducible synthetic train/test splits
│   │   ├── s1_*.npz       Data used for the S1 rows in the README tables
│   │   ├── s1prime_*.npz  Data used for the S1' rows in the README tables
│   │   ├── s2_*.npz       Data used for the S2 rows in the README tables
│   │   ├── s3_*.npz       Data used for the S3 rows in the README tables
│   │   └── plots/         Canonical dataset overview figures
│   └── hasc/
│       ├── splits/        Pre-split HASC windows (train.npz, val.npz, meta.json)
│       └── raw/           Raw HASC CSV files (not committed)
├── models/
│   ├── mlp_s1/            Canonical MLP artifacts; metrics in eval_results.json
│   ├── mlp_s1prime/
│   ├── mlp_s2/
│   ├── mlp_s3/
│   ├── rescnn_s1_paper/   Canonical ResCNN artifacts; metrics in eval_results.json
│   ├── rescnn_s1prime_paper/
│   ├── rescnn_s2_paper/
│   ├── rescnn_s3_paper/
│   ├── autocpd_s1_paper/  AutoCPD MLP artifacts on the same canonical splits
│   ├── autocpd_s1prime_paper/
│   ├── autocpd_s2_paper/
│   └── autocpd_s3_paper/
├── artifacts/
│   └── synthetic/
│       ├── summary.md     Canonical MLP/ResCNN summary shown in the README
│       └── manifest.json  Dataset hashes and artifact provenance
├── output/
│   ├── comparison/
│   │   ├── figure2_comparison.png    Canonical comparison chart shown in Chapter 4
│   │   └── comparison_results.json   Metrics used to draw that chart
├── models/
│   ├── ...
│   └── hasc/
│       └── best_model.pt             Best HASC checkpoint (Val Acc=0.9621)
│   └── hasc_runs2/                   TensorBoard event files for HASC training
├── comparison/
│   ├── results/
│   │   └── AUTOCPD_PAPER_FAITHFUL_SUMMARY.md   AutoCPD table shown in the README
│   └── scripts/
│       └── train_autocpd_paper_faithful.py     Runs the author's MLP on canonical splits
├── docs/
│   └── report/
│       ├── main.tex                  Main LaTeX report
│       ├── 4.experiment.tex          Experiment section (includes HASC results)
│       └── hasc_learning_curve.png   HASC learning curves (extracted from TensorBoard)
├── src/
│   ├── config.py          ExperimentConfig dataclasses + YAML I/O
│   ├── data/
│   │   ├── simulator.py   Sequence simulation (S1/S2/S3 noise profiles)
│   │   ├── paper_faithful.py  Canonical NPZ loading helpers
│   │   ├── transforms.py  Scaling, augmentation, pre-transforms
│   │   └── dataset.py     PyTorch Dataset + DataLoader factory
│   ├── models/
│   │   ├── mlp.py         MLPDetector (full / pruned variants)
│   │   └── rescnn.py      ResidualCNN (shared by synthetic & HASC)
│   ├── training/
│   │   └── trainer.py     Training loop, checkpointing, early stopping
│   ├── inference/
│   │   └── localizer.py   Algorithm 1: sliding-window localization
│   └── evaluation/
│       ├── metrics.py     Power, FPR, localization error, ROC
│       └── baselines.py   Two-sided CUSUM baseline
├── scripts/
│   ├── reproduce_synthetic.py       Canonical one-command synthetic pipeline
│   ├── generate_reproducible_data.py
│   ├── plot_canonical_synthetic_comparison.py
│   ├── split_hasc.py                Preprocess raw HASC → sliding windows + train/val split
│   ├── run_hasc.py                  Train ResCNN on HASC (30-class, TensorBoard logging)
│   ├── train.py
│   ├── evaluate.py
│   ├── locate.py
│   ├── visualize.py
│   └── visualize_paper_faithful_data.py
└── tests/                 Reproducibility and model smoke tests
```

The README tables are backed by concrete files:
- MLP/ResCNN rows are summarized in `artifacts/synthetic/summary.md` and traced back to `models/*/eval_results.json`.
- AutoCPD rows are summarized in `comparison/results/AUTOCPD_PAPER_FAITHFUL_SUMMARY.md` and traced back to `models/autocpd_*/eval_results.json`.
- The Chapter 4 comparison figure is drawn from `output/comparison/comparison_results.json`, which is regenerated from those same saved `eval_results.json` files.
- Dataset provenance and hashes are recorded in `artifacts/synthetic/manifest.json`.

---

## Quickstart

### Environment

```bash
conda env create -f environment.synthetic.yml
conda activate change-point-synthetic
```

### Synthetic experiments (one command)

```bash
python scripts/reproduce_synthetic.py
```

That command will:

1. create or verify the canonical synthetic datasets in `data/paper_faithful/`
2. train the canonical MLP and ResCNN experiments
3. evaluate them on the fixed test splits
4. regenerate the teacher-facing synthetic plots
5. write `artifacts/synthetic/manifest.json` and `artifacts/synthetic/summary.md`

### HASC experiment

```bash
# Step 1: prepare pre-split HASC windows
python scripts/split_hasc.py --hasc_dir data/hasc --out_dir data/hasc/splits

# Step 2: train the 30-class ResCNN
python scripts/run_hasc.py \
    --split_dir data/hasc/splits \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.001 \
    --scheduler cosine \
    --device auto

# Step 3: monitor training (optional)
tensorboard --logdir output/hasc_runs2
```

Best model checkpoint is saved to `models/hasc/best_model.pt`.

### Step-by-step breakdown (synthetic)

```bash
python scripts/generate_reproducible_data.py

python scripts/train.py --config configs/mlp_s1.yaml --device cpu
python scripts/train.py --config configs/rescnn_s1_paper.yaml --device cpu
python scripts/train.py --config configs/mlp_s1prime.yaml --device cpu
python scripts/train.py --config configs/rescnn_s1prime_paper.yaml --device cpu
python scripts/train.py --config configs/mlp_s2.yaml --device cpu
python scripts/train.py --config configs/rescnn_s2_paper.yaml --device cpu
python scripts/train.py --config configs/mlp_s3.yaml --device cpu
python scripts/train.py --config configs/rescnn_s3_paper.yaml --device cpu

python scripts/evaluate.py --experiment mlp_s1 --device cpu
python scripts/evaluate.py --experiment rescnn_s1_paper --device cpu
python scripts/evaluate.py --experiment mlp_s1prime --device cpu
python scripts/evaluate.py --experiment rescnn_s1prime_paper --device cpu
python scripts/evaluate.py --experiment mlp_s2 --device cpu
python scripts/evaluate.py --experiment rescnn_s2_paper --device cpu
python scripts/evaluate.py --experiment mlp_s3 --device cpu
python scripts/evaluate.py --experiment rescnn_s3_paper --device cpu

python scripts/visualize.py --experiment mlp_s1 --device cpu
python scripts/visualize_paper_faithful_data.py
python scripts/plot_canonical_synthetic_comparison.py

python scripts/reproduce_synthetic.py --step manifest
python scripts/reproduce_synthetic.py --step verify
```

### Run tests

```bash
python -m pytest tests/ -v
```

Current lightweight suite: `16` passing tests, including reproducibility checks plus `10` ResCNN unit tests covering residual blocks, forward shape, logits/probabilities, binary helpers, variable-length inputs, and the multiclass head.

---

## Configuration Reference

All experiments are controlled by YAML files. Key fields:

```yaml
experiment_name: mlp_s1

dataset:
  source: simulated
  data_dir: data/paper_faithful
  n: 100
  N: 10000
  noise_type: S1
  rho: 0.0
  sigma: 1.0
  snr_based_mu: true
  seed: 42

model:
  architecture: mlp        # mlp | rescnn
  mlp_variant: pruned      # full | pruned
  use_squared: false
  use_cross_product: false

training:
  epochs: 200
  batch_size: 32
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

Jie Li, Paul Fearnhead, Piotr Fryzlewicz, Tengyao Wang (2024).  
*Automatic Change-Point Detection in Time Series via Deep Learning.*  
Journal of the Royal Statistical Society: Series B.
