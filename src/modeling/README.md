# Modeling Stage — Context-Aware BiLSTM Training GUI

This document covers `train_context_aware_bilstm_gui.py` (internal
changelog: FIX 1–15 — see [§20](#20-key-fixes-in-v3-vs-v2) for FIX 9–15,
the with/without-C_t validation framework added most recently). It's the
desktop GUI that turns the merged CSV from
[`../data_pipeline/`](../data_pipeline/README.md) into trained models and
predictions — see [§19](#19-withwithout-ct-validation-framework-fix-9-13)
for the part of this system that actually answers the research question,
not just trains a model.

> **Predict PM₁ · PM₂.₅ · PM₁₀ · CO₂ · VOC concentrations up to T+10 minutes ahead — before chemical sensors react — using door state, human motion, and machine operational context.**

Developed at **IIT Bombay** — Environmental Science & Engineering / Centre for Machine Intelligence and Data Science.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Research Background](#2-research-background)
3. [System Requirements](#3-system-requirements)
4. [Installation](#4-installation)
5. [Input Data Format](#5-input-data-format)
6. [Running the GUI](#6-running-the-gui)
7. [GUI Controls — Complete Reference](#7-gui-controls--complete-reference)
8. [How the Pipeline Works (Step by Step)](#8-how-the-pipeline-works-step-by-step)
9. [Feature Engineering — What Each Toggle Does](#9-feature-engineering--what-each-toggle-does)
10. [Model Architectures](#10-model-architectures)
11. [Training Details](#11-training-details)
12. [Output Files](#12-output-files)
13. [Understanding the Plots](#13-understanding-the-plots)
14. [Operational Phases](#14-operational-phases)
15. [Key Fixes in v2 (vs v1)](#15-key-fixes-in-v2-vs-v1)
16. [Troubleshooting](#16-troubleshooting)
17. [Code Architecture Map](#17-code-architecture-map)
18. [Configuration Quick Reference](#18-configuration-quick-reference)
19. [With/Without-C_t Validation Framework (FIX 9–13)](#19-withwithout-ct-validation-framework-fix-9-13)
20. [Key Fixes in v3 (vs v2)](#20-key-fixes-in-v3-vs-v2)
21. [Known Gap: Incomplete Vision-Side C_t Features](#21-known-gap-incomplete-vision-side-ct-features)

---

## 1. What This System Does

This is a **desktop GUI application** that:

1. Loads a merged sensor CSV file from a laser fabrication laboratory
2. Engineers multimodal features from door state, human motion, machine operation, and pollutant history
3. Trains one or more deep learning / machine learning models entirely inside the GUI
4. Generates publication-quality plots showing predictions 10 minutes ahead of actual sensor readings
5. Saves a full ablation study, heatmaps, RMSE bar charts, and a prediction CSV

The central claim it proves: **camera-derived behavioural signals (who is in the room, how fast they are moving, whether machine doors are open) predict dangerous air quality events before any chemical sensor can detect them.**

---

## 2. Research Background

### The Problem

Laser cutting machines generate bursts of PM₁, PM₂.₅, PM₁₀, and VOCs through pyrolysis. These concentrations spike within 1–3 minutes of a door opening during a cutting cycle. A conventional IAQ monitor only fires an alarm *after* the sensor measures a dangerous level — by which point the operator has already been exposed.

### The Solution This Code Implements

The code implements a **lead-time forecasting pipeline**:

```
Camera (YOLO) → person count, motion μ, motion σ²
Door sensors  → φ_open per machine, door_open_sum, consecutive open epochs
Machine state → operational state one-hot (IDLE / CUT / EXPOSURE / MAINTENANCE)
Sensor lags   → PM lags 1-3 min, CO₂ lags, VOC diffs, rolling means
         ↓
   Feature vector X[t … t+lookback]
         ↓
   Bi-LSTM / GRU / Transformer / ML model
         ↓
   Prediction ŷ[t + lead_minutes]  ←  10 minutes into the future
```

The system achieves R² = 0.842–0.942 across all five pollutants with a +10 minute warning window.

---

## 3. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.10 or 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU works) | NVIDIA CUDA GPU |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Any recent 64-bit OS |
| Display | 1280 × 800 | 1920 × 1080 |

### Python Packages Required

```
numpy
pandas
scikit-learn
torch            (PyTorch ≥ 2.0)
matplotlib
seaborn
xgboost          (optional — XGBoost model)
```

---

## 4. Installation

### Step 1 — Clone or download the repository

```bash
git clone https://github.com/your-org/iaq-early-detection.git
cd iaq-early-detection
```

Or simply download `train_context_aware_bilstm_gui.py` into a folder of your choice.

### Step 2 — Create a virtual environment (strongly recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install numpy pandas scikit-learn torch matplotlib seaborn

# Optional but recommended
pip install xgboost
```

### Step 4 — Verify installation

```bash
python -c "import torch, pandas, sklearn, matplotlib; print('All OK')"
```

You should see `All OK`. If any import fails, re-run `pip install <package>`.

### Step 5 — Run

```bash
python train_context_aware_bilstm_gui.py
```

The GUI window opens immediately. No further setup is needed.

---

## 5. Input Data Format

The system expects **one merged CSV file** containing all sensor readings at **1-minute resolution**. Every row is one timestep. Columns can be in any order.

### Required Columns

| Column name | Type | Description |
|-------------|------|-------------|
| `timestamp_minute` | datetime string | Timestamp for this row. Format: `DD-MM-YYYY HH:MM:SS` or `YYYY-MM-DD HH:MM:SS` |
| `pm1` | float | PM₁ concentration (μg/m³) |
| `pm2_5` | float | PM₂.₅ concentration (μg/m³) |
| `pm10` | float | PM₁₀ concentration (μg/m³) |
| `co2` | float | CO₂ concentration (ppm) |
| `voc` | float | VOC index / concentration (ppb) |

### Strongly Recommended Columns

| Column name | Type | Description |
|-------------|------|-------------|
| `n_person` | int | Number of people detected by YOLO camera |
| `mu_motion` | float | Mean motion magnitude from optical flow |
| `sigma2_motion` | float | Motion variance from optical flow |
| `M1_phi_open` | float | Machine 1 door open fraction (0=fully open, 1=closed) |
| `M2_phi_open` | float | Machine 2 door open fraction |
| `M3_phi_open` | float | Machine 3 door open fraction |
| `M1_emission_weight` | float | Machine 1 emission weighting factor — ⚠️ not currently produced by the upstream vision pipeline, see [§21](#21-known-gap-incomplete-vision-side-ct-features) |
| `M2_emission_weight` | float | Machine 2 emission weighting factor — ⚠️ same gap |
| `M3_emission_weight` | float | Machine 3 emission weighting factor — ⚠️ same gap |
| `M1_effective_tau` | float | Machine 1 effective exposure time — real, produced by the data-pipeline merge stage's `enrich_operational_state()`, NOT dead (corrected — see [§21](#21-known-gap-incomplete-vision-side-ct-features)). Statefully computed, though; not derivable from a short live window, see §21. |
| `M1_consecutive_full_open` | int | Consecutive minutes M1 door has been fully open — ⚠️ same gap |
| `M1_is_IDLE` | int (0/1) | Machine 1 operational state one-hot — ⚠️ same gap |
| `M1_is_CUT` | int (0/1) | Machine 1 in cutting operation — ⚠️ same gap |
| `M1_is_EXPOSURE` | int (0/1) | Machine 1 in exposure mode — ⚠️ same gap |
| `M1_is_MAINTENANCE` | int (0/1) | Machine 1 in maintenance — ⚠️ same gap |
| `temp` | float | Ambient temperature (°C) |
| `hum` | float | Relative humidity (%) |

> **Missing columns are handled gracefully.** If a column is absent, the system fills it with 0.0 and continues. The more columns you provide, the better the predictions.

### Example CSV (first 3 rows)

```
timestamp_minute,pm1,pm2_5,pm10,co2,voc,n_person,mu_motion,sigma2_motion,M1_phi_open,M2_phi_open,M3_phi_open,temp,hum
12-02-2026 12:58:00,15.2,24.1,31.4,1623,88,0,0.0,0.0,1.0,1.0,1.0,27.2,48.1
12-02-2026 12:59:00,15.5,24.3,31.7,1634,91,0,0.0,0.0,1.0,1.0,1.0,27.3,47.9
12-02-2026 13:00:00,16.1,25.0,32.2,1641,94,1,0.8,0.3,0.5,1.0,1.0,27.4,47.8
```

---

## 6. Running the GUI

```bash
python train_context_aware_bilstm_gui.py
```

The window is 1200 × 940 pixels with two panels:

- **Left panel** — scrollable configuration controls (8 sections)
- **Right panel** — live training log + improvement summary + metrics table

### Minimal Quick-Start (5 steps)

1. Click **Browse** next to "CSV file" → select your merged CSV
2. Set **Start datetime** to your session start (e.g. `12-02-2026 12:58:00`)
3. Set **End datetime** to your session end (e.g. `12-02-2026 18:00:00`)
4. Select at least one model (BiLSTM is checked by default)
5. Click **▶ RUN EARLY DETECTION v2**

Training logs appear in real time on the right. Output files are saved to `results_early_v2/` when training completes.

---

## 7. GUI Controls — Complete Reference

### ① DATA SOURCE

| Control | What it does |
|---------|-------------|
| **CSV file** | Path to your merged IAQ CSV. Click Browse to navigate. |
| **Timestamp column** | Column name containing the datetime string. Default: `timestamp_minute` |
| **Door open column** | Optional override. If blank, the system auto-sums all `M*_phi_open` columns. |
| **Person count column** | Column with YOLO person count. Default: `n_person` |
| **Motion column** | Column with optical flow magnitude. Default: `mu_motion` |

---

### ② TIME WINDOW

| Control | What it does |
|---------|-------------|
| **Start datetime** | First timestamp to include. Format: `DD-MM-YYYY HH:MM:SS`. Leave blank to use full dataset. |
| **End datetime** | Last timestamp to include. Same format. |

> The system filters rows to `start ≤ timestamp ≤ end` before any processing. The plot x-axis starts exactly at this time.

---

### ③ LEAD-TIME SETTINGS

| Control | What it does |
|---------|-------------|
| **Lead steps (minutes)** | How many minutes ahead to forecast. 10 = predict what the sensor will read 10 min from now. Range: 1–60. |
| **Lookback window** | How many past timesteps the model sees as input. 20 = last 20 minutes. Range: 5–120. |
| **Plot prediction at TRIGGER TIME** | When checked (default), the predicted value is plotted at the moment the model fires (the trigger), not at the future time. This visually shows the lead-time gap — the prediction line rises *before* the actual line. |

**Understanding the sequence structure:**

```
Time:  t   t+1  t+2  ...  t+LB          t+LB+LEAD-1
       [←────── X input (lookback) ──────→]    [y target]
                                               ↑ this is what the model predicts
```

---

### ④ FEATURE ENGINEERING

Each checkbox adds a group of features to the input vector.

| Toggle | Features added | When to enable |
|--------|----------------|----------------|
| **PM lags 1-3** | `pm1_lag1/2/3`, `pm25_lag1/2`, `pm10_lag1/2` | **Always** — critical for PM R² |
| **PM momentum diff1** | `pm1_diff1`, `pm25_diff1`, `pm10_diff1`, `pm_total` | **Always** — detects PM spikes |
| **PM rolling mean 5** | `pm1_roll5`, `pm25_roll5`, `pm10_roll5` | Recommended |
| **VOC momentum diff** | `voc_diff1`, `voc_diff2` | Recommended for VOC prediction |
| **Person count Δ** | `n_person`, `person_diff` | Recommended — entry/exit events |
| **Door open sum** | `door_open_sum`, `door_exposure` | Recommended |
| **Door Δ** | `door_diff` | Adds door transition speed |
| **Motion rolling** | `mu_motion`, `motion_roll10`, `trigger_strength` | Recommended |
| **CO₂ lags** | `co2_lag1/2/3`, `co2_roll5` | Required for good CO₂ R² |
| **Raw sensors** | Current `pm1`, `pm2_5`, `pm10`, `co2`, `voc` | Optional |
| **Env (temp/hum)** | `temp`, `hum` | Optional |
| **Emission weight** | `M*_emission_weight`, `M*_phi_open` | Enables if machine data available — ⚠️ `emission_weight` is currently always 0, see [§21](#21-known-gap-incomplete-vision-side-ct-features) |
| **Consecutive open** | `M*_consecutive_full_open` | Captures sustained door opening — ⚠️ currently always 0, see [§21](#21-known-gap-incomplete-vision-side-ct-features) |
| **Op-state one-hot** | `M*_is_IDLE/CUT/EXPOSURE/MAINTENANCE` | Powerful — if op-state data available — ⚠️ currently always 0, see [§21](#21-known-gap-incomplete-vision-side-ct-features) |

> **Recommended minimum set for good results:**  
> ✅ PM lags + PM diff + PM roll + VOC diff + Person Δ + Door sum + Motion rolling + CO₂ lags

---

### ⑤ MODEL SELECTION

| Model | Type | Best for | Notes |
|-------|------|----------|-------|
| **BiLSTM** | Deep — Bidirectional LSTM + Attention | All pollutants balanced | Proposed model. Best overall. |
| **BiGRU** | Deep — Bidirectional GRU + Attention | VOC spike events | Faster than BiLSTM |
| **GRU** | Deep — Unidirectional GRU | Medium datasets | Good balance |
| **LSTM_uni** | Deep — Unidirectional LSTM | CO₂ slow accumulation | Strong cell memory |
| **VanillaRNN** | Deep — Basic RNN | Baseline comparison only | Vanishing gradient issues |
| **TCN** | Deep — Temporal Convolutional Network | Datasets > 1000 rows | Parallel computation |
| **Seq2Seq** | Deep — Encoder-Decoder LSTM | Multi-step forecasting | Struggles with CO₂ long range |
| **CNN_LSTM** | Deep — Conv1D + LSTM | Feature extraction + sequence | Risk of CO₂ degradation |
| **Transformer** | Deep — Self-attention | Large datasets | Needs > 500 rows |
| **PatchTST** | Deep — Patched Transformer | Long-horizon forecasting | Experimental |
| **LinearRegression** | ML | Lower-bound baseline | Shows non-linearity need |
| **Ridge** | ML | Regularised linear | Better than OLS |
| **RandomForest** | ML | Fast baseline | Inflated overall R² due to CO₂ scale |
| **SVR** | ML | Comparison | Slow; often poor CO₂ |
| **XGBoost** | ML | Gradient boosting | Good but no temporal structure |

**Quick select buttons:**
- **All** — selects every model
- **None** — deselects all
- **Deep only** — selects only neural network models
- **ML only** — selects only scikit-learn models

---

### ⑥ TARGET POLLUTANTS

Select which pollutants to predict. All five are checked by default:

| Target | Unit | Primary driver |
|--------|------|----------------|
| **pm1** | μg/m³ | Machine emission at door opening |
| **pm2_5** | μg/m³ | Machine emission + air mixing |
| **pm10** | μg/m³ | Emission + foot traffic resuspension |
| **co2** | ppm | Human occupancy (metabolic) |
| **voc** | ppb | Machine operation (pyrolysis) |

---

### ⑦ TRAINING SETTINGS

| Control | Default | Description |
|---------|---------|-------------|
| **Epochs** | 200 | Maximum training iterations |
| **Hidden units** | 64 | LSTM/GRU hidden state size |
| **Layers** | 1 | Stacked recurrent layers |
| **Dropout** | 0.30 | Fraction of neurons randomly dropped during training |
| **Learning rate** | 0.001 | AdamW optimiser initial step size |
| **Batch size** | 16 | Sequences per gradient update |
| **Train fraction** | 0.70 | 70% of data for training |
| **Val fraction** | 0.15 | 15% for validation (early stopping) |
| **Patience** | 30 | Stop if validation loss doesn't improve for 30 epochs |
| **Random seed** | 42 | Reproducibility seed |
| **Adaptive loss weights** | ✅ ON | Automatically balances PM vs CO₂/VOC targets |

---

### ⑧ OUTPUT

| Control | Description |
|---------|-------------|
| **Output directory** | Folder where all PNGs and CSVs are saved. Default: `results_early_v2/` |
| **Save prediction CSV** | Saves `predictions_lead{N}_v2.csv` with all actual/predicted pairs |
| **Save PNG plots** | Saves per-model per-target causal chain plots |
| **Save lead-gap comparison** | Saves additional lead-gap analysis plots |

---

## 8. How the Pipeline Works (Step by Step)

When you click **▶ RUN**, the following sequence executes in a background thread:

### Step 1 — Load and Filter Data

```python
df = pd.read_csv(csv_path)
df['timestamp_minute'] = pd.to_datetime(df['timestamp_minute'], dayfirst=True)
df = df.sort_values('timestamp_minute')
# Apply start/end time filter
df = df[(df['timestamp_minute'] >= start) & (df['timestamp_minute'] <= end)]
```

The `dayfirst=True` setting correctly parses `12-02-2026` as February 12, not December 2.

### Step 2 — Fill Missing Columns

Any column listed in the expected set (door, motion, machine state) that is absent gets filled with `0.0`. This prevents crashes on partial datasets.

### Step 3 — Compute the Door Signal

```python
# Auto-detect: sum all machine phi_open columns
df['door_open_sum'] = df[['M1_phi_open','M2_phi_open','M3_phi_open']].sum(axis=1)
```

If you specify a manual door column in the GUI, that single column is used instead.

### Step 4 — Feature Engineering

Every feature group selected in the GUI is computed here:

```python
# PM momentum (always fast — 1-3 min lags)
df['pm1_lag1'] = df['pm1'].shift(1).bfill()
df['pm1_diff1'] = df['pm1'].diff(1).bfill()
df['pm1_roll5'] = df['pm1'].rolling(5, min_periods=1).mean()

# VOC momentum
df['voc_diff1'] = df['voc'].diff(1).bfill()

# CO2 slow accumulation (5-10 min lags)
df['co2_lag1'] = df['co2'].shift(1).bfill()
df['co2_roll5'] = df['co2'].rolling(5, min_periods=1).mean()

# Trigger composite
df['trigger_strength'] = df['n_person'].clip(0) * df['mu_motion'].clip(0)
df['door_exposure'] = (df['door_open_sum'] > 0).rolling(10).sum()
```

### Step 5 — Scale (Train-Only Fitting)

```python
# FIX 4: StandardScaler fitted ONLY on training rows — no data leakage
x_scaler = StandardScaler()
x_scaler.fit(X_raw[:train_end])
X_scaled = x_scaler.transform(X_raw)

# Each target gets its own MinMaxScaler (fitted on train only)
for target in ['pm1','pm2_5','pm10','co2','voc']:
    scaler = MinMaxScaler()
    scaler.fit(y_raw[:train_end, i])
```

### Step 6 — Build Sliding-Window Sequences

```python
# Sequence: X[t … t+lookback] → y[t + lookback + lead - 1]
for i in range(len(data) - lookback - lead + 1):
    X_sequences.append(data[i : i + lookback])     # shape: (lookback, n_features)
    y_targets.append(data[i + lookback + lead - 1]) # shape: (5,) — all targets
```

Each input sequence is `lookback` timesteps of all features. The target is the pollutant concentrations `lead` minutes after the end of the input window.

### Step 7 — Compute Adaptive Loss Weights

```python
# FIX 3: Targets with low variance (PM in normalised space) get higher weight
train_variance = y_scaled[:train_end].var(axis=0)
loss_weights = 1.0 / train_variance
loss_weights = loss_weights / loss_weights.mean()  # normalise to mean=1
```

This ensures the model does not ignore PM targets just because their normalised variance is 10-100× smaller than CO₂.

### Step 8 — Train Each Model

```python
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        prediction = model(X_batch)
        loss = ((prediction - y_batch)**2 * loss_weights).mean()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    # Early stopping on validation loss with patience=20
```

### Step 9 — Generate Predictions

```python
model.eval()
with torch.no_grad():
    predictions = model(X_test)
# Inverse-transform back to original units
predictions_original = scaler.inverse_transform(predictions)
```

### Step 10 — Compute Metrics

For each model × target combination:

```python
R²   = r2_score(actual, predicted)
RMSE = sqrt(mean_squared_error(actual, predicted))
MAE  = mean_absolute_error(actual, predicted)
```

### Step 11 — Generate All Plots and Save

The pipeline generates 7 types of output files (detailed in Section 12).

---

## 9. Feature Engineering — What Each Toggle Does

### Why PM Lags Matter So Much

PM concentrations have strong **temporal autocorrelation** — the concentration 1 minute ago is the best single predictor of the concentration now. Without lag features, the model cannot learn this persistence. This was the root cause of negative PM R² in v1.

```
pm1_lag1 = pm1 value 1 minute ago
pm1_lag2 = pm1 value 2 minutes ago
pm1_lag3 = pm1 value 3 minutes ago
```

### Why PM Diff Catches Spikes

The first difference detects whether PM is rising or falling:

```
pm1_diff1 = pm1[t] - pm1[t-1]
```

A positive diff at a door-open event is the earliest signal of a PM spike — the model learns to associate `(door_open + positive_pm_diff)` → "PM will be high in 10 minutes".

### Why trigger_strength Captures the Dangerous Combination

```python
trigger_strength = n_person × mu_motion
```

A room with 15 people moving fast near open laser machines is far more dangerous than 15 stationary people or 1 moving person. This product captures the synergistic risk.

### Why door_exposure Captures Sustained Risk

```python
door_exposure = rolling_10min_sum(door_open_sum > 0)
```

A door open for 10 continuous minutes (door_exposure = 10) creates a much larger PM reservoir than 10 separate 1-minute openings (also door_exposure = 10 but different aerodynamic pattern). Combined with `door_diff` (rate of change), the model distinguishes these scenarios.

---

## 10. Model Architectures

### BiLSTM (Proposed — Best Model)

```
Input: (batch, lookback, n_features)
    ↓
BiLSTM layer 1: 64 units forward + 64 backward = 128 total hidden
    ↓
Attention: weighted sum over time steps → context vector (128,)
    ↓
Dropout (0.20)
    ↓
Per-target heads (5 separate output layers):
    PM targets:  Linear(128 → 64) → GELU → Dropout(0.10) → Linear(64 → 1)
    Gas targets: Linear(128 → 64) → GELU → Linear(64 → 1)
Output: (batch, 5)
```

**Why bidirectional?** The forward pass learns "door opens → PM will rise". The backward pass learns "VOC is falling + door closed → decay phase". Both directions are needed simultaneously for accurate lead-time forecasting.

**Why per-target heads?** PM and gas targets have fundamentally different temporal dynamics. Separate heads let each learn its own mapping from the shared latent representation.

### Attention Mechanism

```python
class Attention(nn.Module):
    def __init__(self, hidden):
        self.w = nn.Linear(hidden, 1)    # score each timestep
    
    def forward(self, lstm_output):
        # lstm_output: (batch, time, hidden)
        scores = softmax(self.w(lstm_output), dim=1)   # (batch, time, 1)
        context = (scores * lstm_output).sum(dim=1)    # (batch, hidden)
        return context
```

The attention layer learns which timesteps within the lookback window matter most for the prediction. For lead-time forecasting, this focuses attention on the **trigger moment** — the specific minute when a door opened or a person entered.

### TCN (Temporal Convolutional Network)

```
Input: (batch, lookback, features)
    ↓ permute to (batch, features, lookback)
ResBlock(dilation=1): causal conv, kernel=3
ResBlock(dilation=2): wider receptive field
ResBlock(dilation=4): even wider
    ↓
Take last timestep [:, :, -1]
    ↓
Linear → (batch, 5)
```

Uses dilated causal convolutions so each output only depends on past inputs (no future leakage).

### Transformer

```
Input → Linear projection to d=64
    ↓
Positional encoding (sinusoidal)
    ↓
TransformerEncoder (1 layer, 4 heads, d_ff=128, dropout=0.2)
    ↓
Take last token output
    ↓
Linear → (batch, 5)
```

> ⚠️ The Transformer needs more data (>500 rows) to outperform recurrent models. On small datasets (386 rows), expect lower PM R² compared to BiLSTM.

---

## 11. Training Details

### Optimiser: AdamW

```python
optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-3    # L2 regularisation — stronger than default 1e-4
)
```

Weight decay was increased in v2 to combat overfitting on small datasets (386 rows).

### Learning Rate Schedule: Cosine Annealing with Warm Restarts

```python
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=30,      # first restart after 30 epochs
    T_mult=2     # each restart period doubles (30, 60, 120...)
)
```

This avoids the premature learning rate collapse problem that occurred in v1 with ReduceLROnPlateau.

### Gradient Clipping

```python
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Prevents exploding gradients during training on spiky PM data.

### Early Stopping

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_checkpoint()
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:  # default: 30 epochs
        break
```

The best checkpoint is restored after training — not the last epoch.

### Data Split

```
Total data (n rows)
├── Training:   70%  (rows 0 to 0.70n)
├── Validation: 15%  (rows 0.70n to 0.85n)
└── Test:       15%  (rows 0.85n to n)
```

The split is **chronological** — no shuffling — to prevent data leakage from future to past.

---

## 12. Output Files

All files are saved to the directory specified in the Output section (default: `results_early_v2/`).

### Per-Model Per-Target Causal Chain Plots

**Filename:** `{ModelName}_{target}_early.png`

**Example:** `BiLSTM_pm2_5_early.png`

4-panel stacked plot showing the complete causal chain:
- Panel 1 (tall): Actual sensor trace + Predicted (+10 min) + ±error band + warning gap + phase bands
- Panel 2: Person count (orange fill)
- Panel 3: Motion magnitude + Motion variance (dual y-axis)
- Panel 4: Door open sum (green fill)

**Example files generated per model:**

```
BiLSTM_pm1_early.png
BiLSTM_pm2_5_early.png
BiLSTM_pm10_early.png
BiLSTM_co2_early.png
BiLSTM_voc_early.png
```

---

### Consolidated All-Models Per-Pollutant Plots

**Filename:** `CONSOLIDATED_{target}.png`

One chart per pollutant with all trained models overlaid, enabling direct visual comparison. Includes the same person/motion/door panels below.

---

### Lead-Gap Comparison

**Filename:** `LEAD_GAP_{target}.png`

Shows the temporal offset between the prediction line and the actual sensor line — visually proving that the prediction rises *before* the sensor reacts.

---

### R² Heatmap

**Filename:** `HEATMAP_R2_lead{N}.png`

Grid of R² values: rows = models, columns = pollutants. Green = high R², red = low R². Immediately shows which models fail on which targets.

---

### RMSE Bar Chart

**Filename:** `RMSE_BAR_lead{N}.png`

Grouped bar chart of RMSE per model per pollutant. Includes a zoomed PM-only panel because PM RMSE values (0.9–2.0 μg/m³) are far smaller than CO₂ RMSE (88–130 ppm) and would be invisible on the full-scale chart.

---

### Ablation Study

**Filename:** `ABLATION_STUDY_lead{N}.png`

Two-panel chart:
- Left: R² as features are removed one group at a time (modality ablation)
- Right: R² at different lead times T+1, T+3, T+5, T+10, T+15, T+20 (lead-time ablation)

**Filename:** `ablation_results_lead{N}.csv`

Numeric table of all ablation results.

---

### Prediction CSV

**Filename:** `predictions_lead{N}_v2.csv`

Columns:

| Column | Description |
|--------|-------------|
| `trigger_timestamp` | Time when the model made the prediction |
| `future_timestamp` | Time the prediction refers to (+N minutes later) |
| `lead_minutes` | Lead time used |
| `model` | Model name |
| `target` | Pollutant (pm1, pm2_5, pm10, co2, voc) |
| `actual` | Actual sensor reading at future_timestamp |
| `predicted` | Model prediction made at trigger_timestamp |
| `error` | actual - predicted |

---

### With/Without-C_t Validation Outputs (FIX 9–13 — see §19 for full explanation)

| File | Contents |
|---|---|
| `regime_stratified_metrics_lead{N}.csv` | R²/RMSE, with-Ct vs without-Ct, split baseline vs onset |
| `phase_stratified_metrics_lead{N}.csv` | Same, split by the 4 operational phases instead |
| `causality_lag_analysis.csv` | Granger causality + cross-correlation-at-lag, each C_t feature vs each pollutant |
| `event_detection_lead{N}.csv` | Precision/recall/mean-lead-time-to-alert per pollutant threshold, with-Ct vs without-Ct |
| `paired_significance_onset_lead{N}.csv` | Diebold-Mariano + paired t-test, onset-window squared residuals |
| `predictions_noct_lead{N}.csv` | Without-Ct per-minute predictions (same schema as `predictions_lead{N}_v2.csv`) |
| `trigger_events_lead{N}.csv` | Real door/motion/cutting trigger timestamps used for the onset/baseline split |

---

## 13. Understanding the Plots

### Reading the Main Panel (Panel 1)

```
         +10 min lead
          ←→
    -----.              .-------    ← Predicted (dashed red)
    ....../:::::::::::::.......     ← ±error band (pink shading)
    |||||||                         ← Warning gap (orange fill between actual and predicted)
    ------.---------.              ← Actual (solid orange)
    ↑                ↑
 Door opens    Actual sensor rises
 (trigger)     10 min later
```

When the alignment checkbox is ON, the predicted line is plotted at the trigger time. This makes the lead visible: **the prediction rises at the same moment the door opens, 10 minutes before the actual sensor responds.**

### The Warning Gap

The orange shaded region between the actual and predicted lines is the **warning gap** — the area where the prediction has already detected a problem but the sensor has not yet confirmed it. The larger this gap, the more useful the system.

### Phase Bands

The background colour of the main panel shows which operational phase is active (see Section 14). Phase labels appear as coloured boxes at the top of the panel.

### Door Open Event Annotations

Green triangle arrows (▲) mark moments when a machine door opened. These are the **causal triggers** for PM and VOC spikes.

---

## 14. Operational Phases

The session is divided into 4 phases defined by time boundaries:

| Phase | Time | Colour | Description |
|-------|------|--------|-------------|
| **Phase 1** | 12:58 – 14:00 | Grey | Baseline / Pre-occupancy. Machines idle, few or no people. |
| **Phase 2** | 14:00 – 15:30 | Orange | High-occupancy demonstration. Many people, active machine use. |
| **Phase 3** | 15:30 – 16:30 | Dark red | Independent fabrication / sustained cutting. Maximum emission. |
| **Phase 4** | 16:30 – end | Blue | Post-occupancy decay. People leave, pollutants decay by ventilation. |

These boundaries are hardcoded to the specific IIT Bombay session date (12-02-2026). For a different session, edit the `_phase_boundaries()` function in the code:

```python
def _phase_boundaries(ts_ref, t_start=None, t_end=None):
    day    = pd.Timestamp(ts_ref[0]).normalize()
    b_1258 = day + pd.Timedelta(hours=12, minutes=58)  # ← change to your start
    b_1400 = day + pd.Timedelta(hours=14, minutes=0)   # ← change phase 2 start
    b_1530 = day + pd.Timedelta(hours=15, minutes=30)  # ← change phase 3 start
    b_1630 = day + pd.Timedelta(hours=16, minutes=30)  # ← change phase 4 start
```

---

## 15. Key Fixes in v2 (vs v1)

These are the specific bugs fixed between version 1 and version 2, documented here so future developers understand the design decisions:

### FIX 1 — PM R² Was Negative
**Root cause:** PM features (lags, diffs) were not included by default. PM₁/PM₂.₅/PM₁₀ are not predictable from VOC/CO₂ triggers alone — they need their own history.  
**Solution:** PM lags (1-3 min) and PM diff are now always included and prominently labelled as "ALWAYS recommended".

### FIX 2 — Overfitting on Small Dataset
**Root cause:** Hidden units of 128 and 2 stacked layers is too complex for 386 rows (~270 training sequences).  
**Solution:** Default hidden = 64, layers = 1. Added stronger L2 weight decay (1e-3 instead of 1e-4).

### FIX 3 — Loss Weights Imbalanced
**Root cause:** Fixed weights [1,1,1,2,2] gave PM and CO₂/VOC equal effective weight, but in normalised space PM variance is 10-100× smaller than CO₂ variance, so MSE on PM was negligible.  
**Solution:** Adaptive weights computed as `1 / train_variance` — targets with smaller variance get proportionally higher weight.

### FIX 4 — Data Leakage via Scaler
**Root cause:** RobustScaler was fitted on the entire dataset before splitting, so the model indirectly saw test data statistics during training.  
**Solution:** StandardScaler and MinMaxScaler are now fitted on training rows only, then applied to all.

### FIX 5 — No PM-Specific Momentum Features
**Root cause:** PM spikes are 1-3 minute events; CO₂ changes over 5-10 minutes. The same lag depth was used for both.  
**Solution:** Short PM lags (1, 2, 3 min) added separately from CO₂ lags (1, 2, 3, 5-min rolling).

### FIX 6 — Feature Scaler Clipped PM Spikes
**Root cause:** RobustScaler clips outliers to the interquartile range, which specifically reduces the amplitude of PM spikes — the most important signal.  
**Solution:** StandardScaler preserves spike magnitude (clips nothing).

### FIX 7 — Fusion Bottleneck Was Counter-Productive
**Root cause:** A Dense(64) fusion layer before the LSTM made no sense when the input was only 25 features — it reduced dimensionality before the temporal encoder could use it.  
**Solution:** Direct LSTM input from the feature vector.

### FIX 8 — No Temporal Focus Mechanism
**Root cause:** Without attention, the LSTM weighted all timesteps equally. The trigger moment (door opening) might be at position t-3 in a 20-step window and get the same gradient as background timesteps.  
**Solution:** Added soft attention over the LSTM output sequence for BiGRU and BiLSTM.

---

## 16. Troubleshooting

### "Too few rows (<100). Aborting."

Your time window filter is too narrow or the CSV has very little data. Either widen Start/End datetime or reduce the lead + lookback settings so more sequences are created.

### "Too few training sequences."

Reduce `Lead steps` or `Lookback window`. The minimum sequences needed is:
```
sequences = total_rows - lookback - lead + 1
training_sequences = sequences × train_fraction
```
Ensure training_sequences ≥ 10.

### PM R² is very low or negative

1. Check that **PM lags 1-3** is checked ✅
2. Check that **PM momentum diff1** is checked ✅
3. Reduce hidden units to 32 or 64 if your dataset is small
4. Increase patience to 30 epochs

### CO₂ R² is poor

1. Check that **CO₂ lags** is checked ✅
2. Ensure the `n_person` column exists and has non-zero values
3. CO₂ is driven by occupancy — if all person counts are 0, the model has no causal signal

### "XGBoost not available"

Install XGBoost: `pip install xgboost`. The system runs without it; XGBoost just won't appear in the model list.

### GUI window is blank or very small

The minimum window size is 980 × 720. If your display is smaller, the window may be truncated. Try scaling your OS display settings.

### Training never stops

Click **■ STOP** to interrupt. The stop flag is checked at each epoch. The model will finish the current batch, then halt cleanly.

### Plot x-axis shows no time labels

This occurs when the bottom subplot (Door open sum, ax4) has its x-tick labels suppressed. Check that this block exists in `plot_early()`:
```python
ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax4.tick_params(axis="x", labelbottom=True, colors=TX_MUTE,
                labelsize=9, rotation=30)
```

---

## 17. Code Architecture Map

```
train_context_aware_bilstm_gui.py  (~3600 lines — was train_context_aware_bilstm_gui.py/train_context_aware_bilstm_gui.py/train_context_aware_bilstm_gui.py in earlier versions)
│
├── THEME CONSTANTS (lines 67-84)
│   └── All hex colour codes, font definitions
│
├── QueueHandler (lines 90-93)
│   └── Routes logging.Logger output to the live log widget
│
├── Widget helpers (lines 99-151)
│   ├── section_card()    — creates a titled card frame
│   ├── row_entry()       — label + text input row
│   ├── row_spin()        — label + spinbox row
│   ├── row_combo()       — label + dropdown row
│   └── divider()         — horizontal separator line
│
└── EarlyDetectionApp (lines 157-2600)  — main tk.Tk subclass
    │
    ├── __init__()                — window setup, queue, thread event
    ├── _style_ttk()              — applies custom dark theme to ttk widgets
    ├── _build_ui()               — creates PanedWindow with left/right panels
    │
    ├── LEFT PANEL
    │   ├── _build_left()         — scrollable canvas container
    │   ├── _sec_data()           — ① CSV file + column mapping
    │   ├── _sec_window()         — ② time window start/end
    │   ├── _sec_lead()           — ③ lead time + lookback + align toggle
    │   ├── _sec_features()       — ④ feature toggle checkboxes
    │   ├── _sec_models()         — ⑤ model selection + quick-select buttons
    │   ├── _sec_targets()        — ⑥ target pollutant selection
    │   ├── _sec_training()       — ⑦ epochs, hidden, dropout, LR, batch
    │   ├── _sec_output()         — ⑧ output dir, save CSV/plots toggles
    │   └── _sec_run()            — ▶ RUN button + ■ STOP + progress bar
    │
    ├── RIGHT PANEL
    │   └── _build_right()        — live log text widget + fixes summary + metrics table
    │
    ├── HELPERS
    │   ├── _sel_all/none/deep/ml()  — model selection shortcuts
    │   ├── _browse_csv()            — file dialog
    │   ├── _browse_outdir()         — directory dialog
    │   ├── _start()                 — validates config, launches _run() thread
    │   ├── _stop()                  — sets stop_flag threading.Event
    │   ├── _validate()              — checks required fields before running
    │   ├── _finish()                — re-enables GUI after run completes
    │   ├── _append_log()            — colourises and writes to log widget
    │   ├── _poll_log()              — drains the log queue every 100ms
    │   ├── _clear_log()             — empties the log
    │   ├── _copy_log()              — copies log to clipboard
    │   └── _update_metrics()        — writes final table to metrics text widget
    │
    └── _run(cfg)  (lines ~720-2595) — ENTIRE PIPELINE runs here in background thread
        │
        ├── Data loading & filtering
        ├── Column fill / forward-fill
        ├── Feature engineering (all pm/voc/co2/door/motion features)
        ├── Feature selection (based on GUI toggles)
        ├── StandardScaler + MinMaxScaler (train-only fit)
        ├── Adaptive loss weight computation
        ├── Sliding window sequence construction
        ├── PyTorch Dataset + DataLoader creation
        │
        ├── Model class definitions (all inline):
        │   ├── Attention           — soft attention module
        │   ├── BiGRU               — bidirectional GRU + attention
        │   ├── BiLSTM              — bidirectional LSTM + attention + per-target heads
        │   ├── GRU                 — unidirectional GRU
        │   ├── LSTM                — unidirectional LSTM
        │   ├── RNN                 — vanilla RNN
        │   ├── TCN                 — dilated causal convolution
        │   ├── S2S (Seq2Seq)       — encoder-decoder LSTM
        │   ├── CNNLSTM             — Conv1D + LSTM
        │   ├── Transformer         — self-attention encoder
        │   └── PatchTST            — patched transformer
        │
        ├── train_model()           — AdamW + CosineAnnealingWarmRestarts + early stopping
        ├── predict_deep()          — inference + inverse scaling
        ├── predict_ml()            — scikit-learn predict + inverse scaling
        ├── metrics()               — R², RMSE, MAE per target + overall
        │
        ├── _phase_boundaries()     — defines 4 time phase boundary Timestamps
        ├── _draw_phase_bands()     — draws coloured bands + labels on axes
        │
        ├── §19 WITH/WITHOUT-CT VALIDATION (FIX 9-13 — see §19 for the "why")
        │   ├── _build_regime_labels()      — baseline/onset labels from RAW signals
        │   ├── _build_phase_labels()       — per-row phase assignment (reuses _phase_boundaries)
        │   ├── _get_trigger_timestamps()   — real door/motion/cutting rising-edge events
        │   ├── _stratified_metrics()       — R²/RMSE split by regime
        │   ├── _phase_stratified_metrics() — R²/RMSE split by phase
        │   ├── _detect_events()            — rising-edge event detection at a threshold
        │   ├── _event_scoring()            — precision/recall/mean-lead-time, greedy matching
        │   ├── _diebold_mariano()          — paired forecast-accuracy significance test
        │   ├── _cross_corr_lag()           — correlation(x[t], y[t+lag]) sweep
        │   ├── _granger_pvalue()           — Granger causality, returns (lag, p, fail_reason) — FIX 11
        │   └── train + predict a matched without-Ct model per architecture, export
        │       predictions_noct_lead{N}.csv / trigger_events_lead{N}.csv — FIX 13
        │
        ├── plot_early()            — 4-panel causal chain plot per model × target
        ├── plot_consolidated_per_pollutant()  — all models on one chart
        ├── plot_lead_gap()         — lead-time offset visualisation
        ├── plot_heatmap()          — R² heatmap grid
        ├── plot_rmse_bar()         — RMSE grouped bar chart
        ├── run_ablation_study()    — feature + lead-time ablation
        │
        └── Final metrics table printed to log + GUI metrics widget
```

---

## 18. Configuration Quick Reference

For advanced users who want to change defaults without touching the GUI, the key constants defined at the start of `_run()` are:

```python
LEAD    = int(cfg["lead"])         # default 10
LB      = int(cfg["lookback"])     # default 20
HIDDEN  = int(cfg["hidden"])       # default 64
N_LAYERS= int(cfg["n_layers"])     # default 1
DROPOUT = float(cfg["dropout"])    # default 0.30
LR      = float(cfg["lr"])         # default 0.001
EPOCHS  = int(cfg["epochs"])       # default 200
BATCH   = int(cfg["batch"])        # default 16
TF      = float(cfg["train_frac"]) # default 0.70
VF      = float(cfg["val_frac"])   # default 0.15
SEED    = int(cfg["seed"])         # default 42
PATIENCE= int(cfg["patience"])     # default 30

ALL_TGT  = ["pm1","pm2_5","pm10","co2","voc"]   # prediction targets
MACHINES = [1, 2, 3]                              # machine numbers
OP_ST    = ["IDLE","CUT","EXPOSURE","MAINTENANCE"] # machine operational states
```

Phase time boundaries (edit for different sessions):

```python
b_1258 = day + pd.Timedelta(hours=12, minutes=58)  # session start
b_1400 = day + pd.Timedelta(hours=14, minutes=0)   # Phase 2 start
b_1530 = day + pd.Timedelta(hours=15, minutes=30)  # Phase 3 start
b_1630 = day + pd.Timedelta(hours=16, minutes=30)  # Phase 4 start
```

---

## 19. With/Without-C_t Validation Framework (FIX 9–13)

**This is the part of the system that actually answers the research
question, and it has no representation anywhere above this section.**
Everything in §1–18 describes training *one* model on *one* feature set.
What actually runs, every time, is two matched training passes per
architecture: the full 68-column feature set ("with C_t") and the same
architecture retrained with all C_t-derived columns stripped ("without
C_t") — see `CT_KEYWORDS` in the code for the exact strip list (`door`,
`phi_open`, `rho_open`, `eps_max`, `emission_weight`, `effective_tau`,
`consecutive_full_open`, `f_trans`, `motion`, `person`, plus the
machine operational-state one-hots). Comparing those two trained models is
the entire point: does the vision-derived context vector actually help, or
does pooled R²/RMSE only make it look that way?

### Why pooled R²/RMSE isn't sufficient on its own (FIX 9)

On a session like this one, roughly 80% of the recording is quiescent —
nothing is happening, and any model gets those minutes almost right just
from each pollutant's own short-term persistence. Quiescent minutes
dominate a pooled metric by sheer volume, so a model can lose essentially
all of its ability to anticipate an actual emission event and still show a
respectable overall R², because the easy 80% of the data is carrying the
score. A single R²/RMSE number cannot tell you whether that's happening.

**What the pipeline does about it — regime-stratified validation.** Every
test-window minute is labeled **baseline** (quiescent) or **onset**
(transition/emission window) using the *raw* door/motion/machine-state
signals — never the model's own features or predictions, so the split is
independent of whatever is being evaluated. R²/RMSE are then reported
separately for each regime, for both the with-C_t and without-C_t model.
**Output files**: `regime_stratified_metrics_lead{N}.csv`,
`REGIME_STRATIFIED_R2.png`. The same idea is applied a second way —
**phase-stratified** validation, splitting by the four fixed operational
phases (§14) instead of a binary onset/baseline split, since a model can
be onset-good overall while still failing badly in one specific phase.
**Output files**: `phase_stratified_metrics_lead{N}.csv`,
`PHASE_STRATIFIED_R2.png`.

### Why R²/RMSE is the wrong metric for an early-warning claim (FIX 10)

A model that "only" loses two points of R² without C_t can still have
silently lost all of its useful lead time — R² doesn't measure *when* a
model got a prediction right, only *how close* on average. Three
complementary analyses replace the R²-only framing:

- **Causal-detection**: Granger-causality and cross-correlation-at-lag
  between each C_t feature and each pollutant series — does C_t
  statistically precede the pollutants it's meant to explain, and at what
  lag? **Output file**: `causality_lag_analysis.csv`.
- **Event-detection precision/recall/mean-lead-time**: for each pollutant
  with a configured alert threshold (§7①, e.g. CO2 > 2000 ppm), a real
  event is a rising edge in the *actual* series; a predicted alert is a
  rising edge in the *model's own forecast* series, timestamped at the
  moment the forecast was made. Each real event is matched to the earliest
  qualifying alert (within `event_match_horizon_var`, default 30 minutes)
  that came before it. `mean_lead_min` is the average, over successfully
  matched pairs only, of how many minutes of genuine advance warning the
  alert gave — computed for with-C_t and without-C_t separately, so you
  can see directly whether C_t buys real warning time or just a better
  average fit. **Output file**: `event_detection_lead{N}.csv`.
- **Paired significance testing**: a Diebold-Mariano test and a paired
  t-test on squared onset-window residuals, with-C_t vs. without-C_t
  against the same ground truth — a statistical test for "is the
  difference real," not just a point-estimate gap. **Output file**:
  `paired_significance_onset_lead{N}.csv`.

### What made all of the above possible to check externally (FIX 13)

Originally only the with-C_t model's aggregated predictions were ever
exported. The without-C_t model's per-minute predictions, and the real
door/motion/cutting trigger timestamps used internally for the onset/
baseline split, were computed but discarded — meaning none of the analysis
above could be reproduced, extended, or audited outside a live GUI run.
**Output files added**: `predictions_noct_lead{N}.csv` (without-C_t,
same schema as the existing `predictions_lead{N}_v2.csv`) and
`trigger_events_lead{N}.csv` (the real trigger timestamps). Together with
the existing with-C_t export, this is enough to build further analysis
(e.g. an external script testing custom alert thresholds, or a continuous
event-proximity-weighted accuracy metric) without needing the raw sensor
CSV or retraining anything.

---

## 20. Key Fixes in v3 (vs v2)

Continues the FIX log from [§15](#15-key-fixes-in-v2-vs-v1) — these are
FIX 9–15, the changes between v2 (`train_context_aware_bilstm_gui.py`/`train_context_aware_bilstm_gui.py`) and v3
(`train_context_aware_bilstm_gui.py`). FIX 9, 10, and 13 are explained in depth in §19 above since
they're inseparable from the validation framework itself; they're
summarized here for a complete chronological log.

### FIX 9 — Pooled R²/RMSE Hides Whether C_t Matters
**Root cause**: ~80% of a typical session is quiescent and trivially
predictable from each pollutant's own short-term persistence; those
minutes dominate a pooled metric by volume. **Solution**: regime-stratified
validation (baseline vs. onset, from raw signals) — see §19.

### FIX 10 — R²/RMSE Is the Wrong Metric for an Early-Warning Claim
**Root cause**: R² measures average closeness, not whether a model
anticipated an event early enough to be useful. **Solution**:
causal-detection (Granger/cross-correlation), event-detection precision/
recall/lead-time, and paired Diebold-Mariano/t-test significance testing —
see §19.

### FIX 11 — Granger Causality Silently Returned Blank P-Values
**Root cause**: `_granger_pvalue()` caught every failure mode identically —
including a plain "`statsmodels` not installed" `ImportError` — and
returned `(None, None)` with no log message, so a missing optional
dependency was indistinguishable from "the test ran and found nothing."
**Solution**: `_granger_pvalue()` now returns a third value, the failure
reason. The pipeline reports `statsmodels`' install status up front, and
logs the first few distinct failure reasons (missing package, too few
usable rows, zero-variance series, or the underlying exception) if pairs
still fail — a blank cell is now always traceable.

### FIX 12 — The Door-State Temporal Encoding Vector (D_t) Was Incomplete
**Root cause**: of D_t's five descriptors — `phi_open`, `rho_open`,
`eps_max`, `effective_tau`, `f_trans` — only `phi_open` (plus the derived
`emission_weight`) ever reached `feat_cols`. The other four were loaded
into the dataframe and used by the standalone causality-lag diagnostic, but
never became model input — meaning every with-C_t/without-C_t comparison
before this fix was never actually testing the complete D_t vector
described in the manuscript. **Solution**: the door-physics feature block
now adds all five D_t descriptors plus `emission_weight`, per machine. No
change was needed to the without-C_t stripping logic — `CT_KEYWORDS`
already defensively listed all five, so the without-C_t variant correctly
excluded them as soon as they started actually being added.

### FIX 13 — No Way to Compare With/Without-C_t Minute-by-Minute Outside the GUI
**Root cause**: only aggregated with-C_t predictions were ever exported.
**Solution**: added `predictions_noct_lead{N}.csv` and
`trigger_events_lead{N}.csv` — see §19.

### FIX 14 — "Door Open Sum" Panel Ran Opposite to Intuition
**Root cause**: `door_open_sum` is built from `phi_open` ("opening-phase
position"), which the manuscript describes as reading LOW while a door is
early in an active opening event and HIGH once it has settled back to
idle — the reverse of "amount of door openness" the panel label implies.
This showed up as a clean anti-correlation with the People-count panel
directly above it (high when the room was empty, low when busy). **Solution
(display-only)**: `_get_ctx_for_widx()` reflects `door_open_sum` around its
own session min/max before plotting, so the chart reads high = more open,
matching its label. The underlying `df["door_open_sum"]` column — and
therefore feature engineering, trigger/onset detection, and the causality
analysis — was untouched by this fix; see FIX 15.

### FIX 15 — Door Orientation Fixed at the Source (tentative)
**Root cause**: FIX 14 only patched the chart. `door_open_sum` was still
backwards everywhere else — `feat_cols`, `door_diff`, `door_exposure`, the
onset/trigger rising-edge detector (`door_rise`), and the causality
cross-correlation were all still reading it in `phi_open`'s original
orientation, meaning `door_rise` was plausibly firing when a door settled
back to idle, not when it opened. **Solution**: `df["door_open_sum"]` is
now reversed once, immediately after it's built (reflected around its own
session min/max), before anything derives from it — every downstream
consumer inherits the corrected orientation automatically. FIX 14's
display-only flip was reverted, since flipping an already-corrected value
again would silently restore the original wrong orientation. **Marked
tentative**: this is inferred from the manuscript's own description of
`phi_open` plus an empirical anti-correlation check, not confirmed against
a real per-machine door sensor. Re-verify if one becomes available before
treating results built on this as final.

---

## 21. Known Gap: Incomplete Vision-Side C_t Features

**Correction (verified against the real v1 training CSV during the
edge-deployment work in `context-aware-bilstm-edge` — see that repo's
`feature_engineering.py`):** this section previously grouped
`M{1,2,3}_effective_tau` in with the three genuinely dead feature groups.
That was wrong. Checked directly against `data/raw/sensor_data_merged_iaq_m2.csv`:
`M{1,2,3}_effective_tau` is **present with real, varying values** (28-34
unique values per machine, range 0-57) — it is NOT constant zero. The
three that actually are constant zero — confirmed the same way — are
`M{1,2,3}_emission_weight` (3), `M{1,2,3}_consecutive_full_open` (3), and
the 12 `M{1,2,3}_is_{IDLE,CUTTING,EXPOSURE,MAINTENANCE}` one-hots (12).
3+3+12 = **18**, which is where that figure actually comes from — not
3+3+3+12=21 as the original wording implied by including `effective_tau`.
`base_cols` silently defaults any of the four groups to `0.0` when absent
from the input CSV (by design, so the GUI degrades gracefully on partial
data) rather than erroring, which is exactly why this went unnoticed
until it was traced end-to-end against real data instead of assumed from
the code's structure.

**Where `effective_tau` actually comes from, and why it's excluded from
the edge-deployed model anyway:**
[`../data_pipeline/merge_vision_and_sensor_data.py`](../data_pipeline/README.md)'s
`enrich_operational_state()` computes it (`effective_tau = tau_open ×
emission_weight`, with `emission_weight` looked up from an
`_classify_state()` call that itself depends on `consecutive_full_open` —
see that file's own CLAUDE.md). It's a **stateful, order-dependent
running counter over the entire history** ("`consecutive_full_open` does
not reset across a gap"), computed row-by-row from the start of the
dataset — not a windowed feature. `sensor_data_merged_iaq_m2.csv` retains
the resulting `effective_tau` column but not the intermediate
`emission_weight`/`op_state`/`consecutive_full_open` columns it was
computed from (this CSV predates, or was exported without,
`enrich_operational_state()`'s full output — see that module's own
docstring history). This statefulness is why `context-aware-bilstm-edge`
deliberately excludes `effective_tau` from the model actually deployed to
the Pi: reproducing it in live inference would require porting
`enrich_operational_state()`'s exact row-by-row classifier to run
continuously on the device, not just a training-config change. See
`context-aware-bilstm-edge/feature_engineering.py`'s docstring for the
full reasoning and the `use_effective_tau` / `use_emission_weight` toggle
split that resulted from this.

**The actual root cause of the 18 dead columns is upstream**, in
[`../vision/extract_context_vector_from_video.py`](../vision/README.md):
`compute_Dt()` — the function that turns YOLO door detections into the D_t
feature vector — only ever computes
`tau_open`, `f_trans`, `rho_open`, `eps_max`, `phi_open`. It has no
`emission_weight`, `consecutive_full_open`, or per-machine
operational-state (`IDLE`/`CUTTING`/`EXPOSURE`/`MAINTENANCE`) logic at all
(that logic lives one stage downstream, in `enrich_operational_state()` —
see above). `M{n}_op_state` — the column this repo's op-state one-hot
block reads — isn't produced by `compute_Dt()` either, for the same
reason; `iaq-edge-pipeline`'s own `compute_Dt()` (used live on the Pi) has
the identical, deliberately-matching set of outputs.

**This does not invalidate the with/without-C_t comparisons in §19** —
those 18 columns are zero in both configurations, so they contribute
nothing to either side of the comparison; `effective_tau` (real, not
dead) IS meaningfully present in the with-C_t configuration and absent
from without-C_t, so it does contribute to that comparison, correctly.
Two ways forward for the 18 genuinely dead columns, not mutually
exclusive:
1. **Implement `emission_weight`/`op_state`/`consecutive_full_open`
   upstream** in `compute_Dt()` if you want them in a *research* model
   (they're already computed downstream by `enrich_operational_state()`
   for `effective_tau`'s sake — extending `compute_Dt()` itself would
   mean duplicating or relocating that logic, a real design decision, not
   a quick fix).
2. **At minimum, strip them from `feat_cols`** until they're real, so the
   68-column feature count in the manuscript matches what's actually being
   tested — reporting "68 features" when 18 are provably inert overstates
   the with-C_t configuration's actual input richness.

---

## License

This software is developed for research purposes at IIT Bombay. Please contact the authors before using in commercial applications.

---

## Citation

If you use this system in your research, please cite:

```bibtex
@article{kahalekar2026iaq,
  title   = {Context-Aware Lead-Time Forecasting of Indoor Air Quality in 
             Laser Fabrication Laboratories: A Multimodal Bi-LSTM Framework},
  author  = {Kahalekar, Sunil and Sahu, Manoranjan},
  journal = {IEEE Transactions on Industrial Informatics},
  year    = {2026},
  institution = {IIT Bombay}
}
```

---

*Documentation generated from `train_context_aware_bilstm_gui.py` — IAQ Early Detection System v2*
