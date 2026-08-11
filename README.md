# Context-Aware BiLSTM — Vision-Grounded Lead-Time IAQ Forecasting

**Does a vision-derived context vector (C_t — door state + human motion,
extracted from CCTV via a custom-trained YOLO model) let a deep sequence
model forecast indoor air quality 10 minutes ahead, before the chemical
sensors themselves react — and is that benefit real, or a pooled-metric
illusion?**

This repo is the complete pipeline that answers that question, end to end:
raw CCTV footage in, a statistically-validated verdict on C_t's effect out.
Every stage below has its own `README.md` (how to run it) and `CLAUDE.md`
(internals, for anyone modifying the code) — this file is the map so a new
reader can see how the pieces fit together before diving into any one of
them.

## The pipeline, stage by stage

```
 1. CCTV capture (.3gp)
        │
 2. Roboflow annotation (4,110 frames: person, M{1,2,3}_door_open/closed)
        │
 3. YOLO fine-tuning ──────────────────────► src/vision/train_door_person_detector.py
        │
 4. Frame extraction + inference (1 fps) ──► src/vision/extract_context_vector_from_video.py
        │   produces: tracking_telemetry, per_second_analytics, Ct_vectors
        │
 5. Merge with pollutant sensor log ───────► src/data_pipeline/merge_vision_and_sensor_data.py
        │   produces: data/raw/sensor_data_merged_iaq_m2.csv (one row per minute)
        │
 6. Feature engineering + model training ──► src/modeling/train_context_aware_bilstm_gui.py
        │   68-column tensor, with-C_t vs. without-C_t, BiLSTM/BiGRU/GRU/...
        │   produces: data/runs/v1/predictions_lead10_v2.csv, event_detection_lead10.csv, etc.
        │
 7. Statistical validation — does C_t actually help? ─► src/analysis/ct_significance_testing/
        │   phase-stratified significance, Granger causality, cross-run reliability
        │
 8. Phase-wise & lead-time deep dives ─────► src/analysis/phase_wise_prediction_traces/
            prediction traces, event-detection early-warning     src/analysis/forecast_horizon_and_early_warning/
```

### Stage 1–3 — Capture, annotation, YOLO training
CCTV footage of the fabrication cell is annotated on Roboflow across three
classes (`person`, `machine_N_door_open`, `machine_N_door_closed`) — 4,110
frames (3,636 train / 319 val / 155 test), augmented with rotation, scaling,
histogram equalisation, and grayscale conversion. A YOLO model is fine-tuned
on this corpus. **Full detail**: [`src/vision/README.md`](src/vision/README.md).

### Stage 4 — Frame extraction & feature computation
Reads raw video at 1 fps (one keyframe per second — chosen to suppress
transient motion artefacts while still catching emission-triggering
events), runs the trained YOLO model on each keyframe, tracks people with
stable cross-occlusion IDs, and debounces per-machine door detections into
a committed open/closed state. Every 60 keyframe-seconds it flushes one row
of vision-derived features: `M{n}_tau_open, M{n}_f_trans, M{n}_rho_open,
M{n}_eps_max, M{n}_phi_open, n_person, mu_motion, sigma2_motion`. **Full
detail**: [`src/vision/README.md`](src/vision/README.md) (Part 2).

> **Known gap**: the modeling stage's feature list also expects
> `M{n}_emission_weight`, `M{n}_effective_tau`, `M{n}_consecutive_full_open`,
> and per-machine operational-state one-hots. `compute_Dt()` here **does not
> compute any of these** — they were never implemented at this stage, not
> lost in the merge. The modeling GUI silently defaults them to 0.0 when
> absent rather than erroring, so every run to date has trained on 18 of 68
> "with-C_t" columns that are constant zero. This is a real, fixable
> extension to `compute_Dt()`, not a bug to work around downstream — see
> [`src/modeling/README.md`](src/modeling/README.md) §21 for the full
> explanation and its (currently negligible) practical impact.

### Stage 5 — Merge with pollutant data
Joins the vision pipeline's `Ct_vectors_<tag>.csv` against the per-minute
pollutant sensor log (PM1/PM2.5/PM10/CO2/VOC, temp/hum) on timestamp,
producing `sensor_data_merged_iaq_m2.csv` — the single input file
everything downstream consumes (checked into `data/raw/`). **Full detail**:
[`src/data_pipeline/README.md`](src/data_pipeline/README.md).

### Stage 6 — Feature engineering + model training
Expands the merged CSV's ~11 raw columns into a 68-column tensor and trains
sequence models. The tensor decomposes as:

| Group | Dimension | Contents |
|---|---|---|
| Environmental | 2 | `temp`, `hum` |
| Pollutant history | 25 | raw pm1/pm2_5/pm10/co2/voc (5) + PM lags (7) + PM diff (4) + PM roll (3) + VOC diff (2) + CO2 lags/roll (4) |
| Context (C_t) | 41 | person + motion + door (11) + emission_weight/phi_open/rho_open/eps_max/effective_tau/f_trans ×3 machines (18) + consecutive_full_open ×3 (3) + op-state one-hot ×3×4 (12) |

(2+25+41 = 68, verified against the pipeline's own logged `Feature set (68
cols)` output. If you've seen this described elsewhere as E_t∈ℝ⁴/P_t∈ℝ²³/C_t∈ℝ³³,
that's a different grouping convention that doesn't sum to 68 — reconcile
before publishing both versions.)

Models are trained twice per configuration — once on all 68 columns
("with C_t"), once with all 41 C_t columns stripped ("without C_t") — at
T+10 minutes, the target this project settled on because it's the shortest
horizon that still sits past where a naive "nothing changed" baseline stops
working (see the forecast-horizon analysis, stage 8). **Full detail**:
[`src/modeling/README.md`](src/modeling/README.md), including the complete
FIX 1–15 changelog and the with/without-C_t validation framework (§19).

### Stage 7 — Does C_t actually help?
The core validation question, answered from multiple angles because pooled
R²/RMSE alone is misleading (the original motivation for this whole
validation framework): phase-stratified significance testing
(Diebold-Mariano, FDR-corrected), Granger causality (does C_t statistically
precede the pollutants it's meant to explain?), cross-run reliability (does
the effect reproduce across independent training runs?). **Full detail**:
[`src/analysis/ct_significance_testing/README.md`](src/analysis/ct_significance_testing/README.md).

### Stage 8 — Phase-wise and lead-time deep dives
`phase_wise_prediction_traces/` — actual predicted-vs-actual trace plots,
with vs. without C_t, broken out by operational phase and pollutant.
`forecast_horizon_and_early_warning/` — how far ahead this system can
usefully forecast (persistence-baseline decay, bootstrap confidence bands)
and the operationally meaningful version of the same question: how many
minutes of real advance warning does C_t give before an actual pollution
spike. **Full detail**:
[`src/analysis/phase_wise_prediction_traces/README.md`](src/analysis/phase_wise_prediction_traces/README.md),
[`src/analysis/forecast_horizon_and_early_warning/README.md`](src/analysis/forecast_horizon_and_early_warning/README.md).

## The central finding, one level up from any single chart

C_t's benefit is real but **architecture-dependent, not universal** — do
not report it as a blanket "vision context always helps." BiLSTM shows the
cleanest, most reproducible benefit (event-level early-warning: 9–10 min of
advance warning for CO2/PM2.5 with C_t vs. none at all without it); BiGRU
shows the opposite for particulate matter, consistent with an
architecture-capacity failure rather than C_t being uninformative. See
[`src/analysis/ct_significance_testing/CLAUDE.md`](src/analysis/ct_significance_testing/CLAUDE.md)'s
standing instruction on this — it was added after an explicit back-and-forth
about not overstating this result, and it governs every downstream document
in this repo.

## Repository layout

```
context-aware-bilstm/
├── README.md / CLAUDE.md          you are here / master internals reference
├── requirements.txt                 consolidated dependencies (vision + modeling + analysis)
├── src/
│   ├── vision/                      YOLO training + video → C_t feature extraction
│   ├── data_pipeline/               merge vision + sensor data
│   ├── modeling/                    the training GUI (with/without-C_t, all architectures)
│   └── analysis/
│       ├── ct_significance_testing/         does C_t help, statistically?
│       ├── phase_wise_prediction_traces/    what do the prediction curves look like?
│       └── forecast_horizon_and_early_warning/  how far ahead, and how much warning?
├── data/
│   ├── raw/                         sensor_data_merged_iaq_m2.csv — input to everything downstream
│   └── runs/v1/                     small derived CSVs from the v1 training run (no checkpoints/video)
└── docs/figures/                    curated figures
```

## Getting started

```bash
git clone https://github.com/sunilkahalekar/context-aware-bilstm.git
cd context-aware-bilstm
pip install -r requirements.txt
```

To reproduce the statistical validation without retraining anything (uses
the checked-in `data/runs/v1/` CSVs):
```bash
python src/analysis/ct_significance_testing/scripts/phase_significance_analysis.py \
    --dir data/runs/v1 --lead 10 --outdir src/analysis/ct_significance_testing/data
```

To run the full pipeline from scratch, work through the stages in order —
each `src/*/README.md` has its own runnable quick-start.

## Known open issues (see the relevant CLAUDE.md for full detail)

- **BiLSTM-specific lead-time sweep (T+1 through T+20) is unresolved** —
  every training attempt so far has produced deeply negative R² beyond
  T+1. See `src/analysis/forecast_horizon_and_early_warning/CLAUDE.md`'s
  full debugging log before attempting another fix.
- **18 of 68 "with-C_t" features are constant zero** (see Stage 4's "Known
  gap" above) — doesn't invalidate the with/without-C_t comparisons, but
  means the true informative feature count is smaller than 68.
- **The door-orientation fix (FIX 15) is tentative** — inferred from the
  manuscript's own description of `phi_open` plus an empirical
  anti-correlation check, not confirmed against a real per-machine door
  sensor. See `src/modeling/README.md` §20.
