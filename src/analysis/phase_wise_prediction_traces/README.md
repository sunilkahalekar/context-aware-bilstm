# Phase-Wise Calculation — Prediction Traces, With C_t vs Without C_t

Visual companion to `src/analysis/ct_significance_testing/` (which answers "how big is C_t's effect"
as summary numbers). This folder answers "what does that effect actually
look like" — the real predicted curves, phase by phase, for each model.

Source run: v1 (same run as `src/analysis/ct_significance_testing/`; T+10
min lead). Small derived CSVs from that run's raw output are checked into
`data/runs/v1/` at the repo root.

## What "with Ct" / "without Ct" means here

Both come straight from the pipeline's own per-minute prediction exports —
no retraining, no recomputation of predictions:

- **With C_t**: `predictions_lead10_v2.csv` — model trained on the full
  feature set, including the vision-derived context vector C_t (door-state
  + human-motion descriptors).
- **Without C_t**: `predictions_noct_lead10.csv` — the *same* architecture,
  *same* training run, with only the C_t-derived columns removed from the
  feature set. Every other feature (pollutant lags, temp/humidity, rolling
  stats, etc.) is identical between the two.

They're merged on `(trigger_timestamp, future_timestamp, model, target)` so
every row is a genuinely paired comparison: same moment, same ground truth,
two competing forecasts.

## Folder contents

```
src/analysis/phase_wise_prediction_traces/
├── README.md                 <- you are here
├── scripts/
│   └── phase_wise_predictions.py   Does everything below.
│                                    Usage: python phase_wise_predictions.py
│                                           --dir <run_folder> --lead 10
├── figures/
│   └── {Model}_phase_wise_predictions.png   One per model (7 total).
│                                    5 rows (PM1, PM2.5, PM10, CO2, VOC) x
│                                    4 columns (Phase 1-4). Each cell:
│                                    Actual (black) vs Predicted-with-Ct
│                                    (blue) vs Predicted-without-Ct (orange)
│                                    as a time series over that phase's own
│                                    window.
└── data/
    └── phase_wise_prediction_metrics_lead10.csv
                                     One row per (model, phase, pollutant):
                                     n, RMSE_with_Ct, RMSE_without_Ct,
                                     R2_with_Ct, R2_without_Ct, RMSE gap,
                                     % change. The plain, unnormalized
                                     numbers behind the plots above.
```

## How the calculation works

For each (model, phase, pollutant) cell:

1. Take every prediction row falling in that phase (phase assigned from
   `future_timestamp` using the same fixed clock-time boundaries as the
   main pipeline — Phase 1 <=14:00, Phase 2 <=15:30, Phase 3 <=16:30,
   Phase 4 the rest of the session).
2. `RMSE = sqrt(mean((actual - predicted)^2))`, computed separately for
   the with-Ct and without-Ct predicted columns, in the pollutant's own
   raw units (µg/m³ for PM, ppm for CO2, ppb for VOC) — no cross-pollutant
   normalization here, unlike `src/analysis/ct_significance_testing`'s significance-tested figure.
3. `R^2 = 1 - SS_res/SS_tot` against the phase's own actual values.
4. `RMSE gap = RMSE_without_Ct - RMSE_with_Ct` (positive = C_t helps).

This is intentionally the plain, un-normalized calculation — it exists to
be directly readable next to the plots, not to make a magnitude claim.
**For the normalized, significance-tested magnitude claim (the one to cite
in a paper), use `src/analysis/ct_significance_testing/figures/OVERALL_AND_PHASE_EFFECT_SIGNIF.png`
and `src/analysis/ct_significance_testing/data/phase_significance_cells_lead10.csv` instead** — that
version corrects for CO2/VOC's larger raw scale drowning out PM, and for
the ceiling-compression effect that makes small RMSE differences look huge
as a raw percentage. This folder's numbers will disagree with that one in
magnitude for exactly those reasons; that's expected, not a bug.

## Reading the figures

- **Blue tracking Actual (black) more closely than orange** = C_t helping
  in that phase, for that pollutant, for that model.
- **Orange tracking Actual more closely than blue** = C_t hurting.
- **Blue and orange overlapping** = C_t made little difference there.
- Look at CO2 and VOC rows for BiLSTM's Phase 1 as a clear example of the
  first case (orange is offset from Actual by a near-constant gap for the
  whole phase; blue sits almost on top of Actual) — visually consistent
  with BiLSTM being this project's most reliably C_t-positive architecture
  (see `src/analysis/ct_significance_testing/CLAUDE.md`).

## Reproducing / regenerating

```bash
python scripts/phase_wise_predictions.py --dir <run_folder> --lead 10
```

Defaults to writing into this folder's own `figures/`/`data/` regardless
of where it's run from; pass `--outdir` to redirect.
