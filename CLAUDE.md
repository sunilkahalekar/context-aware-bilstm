# CLAUDE.md — master internals/history reference

Read this before making cross-cutting changes (renaming a shared column,
changing a phase boundary, retraining a model whose output other folders
depend on). For any single stage's internals, go straight to that stage's
own `CLAUDE.md` — this file only covers things that span more than one.

## How this repo came together

This is a consolidation of what were previously two separate repositories
(`IAQ_video_Processing`, `IAQ_Early_Detection_GUI_code`) plus local-only
analysis work, reorganized into one coherent structure under `src/`. The
reorganization renamed **folders and script filenames only** — data/output
CSV names, PNG names, and internal column schemas were deliberately left
untouched, because dozens of scripts across every stage hardcode those
filenames to read each other's output, and there was no way to test the
full pipeline end-to-end during the consolidation (this environment has no
PyTorch — see `src/modeling/CLAUDE.md`). If you rename a data file, you
must manually find and update every script that reads it — grep first.

## Standing instruction — do not violate this, it governs every document in this repo

The user explicitly asked (after initially requesting the opposite) **not**
to force a universal "C_t always helps every model" claim. The correct
framing: report the real per-model, per-phase magnitude honestly, including
negative/near-zero cases, and explain them mechanistically (architecture
capacity, training convergence, phase-specific dynamics) rather than
smoothing them into a blanket claim. Full context in
`src/analysis/ct_significance_testing/CLAUDE.md`.

## The three different "how far ahead can we forecast" questions

Don't conflate these — a user or future session asking this needs to be
pointed at the *right one*, or answered with all three explicitly labeled:

1. **Statistical**: how far ahead does the raw pollutant signal remain
   self-predictable at all, ignoring any model? →
   `src/analysis/forecast_horizon_and_early_warning/` (persistence
   baseline + bootstrap CI).
2. **Model-specific**: does a real trained architecture (BiLSTM) hold up
   across lead times, and does C_t change that? → same folder,
   `train_bilstm_lead_sweep*.py`. **Unresolved** — full debugging history
   in that folder's `CLAUDE.md`.
3. **Operational**: how many minutes of real advance warning does C_t give
   a facility manager before an actual pollution spike? → the modeling
   stage's own `event_detection_lead10.csv` (`data/runs/v1/`), visualized
   in `src/analysis/forecast_horizon_and_early_warning/figures/BiLSTM_EVENT_WARNING*.png`.

Quoting the operational 9-minute CO2 warning number as if it validates the
statistical persistence-baseline finding would be a real error, not just
imprecise writing — they're different analyses answering different
questions.

## Cross-cutting gotchas

- **Phase boundaries (14:00 / 15:30 / 16:30) are defined independently in
  at least 3 places**: `src/modeling/train_context_aware_bilstm_gui.py`'s
  `_phase_boundaries()`, and standalone numpy re-implementations in
  `src/analysis/phase_wise_prediction_traces/scripts/phase_wise_predictions.py`
  and `src/analysis/ct_significance_testing/scripts/*.py`. They are NOT
  imported from a shared source — this is deliberate (the analysis
  scripts shouldn't need to import a tkinter+torch GUI application as a
  library) but means a phase-boundary change must be applied by hand in
  every one of these files, or downstream phase labels will silently
  disagree with the GUI's own plots.
- **The door-orientation convention (FIX 15) is tentative project-wide.**
  `door_open_sum` is reversed once at the source in the modeling stage;
  every analysis script that touches door state inherits this correction
  automatically *if* it reads the already-corrected column — but any
  script that recomputes `door_open_sum` from raw `M{n}_phi_open` values
  independently (several analysis scripts do, to avoid depending on the
  modeling stage's internals) must apply the same reversal by hand. Search
  for "FIX 15" across `src/` before trusting a door-related result that
  doesn't cite it.
- **`data/runs/v1/` is intentionally incomplete.** It has the small CSVs
  needed to reproduce this repo's analysis (predictions, metrics,
  significance tests), not the full raw run output (model checkpoints,
  per-model PNG plots, training logs) — those weren't checked in (too
  large, and not needed to reproduce anything in `src/analysis/`). If you
  need to retrain or inspect a checkpoint, that lives only wherever the
  original training run's `--outdir` pointed.
- **`data/raw/sensor_data_merged_iaq_m2.csv` and the CSVs in
  `data/runs/v1/` are a specific, frozen snapshot** (the "v1" run,
  10-Aug-2026). Regenerating any upstream stage (new video, re-run YOLO,
  re-merge) produces a *different* dataset — don't mix files from a new
  run with `data/runs/v1/`'s files under the assumption they're
  interchangeable; every analysis script's default paths assume all its
  inputs come from the same run.

## Known open issues (see linked CLAUDE.md for full debugging history)

- **BiLSTM lead-time sweep (T+1–T+20) is unresolved.** Three attempts,
  all producing deeply negative R² beyond T+1. Full log:
  `src/analysis/forecast_horizon_and_early_warning/CLAUDE.md`.
- **18 of 68 "with-C_t" features are constant zero** — root cause is in
  `src/vision/extract_context_vector_from_video.py`'s `compute_Dt()`,
  which never implements `emission_weight`/`effective_tau`/
  `consecutive_full_open`/op-state. Full explanation:
  `src/modeling/README.md` §21.
- **Granger causality's original pipeline output is still blank** in the
  v1 run (`data/runs/v1/causality_lag_analysis.csv`'s `granger_best_lag`/
  `granger_min_pvalue` columns) — FIX 11 made the pipeline log *why*
  instead of failing silently, but that log has never actually been
  checked. `src/analysis/ct_significance_testing/scripts/granger_causality_analysis.py`
  computes it independently instead, with no dependency on the pipeline's
  runtime.
- **Fisher's method should not be used to combine per-pollutant
  significance tests on this dataset** — PM1/PM2.5/PM10/CO2/VOC share
  timestamps and the same underlying process, violating the independence
  assumption Fisher's method requires. Use majority-rule (≥3 of 5
  pollutants individually FDR-significant) instead. Full explanation:
  `src/analysis/ct_significance_testing/CLAUDE.md`.

## Where to find the full FIX 1–15 changelog

`src/modeling/train_context_aware_bilstm_gui.py`'s own module docstring, or
the human-readable version with Root-cause/Solution writeups in
`src/modeling/README.md` §15 (FIX 1–8) and §20 (FIX 9–15). FIX 9, 10, and
13 are inseparable from the with/without-C_t validation framework and are
explained in full in that same README's §19.
