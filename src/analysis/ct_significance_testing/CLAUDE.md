# CLAUDE.md — context for future sessions on this project

Read this before doing new analysis on the IAQ / C_t project. It exists so a
fresh session doesn't have to re-derive the history in this folder's
`README.md` companion, or re-discover bugs that have already been found
and fixed.

## What this project is

An indoor-air-quality early-warning system for a laser-cutting fabrication
lab (386-minute recorded session, 4 operational phases). Forecasts 5
pollutants (PM1, PM2.5, PM10, CO2, VOC) at T+10 min lead time. The central
research question: does the vision-derived context vector **C_t**
(door-state descriptors + human-motion descriptors, extracted via YOLO +
optical flow) actually improve forecasting accuracy, or does pooled R²/RMSE
just make it look that way ("statistical masking" — quiescent minutes
dominate the aggregate and are trivially predictable either way)?

That masking concern is why this project's validation is never a single
pooled R²/RMSE comparison. It's always at least: **onset vs. baseline
regime**, **phase-wise** (4 phases), **paired significance** (Diebold-Mariano
+ paired t-test on onset-window squared errors), and increasingly,
**event-proximity-weighted** (continuous τ-decay weighting instead of a
binary onset cutoff).

## Where things live

| What | Where |
|---|---|
| Main pipeline (current) | `src/modeling/train_context_aware_bilstm_gui.py` |
| Standalone analysis scripts | `scripts/` in this folder and its sibling analysis folders (`../phase_wise_prediction_traces/scripts/`, `../forecast_horizon_and_early_warning/scripts/`) |
| Per-run raw pipeline output (checkpoints, per-model plots) | not tracked in this repo — lives wherever the GUI's `--outdir` was pointed for that run; only the small derived CSVs needed to reproduce this folder's analysis are checked into `data/runs/v1/` at the repo root |
| Packaged analysis (this folder's siblings) | `src/analysis/{ct_significance_testing, phase_wise_prediction_traces, forecast_horizon_and_early_warning}/` — derived summary CSVs + figures + the scripts that made them |

Known run folders referenced so far: `9-8-26/v9`, `9-8-26/v10`, `9-8-26/v16`,
`10-8-26/v1`. **v9 and v10 predate the door-orientation fix (see below) —
do not compare them directly to v16/v1 without noting the convention
changed.**

## Bugs found and fixed — read this before trusting old numbers

1. **Dt vector was incomplete** (pre-FIX-12): only `phi_open` and the
   derived `emission_weight` were fed to the model; `rho_open`, `eps_max`,
   `effective_tau`, `f_trans` existed in the dataframe (used only by the
   causality diagnostic) but never reached `feat_cols`. Fixed in FIX 12 —
   all 5 Dt descriptors are now real input features.
2. **`door_open_sum` was oriented backwards.** It's built by summing
   `M{m}_phi_open` across machines. The manuscript's own description of
   φ_open ("early-window openings" = φ_open < 0.5) implies φ_open reads
   **low while a door is actively/freshly opening and high once settled
   back to idle** — the opposite of what a column named "door open sum"
   suggests. Confirmed empirically: in the plotted panel, door_open_sum
   was anti-correlated with the People-count panel (high when the room was
   empty, low when busy) — backwards from what you'd expect.
   - **FIX 14** patched this for chart display only (reflected around
     session min/max), leaving the training feature untouched.
   - **FIX 15** (in `iaq_early_detection_gui_v3.py`) fixes it at the
     source — reverses `df["door_open_sum"]` once, immediately after it's
     built, so every downstream consumer (the `door_open_sum`/`door_diff`/
     `door_exposure` features, the onset-trigger rising-edge detector in
     `_build_regime_labels`/`_get_trigger_timestamps`, and the causality
     cross-correlation) inherits the corrected orientation automatically.
     FIX 14's display-only patch was reverted in v3.py since it would
     otherwise double-flip and silently restore the original bug.
   - **This is still tentative** — it's inferred from the manuscript's own
     text and an empirical anti-correlation check, not confirmed against a
     ground-truth per-machine door sensor. Re-verify if one becomes
     available.
3. **Granger causality silently returned nothing** for every pair in
   earlier runs — root cause was `statsmodels` not installed in the run
   environment, and the code caught that `ImportError` and returned
   `(None, None)` with no log message. FIX 11 makes the pipeline report
   statsmodels' install status up front and log *why* a pair failed
   instead of leaving a silently blank cell. If you see blank Granger
   p-values again, check the log for this message before assuming the
   test ran and found nothing. **Still unresolved as of v1**:
   `causality_lag_analysis.csv`'s `granger_best_lag`/`granger_min_pvalue`
   columns are still blank in the v1 run itself — whatever FIX 11's log
   would say, it hasn't been checked yet. Rather than keep waiting on the
   GUI pipeline's environment, `src/analysis/ct_significance_testing/scripts/granger_causality_
   analysis.py` computes Granger causality independently, straight from
   the raw per-minute session file
   (`data_input/sensor_data_merged_iaq_m2.csv`, 386 rows) — no dependency
   on the pipeline's runtime at all. See its docstring for the four
   scenarios it covers (forward, reverse/placebo, phase-stratified,
   stationarity-checked) and `README.md`'s Granger section for the
   results and how to read them.
4. **Without-Ct per-minute predictions and real trigger timestamps weren't
   exported** before FIX 13 — only aggregated with-Ct predictions were
   saved. `predictions_noct_lead{N}.csv` and `trigger_events_lead{N}.csv`
   (added in FIX 13) are required inputs for
   `event_proximity_weighted_rmse.py`; older run folders won't have them.

## Behavioral facts worth remembering (not bugs — just how this dataset behaves)

- **386 rows is small enough that architecture ranking and even the *sign*
  of Ct's effect for a given model can flip between runs with different
  random seeds.** This has now been observed directly: BiGRU has shown
  Ct helping (early runs), Ct hurting only PM (v16), and Ct hurting all 5
  pollutants (v1) — three different pictures from the same architecture.
  **Never trust a single run's result for a specific architecture without
  checking whether that architecture's *absolute* accuracy (its own R²)
  was reasonable in that run** — if the with-Ct model itself trained
  poorly (e.g. BiGRU in v1: PM1 R²=0.222, PM2.5 R²=0.133), the with/without
  comparison for that run is confounded by a bad training run, not
  evidence about Ct.
- **BiLSTM and Seq2Seq have been the most consistently Ct-positive
  architectures across every run tracked so far.** BiLSTM in particular
  has never shown a negative phase in any run.
- **CNN_LSTM has been consistently near-null-to-mixed on Ct across every
  run tracked so far** (phase-mean roughly −9% to +5%, but volatile
  phase-to-phase within a run — e.g. v1: −77.3 in Phase 1, +27.7 in
  Phase 3). This is the best-supported candidate for "this architecture's
  own structure doesn't integrate C_t well," as opposed to noise.
- **VanillaRNN and BiGRU are the two architectures most prone to poor
  training convergence** (single-layer, minimal regularization by design
  for VanillaRNN in this pipeline). Catastrophic negative R² on PM has
  been seen for both in different runs. Treat their with/without-Ct
  comparisons with extra scrutiny.
- **CO2 and VOC (the "slow gas" pollutants) are where isolated
  significant-but-opposite-direction exceptions have shown up** (e.g.
  VanillaRNN/CO2, CNN_LSTM/VOC in v16) even in models that are otherwise
  clean Ct-positive. Not yet explained; worth watching for recurrence.
- **Relative-% and normalized-point-gap can tell visually different
  stories about the same result, and both are "correct."** BiLSTM's v1
  overall effect is +268% in raw relative terms but only +4 to +8
  percentage points on the normalized-range scale (see
  `phase_significance_analysis.py` / `OVERALL_AND_PHASE_EFFECT_SIGNIF.png`).
  The relative number is inflated by a small with-Ct baseline RMSE — the
  same ceiling-compression effect already documented for R². Report the
  point-gap when the question is "how big is the effect," and the relative
  % only when the question is "how much does removing Ct multiply the
  error by." Don't drop either without saying which one you're using.
- **Per-pollutant significance tests should not be combined via Fisher's
  method for this dataset.** PM1/PM2.5/PM10/CO2/VOC share timestamps and
  the same underlying process, so their errors are correlated, and
  Fisher's method assumes independence — combining them that way returned
  27/28 "significant" cells including a 0.3-point gap, which was the tell
  that it was anti-conservative. Use a majority-rule (≥3 of 5 pollutants
  individually FDR-significant) instead; see `phase_significance_analysis.py`.

## A standing instruction from the user — do not violate this

The user explicitly asked (after initially requesting the opposite) **not**
to force a universal "Ct always helps every model" claim. The correct,
requested framing is: report the real per-model, per-phase magnitude
honestly, including negative/near-zero cases, and explain them mechanistically
(architecture capacity, training convergence, phase-specific dynamics)
rather than smoothing them into a blanket claim. This is a stronger,
more defensible position for a paper, and it's what every analysis in
this project should default to.

## Lead-time (T+1 to T+20 min) accuracy sweep — whose model is it, really?

`ablation_results_lead10.csv` (and `scripts/lead_time_accuracy.py`, which
repackages it) contains a lead-time sweep that looks like it comes from
the project's real architectures but doesn't. The pipeline's ablation
study trains its own lightweight single-layer GRU proxy (`_AblGRU`,
`iaq_early_detection_gui_v3.py:2338`) for speed, separate from the 7
production models everywhere else in this project. The raw ablation
chart's title ("Model=BiGRU") is a display-label artifact from a
name-preference list, not evidence it's a real BiGRU run. If a future
session is asked for lead-time accuracy "for BiLSTM" or "for BiGRU"
specifically, this file is NOT that — flag the distinction rather than
silently reusing these numbers as if they were architecture-specific.
Also note: this sweep only covers the with-C_t (68-feature) configuration
— no without-C_t comparison exists at any lead time except T+10, where
`predictions_noct_lead10.csv` was exported.

## BiLSTM-specific lead-time sweep — blocked here, script ready elsewhere

`scripts/train_bilstm_lead_sweep.py` exists but has never been run — this
analysis environment has no PyTorch and no network access to install it
(pip install fails on SSL cert verification). The script is a verbatim
port of the real BiLSTM class, feature engineering, and default
hyperparameters from `iaq_early_detection_gui_v3.py`, parameterized over
lead ∈ {1,3,5,10,15,20}. Run it in whatever environment already produced
`BiLSTM_ck.pt` for v1 (that environment has torch). Do not present the
existing `LEAD_TIME_ACCURACY.png` as a BiLSTM result — it's the
ablation-proxy GRU (see the section above).

**Incidental finding while extracting this code**: `M{1,2,3}_op_state`,
`M{1,2,3}_emission_weight`, and `M{1,2,3}_consecutive_full_open` don't
exist in `sensor_data_merged_iaq_m2.csv`. The pipeline silently defaults
all of them to 0.0 (`v3.py` line ~1024) rather than erroring — meaning
**18 of the 68 "with-C_t" feature columns in every run so far are constant
zero**, contributing nothing. Doesn't invalidate the with/without-C_t
comparisons (those 18 are zero in both), but the *true* informative
C_t feature count is smaller than 68 suggests. Worth raising with whoever
owns the upstream vision pipeline — those signals may exist and simply
aren't being exported into this CSV.

## Useful next steps discussed but not yet done

- ~~Per-phase significance testing~~ **Done** — `phase_significance_analysis.py`
  now runs a Diebold-Mariano test per (model, phase, pollutant), FDR-corrects
  across all 140 tests, and marks each (model, phase) heatmap cell
  significant only if a majority of pollutants clear it. See the README
  section "Significance-tested version" for the full method and rationale.
- **Multi-seed replication**: run the same architecture N times with
  different seeds and use `cross_run_ct_reliability.py` (or extend it) to
  get a proper confidence interval on the Ct effect per architecture,
  rather than treating any single run as final. This is the single most
  requested-but-not-yet-executed piece of rigor in this project.
- **Lag-shifted event-proximity weighting**: `event_proximity_weighted_
  rmse.py` currently centers its τ-decay weight on the raw trigger
  timestamp. The known ~10–20 minute physical transport lag between a
  door/motion trigger and the pollutant's actual response means the
  "hardest to predict" moment is *after* the trigger, not at it. Adding a
  `--lag-shift` argument to re-center the weighting was proposed but not
  implemented.
- **Verify φ_open's true convention** against a ground-truth per-machine
  door sensor if one ever becomes available (see bug #2 above).
- Install `statsmodels` in whatever environment actually runs the
  pipeline if Granger causality p-values are still coming back blank.
