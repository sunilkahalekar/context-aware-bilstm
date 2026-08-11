# CLAUDE.md — internals reference for src/analysis/phase_wise_prediction_traces/

Scope: `scripts/phase_wise_predictions.py`. For what this folder is and how
to read its output, see [README.md](README.md) first. This file is for
modifying the code.

## Function reference

- **`assign_phase(ts)`** — vectorized phase assignment from an array of
  timestamps, using the *same* fixed clock-time boundaries as the main
  GUI's `_phase_boundaries()` (14:00 / 15:30 / 16:30). Kept as a standalone
  numpy re-implementation rather than importing the GUI (which is a
  different repo/file entirely) — if the GUI's phase boundaries ever
  change, this must be updated by hand to match, or the phase labels here
  will silently disagree with the GUI's own plots.
- **`load_merged(run_dir, lead)`** — reads both prediction exports, keeps
  only the columns needed, inner-joins on
  `(trigger_timestamp, future_timestamp, model, target)` so every row is a
  genuinely paired with/without-C_t comparison at the same moment, filters
  to the 7 sequence models this project's analysis covers (drops
  LinearRegression/Ridge/RandomForest/SVR if present in the source CSVs —
  see `MODEL_ORDER`), and tags each row with its phase.
- **`r2_score(actual, pred)`** — plain, un-normalized R² in the target's
  own raw units. Deliberately NOT the same calculation as
  `src/analysis/ct_significance_testing/scripts/phase_significance_analysis.py`'s normalized version
  — see the README's "How the calculation works" section for why both
  exist and disagree on purpose.
- **`compute_metrics(m)`** — one row per (model, phase, target): RMSE/R²
  for both variants, plus the raw gap and % change. This is the numeric
  table the figures are built from; nothing in the figure-drawing function
  recomputes anything.
- **`make_model_figure(m, model, outdir)`** — one PNG per model, a
  5 (pollutant) × 4 (phase) grid via `plt.subplots`. Each cell is an
  independent time-series plot over just that phase's rows (`sub =
  row_sub[row_sub.phase == phase]`) — phases are NOT drawn on a shared,
  continuous x-axis, so don't read continuity across a phase boundary in
  the figure; each cell's x-axis only spans that phase's own window.
  Empty cells (`len(sub) == 0`) are turned off (`ax.axis("off")`) rather
  than left as a blank/misleading empty plot.

## Gotchas for future changes

- **The inner join in `load_merged` silently drops any (trigger, future,
  model, target) combination not present in BOTH prediction files.** If a
  future run's with-Ct and without-Ct exports don't cover exactly the same
  trigger timestamps (e.g. one crashed partway through), rows will vanish
  from the merge with no warning. Worth adding a row-count sanity check
  (`len(with_df) == len(without_df) == len(m)` before filtering to
  `MODEL_ORDER`) if this is ever run on a new dataset.
- **`MODEL_ORDER` is a hardcoded allowlist**, not derived from the data. If
  the GUI adds a new sequence architecture, it won't appear here until
  this list is updated by hand.
- **Figure y-axis is auto-scaled per subplot** (no explicit `ylim` call) —
  unlike `src/analysis/forecast_horizon_and_early_warning/scripts/bilstm_event_warning_chart.py`, which had a
  real bug from a *hardcoded* y-axis that clipped bars off-canvas. This
  script doesn't have that failure mode, but also means each phase's panel
  for a given pollutant can have a different y-scale from its neighbors —
  intentional (each phase's own dynamic range is what matters for reading
  the with/without-Ct gap), but worth calling out if someone expects a
  shared scale across a row.

## Standing instructions inherited from src/analysis/ct_significance_testing/CLAUDE.md

Same project, same rules — repeated here because this folder is often
opened independently of `src/analysis/ct_significance_testing/`:

- **Never report "C_t always helps" as a blanket claim.** This is exactly
  the folder where the counterexamples are most visible — BiGRU's PM1/
  PM2.5/PM10 traces in Phase 1/3 show the with-C_t (blue) line detaching
  from Actual while without-C_t (orange) tracks it fine. Report that
  honestly when it's what the chart shows.
- **This folder's RMSE/R² numbers are in raw units and will not match
  `src/analysis/ct_significance_testing`'s normalized, significance-tested numbers for the same
  model/phase/pollutant.** That's expected (see README) — don't treat a
  mismatch between the two folders as a bug without first checking whether
  it's just the normalization difference.
