# C_t Effect Analysis — Run v1 (10-8-26)

This folder is a self-contained analysis package for one pipeline run: the
IAQ early-detection model's with-C_t vs. without-C_t comparison, run on
**10-Aug-2026, output folder `v1`**. It answers: *does the vision-derived
context vector (C_t — door state + motion) actually improve pollutant
forecasting, overall and phase-by-phase, for each of the 7 sequence
architectures?*

Source pipeline: `iaq_early_detection_gui_v3.py` (see that file's own
changelog for FIX 1–15; this run uses the corrected door-orientation logic
from FIX 15). Raw run output (checkpoints, per-model plots, full per-minute
prediction CSVs) lives at:
`data/runs/v1/` (repo root) for the small CSVs needed to reproduce this
analysis; the full raw run output (checkpoints, per-model plots) is not
tracked in this repo — this folder only holds the *derived* analysis (summary CSVs, figures,
the scripts that produced them), not the raw training artifacts.

## Headline result

| Model | Overall RMSE change without C_t | Verdict |
|---|---|---|
| BiLSTM | **+268.1%** | C_t required, consistent across all 4 phases |
| Seq2Seq | +74.2% | C_t helps |
| GRU | +51.3% | C_t helps |
| LSTM_uni | +34.4% | C_t helps |
| VanillaRNN | +22.8% | C_t helps |
| CNN_LSTM | −7.3% | C_t roughly neutral, volatile phase-to-phase |
| BiGRU | −7.4% | C_t roughly neutral, volatile phase-to-phase |

**5 of 7 architectures show C_t provides a real, positive benefit; 2 show a
small negative effect.** See `data/overall_ct_effect.csv` and
`data/phase_wise_ct_effect.csv` for the numbers, `figures/OVERALL_AND_PHASE_
EFFECT.png` for the chart. The phase-wise breakdown matters: several models
that are net-positive overall (GRU, LSTM_uni, VanillaRNN) still dip negative
in exactly one phase — expected architecture-specific variation, not a
contradiction of the overall result. CNN_LSTM and BiGRU are qualitatively
different: they swing large in *both* directions phase-to-phase (CNN_LSTM:
−77.3 to +27.7; BiGRU: −39.8 to +83.3), which reads as genuine instability
in how those two architectures integrate C_t, not a clean "doesn't need it."

### Significance-tested version — use this one for the paper

`figures/OVERALL_AND_PHASE_EFFECT_SIGNIF.png` (built by
`scripts/phase_significance_analysis.py`) is a corrected, publication-grade
version of panel (b). Two problems with the original panel (b) motivated it:

1. **The original `overall_RMSE` pools all 5 pollutants in raw units**,
   which means CO2 (RMSE in the hundreds) dominates almost completely and
   PM's contribution is nearly invisible — a masking problem at the
   pollutant level, structurally the same complaint the reviewer raised
   about pooling across time. The fix: express each pollutant's RMSE as %
   of *that pollutant's own whole-session observed range*, then average
   the 5 **percentage-point gaps** with equal weight. CO2 can no longer
   drown out PM, and a point-gap (not a ratio) avoids the ceiling-
   compression distortion documented earlier for R².
   - This also explains something worth reporting explicitly: **BiLSTM's
     headline "+268%" in panel (a) corresponds to only +4 to +8
     percentage points of normalized RMSE in panel (b).** Both numbers are
     correct — the relative % is inflated by a very small with-Ct
     baseline RMSE in raw units, the same small-denominator effect
     already flagged for R². The point-gap is the more honest magnitude
     to quote for "how much does C_t help," even though BiLSTM is still
     the most consistent performer.
2. **No significance test existed per phase** — only the pooled onset
   window had one. Panel (b) cells are now annotated with `n` (phase
   sample size) and a significance mark. **23 of 28 (model, phase) cells
   are significant**; the 5 that aren't (BiLSTM/P4, Seq2Seq/P3, GRU/P3,
   GRU/P4, LSTM_uni/P3) all have the smallest point-gaps (0.3–4.1 pts) —
   an internally consistent result, not noise.

**Statistical method used, and why:** per-pollutant Diebold-Mariano test
(paired, same test already used elsewhere in this project for the pooled
onset window) on normalized squared errors, run separately for all 140
(model × phase × pollutant) combinations, then Benjamini-Hochberg FDR
correction applied across all 140 at once (the correct granularity — you
correct at the level tests were actually run, not after pre-aggregating).
A cell is marked significant only if a **majority (≥3 of 5) of pollutants**
are individually FDR-significant. This was chosen over combining the 5
pollutants' p-values into one number (e.g. Fisher's method) because Fisher's
method assumes the 5 tests are independent — but PM1/PM2.5/PM10/CO2/VOC
share the same timestamps and the same underlying physical process, so
their errors are almost certainly correlated, which makes Fisher's combined
p-value anti-conservative. An earlier version of this exact analysis used
Fisher's method and returned 27/28 "significant" cells, including gaps as
small as 0.3 points — not credible, and the tell that the independence
assumption had failed. Fisher's p-value is still reported in
`data/phase_significance_cells_lead10.csv` as a supplementary (optimistic)
diagnostic, not as the basis for the chart's significance marks. A Wilcoxon
signed-rank p-value (distribution-free) is reported per pollutant in
`data/phase_significance_detail_lead10.csv` as a further robustness check,
useful because Phase 1 has only ~34 minutes — the regime where the
DM-test's asymptotic-normality assumption is weakest.

## Folder contents

```
v1/
├── README.md              <- you are here
├── CLAUDE.md               <- context for Claude in future sessions
├── scripts/
│   ├── overall_and_phase_summary.py   Computes the two tables + chart above,
│   │                                   directly from a run's own CSVs.
│   │                                   Usage: python overall_and_phase_summary.py
│   │                                          --dir <run_folder> --lead 10
│   ├── phase_significance_analysis.py  Significance-tested, normalized-range
│   │                                   version of panel (b) — see section
│   │                                   above. Needs predictions_lead{N}_v2.csv
│   │                                   and predictions_noct_lead{N}.csv.
│   │                                   Usage: python phase_significance_analysis.py
│   │                                          --dir <run_folder> --lead 10 --outdir <data_folder>
│   ├── event_proximity_weighted_rmse.py  Continuous tau-weighted RMSE gap
│   │                                   (generalizes the onset/baseline split
│   │                                   into a smooth decay-weighted metric).
│   │                                   Needs predictions_lead{N}_v2.csv,
│   │                                   predictions_noct_lead{N}.csv, and
│   │                                   trigger_events_lead{N}.csv (all
│   │                                   produced by the main pipeline).
│   └── cross_run_ct_reliability.py    Compares the SAME model's Ct effect
│                                       across multiple independent runs
│                                       (different seeds/configs) to tell
│                                       robust findings from lucky seeds.
├── figures/
│   ├── OVERALL_AND_PHASE_EFFECT.png   The headline chart (this run only)
│   ├── OVERALL_AND_PHASE_EFFECT_SIGNIF.png   Significance-tested, normalized-
│   │                                   range version of panel (b) — use this
│   │                                   one for the paper (see section above)
│   ├── PHASE_STRATIFIED_R2.png        Pipeline's own per-model R² by phase
│   ├── REGIME_STRATIFIED_R2.png       Pipeline's own onset-vs-baseline R²
│   ├── EVENT_DETECTION_LEADTIME.png   Alert precision/recall/lead-time
│   ├── EVENT_PROXIMITY_WEIGHTED_RMSE_lead10.png   Per-model tau-sweep curves
│   ├── EVENT_PROXIMITY_GAP_SUMMARY_lead10.png     Pooled tau-sweep curve
│   └── CROSS_RUN_CT_RELIABILITY.png   v9/v10/v16/v1 compared (see caveat below)
└── data/
    ├── overall_ct_effect.csv / phase_wise_ct_effect.csv   This run's tables
    ├── phase_stratified_metrics_lead10.csv                Pipeline output
    ├── regime_stratified_metrics_lead10.csv                Pipeline output
    ├── paired_significance_onset_lead10.csv                DM-test / t-test
    ├── event_detection_lead10.csv                          Alert P/R/lead-time
    ├── causality_lag_analysis.csv                          Cross-corr-at-lag
    ├── event_proximity_weighted_rmse_{raw,normalized}_lead10.csv
    ├── phase_significance_cells_lead10.csv                 One row per (model,phase):
    │                                                        gap_pts, n, majority-rule verdict
    ├── phase_significance_detail_lead10.csv                One row per (model,phase,pollutant):
    │                                                        DM-test + Wilcoxon p-values, FDR q-values
    └── cross_run_ct_reliability.csv                        v9/v10/v16/v1 (see caveat)
```

## Granger causality: does C_t actually precede the pollutants?

`figures/GRANGER_CAUSALITY_FORWARD.png` and the CSVs `data/granger_
causality_forward_reverse.csv` / `data/granger_causality_phase_
stratified.csv` (built by `scripts/granger_causality_analysis.py`) test a
different question than everything else in this folder: not "does C_t
improve forecast accuracy" but "do C_t's own past values statistically
precede the pollutants, independent of any trained model." This matters
because the pipeline's own `causality_lag_analysis.csv` has always shipped
with blank Granger columns (see `CLAUDE.md`) — this script computes it
independently from the raw 386-row per-minute session file, with no
dependency on the pipeline's environment.

**What Granger causality tests, precisely**: whether past values of X
reduce a linear model's forecast error for Y beyond what Y's own past
already provides. It is a predictive-precedence test, not proof of
physical causation — a significant result doesn't rule out both series
being driven by an unmeasured third factor. Used here as a second,
independent statistical method (VAR-based) to corroborate or challenge the
lagged relationship the rest of this project's evidence already implies.

**Full-session result**: 7 of 25 (C_t feature, pollutant) pairs are
FDR-significant (Benjamini-Hochberg, q<0.05, across all 25 pairs) —
notably `door_open_sum`→PM10, `M2_rho_open`→PM1 and →PM10 (forward-only,
no reverse significance — clean directional evidence), and every C_t
feature tested against VOC. **3 of those 7 are also significant in
reverse** (VOC "Granger-causing" `door_open_sum`, `M1_rho_open`, and
`M3_rho_open` just as strongly as the forward direction) — flagged as a
placebo warning, not swept under the rug: VOC's relationship with door/
motion activity likely reflects occupancy or cutting activity driving both
simultaneously (e.g. someone opening a door *because* VOC/fumes built up,
as well as door-opening changing ventilation), rather than clean one-way
precedence. Read the PM10/PM1 results as the stronger causal evidence and
the VOC results with that caveat attached.

**Phase-stratified result**: far fewer pairs stay significant once split
into phases (n=57-88 for Phases 1-3), which is expected — full-session
pooling has roughly 4-6x the statistical power of any single phase, so
this drop in significance count reflects reduced power, not proof the
relationship is phase-specific. Note also that this script's "Phase 4" is
larger (n=177) than in the prediction-based analyses (n=99) because it
uses the raw sensor file's true extent (12:58-19:30) rather than being
capped by how far the model's T+10 forecasts reach (18:10) — don't
directly compare phase sample sizes across the two folders.

## Prediction accuracy (R²) vs. forecast lead time, T+1 to T+20 min

`figures/LEAD_TIME_ACCURACY.png` and `data/lead_time_accuracy.csv` (built by
`scripts/lead_time_accuracy.py`) answer: as the forecast horizon lengthens,
how much accuracy is actually lost? Source data already existed in the v1
run's own `ablation_results_lead10.csv` (the `type == "src/analysis/forecast_horizon_and_early_warning"` rows) —
no retraining involved, just cleanly repackaged and charted.

**Read the caveat before citing these numbers as a model result.** This
sweep is *not* produced by any of the 7 production architectures discussed
everywhere else in this project. The pipeline's ablation study trains a
separate, lightweight single-layer GRU proxy purely for ablation speed
(`_AblGRU`, hidden size capped at 64, 100 epochs —
`iaq_early_detection_gui_v3.py:2338`). The raw chart's title says
"Model=BiGRU," which is a labeling artifact from a name-preference list
used only to pick a display string, not the architecture actually trained
for this sweep — don't read it as a real BiGRU result. It also only used
the full 68-feature (with-C_t) set; no without-C_t comparison exists at
lead times other than 10 minutes, since predictions were only exported
with and without C_t at T+10.

**What it shows**: overall R² barely moves (0.9972 at T+1 → 0.9841 at
T+20, a drop of just 0.013), while overall RMSE more than doubles (38.06
→ 84.89, ×2.23) over the same range — the same ceiling-compression effect
already documented for the C_t comparison: R² alone understates how much
harder the forecasting problem gets at longer horizons. VOC is the
standout case — its R² holds up with the others through T+10, then falls
off a cliff at T+15 (0.88 → 0.60) while its indexed RMSE nearly triples,
well past every other pollutant. CO2 degrades the most gently of the five
(R² 0.968 → 0.817), consistent with it being the slowest-moving pollutant
in this dataset.

## Reproducing this analysis on a new run

```bash
python scripts/overall_and_phase_summary.py --dir <new_run_folder> --lead 10
python scripts/phase_significance_analysis.py --dir <new_run_folder> --lead 10 --outdir <new_run_folder>
python scripts/event_proximity_weighted_rmse.py --dir <new_run_folder> --lead 10
```

## Important caveat: `cross_run_ct_reliability.csv` mixes eras

That file (and `CROSS_RUN_CT_RELIABILITY.png`) includes v9 and v10, which
predate the door-orientation fix (FIX 14/15 in the pipeline) — they used a
different, likely-backwards convention for `door_open_sum`. **Per the
explicit direction for this v1 analysis: treat v9/v10 as historical only.
The two most relevant runs for current conclusions are v16 and v1
(both post-fix).** This file is kept for the record, not as the basis for
the headline result above, which is v1-only.
