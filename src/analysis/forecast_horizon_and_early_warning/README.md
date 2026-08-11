# Lead-Time Analysis — How Far Ahead Can This System Actually Forecast?

Self-contained package for one question this project kept circling back to:
**how many minutes ahead can the IAQ system usefully forecast, and how do
we know that number is real rather than assumed?** It pulls together
everything built around that question across several sessions: the
pipeline's own event-detection early-warning numbers, a from-scratch
persistence-baseline sweep, a bootstrap-confidence-band analysis, and
(currently blocked/in-progress) a genuine BiLSTM-specific lead-time sweep.

Source run: v1 (10-Aug-2026). Small derived CSVs from that run's raw output
are checked into `data/runs/v1/` at the repo root. Raw per-minute session
file (needed by every script in `scripts/` except `lead_time_accuracy.py`):
`data/raw/sensor_data_merged_iaq_m2.csv` (repo root, 386 rows).

## The headline findings, in order

1. **The room's own pollutant readings "forget" themselves after about
   8 minutes.** A naive "predict no change" baseline, swept across every
   lead time from 1 to 20 minutes, crosses from useful to actively
   misleading (R² crosses zero) at lead ≈ 8 min
   (`figures/PERSISTENCE_BASELINE.png`).
2. **That 8-minute estimate is less certain than it looks.** Bootstrap
   resampling (2,000 resamples/lead, accounting for this dataset's small
   test windows — as few as 38 origins at T+20) shows the 95% confidence
   band doesn't fully drop below zero until lead ≈ 11 min
   (`figures/EFFECTIVE_HORIZON_PERSISTENCE.png`). T+10, the lead time used
   everywhere else in this project, sits deliberately inside that 8–11
   minute zone of genuine uncertainty — late enough that naive
   extrapolation is expected to have failed, which is what makes a
   model's success there meaningful evidence of real anticipatory
   forecasting rather than borrowed autocorrelation.
3. **The operationally meaningful version of this question — "how much
   warning does vision context (C_t) actually give a facility manager
   before a real pollution spike" — is answered by the pipeline's own
   event-detection analysis, not by anything built in this folder.** For
   BiLSTM specifically: **9.0–9.5 minutes of advance warning for CO2 and
   PM2.5 with vision context, and *zero* advance warning for either
   pollutant without it** (0 of the real spikes were caught early at all
   without C_t). VOC is the exception — BiLSTM's vision-aware version
   gives no advance warning there, and the non-vision version actually
   caught VOC spikes 8 minutes early. See `figures/EVENT_DETECTION_
   LEADTIME.png` and `data/event_detection_lead10.csv`, and the section
   below for exactly how that file is produced.
4. **A genuine BiLSTM-specific version of finding #1 (does BiLSTM's own
   accuracy hold up across lead times) is not yet working.** Every
   training attempt so far has produced deeply negative R² beyond T+1 —
   see `data/bilstm_lead_time_sweep.csv` / `_v2.csv` and the CLAUDE.md
   troubleshooting log. This is the single most important unresolved
   thread in this folder.

## Folder contents

```
src/analysis/forecast_horizon_and_early_warning/
├── README.md                  <- you are here
├── CLAUDE.md                  <- context + full debugging history for future sessions
├── scripts/
│   ├── persistence_baseline_sweep.py   Naive "predict no change" baseline,
│   │                                    T+1..T+20, no model/torch needed.
│   │                                    Usage: python persistence_baseline_sweep.py
│   │                                           --raw <sensor_data_merged_iaq_m2.csv> --outdir <this folder>
│   ├── lead_time_effective_horizon.py  Bootstrap-CI version of the above, and
│   │                                    (once a model prediction export exists)
│   │                                    the real model-vs-persistence gap plot.
│   │                                    Usage: python lead_time_effective_horizon.py
│   │                                           --raw <raw_csv> [--model-predictions <export.csv>] --outdir <this folder>
│   ├── train_bilstm_lead_sweep.py      Real BiLSTM architecture (ported verbatim
│   │                                    from iaq_early_detection_gui_v3.py),
│   │                                    hidden=160. BROKEN -- see CLAUDE.md.
│   │                                    Requires PyTorch (this analysis env has
│   │                                    none -- run in the env that produced
│   │                                    BiLSTM_ck.pt for v1).
│   ├── train_bilstm_lead_sweep_v2.py   Same, hidden=64, raised dropout/weight-
│   │                                    decay, --export-predictions flag added.
│   │                                    ALSO STILL BROKEN beyond T+1 -- see
│   │                                    CLAUDE.md before trusting any output.
│   ├── lead_time_accuracy.py           Charts ANY lead-time sweep CSV (schema-
│   │                                    detects ablation_results_lead10.csv vs.
│   │                                    a pre-filtered bilstm_*.csv). Auto-labels
│   │                                    the chart from the filename -- pass
│   │                                    --model-label explicitly to be safe.
│   ├── regenerate_event_detection.py   Re-scores event-detection lead time at
│   │                                    custom alert thresholds, reusing the
│   │                                    existing valid T+10 predictions --
│   │                                    no retraining. See the calculation
│   │                                    section below for the full procedure.
│   └── bilstm_event_warning_chart.py   Renders the "how much warning" bar
│                                        chart from any event_detection_lead10
│                                        *.csv. Threshold labels and y-axis
│                                        scale are read from the data, not
│                                        hardcoded (see calculation section).
├── figures/
│   ├── LEAD_TIME_ACCURACY.png          Ablation-proxy GRU sweep (NOT a real
│   │                                    architecture -- see CLAUDE.md).
│   ├── LEAD_TIME_ACCURACY_BiLSTM.png   First BiLSTM attempt (hidden=160,
│   │                                    dropout=0.1) -- broken, kept for record.
│   ├── PERSISTENCE_BASELINE.png        Finding #1 above.
│   ├── EFFECTIVE_HORIZON_PERSISTENCE.png   Finding #2 above.
│   ├── EVENT_DETECTION_LEADTIME.png    Finding #3 above (pipeline's own output,
│   │                                    all 7 models, original thresholds).
│   └── BiLSTM_EVENT_WARNING*.png       BiLSTM-only re-renders at custom
│                                        thresholds (e.g. VOC>100 ppb) -- see
│                                        the calculation section below.
└── data/
    ├── ablation_results_lead10.csv     Raw source for the ablation-proxy GRU sweep.
    ├── bilstm_lead_time_sweep.csv      BiLSTM attempt 1 (hidden=160) -- broken.
    ├── bilstm_lead_time_sweep_v2.csv   BiLSTM attempt 2 (hidden=64) -- also broken.
    ├── lead_time_accuracy.csv          Ablation-proxy GRU, repackaged.
    ├── lead_time_accuracy_BiLSTM.csv   Attempt-1 BiLSTM, repackaged.
    ├── persistence_baseline_sweep.csv  Finding #1's numbers.
    ├── effective_horizon_persistence_ci.csv   Finding #2's numbers.
    ├── event_detection_lead10.csv      Finding #3's numbers (pipeline's own output).
    └── event_detection_lead10_voc100.csv   VOC re-scored at >100 ppb; CO2/PM2.5
                                             kept at their originally-published
                                             values (see calculation section).
```

## How `event_detection_lead10.csv` is generated (finding #3)

This file is **not** produced by anything in `scripts/` — it comes directly
from the main pipeline, `iaq_early_detection_gui_v3.py`, during a normal
run with the "Event detection" validation option enabled. Exact mechanism:

1. **`_detect_events(actual_1d, ts_actual, thr, min_gap_min=5)`**
   (`iaq_early_detection_gui_v3.py:2924`) scans the real, observed
   pollutant series for every *rising edge* across a threshold (e.g. CO2
   crossing above 2000 ppm) — the moment a real event "starts" — merging
   any two crossings less than 5 minutes apart into one event, so a
   single spike isn't double-counted.
2. **`_event_scoring(actual_1d, ts_actual, pred_1d, ts_pred, thr,
   horizon_min)`** (`:2936`) does the same rising-edge detection on the
   *model's own predicted* series (`pred_1d` = the model's T+LEAD-ahead
   forecast, timestamped at the moment the forecast was made — i.e. an
   "alert" fires now, based on what the model thinks will happen LEAD
   minutes from now) — merging predicted alerts less than 5 minutes apart
   the same way. It then greedily matches each real event to the
   *earliest* predicted alert that (a) came before it and (b) came no
   more than `horizon_min` minutes before it (GUI default: 30 minutes,
   `event_match_horizon_var`, `:691`) — each alert can only be matched to
   one event. `mean_lead_min` is the average, over matched (true-positive)
   pairs only, of how many minutes before the real event the alert fired.
   Precision/recall follow directly from TP/FP/FN counts.
3. This runs once per (model, pollutant, with/without-C_t variant) inside
   the main validation loop (`:3293-3325`), for every pollutant with a
   non-blank alert threshold configured in the GUI (v1's config: CO2>2000
   ppm, PM2.5>35 µg/m³, VOC>200 ppb — PM1/PM10 were left blank and
   skipped). Results accumulate into `event_rows` and are written to
   `event_detection_lead{LEAD}.csv` (`:3429`).

**Read `mean_lead_min` as "if this alert fires, how many minutes of real
warning does it give" — but note it's an average over very few events per
pollutant (2–4 real threshold crossings in the whole v1 test window).
These are directionally real findings, not statistically precise ones.**

## How the BiLSTM early-warning chart (custom thresholds) was calculated

`figures/BiLSTM_EVENT_WARNING*.png` (e.g. `_co2-2000_voc-100_ymax20.png`) is
a derived, re-scoreable version of finding #3 above — same underlying
question ("how many minutes of warning does vision context give before a
real spike"), but computed independently so the alert thresholds can be
changed (e.g. VOC from 200 to 100 ppb) without retraining any model. Full
procedure, in order:

1. **Start from the model's existing, valid T+10 predictions — not a new
   training run.** `predictions_lead10_v2.csv` (with C_t) and
   `predictions_noct_lead10.csv` (without C_t) are the *original* v1
   production output, already known-good (BiLSTM R²=0.9979 at T+10). The
   BiLSTM *lead-time sweep* (T+1/3/5/15/20) is the part that's broken (see
   above) — this chart only ever uses the working T+10 predictions, so
   that failure doesn't touch it.
2. **Detect real events**: for each pollutant, scan the *actual* observed
   values for the moment they first cross the alert threshold (a rising
   edge), merging anything within 5 minutes into a single event — the
   same `_detect_events` logic the pipeline itself uses, reimplemented in
   `regenerate_event_detection.py` so it has no dependency on re-running
   the GUI.
3. **Detect predicted alerts the same way, on the model's forecast**: scan
   the model's *predicted* values for the same threshold-crossing pattern.
   Each predicted alert is timestamped at the moment the forecast was
   made (the trigger time), not the future moment it's predicting about —
   an alert firing "now" is the model saying "in 10 minutes, this will be
   over the line."
4. **Match alerts to real events, greedily, earliest-first**: each real
   event is paired with the earliest not-yet-used predicted alert that
   came before it and within a 30-minute horizon. `mean_lead_min` is the
   average gap (event time − alert time) over successful matches only;
   unmatched real events count as misses (FN), unmatched alerts count as
   false alarms (FP).
5. **Verify the port is faithful before trusting any new number.** Before
   changing anything, the script was run with thresholds set back to the
   *original* values (CO2>2000, PM2.5>35, VOC>200) and checked against the
   already-published `event_detection_lead10.csv`. VOC matched exactly
   (same TP, same 0.0/8.0 min leads) — confirming the ported logic is
   correct. CO2 and PM2.5 did **not** match, which told us
   `predictions_lead10_v2.csv` on disk has drifted from whatever run
   originally produced the published CO2/PM2.5 numbers (see CLAUDE.md).
   Consequence: **any chart showing CO2 or PM2.5 at their original 2000 /
   35 thresholds uses the originally-published rows, not a recomputation**
   — only VOC (or a threshold change you explicitly ask for, like the
   CO2>1000 exploratory version) comes from this script.
6. **Re-run step 2–4 at the new threshold.** Lowering VOC from 200→100 ppb
   doesn't just rescore the same events — a lower bar means more real
   crossings exist to find in the first place (2 events at 200 ppb → 6 at
   100 ppb), which is why the with-C_t VOC result changed so much (0 min →
   10.3 min): there was more real signal available to detect, not a
   different model.
7. **Render**: bar height = `mean_lead_min`; a hatched marker (not a
   height-0 bar) replaces any (model, pollutant, variant) with zero true
   positives, so "never caught it" is never visually confused with "caught
   it right as it happened" (both are 0/blank in the raw numbers but mean
   different things); every bar is annotated with "caught k of n" so the
   sample size behind each average is never hidden; the x-axis threshold
   label and the y-axis scale are both read from the data itself (the
   CSV's own `threshold` column; the data's own max lead time) rather than
   hardcoded, after a fixed y-axis cap of 11 silently clipped a 27.5-minute
   bar off the canvas in an earlier version of this chart.

## Reproducing / extending this analysis

```bash
# no torch needed:
python scripts/persistence_baseline_sweep.py --raw <raw_csv> --outdir .
python scripts/lead_time_effective_horizon.py --raw <raw_csv> --outdir .

# needs torch, run in the env that produced BiLSTM_ck.pt:
python scripts/train_bilstm_lead_sweep_v2.py --raw <raw_csv> --outdir data --export-predictions
python scripts/lead_time_effective_horizon.py --raw <raw_csv> \
    --model-predictions data/bilstm_predictions_export.csv --outdir .
```

`event_detection_lead10.csv` can only be regenerated by re-running the main
GUI pipeline itself (with "Event detection" enabled) — there is no
standalone script for it here, by design, since it needs the trained model
checkpoints, not just the raw session file.
