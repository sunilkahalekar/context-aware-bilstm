# CLAUDE.md — context for future sessions on lead-time analysis

Read this before touching anything in this folder, especially before
trusting any BiLSTM lead-time number. This folder exists because the
"what forecast lead time should this system target, and why" question
turned out to need several different analyses, not one chart.

## What question this folder answers, and why it needed several attempts

The project had been assuming T+10 minutes as the forecast horizon
without ever deriving it. Three different framings of "how far ahead can
we forecast" turned out to give three different, complementary answers:

1. **Statistical**: how far ahead does the raw pollutant signal remain
   self-predictable at all (ignoring any model)? →
   `persistence_baseline_sweep.py` / `lead_time_effective_horizon.py`.
2. **Model-specific**: does a real trained architecture (BiLSTM) hold up
   across lead times, and does C_t change that? → `train_bilstm_lead_
   sweep*.py`. **Still unresolved — see the debugging log below.**
3. **Operational**: how many minutes of real advance warning does the
   vision context give a facility manager before an actual pollution
   spike, so they can act? → the pipeline's own `event_detection_
   lead10.csv` (not built in this folder — see README.md for exactly how
   it's generated).

Don't conflate these. A user or future session asking "how far ahead can
we forecast" needs to be pointed at the *right one* of these three, or
answered with all three explicitly labeled — mixing them (e.g. quoting
the operational 9-minute CO2 warning number as if it validates the
statistical persistence-baseline finding) would be a real error, not just
sloppy writing.

## BiLSTM lead-time sweep — full debugging history (unresolved)

Every attempt at a genuine BiLSTM-specific lead-time sweep has failed so
far. In order:

1. **`train_bilstm_lead_sweep.py`, hidden=160, dropout=0.1** (user's
   original hyperparameter spec: lookback=15, epochs=500, batch=24,
   lr=0.0003, hidden=160, dropout=0.1, early-stop patience=50,
   70/15/15 split). Result: overall R² deeply negative at every lead,
   −9.6 (T+1) to −24.0 (T+15), getting WORSE at longer leads. Diagnosed
   as likely overfitting: hidden=160 is a large model for ~230 training
   sequences, and longer leads leave even fewer (`make_seqs` loses
   `lookback + lead` rows off the front).
2. **Same file, dropout raised 0.1→0.35, weight_decay raised 1e-3→3e-3**
   (still hidden=160), on the theory that more regularization would fix
   the overfitting. Result: barely changed (−10.0 to −22.9). **This
   ruled OUT "insufficient regularization" as the cause** — important
   negative result, don't re-try more dropout/weight-decay on a
   hidden=160 model expecting a different outcome without new evidence.
3. **`train_bilstm_lead_sweep_v2.py`, hidden dropped 160→64** (matching
   the GUI pipeline's own default, and the exact capacity that achieved
   real BiLSTM R²=0.9979 at lead=10 in the v1 production run), dropout/
   weight_decay left at the raised values from attempt 2. Result:
   **partially different, still broken** — R²≈0.85–0.9 at T+1 (usable!),
   but every other lead (T+3 through T+20) still deeply negative and
   off the visible axis range in `LEAD_TIME_ACCURACY_BiLSTM.png`-style
   charts. NOT confirmed whether this run was actually v1 or v2 code —
   `lead_time_accuracy.py`'s filename-based auto-labeling can't
   currently distinguish `bilstm_lead_time_sweep.csv` from `_v2.csv`
   well enough to be sure from the chart title alone. **Verify which
   script produced any BiLSTM chart you're looking at before trusting
   it** — check the actual CSV filename, not just the PNG title.

**Where this stands**: shrinking the model helped the shortest lead but
did not fix the longer ones. The next things to check, in order of
likely payoff, before trying another hyperparameter shot in the dark:
- Print/inspect `n_train` (now included in v2's output row) across
  leads — confirm training-sequence count isn't collapsing to something
  unreasonably small at the longer leads with lookback=15.
- Actually look at the per-50-epoch validation loss log v2 now prints
  (this was added specifically because v1 never surfaced it) — confirm
  whether training is diverging, plateauing immediately, or genuinely
  converging to a bad optimum.
- Consider whether early-stop patience=50 against a cosine-annealing-
  with-warm-restarts schedule (T_0=30) is fighting itself — a restart
  every 30 epochs could be repeatedly kicked out of a good minimum right
  as patience is about to trigger.
- If still broken: try the GUI pipeline's own full defaults (lookback=20,
  hidden=64, dropout=0.3, lr=0.001) with only LEAD swept, as a sanity
  check that the *architecture port itself* is faithful — if that also
  fails to reproduce the production run's R²=0.9979 at lead=10, the bug
  is in the ported feature engineering or training loop, not the
  hyperparameters.

## A subtlety worth remembering: R² baseline choice

The R² used in `train_bilstm_lead_sweep*.py` (and everywhere else in this
project) compares a model to "always predict the test window's own
mean" — for a **trending** series (CO2 and VOC drift steadily within a
phase), that's a weak bar. `persistence_baseline_sweep.py` showed that
the much tougher "predict no change" baseline actually stays *positive*
out to ~8 minutes — meaning a model failing to clear even the weak
mean-baseline (as BiLSTM has at every lead past 1 minute) is failing
something more basic than "not beating C_t" or "not beating persistence."
This is a real training/convergence problem, not a metric artifact.

## Standing instructions carried over from the rest of this project

- **Never force a universal "C_t always helps" claim** (see
  `src/analysis/ct_significance_testing/CLAUDE.md` — this instruction originated there and applies
  here identically). The event-detection findings already show this
  directly: C_t gives BiLSTM strong early warning for CO2/PM2.5 but *no*
  early warning for VOC, where the non-vision variant did better. Report
  that honestly, per-pollutant, not smoothed into "vision context always
  helps."
- **Small-sample caveats are not optional footnotes.** Every number in
  `event_detection_lead10.csv` rests on 2–4 real events; every bootstrap
  CI in this folder exists specifically because point estimates on this
  dataset (386 rows) have already been shown to be unstable (the BiLSTM
  debugging history above is the clearest possible demonstration of
  this). State the sample size next to any lead-time number quoted from
  this folder.

## Useful next steps not yet done

- Fix `lead_time_accuracy.py`'s auto-label logic to distinguish
  `bilstm_lead_time_sweep.csv` (v1, hidden=160) from `_v2.csv`
  (hidden=64) by filename, not just "contains 'bilstm'" — right now both
  produce the same generic "BiLSTM" label and source-note text.
- Once a BiLSTM sweep produces sane R² at every lead (not just T+1), run
  `train_bilstm_lead_sweep_v2.py --export-predictions` and feed the
  result into `lead_time_effective_horizon.py --model-predictions` to
  get the actual model-vs-persistence gap-with-confidence-band plot —
  this was requested explicitly and is currently blocked purely on
  BiLSTM training working at all.
- Consider whether `event_detection_lead10.csv`'s 2–4-events-per-pollutant
  sample size can be improved by lowering alert thresholds (more events,
  less individually meaningful) or is simply a hard limit of this
  386-minute recording — worth discussing with whoever owns data
  collection before quoting these numbers in anything more formal than
  an internal report.
