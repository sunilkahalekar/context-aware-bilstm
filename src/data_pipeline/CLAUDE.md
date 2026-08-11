# CLAUDE.md — internals reference for src/data_pipeline/

Scope: `merge_vision_and_sensor_data.py` (was `iaq_merge.py`). For what it
does and how to run it, see [README.md](README.md) first. This file is for
anyone about to modify the code.

## Mental model: two independent time series joined on a lossy key

1. **Sensor readings** — arbitrary, often sub-minute frequency. Timestamp
   column: `created_at` (`Config.SENSOR_TS_COL`).
2. **Video C_t vectors** — exactly one row per 60-second window, produced
   upstream by
   [`../vision/extract_context_vector_from_video.py`](../vision/CLAUDE.md).
   Timestamp column: `window_start` (`Config.VIDEO_TS_COL`).

Both collapse to a common, lossy join key: timestamp truncated to the
minute (`floor_to_minute`). Intentionally lossy — a reading at `14:22:59`
and one at `14:22:01` both become `minute_key = 14:22:00`. This is why
sensor rows are aggregated (mean, by default) before the join: multiple
readings can legitimately collide on the same key, but the join itself is a
simple 1:1 merge on `minute_key` and can't handle collisions on its own.

Video rows are **not** aggregated the same way — the upstream pipeline
should never emit two windows for the same minute. If it does anyway
(`load_video`'s duplicate check), the extras are dropped with a warning
rather than averaged (`f_trans`, `eps_max`, etc. aren't meaningfully
averageable across two independent windows).

## Module layout (top to bottom)
```
Config / cfg               — all tunable constants, single object
EMISSION_WEIGHT             — state → weight lookup table
TIMESTAMP PARSERS           — parse_sensor_ts, parse_video_ts, floor_to_minute
FILE LOADERS                — load_sensor, load_video
merge                       — the actual pd.merge + coverage report
OPERATIONAL STATE           — _classify_state, enrich_operational_state
reorder_columns             — cosmetic, final column ordering
print_summary                — terminal report after everything else runs
ask_path                    — interactive CLI prompt helper
main()                      — orchestrates everything, in file-execution order
```

## Function reference

**`parse_sensor_ts(raw)` / `parse_video_ts(raw)`** — two parsers because
the upstream systems format timestamps differently. Sensor: tries
`datetime.fromisoformat` first (handles ISO variants including a tz offset,
immediately stripped via `.replace(tzinfo=None)` — the rest of the pipeline
is naive-datetime throughout), falls back to 3 explicit `strptime` formats.
Video: tries 5 explicit formats, prioritizing `%d-%m-%Y %H:%M` (day-first —
matches the vision script's `session_tag` output). **Don't swap the
priority order** without checking your actual `Ct_vectors_*.csv` format;
day-first vs month-first ambiguity (`03-04-2026`) is a silent-wrong-answer
risk, not a crash risk. Both return `None` (+ log warning) on total parse
failure; callers `dropna()` those rows rather than crashing the run.

**`floor_to_minute(dt)`** — `dt.replace(second=0, microsecond=0)`. The join
key generator; every "what minute is this" need calls this, never
re-derives it inline.

**`load_sensor(path)`** — reads CSV, validates `SENSOR_TS_COL` exists
(raises `ValueError` with the actual column list — caught in `main()` and
turned into a clean CLI error, not a traceback). Parses timestamps, drops
unparseable rows, computes `minute_key`. Builds an aggregation dict
**dynamically from the dataframe's own dtypes**: numeric columns use
`Config.SENSOR_AGG` (mean by default), non-numeric use `"first"` — adapts
automatically to extra sensor columns, no code change needed for schema
changes (only `Config.SENSOR_AGG` for a different strategy). Then
`groupby("minute_key").agg(...)`.

**`load_video(path)`** — same shape, but does **not** aggregate same-key
rows (see "Mental model"). A duplicate `minute_key` is a data-quality
warning, kept as `first`.

**`merge(sensor_df, video_df)`** — thin wrapper around
`pd.merge(..., how=Config.JOIN_TYPE)` (outer by default — no data silently
dropped even on partial overlap) plus a coverage report (`both`/`video
only`/`sensor only` counts) logged for the user. `timestamp_minute` (a
string, not `datetime`) replaces the raw `minute_key` right after merge —
kept as a string specifically so pandas/Excel don't reinterpret and shift
the display format.

**`_classify_state(tau, f_trans, eps, consec)`** — pure function, no side
effects, no dataframe access (deliberately unit-testable in isolation from
`enrich_operational_state`'s row iteration). Priority order is significant,
encoded as sequential `if` returns, not a table:
```
tau == 0                                                        → IDLE
f_trans >= CUTTING_FTRANS_MIN                                   → CUTTING
eps >= EPS_FULL_THRESHOLD and consec >= MAINTENANCE_WINDOW_MIN   → MAINTENANCE
eps >= EPS_FULL_THRESHOLD (consec below threshold)               → EXPOSURE
else                                                              → CUTTING (fallback)
```
Rule 2 (`f_trans`) is checked **before** rules 3/4: a cycling door is
always `CUTTING` regardless of open duration, because cycling itself is
evidence of active work. `eps_max` is only diagnostic once you already know
the door *isn't* cycling.

**`enrich_operational_state(df)`** — per machine, iterates the merged
dataframe **row by row** (`df.iterrows()`) because `consecutive_full_open`
is a running, order-dependent counter — not cleanly vectorizable without
either a manual rolling-reset condition or `groupby` tricks less readable
than a plain loop here. Intentionally O(n) per machine; not a performance
concern at 1-row-per-minute scale (a full year ≈ 525,600 rows, still fast
for a one-off CLI transform) — revisit only if repurposed for sub-minute
granularity or many-year batch runs.

Key gap behavior: on `pd.isna(tau)` (sensor-only row, no video data that
minute), the row gets `NaN` for all 4 new columns **and `consec` is left
untouched** — a gap means "unknown," not "door closed." Flipping this to
reset `consec` on a gap changes MAINTENANCE detection across any real gap
(dropped frames, sensor offline) — don't change without re-reading the
code's stated rationale.

Column insertion: new columns inserted immediately after each machine's
`M{n}_phi_open` (`df.columns.get_loc(...) + 1`), in fixed order
(`op_state`, `emission_weight`, `effective_tau`, `consecutive_full_open`)
per machine before moving to the next. `reorder_columns` re-asserts this
order later — keep its `video_preferred` list in sync if you add a 5th
new column.

**`reorder_columns(df)`** — cosmetic only, static preferred-order list from
`Config.MACHINES`, falls back to appending anything not listed (`extras`)
at the end — no column ever silently dropped, only repositioned.

**`print_summary(df)`** — terminal-only, no return, doesn't mutate `df`.
Picks whichever of `pm2_5`/`pm1`/`pm10` exists (tolerant of sensor CSVs
missing one). Aggregates `op_state` across **all** machines into one
combined distribution (separate from the per-machine ones logged earlier
during enrichment).

## Execution flow (`main()`)
```
Parse sys.argv (sensor_path, video_path, out_path)
  → not enough args? prompt interactively (ask_path)
  → validate both files exist
  → out_path not given? derive: sensor_stem + OUTPUT_SUFFIX
  → load_sensor → load_video → merge (outer join on minute_key)
  → enrich_operational_state (per machine, per row)
  → reorder_columns → print_summary → df.to_csv
  → print first-5-rows preview
```

### Error paths
- Missing required timestamp column → `ValueError` → caught in `main()`,
  logged, `sys.exit(1)`. No traceback shown.
- File path doesn't exist → checked in `ask_path`'s interactive loop
  (re-prompts) and again for argument-based paths (`sys.exit(1)`, no
  re-prompt available).
- Zero rows matched → not fatal, just a logged warning. Output file is
  still written (likely mostly `NaN` on one side) — useful for debugging
  why the dates don't line up.

## Gotchas for future changes

- **`_classify_state`'s rule order is load-bearing.** Rule 2 (`f_trans`)
  must stay checked before rules 3/4. Reordering silently changes which
  windows get `CUTTING` vs `MAINTENANCE`/`EXPOSURE` with no error raised.
- **The video-timestamp format list in `parse_video_ts` is priority-ordered,
  not a set of alternatives.** `%d-%m-%Y` tried first. If the video CSV
  ever switches to month-first, dates with day ≤ 12 parse *silently
  wrong* (swapped day/month) — no ambiguity check exists.
- **`consecutive_full_open` does not reset across a NaN gap** —
  intentional (see `enrich_operational_state` above), but a sensor outage
  spanning a real door-close event can make the counter overcount when
  data resumes. "Reset on any gap" is a behavior change, not a bug fix —
  confirm with whoever owns the downstream model before changing it.
- **Adding a machine (M4+)**: extend `Config.MACHINES` — the enrichment
  loop and `reorder_columns`'s `video_preferred` loop already iterate over
  it. No other change needed *unless* the video CSV's column names deviate
  from the `M{n}_tau_open`/`f_trans`/`eps_max`/`phi_open` pattern.
- **Changing `EPS_FULL_THRESHOLD`/`MAINTENANCE_WINDOW_MIN`**: read live
  from `cfg` inside `_classify_state`, so a single `Config` edit suffices.
  Re-run against a known video CSV and check the printed state
  distribution before trusting new thresholds on real data.
- **No automated tests.** If you vectorize `enrich_operational_state`'s
  row-by-row loop for performance, manually diff the output CSV against
  the current implementation on a real file first — the gap/`consec`
  behavior above is the part most likely lost in a vectorized rewrite.
