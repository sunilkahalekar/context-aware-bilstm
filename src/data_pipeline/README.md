# Data Pipeline — Merge Vision + Sensor Data

`merge_vision_and_sensor_data.py` (was `iaq_merge.py`) takes two
spreadsheets you already have —

1. a **sensor CSV** (air-quality readings: PM, CO2, VOC, temperature, humidity…)
2. a **video CSV** (`Ct_vectors_*.csv`, produced by
   [`../vision/extract_context_vector_from_video.py`](../vision/README.md)
   from CCTV footage — when each machine's door was open/closed)

— and glues them together into **one CSV**, lined up minute-by-minute, with
extra "what was actually happening" columns added in. This is the file that
becomes `data/raw/sensor_data_merged_iaq_m2.csv`, the input to everything in
`src/modeling/` and `src/analysis/`.

If you just want to run it, skip to **Quick start**. If you want to
understand *why* the tool exists and what the new columns mean, read
**What problem this solves** first.

## What problem this solves

A laser-cutting machine's door sensor shows "open" for 6 minutes straight.
Two very different things could be happening:

- **The laser is cutting.** The door cycles open/closed rapidly as parts
  are loaded/unloaded — produces smoke and particulate matter (PM).
- **Someone propped the door open** for cleaning/cooldown/maintenance. No
  laser running, no smoke — the open door just lets room air circulate.

A model trying to predict air quality from "is the door open" alone cannot
tell these apart, and they have opposite effects. This tool looks at the
*pattern* of door activity (not just open/closed) and labels every
1-minute window:

| Label | Meaning | "Emission" assumed |
|---|---|---|
| `IDLE` | Door never opened this minute | None (0%) |
| `CUTTING` | Door cycling open/closed — active work | Full (100%) |
| `EXPOSURE` | Open a while, not long enough for maintenance yet | Most (80%) |
| `MAINTENANCE` | Continuously open 4+ minutes straight | Almost none (20%) |

That "how much emission to assume" number is the **emission weight**,
multiplied by raw open-door time to produce **effective tau** — feed this
into any downstream model instead of raw door-open time.

## Requirements

```bash
pip install pandas numpy
```
Python 3.10+ (uses modern type hints like `str | None`).

## Quick start

```bash
python merge_vision_and_sensor_data.py                                    # prompts for paths interactively
python merge_vision_and_sensor_data.py sensor_data.csv Ct_vectors_2026-08-10.csv                # direct
python merge_vision_and_sensor_data.py sensor_data.csv Ct_vectors_2026-08-10.csv my_output.csv  # custom output name
```
No output filename given → auto-created next to the sensor file as
`<sensor-file-name>_merged_iaq.csv`.

## What happens when you run it

1. **Loads the sensor file**, reports row/minute counts. Multiple readings
   in the same minute are averaged into one row.
2. **Loads the video file** the same way, warning on duplicate minutes
   (keeps the first, drops the rest).
3. **Merges** by matching shared minutes; reports how many minutes had
   both / only sensor / only video data.
4. **Classifies each machine's activity** per minute (table above), prints
   a percentage breakdown per machine.
5. **Saves** the result, prints the exact path.
6. **Previews** the first 5 rows of key columns in the terminal.

## Understanding the output file

One row per minute:

| Column | What it is |
|---|---|
| `timestamp_minute` | The minute, e.g. `2026-08-10 14:22` |
| `created_at`, `entry_id` | From the sensor file |
| `pm1`, `pm2_5`, `pm10` | Particulate readings |
| `temp`, `hum`, `co2`, `voc`, `rawVoc` | Other sensor readings |
| `window_start` | From the video file |
| `M1_tau_open`, `M2_tau_open`, `M3_tau_open` | Seconds each door was open this minute (0–60) |
| `M{n}_f_trans` | Open↔closed flips this minute |
| `M{n}_rho_open` | Fraction of the minute open (0.0–1.0) |
| `M{n}_eps_max` | Longest continuous open stretch, seconds |
| `M{n}_phi_open` | How far into the minute the door first opened (0.0=immediately, 1.0=never) |
| **`M{n}_op_state`** | This minute's label: `IDLE`/`CUTTING`/`EXPOSURE`/`MAINTENANCE` |
| **`M{n}_emission_weight`** | The 0.0–1.0 multiplier for that label |
| **`M{n}_effective_tau`** | **Use this one.** `tau_open × emission_weight` |
| **`M{n}_consecutive_full_open`** | Minutes in a row (incl. this one) door has been almost entirely open |
| `n_person` | Max people seen near machines this minute |
| `mu_motion`, `sigma2_motion` | Mean/variance of people's movement |

> **Note**: this tool *does* produce `M{n}_op_state`, `M{n}_emission_weight`,
> `M{n}_effective_tau`, `M{n}_consecutive_full_open` — but the *upstream*
> vision extraction script (`../vision/extract_context_vector_from_video.py`)
> does not yet supply the raw signals these are computed from beyond what's
> already in `Ct_vectors_*.csv`. See `../modeling/README.md`'s "Known Gap"
> section for how this currently plays out downstream.

**Empty cells matter** — a row with sensor but no video data (or vice
versa) means that minute wasn't covered by both files (normal at
session start/end, or if a device was offline).

## FAQ

**`tau_open` or `effective_tau`?** Use `effective_tau` — corrected to
near-zero during maintenance so the model doesn't learn that maintenance
causes the same air-quality effect as cutting.

**Why does a 6-minute open door sometimes say `CUTTING`, sometimes
`MAINTENANCE`?** Depends on whether it *cycled* (opening/closing
repeatedly — active loading/unloading, → `CUTTING`) or was held open
continuously (→ `EXPOSURE`/`MAINTENANCE`). Check `f_trans`: ≥2 means it
cycled.

**Fewer output rows than expected?** Check the "Video only"/"Sensor only"
counts printed during the run — if the two files' dates don't overlap at
all, you'll see "NO rows matched."

**Change thresholds (e.g. the "4 minutes" for maintenance)?** Edit the
`Config` section at the top of the script — see
[CLAUDE.md](CLAUDE.md) for full internals before changing anything.

## Troubleshooting

| Message | Meaning | Fix |
|---|---|---|
| `Sensor file missing 'created_at' column` | Timestamp column misnamed | Rename column, or change `Config.SENSOR_TS_COL` |
| `Video file missing 'window_start' column` | Same, video file | Check you're pointing at the right file, or `Config.VIDEO_TS_COL` |
| `Cannot parse sensor timestamp: '...'` | Unrecognized format on one row | Usually harmless (row skipped) unless frequent |
| `NO rows matched` | Files don't share a common minute | Confirm same recording session/date |
| `M2: 'M2_tau_open' not found — skipping` | Video file lacks M2's columns | Expected if fewer than 3 machines |

## Related files
- [merge_vision_and_sensor_data.py](merge_vision_and_sensor_data.py) — the script
- [CLAUDE.md](CLAUDE.md) — internals reference
- [../vision/](../vision/README.md) — produces the video CSV this consumes
