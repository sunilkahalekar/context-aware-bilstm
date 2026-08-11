"""
IAQ Data Merger — Sensor CSV + Video C_t CSV → Enriched Combined CSV
=====================================================================

PURPOSE
  Joins sensor data and video C_t vectors on a 1-minute timestamp key,
  then enriches every machine's D_t descriptors with operational state
  classification — distinguishing laser cutting from maintenance or
  accidental door-open events.

WHAT CHANGED FROM PREVIOUS VERSION
  One new step is inserted after the merge: enrich_operational_state().
  It adds four columns per machine directly in the compiled output CSV:
    M{n}_op_state              IDLE | CUTTING | EXPOSURE | MAINTENANCE
    M{n}_emission_weight       0.0  |  1.0    |  0.8     |  0.2
    M{n}_effective_tau         tau_open × emission_weight
    M{n}_consecutive_full_open consecutive windows door was fully open

  Nothing else is changed. All original merge logic is identical.

WHY OPERATIONAL STATE MATTERS
  Door open ≠ laser firing. If a door is left open for maintenance,
  cleaning, or cooling (4+ consecutive minutes) the laser is not
  running — no thermal decomposition, no PM spike. The open door is
  actually a VENTILATION path. Treating maintenance open time the same
  as cutting open time corrupts the IAQ model's causal understanding.

  effective_tau is the recommended feature for Bi-LSTM training instead
  of raw tau_open. It correctly encodes near-zero emission during
  maintenance and full emission during cutting.

OPERATIONAL STATE DECISION LOGIC
  ┌─────────────────────────────────────────────────────────────────┐
  │  tau_open == 0                          → IDLE        (em=0.0)  │
  │  f_trans >= 2  (at least one cycle)     → CUTTING     (em=1.0)  │
  │  eps_max >= 55  AND  consec >= 4 min    → MAINTENANCE (em=0.2)  │
  │  eps_max >= 55  AND  consec <  4 min    → EXPOSURE    (em=0.8)  │
  │  fallback (short single open)           → CUTTING     (em=1.0)  │
  └─────────────────────────────────────────────────────────────────┘

MATCHING STRATEGY  (unchanged)
  Both timestamps are truncated to the minute boundary.
  A sensor row at 14:34:48 and a video window at 14:34:00 → same key.
  Multiple sensor readings per minute → mean aggregation.

OUTPUT COLUMNS (in order)
  timestamp_minute
  │ Sensor:  created_at  entry_id  pm1  pm2_5  pm10  temp  hum  co2  voc  rawVoc
  │ Video:   window_start
  │          M{1,2,3}_tau_open  f_trans  rho_open  eps_max  phi_open
  │          M{1,2,3}_op_state  emission_weight  effective_tau
  │                             consecutive_full_open           ← NEW
  │          n_person  mu_motion  sigma2_motion

USAGE
  python iaq_merge.py                          # interactive prompts
  python iaq_merge.py sensor.csv video.csv     # direct arguments
  python iaq_merge.py sensor.csv video.csv out.csv
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────
class Config:
    SENSOR_TS_COL: str  = "created_at"
    VIDEO_TS_COL:  str  = "window_start"
    JOIN_TYPE:     str  = "outer"    # "inner" | "left" | "outer"
    SENSOR_AGG:    str  = "mean"     # "mean" | "median" | "first"
    OUTPUT_SUFFIX: str  = "_merged_iaq.csv"
    MACHINES:      list = [1, 2, 3]

    # ── Operational state thresholds ──────────────────────────────────────────
    # Seconds: eps_max at or above this = "door essentially fully open" for window
    EPS_FULL_THRESHOLD:     int   = 55
    # Consecutive fully-open windows before flagging MAINTENANCE (1 window = 1 min)
    MAINTENANCE_WINDOW_MIN: int   = 4
    # Minimum f_trans to classify as CUTTING (at least one open-close cycle)
    CUTTING_FTRANS_MIN:     int   = 2


cfg = Config()

# Emission weight per operational state
EMISSION_WEIGHT = {
    "IDLE":        0.0,
    "CUTTING":     1.0,
    "EXPOSURE":    0.8,
    "MAINTENANCE": 0.2,
}


# ─── Timestamp parsers ────────────────────────────────────────────────────────

def parse_sensor_ts(raw: str) -> datetime | None:
    raw = str(raw).strip()
    try:
        dt = datetime.fromisoformat(raw)
        return dt.replace(tzinfo=None)
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            pass
    log.warning(f"  Cannot parse sensor timestamp: '{raw}'")
    return None


def parse_video_ts(raw: str) -> datetime | None:
    raw = str(raw).strip()
    for fmt in ["%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S",
                "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M"]:
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            pass
    log.warning(f"  Cannot parse video timestamp: '{raw}'")
    return None


def floor_to_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


# ─── File loaders ─────────────────────────────────────────────────────────────

def load_sensor(path: str) -> pd.DataFrame:
    log.info(f"Loading sensor file: {path}")
    df = pd.read_csv(path)
    if cfg.SENSOR_TS_COL not in df.columns:
        raise ValueError(
            f"Sensor file missing '{cfg.SENSOR_TS_COL}' column. "
            f"Found: {list(df.columns)}")

    df["_ts_parsed"] = df[cfg.SENSOR_TS_COL].apply(parse_sensor_ts)
    df = df.dropna(subset=["_ts_parsed"])
    df["minute_key"] = df["_ts_parsed"].apply(floor_to_minute)
    log.info(f"  Rows: {len(df)}  Unique minutes: {df['minute_key'].nunique()}")

    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    non_numeric      = [c for c in df.columns
                        if c not in numeric_cols + ["minute_key", "_ts_parsed",
                                                    cfg.SENSOR_TS_COL]]
    agg_dict         = {c: cfg.SENSOR_AGG for c in numeric_cols}
    agg_dict.update({c: "first" for c in non_numeric})
    agg_dict[cfg.SENSOR_TS_COL] = "first"

    df_agg = (df.drop(columns=["_ts_parsed"])
                .groupby("minute_key", as_index=False)
                .agg(agg_dict))

    multi = (df.groupby("minute_key").size() > 1).sum()
    if multi:
        log.info(f"  {multi} minutes had multiple readings → "
                 f"aggregated with '{cfg.SENSOR_AGG}'")
    return df_agg


def load_video(path: str) -> pd.DataFrame:
    log.info(f"Loading video file:  {path}")
    df = pd.read_csv(path)
    if cfg.VIDEO_TS_COL not in df.columns:
        raise ValueError(
            f"Video file missing '{cfg.VIDEO_TS_COL}' column. "
            f"Found: {list(df.columns)}")

    df["_ts_parsed"] = df[cfg.VIDEO_TS_COL].apply(parse_video_ts)
    df = df.dropna(subset=["_ts_parsed"])
    df["minute_key"] = df["_ts_parsed"].apply(floor_to_minute)

    dups = df.duplicated(subset="minute_key", keep=False).sum()
    if dups:
        log.warning(f"  {dups} duplicate minute_key rows — keeping first")
        df = df.drop_duplicates(subset="minute_key", keep="first")

    log.info(f"  Rows: {len(df)}  Unique minutes: {df['minute_key'].nunique()}")
    return df.drop(columns=["_ts_parsed"])


# ─── Merge ────────────────────────────────────────────────────────────────────

def merge(sensor_df: pd.DataFrame, video_df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"Merging on minute_key (join='{cfg.JOIN_TYPE}') …")

    merged = pd.merge(video_df, sensor_df,
                      on="minute_key", how=cfg.JOIN_TYPE,
                      suffixes=("_video", "_sensor"))
    merged = merged.sort_values("minute_key").reset_index(drop=True)
    merged.insert(0, "timestamp_minute",
                  merged["minute_key"].dt.strftime("%Y-%m-%d %H:%M"))
    merged = merged.drop(columns=["minute_key"])

    has_video  = merged[cfg.VIDEO_TS_COL].notna().sum()
    has_sensor = merged[cfg.SENSOR_TS_COL].notna().sum()
    both       = (merged[cfg.VIDEO_TS_COL].notna() &
                  merged[cfg.SENSOR_TS_COL].notna()).sum()

    log.info(f"  Total rows    : {len(merged)}")
    log.info(f"  Both sources  : {both}")
    log.info(f"  Video only    : {has_video  - both}")
    log.info(f"  Sensor only   : {has_sensor - both}")

    if both == 0:
        log.warning("  NO rows matched. Check that files cover the same dates.")

    return merged


# ─── Operational state classification ─────────────────────────────────────────
# NEW: inserted as a single enrichment step after the merge.
# Reads existing D_t scalar columns (tau_open, eps_max, f_trans),
# applies the classification rules, and appends four new columns per machine.

def _classify_state(tau: int, f_trans: int, eps: int, consec: int) -> str:
    """
    Classify one window into IDLE / CUTTING / EXPOSURE / MAINTENANCE.

    Rules (applied in priority order):
      1. tau == 0                               → IDLE
      2. f_trans >= CUTTING_FTRANS_MIN          → CUTTING
         (at least one complete open-close cycle = active operation)
      3. eps >= EPS_FULL_THRESHOLD AND
         consec >= MAINTENANCE_WINDOW_MIN       → MAINTENANCE
         (4+ consecutive fully-open windows = not cutting)
      4. eps >= EPS_FULL_THRESHOLD AND
         consec <  MAINTENANCE_WINDOW_MIN       → EXPOSURE
         (door fully open but maintenance not confirmed yet)
      5. fallback                               → CUTTING
    """
    if tau == 0:
        return "IDLE"
    if f_trans >= cfg.CUTTING_FTRANS_MIN:
        return "CUTTING"
    if eps >= cfg.EPS_FULL_THRESHOLD:
        if consec >= cfg.MAINTENANCE_WINDOW_MIN:
            return "MAINTENANCE"
        return "EXPOSURE"
    return "CUTTING"


def enrich_operational_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add four new columns per machine to the merged DataFrame:

      M{n}_op_state              str    IDLE | CUTTING | EXPOSURE | MAINTENANCE
      M{n}_emission_weight       float  0.0  | 1.0     | 0.8      | 0.2
      M{n}_effective_tau         float  tau_open × emission_weight
      M{n}_consecutive_full_open int    running count of fully-open windows

    The consecutive counter is reset to 0 whenever eps_max drops below
    EPS_FULL_THRESHOLD, preserving cross-window maintenance detection.
    Rows where the D_t columns are NaN (sensor-only rows) receive NaN
    for all new columns so the merge coverage report stays accurate.

    This function modifies df in-place and returns it.
    """
    log.info("Enriching with operational state classification …")

    for m in cfg.MACHINES:
        tau_col = f"M{m}_tau_open"
        eps_col = f"M{m}_eps_max"
        ftr_col = f"M{m}_f_trans"

        # If the D_t columns are absent (old C_t format), skip gracefully
        if tau_col not in df.columns:
            log.warning(f"  M{m}: '{tau_col}' not found — skipping")
            continue

        op_states, em_weights, eff_taus, consecs = [], [], [], []
        consec = 0   # persistent across rows, reset each machine

        for _, row in df.iterrows():
            tau = row.get(tau_col)
            eps = row.get(eps_col)
            ftr = row.get(ftr_col)

            # NaN means this is a sensor-only row (no video data)
            if pd.isna(tau):
                op_states.append(np.nan)
                em_weights.append(np.nan)
                eff_taus.append(np.nan)
                consecs.append(np.nan)
                # Do NOT update consec — treat gap as unknown, not as reset
                continue

            tau = int(tau); eps = int(eps); ftr = int(ftr)

            # Update consecutive fully-open counter
            if eps >= cfg.EPS_FULL_THRESHOLD:
                consec += 1
            else:
                consec = 0

            state    = _classify_state(tau, ftr, eps, consec)
            em_w     = EMISSION_WEIGHT[state]
            eff_tau  = round(tau * em_w, 2)

            op_states.append(state)
            em_weights.append(em_w)
            eff_taus.append(eff_tau)
            consecs.append(consec)

        # Insert the four new columns right after M{n}_phi_open
        # so each machine's block stays self-contained in the CSV
        phi_col = f"M{m}_phi_open"
        insert_after = (df.columns.get_loc(phi_col) + 1
                        if phi_col in df.columns
                        else len(df.columns))

        # Build a temporary frame and insert columns at the correct position
        new_cols = {
            f"M{m}_op_state":              op_states,
            f"M{m}_emission_weight":       em_weights,
            f"M{m}_effective_tau":         eff_taus,
            f"M{m}_consecutive_full_open": consecs,
        }
        for offset, (col_name, values) in enumerate(new_cols.items()):
            df.insert(min(insert_after + offset, len(df.columns)),
                      col_name, values)

        # Log state distribution for this machine
        state_series = pd.Series(op_states).dropna()
        if len(state_series):
            dist = state_series.value_counts()
            log.info(f"  M{m} state distribution  "
                     f"({len(state_series)} windows with video data):")
            for state in ["CUTTING", "MAINTENANCE", "EXPOSURE", "IDLE"]:
                count = dist.get(state, 0)
                pct   = count / len(state_series) * 100
                log.info(f"      {state:12s}: {count:>4}  ({pct:5.1f}%)")

    return df


# ─── Column ordering ──────────────────────────────────────────────────────────

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final column order:
      timestamp_minute
      sensor columns (created_at … rawVoc)
      video columns per machine (original D_t + new state columns)
      motion columns
    """
    sensor_preferred = [
        cfg.SENSOR_TS_COL, "entry_id",
        "pm1", "pm2_5", "pm10",
        "temp", "hum", "co2", "voc", "rawVoc",
    ]

    video_preferred = [cfg.VIDEO_TS_COL]
    for m in cfg.MACHINES:
        video_preferred += [
            f"M{m}_tau_open",   f"M{m}_f_trans",   f"M{m}_rho_open",
            f"M{m}_eps_max",    f"M{m}_phi_open",
            # NEW columns — sit immediately after original D_t per machine
            f"M{m}_op_state",
            f"M{m}_emission_weight",
            f"M{m}_effective_tau",
            f"M{m}_consecutive_full_open",
        ]
    video_preferred += ["n_person", "mu_motion", "sigma2_motion"]

    ordered = ["timestamp_minute"]
    for c in sensor_preferred + video_preferred:
        if c in df.columns and c not in ordered:
            ordered.append(c)
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    log.info("=" * 65)
    log.info("MERGED + ENRICHED DATASET SUMMARY")
    log.info("=" * 65)
    log.info(f"  Rows        : {len(df)}")
    log.info(f"  Columns     : {len(df.columns)}")
    if len(df):
        log.info(f"  Time range  : {df['timestamp_minute'].iloc[0]}"
                 f" → {df['timestamp_minute'].iloc[-1]}")

    pm_col  = next((c for c in ["pm2_5","pm1","pm10"] if c in df.columns), None)
    tau_col = next((c for c in df.columns if "tau_open" in c), None)
    if pm_col:
        n = df[pm_col].notna().sum()
        log.info(f"  Sensor rows : {n} / {len(df)}  ({n/len(df)*100:.1f}%)")
    if tau_col:
        n = df[tau_col].notna().sum()
        log.info(f"  Video rows  : {n} / {len(df)}  ({n/len(df)*100:.1f}%)")

    for col in ["pm2_5", "co2", "M2_effective_tau", "n_person"]:
        if col in df.columns and df[col].notna().any():
            s = df[col].dropna()
            log.info(f"  {col:22s}: "
                     f"min={s.min():.2f}  mean={s.mean():.2f}  max={s.max():.2f}")

    # Overall state distribution across all machines
    all_states = []
    for m in cfg.MACHINES:
        col = f"M{m}_op_state"
        if col in df.columns:
            all_states.extend(df[col].dropna().tolist())
    if all_states:
        from collections import Counter
        dist  = Counter(all_states)
        total = len(all_states)
        log.info(f"  Overall op_state (all machines × all windows):")
        for state in ["CUTTING", "MAINTENANCE", "EXPOSURE", "IDLE"]:
            n   = dist.get(state, 0)
            pct = n / total * 100
            log.info(f"      {state:12s}: {n:>5}  ({pct:5.1f}%)")
    log.info("=" * 65)


# ─── File path helper ─────────────────────────────────────────────────────────

def ask_path(prompt: str) -> str:
    while True:
        path = input(f"\n  {prompt}: ").strip().strip('"').strip("'")
        if not path:
            print("  (no path entered — exiting)")
            sys.exit(0)
        if not os.path.exists(path):
            print(f"  File not found: {path}"); continue
        return path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("IAQ Data Merger + Operational State Enrichment")
    log.info("-" * 50)

    args = sys.argv[1:]
    if len(args) >= 2:
        sensor_path = args[0]
        video_path  = args[1]
        out_path    = args[2] if len(args) >= 3 else None
    else:
        print()
        print("  Usage: python iaq_merge.py sensor.csv video.csv [output.csv]")
        sensor_path = ask_path("Sensor CSV  (e.g. sensor_data.csv)")
        video_path  = ask_path("Video C_t CSV  (e.g. Ct_vectors.csv)")
        out_path    = None

    for p, label in [(sensor_path, "sensor"), (video_path, "video")]:
        if not os.path.exists(p):
            log.error(f"{label} file not found: {p}"); sys.exit(1)

    if out_path is None:
        base     = Path(sensor_path).stem
        out_path = str(Path(sensor_path).parent / (base + cfg.OUTPUT_SUFFIX))

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        sensor_df = load_sensor(sensor_path)
        video_df  = load_video(video_path)
    except ValueError as e:
        log.error(str(e)); sys.exit(1)

    # ── Merge on timestamp ────────────────────────────────────────────────────
    merged = merge(sensor_df, video_df)

    # ── Enrich with operational state ── NEW STEP ────────────────────────────
    merged = enrich_operational_state(merged)

    # ── Reorder columns ───────────────────────────────────────────────────────
    merged = reorder_columns(merged)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(merged)

    # ── Save ──────────────────────────────────────────────────────────────────
    merged.to_csv(out_path, index=False, float_format="%.6f")
    log.info(f"Saved: {out_path}")

    # ── Preview ───────────────────────────────────────────────────────────────
    preview_cols = ["timestamp_minute"] + [
        c for c in ["pm2_5", "co2",
                    "M2_tau_open", "M2_op_state", "M2_emission_weight",
                    "M2_effective_tau", "M3_op_state", "n_person"]
        if c in merged.columns
    ]
    log.info("\nFirst 5 rows preview (key columns):")
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.width", 160)
    print(merged[preview_cols].head().to_string(index=False))
    log.info(f"\nNew columns added per machine: "
             f"op_state, emission_weight, effective_tau, "
             f"consecutive_full_open")
    log.info("Use M{{n}}_effective_tau instead of M{{n}}_tau_open "
             "in the Bi-LSTM training data.")


if __name__ == "__main__":
    main()
