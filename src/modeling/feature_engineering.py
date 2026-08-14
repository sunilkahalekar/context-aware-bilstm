"""
feature_engineering.py
=======================
Canonical, GUI-free extraction of the preprocessing pipeline that used to
live only inline inside train_context_aware_bilstm_gui.py's _pipeline()
method (roughly its FIX 1 / FIX 5 / FIX 12 / FIX 15 blocks).

WHY THIS EXISTS
----------------
The GUI script computed features, fit scalers, and built sequences inline,
using local variables closed over by _pipeline(). That made two things true
that this module fixes:

  1. There was no way to reproduce the exact same preprocessing anywhere
     other than inside a running GUI training session -- in particular,
     not from the iaq-edge-pipeline repo at inference time. Any edge-side
     reimplementation would silently drift from what the model was
     actually trained on (train/serve skew), which is exactly the kind of
     bug FIX 12 already found once (see module docstring history in
     train_context_aware_bilstm_gui.py).
  2. Scalers (the StandardScaler on X and the five per-target MinMaxScalers
     on y) were never persisted anywhere -- grep the old pipeline for
     "joblib"/"pickle"/"dump" and you get nothing. Only `model.state_dict()`
     was ever saved. A checkpoint alone was therefore not usable for
     correct inference; this module's save_bundle()/load_bundle() close
     that gap.

This module intentionally has ZERO tkinter and ZERO torch dependency, so
it can be imported by:
  - train_context_aware_bilstm_gui.py (training, this repo)
  - iaq_forecast.py (edge inference, iaq-edge-pipeline repo) -- vendor a
    copy there the same way export_model.py's output gets copied to the
    Pi: this file is exported/copied at model-bundle-build time, not
    imported live across repos.

KNOWN CAVEATS THIS MODULE DELIBERATELY PRESERVES, NOT SILENTLY "FIXES"
------------------------------------------------------------------------
- Verified directly against data/raw/sensor_data_merged_iaq_m2.csv (the
  real v1 training data, 386 rows): `M{m}_effective_tau` IS present with
  real, varying values (derived upstream by
  src/data_pipeline/merge_vision_and_sensor_data.py from tau_open and a
  door-physics op-state model) -- it is NOT one of the dead columns,
  despite this repo's top-level CLAUDE.md grouping it together with the
  three that genuinely are dead. Confirmed dead in the real data (all
  values fillna(0)==0):
    - `M{m}_emission_weight` (3 cols) -- computed by
      merge_vision_and_sensor_data.py but not present in this particular
      exported CSV.
    - `M{m}_consecutive_full_open` (3 cols) -- same.
    - `M{m}_is_{IDLE,CUTTING,EXPOSURE,MAINTENANCE}` (12 cols) -- because
      `M{m}_op_state` itself (the column the one-hot is built from) isn't
      in this CSV, build_base_columns() fills it with 0.0, so every
      is_{state} comparison is constant False.
  That's exactly 18 of 68 -- reconciling the repo's own "18 of 68
  constant zero" figure precisely, once effective_tau is correctly
  excluded from the dead list.

  This module's feat_cfg keys mirror the GUI's checkboxes exactly EXCEPT
  two additions the GUI has no equivalent for: `use_emission_weight` and
  `use_effective_tau`. The GUI's single `use_emission_wt` checkbox bundles
  five descriptors together (emission_weight, phi_open, rho_open, eps_max,
  effective_tau, f_trans -- six, actually, counting f_trans); this module
  splits it three ways:
    - `use_emission_wt` -- phi_open/rho_open/eps_max/f_trans only. Real
      AND stateless (recomputed fresh per 60s window, no history needed).
      DEFAULT_FEATURE_CONFIG's recommended setting for edge deployment.
    - `use_effective_tau` -- real but STATEFUL (see above) -- off by
      default for edge configs, fine for offline/research training.
    - `use_emission_weight` -- verified dead AND stateful -- off by
      default always, unless you're deliberately reproducing an old run.
  Setting all three True together, plus use_consecutive and
  use_op_state_onehot, reproduces the original GUI's `use_emission_wt=True`
  behavior's column SET exactly (verified against the actual v1 training
  run's logged 68-column feature list; column order differs slightly
  since effective_tau/emission_weight are now appended by separate,
  later toggles instead of interleaved into one block -- order doesn't
  affect model correctness since each run's exact feat_cols list is saved
  in its preprocessing bundle and reused consistently for that run's
  training and inference).

  Do NOT set use_op_state_onehot / use_consecutive / use_emission_weight /
  use_effective_tau True in a config used for a real edge-deployed model
  unless you have first (a) implemented real values for the three dead
  ones upstream, and (b) for effective_tau specifically, ported
  enrich_operational_state()'s stateful classifier to run continuously on
  the Pi (see context-aware-bilstm-edge/iaq_forecast.py's docstring for
  why this was deliberately deferred, not solved).

- apply_door_orientation_fix() reproduces "FIX 15": door_open_sum is
  reversed once at the source. The original comment called this
  "tentative ... verify against a ground-truth per-machine door sensor
  before citing results built on this." That caveat still applies here.
  It has NOT been verified. Do not remove this warning when the caveat
  is eventually resolved one way or the other -- update it instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

MACHINES = [1, 2, 3]
OP_STATES = ["IDLE", "CUTTING", "EXPOSURE", "MAINTENANCE"]
ALL_TARGETS = ["pm1", "pm2_5", "pm10", "co2", "voc"]

# Feature-group toggles, matching the GUI's _feat_vars keys exactly so a
# saved bundle's feat_cfg can be replayed identically. Defaults here match
# the GUI's original defaults EXCEPT the three groups noted above, which
# default to False in this module (the GUI still defaults them per its own
# checkboxes) -- see the module docstring caveat before flipping these True.
DEFAULT_FEATURE_CONFIG = {
    "use_env": True,
    "use_raw_sensors": True,
    "use_pm_lags": True,
    "use_pm_diff": True,
    "use_pm_roll": True,
    "use_voc_diff": True,
    "use_person_diff": True,
    "use_door_sum": True,
    "use_door_diff": True,
    "use_motion_roll": True,
    "use_co2_lags": True,
    "use_emission_wt": True,      # phi_open/rho_open/eps_max/f_trans --
                                   # verified real AND stateless: each is
                                   # recomputed fresh per 60s window by
                                   # compute_Dt(), no cross-window history
                                   # needed. Safe for live edge inference.
    "use_effective_tau": False,   # M{m}_effective_tau -- verified real in
                                   # the training data, but computed by
                                   # merge_vision_and_sensor_data.py's
                                   # enrich_operational_state(), a STATEFUL,
                                   # order-dependent running counter over
                                   # the entire history (consecutive_full_open
                                   # "does not reset across a gap"). Not
                                   # derivable from a short live window
                                   # without porting that whole state
                                   # machine to run continuously on the Pi
                                   # -- deliberately excluded from the edge
                                   # default for that reason, not because
                                   # it's dead. NEW key, no GUI equivalent
                                   # (the GUI bundles it into
                                   # use_emission_wt; this module splits it
                                   # out specifically for the edge case).
    "use_emission_weight": False, # M{m}_emission_weight only -- verified
                                   # dead (constant zero) in the real v1
                                   # data, AND same statefulness problem as
                                   # effective_tau even if it weren't.
                                   # NEW key, no GUI equivalent; defaults
                                   # off so a fresh bundle doesn't silently
                                   # train on constant zeros.
    "use_consecutive": False,     # M{m}_consecutive_full_open -- verified
                                   # dead in this data, AND itself the
                                   # stateful counter effective_tau depends
                                   # on -- same live-inference problem.
    "use_op_state_onehot": False, # M{m}_is_{state} one-hot -- verified
                                   # dead (M{m}_op_state itself is absent
                                   # upstream), same reason.
}


def build_base_columns(
    df: pd.DataFrame,
    machines: list[int] = MACHINES,
    op_states: list[str] = OP_STATES,
) -> pd.DataFrame:
    """
    Adds op-state one-hot columns, then guarantees every "base" column the
    downstream feature builder expects exists -- filling with 0.0 and
    ffill/bfill exactly as the original inline code did. Mutates and
    returns df for convenient chaining.
    """
    df = df.copy()
    for m in machines:
        col = f"M{m}_op_state"
        for s in op_states:
            df[f"M{m}_is_{s}"] = (
                (df[col] == s).astype(float) if col in df.columns else 0.0
            )

    base_cols = (
        ["temp", "hum", "pm1", "pm2_5", "pm10", "co2", "voc"]
        + [f"M{m}_f_trans" for m in machines]
        + [f"M{m}_rho_open" for m in machines]
        + [f"M{m}_eps_max" for m in machines]
        + [f"M{m}_phi_open" for m in machines]
        + [f"M{m}_emission_weight" for m in machines]
        + [f"M{m}_effective_tau" for m in machines]
        + [f"M{m}_consecutive_full_open" for m in machines]
        + [f"M{m}_is_{s}" for m in machines for s in op_states]
        + ["n_person", "mu_motion", "sigma2_motion"]
    )
    for c in base_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[base_cols] = df[base_cols].ffill().bfill().fillna(0.0)
    return df


def apply_door_orientation_fix(
    df: pd.DataFrame,
    machines: list[int] = MACHINES,
    user_door_col: str | None = None,
    fixed_bounds: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, float, float]:
    """
    Builds door_open_sum and applies "FIX 15" -- see module docstring.
    UNVERIFIED against ground truth.

    The reversal is `hi + lo - value`, where hi/lo are the min/max of
    door_open_sum. AT TRAINING TIME these should come from the full
    historical dataset (pass fixed_bounds=None, they're computed here and
    returned so the caller can persist them in the preprocessing bundle).
    AT INFERENCE TIME you MUST pass the training-time bounds back in via
    fixed_bounds -- recomputing min/max from whatever small rolling window
    you're running inference on gives a different, unstable range than
    training saw, silently corrupting every door-derived feature (this was
    caught during the edge-deployment work, not present in the original
    GUI code, which never needed an inference-time code path at all).

    Returns (df, lo, hi) -- lo/hi are either freshly computed or the
    fixed_bounds passed in, so the caller always has a value to persist
    or has confirmation of what was actually used.
    """
    df = df.copy()
    door_cols_avail = [
        f"M{m}_phi_open" for m in machines if f"M{m}_phi_open" in df.columns
    ]
    if user_door_col and user_door_col in df.columns:
        df["door_open_sum"] = df[user_door_col]
    else:
        df["door_open_sum"] = (
            df[door_cols_avail].sum(axis=1) if door_cols_avail else 0.0
        )

    if fixed_bounds is not None:
        lo, hi = fixed_bounds
    else:
        lo = float(df["door_open_sum"].min())
        hi = float(df["door_open_sum"].max())
    df["door_open_sum"] = hi + lo - df["door_open_sum"]

    return df, lo, hi


def add_engineered_columns(
    df: pd.DataFrame,
    person_col: str = "n_person",
    motion_col: str = "mu_motion",
) -> pd.DataFrame:
    """
    FIX 1 (PM lags/diff/roll) + FIX 5 (trigger/door/motion features).
    Requires build_base_columns() and apply_door_orientation_fix() to have
    already run (needs pm1/pm2_5/pm10/co2/voc/door_open_sum present).
    Mutates and returns df.
    """
    df = df.copy()

    df["pm1_diff1"] = df["pm1"].diff(1).bfill()
    df["pm25_diff1"] = df["pm2_5"].diff(1).bfill()
    df["pm10_diff1"] = df["pm10"].diff(1).bfill()
    df["pm1_lag1"] = df["pm1"].shift(1).bfill()
    df["pm1_lag2"] = df["pm1"].shift(2).bfill()
    df["pm1_lag3"] = df["pm1"].shift(3).bfill()
    df["pm25_lag1"] = df["pm2_5"].shift(1).bfill()
    df["pm25_lag2"] = df["pm2_5"].shift(2).bfill()
    df["pm10_lag1"] = df["pm10"].shift(1).bfill()
    df["pm10_lag2"] = df["pm10"].shift(2).bfill()
    df["pm_total"] = df["pm1"] + df["pm2_5"] + df["pm10"]
    df["pm1_roll5"] = df["pm1"].rolling(5, min_periods=1).mean()
    df["pm25_roll5"] = df["pm2_5"].rolling(5, min_periods=1).mean()
    df["pm10_roll5"] = df["pm10"].rolling(5, min_periods=1).mean()

    df["voc_diff1"] = df["voc"].diff(1).bfill()
    df["voc_diff2"] = df["voc"].diff(2).bfill()
    df["co2_lag1"] = df["co2"].shift(1).bfill()
    df["co2_lag2"] = df["co2"].shift(2).bfill()
    df["co2_lag3"] = df["co2"].shift(3).bfill()
    df["co2_roll5"] = df["co2"].rolling(5, min_periods=1).mean()

    df["person_diff"] = df[person_col].diff(1).bfill()
    df["door_diff"] = df["door_open_sum"].diff(1).bfill()
    df["motion_roll10"] = df[motion_col].rolling(10, min_periods=1).mean()
    df["door_exposure"] = (
        (df["door_open_sum"] > 0).astype(float).rolling(10, min_periods=1).sum()
    )
    df["trigger_strength"] = df[person_col].clip(0) * df[motion_col].clip(0)

    return df


def select_feature_columns(
    df: pd.DataFrame,
    feat_cfg: dict,
    machines: list[int] = MACHINES,
    op_states: list[str] = OP_STATES,
    person_col: str = "n_person",
    motion_col: str = "mu_motion",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Applies feat_cfg toggles to build the final feat_cols list, exactly
    matching the GUI's SELECT FEATURES block (dedup, fallback list, fill).
    Returns (df_with_filled_features, feat_cols). df must already have
    build_base_columns / apply_door_orientation_fix / add_engineered_columns
    applied.
    """
    df = df.copy()
    feat_cols: list[str] = []

    if feat_cfg.get("use_env"):
        feat_cols += ["temp", "hum"]
    if feat_cfg.get("use_raw_sensors"):
        feat_cols += ["pm1", "pm2_5", "pm10", "co2", "voc"]
    if feat_cfg.get("use_pm_lags"):
        feat_cols += ["pm1_lag1", "pm1_lag2", "pm1_lag3",
                      "pm25_lag1", "pm25_lag2", "pm10_lag1", "pm10_lag2"]
    if feat_cfg.get("use_pm_diff"):
        feat_cols += ["pm1_diff1", "pm25_diff1", "pm10_diff1", "pm_total"]
    if feat_cfg.get("use_pm_roll"):
        feat_cols += ["pm1_roll5", "pm25_roll5", "pm10_roll5"]
    if feat_cfg.get("use_voc_diff"):
        feat_cols += ["voc_diff1", "voc_diff2"]
    if feat_cfg.get("use_person_diff"):
        feat_cols += [person_col, "person_diff"]
    if feat_cfg.get("use_door_sum"):
        feat_cols += ["door_open_sum", "door_exposure"]
    if feat_cfg.get("use_door_diff"):
        feat_cols += ["door_diff"]
    if feat_cfg.get("use_motion_roll"):
        feat_cols += [motion_col, "motion_roll10", "trigger_strength"]
    if feat_cfg.get("use_co2_lags"):
        feat_cols += ["co2_lag1", "co2_lag2", "co2_lag3", "co2_roll5"]
    if feat_cfg.get("use_emission_wt"):
        # Verified real AND stateless (see module docstring) -- deliberately
        # excludes emission_weight and effective_tau, unlike the original
        # GUI's single checkbox.
        feat_cols += (
            [f"M{m}_phi_open" for m in machines]
            + [f"M{m}_rho_open" for m in machines]
            + [f"M{m}_eps_max" for m in machines]
            + [f"M{m}_f_trans" for m in machines]
        )
    if feat_cfg.get("use_effective_tau"):
        # Verified real but STATEFUL -- see module docstring. Off by
        # default for the edge config; fine to enable for offline/research
        # training runs that don't need to reproduce this feature live.
        feat_cols += [f"M{m}_effective_tau" for m in machines]
    if feat_cfg.get("use_emission_weight"):
        # Verified dead (constant zero) in the real v1 data -- separate,
        # off-by-default toggle. See module docstring.
        feat_cols += [f"M{m}_emission_weight" for m in machines]
    if feat_cfg.get("use_consecutive"):
        feat_cols += [f"M{m}_consecutive_full_open" for m in machines]
    if feat_cfg.get("use_op_state_onehot"):
        feat_cols += [f"M{m}_is_{s}" for m in machines for s in op_states]

    seen: set[str] = set()
    deduped: list[str] = []
    for c in feat_cols:
        if c in df.columns and c not in seen:
            seen.add(c)
            deduped.append(c)
    feat_cols = deduped or ["pm1_lag1", "pm25_lag1", "voc_diff1",
                             "door_open_sum", "person_diff", "motion_roll10"]

    df[feat_cols] = df[feat_cols].ffill().bfill().fillna(0.0)
    return df, feat_cols


def make_sequences(
    Xs: np.ndarray, ys: np.ndarray, lookback: int, lead: int
) -> tuple[np.ndarray, np.ndarray]:
    """X[t:t+lookback] -> y[t+lookback+lead-1]. Identical to the GUI's make_seqs()."""
    Xo, yo = [], []
    for i in range(len(Xs) - lookback - lead + 1):
        Xo.append(Xs[i : i + lookback])
        yo.append(ys[i + lookback + lead - 1])
    return (
        np.array(Xo, dtype=np.float32),
        np.array(yo, dtype=np.float32),
    )


def fit_scalers(
    X_raw: np.ndarray, y_raw: np.ndarray, train_end: int,
    all_targets: list[str] = ALL_TARGETS,
) -> tuple[StandardScaler, dict[str, MinMaxScaler]]:
    """Fits on the train split only (FIX 4 -- no leakage)."""
    x_scaler = StandardScaler()
    x_scaler.fit(X_raw[:train_end])

    y_scalers: dict[str, MinMaxScaler] = {}
    for i, t in enumerate(all_targets):
        sc = MinMaxScaler()
        sc.fit(y_raw[:train_end, i].reshape(-1, 1))
        y_scalers[t] = sc
    return x_scaler, y_scalers


def apply_scalers(
    X_raw: np.ndarray, y_raw: np.ndarray,
    x_scaler: StandardScaler, y_scalers: dict[str, MinMaxScaler],
    all_targets: list[str] = ALL_TARGETS,
) -> tuple[np.ndarray, np.ndarray]:
    Xsc = x_scaler.transform(X_raw).astype(np.float32)
    cols = [
        y_scalers[t].transform(y_raw[:, i].reshape(-1, 1))
        for i, t in enumerate(all_targets)
    ]
    ysc = np.hstack(cols).astype(np.float32)
    return Xsc, ysc


def inverse_transform_targets(
    arr: np.ndarray, y_scalers: dict[str, MinMaxScaler],
    all_targets: list[str] = ALL_TARGETS,
) -> np.ndarray:
    out = np.zeros_like(arr)
    for i, t in enumerate(all_targets):
        out[:, i] = y_scalers[t].inverse_transform(arr[:, i].reshape(-1, 1)).ravel()
    return out


def save_bundle(
    bundle_dir: str | Path,
    x_scaler: StandardScaler,
    y_scalers: dict[str, MinMaxScaler],
    feat_cols: list[str],
    feat_cfg: dict,
    lookback: int,
    lead: int,
    door_lo: float,
    door_hi: float,
    person_col: str = "n_person",
    motion_col: str = "mu_motion",
    door_col: str | None = None,
    all_targets: list[str] = ALL_TARGETS,
    machines: list[int] = MACHINES,
    op_states: list[str] = OP_STATES,
    extra_metadata: dict | None = None,
) -> None:
    """
    Writes everything needed to reproduce this exact preprocessing at
    inference time, closing the "scalers were never persisted" gap.
    Layout:
        bundle_dir/
            x_scaler.joblib
            y_scalers.joblib
            manifest.json   -- feat_cols, feat_cfg, lookback, lead,
                                person_col, motion_col, door_col, door_lo,
                                door_hi, all_targets, machines, op_states

    door_lo/door_hi: the door_open_sum min/max computed during THIS
    training run (from build_features_for_training's return values) --
    required at inference time to reproduce the door-orientation reversal
    correctly. See apply_door_orientation_fix()'s docstring.
    """
    import joblib

    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(x_scaler, bundle_dir / "x_scaler.joblib")
    joblib.dump(y_scalers, bundle_dir / "y_scalers.joblib")

    manifest = {
        "feat_cols": feat_cols,
        "feat_cfg": feat_cfg,
        "lookback": lookback,
        "lead": lead,
        "person_col": person_col,
        "motion_col": motion_col,
        "door_col": door_col,
        "door_lo": door_lo,
        "door_hi": door_hi,
        "all_targets": all_targets,
        "machines": machines,
        "op_states": op_states,
    }
    if extra_metadata:
        manifest["extra"] = extra_metadata
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_bundle(bundle_dir: str | Path) -> dict:
    """Inverse of save_bundle(). Returns a dict with keys x_scaler, y_scalers,
    plus everything from manifest.json."""
    import joblib

    bundle_dir = Path(bundle_dir)
    x_scaler = joblib.load(bundle_dir / "x_scaler.joblib")
    y_scalers = joblib.load(bundle_dir / "y_scalers.joblib")
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    manifest["x_scaler"] = x_scaler
    manifest["y_scalers"] = y_scalers
    return manifest


def build_features_for_training(
    df: pd.DataFrame,
    feat_cfg: dict,
    person_col_cfg: str | None = None,
    motion_col_cfg: str | None = None,
    door_col_cfg: str | None = None,
    machines: list[int] = MACHINES,
    op_states: list[str] = OP_STATES,
    door_bounds: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, list[str], str, str, float, float]:
    """
    Convenience wrapper chaining all four steps in the same order the
    original inline GUI code ran them. Returns (df, feat_cols, person_col,
    motion_col, door_lo, door_hi) for downstream use in
    make_sequences()/fit_scalers()/save_bundle().

    Pass door_bounds=None at TRAINING time (bounds are computed fresh from
    the full historical df and returned for you to persist via
    save_bundle). Pass door_bounds=(saved_lo, saved_hi) at INFERENCE time
    -- see apply_door_orientation_fix()'s docstring for why reusing the
    training-time bounds is required, not optional, for correct live
    predictions.
    """
    df = build_base_columns(df, machines=machines, op_states=op_states)
    df, door_lo, door_hi = apply_door_orientation_fix(
        df, machines=machines, user_door_col=door_col_cfg, fixed_bounds=door_bounds,
    )

    person_col = person_col_cfg if person_col_cfg and person_col_cfg in df.columns else "n_person"
    motion_col = motion_col_cfg if motion_col_cfg and motion_col_cfg in df.columns else "mu_motion"

    df = add_engineered_columns(df, person_col=person_col, motion_col=motion_col)
    df, feat_cols = select_feature_columns(
        df, feat_cfg, machines=machines, op_states=op_states,
        person_col=person_col, motion_col=motion_col,
    )
    return df, feat_cols, person_col, motion_col, door_lo, door_hi
