"""
Granger Causality: Does C_t Actually Precede the Pollutants It's Meant to Explain?
======================================================================================
The pipeline's own causality_lag_analysis.csv has always shipped with its
granger_best_lag / granger_min_pvalue columns blank (verified still true
for the v1 run) -- FIX 11 made the pipeline log *why* Granger fails instead
of silently returning nothing, but whatever the runtime issue is, it has
never actually produced a number in any run tracked in this project. This
script computes Granger causality independently, directly from the raw
per-minute session file the pipeline itself is built from
(data_input/sensor_data_merged_iaq_m2.csv, 386 rows, 1-minute cadence),
so it has no dependency on the GUI pipeline's environment at all.

WHAT GRANGER CAUSALITY ACTUALLY TESTS (read this before trusting a p-value)
Granger causality does NOT test physical causation. It tests a narrower,
purely predictive claim: "do past values of X reduce the error of a linear
model forecasting Y, beyond what Y's own past already gives you?" A
significant result means X has predictive precedence over Y in a linear,
time-lagged sense -- it does not rule out both being driven by a third,
unmeasured factor, and it says nothing about mechanism. Here it is used
for exactly what it's suited to: corroborating (or falsifying) the
directional, lagged story the rest of this project's evidence already
points to -- door/motion preceding pollutant changes -- with a second,
independent statistical method (VAR-based, not just cross-correlation).

FOUR SCENARIOS, EACH ANSWERING A DIFFERENT QUESTION
  1. FORWARD, FULL SESSION: does each C_t descriptor Granger-cause each
     pollutant, pooling the whole session? The baseline question -- also
     what causality_lag_analysis.csv attempted and never completed.
  2. REVERSE, FULL SESSION (placebo/falsification control): does each
     pollutant Granger-cause the C_t descriptor? If the reverse direction
     is as significant as the forward one, that's a red flag for a shared
     confound or feedback loop, not clean forward precedence -- a result
     this project should actively look for, not assume away.
  3. PHASE-STRATIFIED FORWARD: the same forward test, run separately
     within each of the 4 operational phases. Pooling the full session
     can hide a relationship that only holds during, say, Fabrication --
     the same masking concern that motivates every other phase-split
     analysis in this project.
  4. STATIONARITY-AWARE: Granger causality assumes stationary input series;
     several of these pollutant trends drift monotonically within a phase
     (see the phase_wise_calculation trace plots). Each series is checked
     with an Augmented Dickey-Fuller test and first-differenced if
     non-stationary before testing -- skipping this step is the single
     most common way to get a spurious "significant" Granger result on
     trending environmental time series.

HOW TO APPLY THIS INTELLIGENTLY (not just mechanically)
  - Multiple lags are tested (1..maxlag) per pair; the MINIMUM p-value
    across lags is reported as the summary, but this is a soft, exploratory
    summary, not a clean single hypothesis test -- scanning several lags
    and keeping the best one inflates the false-positive rate on its own,
    on top of testing many (feature, pollutant) pairs. Benjamini-Hochberg
    FDR correction is applied across the full grid of pairs actually
    tested (not across lags within a pair) to control that.
  - Small phase samples (n as low as ~30-60 minutes) cannot support a
    maxlag of 10 in a VAR model (each lag costs 2 parameters per equation;
    the model becomes unidentified as maxlag approaches n/5 or so).
    maxlag is capped adaptively per phase.
  - door_open_sum is rebuilt here from M1/M2/M3 phi_open with the same
    orientation correction as FIX 15 in iaq_early_detection_gui_v3.py, for
    consistency with the rest of this project -- see analysis/v1/CLAUDE.md.

Usage:
    python granger_causality_analysis.py --raw <sensor_data_merged_iaq_m2.csv> --outdir <data_folder>
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

warnings.filterwarnings("ignore")

CT_FEATURES = ["door_open_sum", "mu_motion", "M1_rho_open", "M2_rho_open", "M3_rho_open"]
TARGETS = ["pm1", "pm2_5", "pm10", "co2", "voc"]
PHASES = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
MAXLAG_FULL = 10


def assign_phase(ts):
    ts = pd.to_datetime(ts)
    day = ts.min().normalize()
    b1400 = day + pd.Timedelta(hours=14)
    b1530 = day + pd.Timedelta(hours=15, minutes=30)
    b1630 = day + pd.Timedelta(hours=16, minutes=30)
    return np.where(ts <= b1400, "Phase 1",
           np.where(ts <= b1530, "Phase 2",
           np.where(ts <= b1630, "Phase 3", "Phase 4")))


def load_raw(path):
    df = pd.read_csv(path)
    df["timestamp_minute"] = pd.to_datetime(df["timestamp_minute"], format="%m-%d-%Y %H:%M")
    df = df.sort_values("timestamp_minute").reset_index(drop=True)

    door_sum = df[["M1_phi_open", "M2_phi_open", "M3_phi_open"]].sum(axis=1)
    lo, hi = door_sum.min(), door_sum.max()
    df["door_open_sum"] = hi + lo - door_sum  # FIX-15-consistent orientation

    df["phase"] = assign_phase(df["timestamp_minute"])
    return df


def maybe_difference(series):
    """ADF test; first-difference once if non-stationary (p > 0.05). Returns
    (series_to_use, was_differenced)."""
    s = series.dropna().values
    if len(s) < 8 or np.std(s) < 1e-9:
        return series, False
    try:
        p = adfuller(s, autolag="AIC")[1]
    except Exception:
        return series, False
    if p <= 0.05:
        return series, False
    return series.diff().dropna(), True


def granger_pair(y, x, maxlag):
    """Does x Granger-cause y? Returns (best_lag, min_pvalue) using the
    ssr F-test, or (nan, nan) if the test can't run (too few points,
    singular matrix, etc.)."""
    y_s, y_diff = maybe_difference(y)
    x_s, x_diff = maybe_difference(x)
    data = pd.concat([y_s, x_s], axis=1).dropna()
    n = len(data)
    eff_maxlag = min(maxlag, max(1, n // 5 - 1))
    if n < 12 or eff_maxlag < 1:
        return np.nan, np.nan, y_diff or x_diff, n
    try:
        res = grangercausalitytests(data.values, maxlag=eff_maxlag, verbose=False)
    except Exception:
        return np.nan, np.nan, y_diff or x_diff, n
    pvals = {lag: res[lag][0]["ssr_ftest"][1] for lag in res}
    best_lag = min(pvals, key=pvals.get)
    return best_lag, pvals[best_lag], (y_diff or x_diff), n


def bh_fdr(pvals):
    pvals = np.asarray(pvals, float)
    out = np.full(pvals.shape, np.nan)
    valid = ~np.isnan(pvals)
    p_valid = pvals[valid]
    m = len(p_valid)
    if m == 0:
        return out
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out_valid = np.empty(m)
    out_valid[order] = adj
    out[valid] = out_valid
    return out


def scenario_forward_reverse(df):
    rows = []
    for feat in CT_FEATURES:
        for tgt in TARGETS:
            f_lag, f_p, f_diff, f_n = granger_pair(df[tgt], df[feat], MAXLAG_FULL)
            r_lag, r_p, r_diff, r_n = granger_pair(df[feat], df[tgt], MAXLAG_FULL)
            rows.append({
                "ct_feature": feat, "pollutant": tgt, "n": f_n,
                "forward_best_lag": f_lag, "forward_p": f_p,
                "reverse_best_lag": r_lag, "reverse_p": r_p,
                "differenced_for_stationarity": f_diff,
            })
    out = pd.DataFrame(rows)
    out["forward_q"] = bh_fdr(out["forward_p"].values)
    out["reverse_q"] = bh_fdr(out["reverse_p"].values)
    out["forward_sig_fdr05"] = out["forward_q"] < 0.05
    out["reverse_sig_fdr05"] = out["reverse_q"] < 0.05
    out["placebo_flag"] = out["reverse_sig_fdr05"] & out["forward_sig_fdr05"]
    return out


def scenario_phase_stratified(df):
    rows = []
    for phase in PHASES:
        sub = df[df.phase == phase]
        n_phase = len(sub)
        maxlag = max(1, min(MAXLAG_FULL, n_phase // 6))
        for feat in ["door_open_sum", "mu_motion"]:
            for tgt in TARGETS:
                lag, p, diffed, n = granger_pair(sub[tgt], sub[feat], maxlag)
                rows.append({"phase": phase, "ct_feature": feat, "pollutant": tgt,
                             "n": n, "maxlag_used": maxlag, "best_lag": lag, "p": p,
                             "differenced_for_stationarity": diffed})
    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out["p"].values)
    out["sig_fdr05"] = out["q"] < 0.05
    return out


def make_heatmap(fwd_rev, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Sequential blue ramp (light->dark), per this project's dataviz palette.
    blue_steps = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", blue_steps, N=256)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    })

    piv_p = fwd_rev.pivot(index="ct_feature", columns="pollutant", values="forward_p").reindex(
        index=CT_FEATURES, columns=TARGETS)
    piv_sig = fwd_rev.pivot(index="ct_feature", columns="pollutant", values="forward_sig_fdr05").reindex(
        index=CT_FEATURES, columns=TARGETS)
    piv_lag = fwd_rev.pivot(index="ct_feature", columns="pollutant", values="forward_best_lag").reindex(
        index=CT_FEATURES, columns=TARGETS)

    neglogp = -np.log10(piv_p.values.astype(float).clip(min=1e-12))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(neglogp, cmap=cmap, aspect="auto", vmin=0, vmax=max(4, np.nanmax(neglogp)))
    for i in range(neglogp.shape[0]):
        for j in range(neglogp.shape[1]):
            p = piv_p.values[i, j]
            if np.isnan(p):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5, color="#999999")
                continue
            sig = bool(piv_sig.values[i, j])
            lag = piv_lag.values[i, j]
            txt_color = "white" if neglogp[i, j] > (np.nanmax(neglogp) * 0.55) else "#1a1a1a"
            marker = " *" if sig else ""
            ax.text(j, i - 0.13, f"p={p:.3f}{marker}", ha="center", va="center",
                     fontsize=7.6, color=txt_color, fontweight="bold" if sig else "normal")
            ax.text(j, i + 0.18, f"lag={int(lag)}min", ha="center", va="center",
                     fontsize=6.6, color=txt_color, alpha=0.85)

    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels(["PM1", "PM2.5", "PM10", "CO2", "VOC"], fontsize=9)
    ax.set_yticks(range(len(CT_FEATURES)))
    ax.set_yticklabels(CT_FEATURES, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(TARGETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CT_FEATURES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Does each C$_t$ descriptor Granger-cause each pollutant?\n"
                  "(full session, best lag over 1-10 min) · * = FDR-significant (BH, q<0.05, across all 25 pairs)",
                  fontsize=9.5, fontweight="bold", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("-log$_{10}$(p)  (darker = stronger evidence)", fontsize=8)
    plt.tight_layout()
    out_png = os.path.join(outdir, "GRANGER_CAUSALITY_FORWARD.png")
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="Path to sensor_data_merged_iaq_m2.csv (or equivalent raw per-minute file)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    fig_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df = load_raw(args.raw)
    print(f"Loaded raw session: {len(df)} rows, {df['timestamp_minute'].min()} to {df['timestamp_minute'].max()}")

    fwd_rev = scenario_forward_reverse(df)
    fwd_rev_fp = os.path.join(outdir, "granger_causality_forward_reverse.csv")
    fwd_rev.round(4).to_csv(fwd_rev_fp, index=False)
    print(f"\n=== Forward + reverse (placebo) Granger causality, full session ===")
    with pd.option_context("display.width", 160):
        print(fwd_rev.round(3).to_string(index=False))
    n_fwd_sig = int(fwd_rev["forward_sig_fdr05"].sum())
    n_placebo = int(fwd_rev["placebo_flag"].sum())
    print(f"\n{n_fwd_sig}/{len(fwd_rev)} forward pairs FDR-significant.")
    print(f"{n_placebo}/{len(fwd_rev)} pairs ALSO significant in reverse (placebo flag -- interpret forward direction cautiously for these).")
    print(f"Saved: {fwd_rev_fp}")

    phase_out = scenario_phase_stratified(df)
    phase_fp = os.path.join(outdir, "granger_causality_phase_stratified.csv")
    phase_out.round(4).to_csv(phase_fp, index=False)
    print(f"\n=== Phase-stratified forward Granger causality (door_open_sum, mu_motion only) ===")
    with pd.option_context("display.width", 160):
        print(phase_out.round(3).to_string(index=False))
    print(f"Saved: {phase_fp}")

    png = make_heatmap(fwd_rev, fig_dir)
    print(f"\nSaved: {png}")


if __name__ == "__main__":
    main()
