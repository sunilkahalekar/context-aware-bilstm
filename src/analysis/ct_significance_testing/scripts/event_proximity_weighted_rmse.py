"""
Event-Proximity-Weighted Accuracy Analysis
===========================================
Generalizes the binary onset/baseline regime split (used elsewhere in the
validation pipeline) into a continuous, smoothly-decaying weight based on
each test-window minute's distance to the nearest REAL trigger event
(door-open, motion-rise, or cutting-start -- labelled from raw sensor
state, never from a model's own output).

Why this exists
----------------
The onset/baseline split uses a hard cutoff (e.g. 15 minutes after a
trigger = "onset", everything else = "baseline"). That is an arbitrary
researcher degree of freedom: a minute 1 minute after a door-open and a
minute 14 minutes after are treated identically, while a minute at 16
minutes is treated as if nothing happened at all. This script replaces
the step function with a smooth exponential-decay weight

    weight(t) = exp( -|t - nearest_trigger| / tau )

and reports RMSE (and its with-Ct vs without-Ct gap) as a continuous
function of tau. As tau -> infinity every minute gets equal weight and
the curve must converge to the plain, unconditioned pooled RMSE gap; as
tau -> 0 only minutes essentially on top of a trigger event count, and
the curve should show the with/without-Ct gap at its most extreme if the
"Ct matters most near transitions" hypothesis is correct.

Because RMSE is not comparable across pollutants of different physical
scale (ppm CO2 vs ppb VOC vs ug/m3 PM), this script computes:
  1. a raw, per-(model, target) weighted RMSE table (rigorous, no unit
     mixing) -- event_proximity_weighted_rmse_raw_lead{N}.csv
  2. a normalized version (RMSE / observed range x 100, same convention
     already used elsewhere in this project) that can be safely averaged
     across pollutants and models for a single summary curve per model
     -- event_proximity_weighted_rmse_normalized_lead{N}.csv
  3. two figures: per-model curves (raw + normalized), and a pooled
     "gap vs tau" summary curve.

Inputs required (all produced by iaq_early_detection_gui_v2.py's
rigorous-validation module -- FIX 13 -- run with the GUI's "Rigorous
Validation" section enabled):
    predictions_lead{N}_v2.csv       with-Ct per-minute predictions
    predictions_noct_lead{N}.csv     without-Ct per-minute predictions
    trigger_events_lead{N}.csv       real door/motion/cutting trigger times

If predictions_noct_lead{N}.csv or trigger_events_lead{N}.csv are
missing (i.e. you are pointing this at an older run's output folder that
pre-dates FIX 13), the script still runs in with-Ct-only mode so you can
sanity-check the weighting mechanics, but it cannot compute a gap -- it
will say so explicitly rather than silently producing a one-sided result.

Usage
-----
    python event_proximity_weighted_rmse.py --dir <results_folder> --lead 10

Optional:
    --tau 1,2,3,5,8,12,15,20,30,45,60,90,120     (minutes; comma-separated)
    --models BiLSTM,LSTM_uni,GRU                 (default: all deep models present)
    --outdir <folder>                             (default: same as --dir)
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ALL_TARGETS = ["pm1", "pm2_5", "pm10", "co2", "voc"]
DEEP_MODELS = ["BiGRU", "BiLSTM", "GRU", "LSTM_uni", "VanillaRNN",
                "Seq2Seq", "CNN_LSTM"]
DEFAULT_TAUS = [1, 2, 3, 5, 8, 12, 15, 20, 30, 45, 60, 90, 120]


# ══════════════════════════════════════════════════════════════════════════
# CORE MATH
# ══════════════════════════════════════════════════════════════════════════
def minutes_to_nearest_trigger(timestamps, trigger_timestamps):
    """
    For each entry in `timestamps`, the absolute distance (in minutes) to
    the nearest entry in `trigger_timestamps`. Vectorized via a sorted
    searchsorted lookup rather than a full pairwise matrix, so this stays
    fast even with a large number of trigger events.
    """
    ts = pd.to_datetime(pd.Series(timestamps)).astype("int64").to_numpy() // 60_000_000_000
    tg = pd.to_datetime(pd.Series(trigger_timestamps)).astype("int64").to_numpy() // 60_000_000_000
    if len(tg) == 0:
        return np.full(len(ts), np.inf)
    tg = np.sort(tg)
    idx = np.searchsorted(tg, ts)
    idx_lo = np.clip(idx - 1, 0, len(tg) - 1)
    idx_hi = np.clip(idx, 0, len(tg) - 1)
    d_lo = np.abs(ts - tg[idx_lo])
    d_hi = np.abs(ts - tg[idx_hi])
    return np.minimum(d_lo, d_hi).astype(float)


def weighted_rmse(actual, predicted, distance_min, tau):
    """
    Exponential-decay-weighted RMSE. weight(t) = exp(-distance/tau), so
    minutes right next to a trigger event dominate at small tau, and all
    minutes converge to equal weight as tau grows large.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    w = np.exp(-distance_min / tau)
    if w.sum() <= 0:
        return np.nan
    wmse = np.sum(w * (actual - predicted) ** 2) / np.sum(w)
    return float(np.sqrt(wmse))


def effective_sample_size(distance_min, tau):
    """Kish effective sample size for the weight vector at this tau --
    reported alongside the RMSE so a reader can see when a small-tau
    estimate is only resting on a handful of effectively-weighted minutes
    and should be treated cautiously."""
    w = np.exp(-distance_min / tau)
    return float((w.sum() ** 2) / np.sum(w ** 2)) if w.sum() > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════
def load_inputs(indir, lead):
    with_path = os.path.join(indir, f"predictions_lead{lead}_v2.csv")
    noct_path = os.path.join(indir, f"predictions_noct_lead{lead}.csv")
    trig_path = os.path.join(indir, f"trigger_events_lead{lead}.csv")

    if not os.path.exists(with_path):
        sys.exit(f"ERROR: required file not found: {with_path}")

    pred_with = pd.read_csv(with_path)
    pred_with["future_timestamp"] = pd.to_datetime(pred_with["future_timestamp"])

    pred_noct = None
    if os.path.exists(noct_path):
        pred_noct = pd.read_csv(noct_path)
        pred_noct["future_timestamp"] = pd.to_datetime(pred_noct["future_timestamp"])
    else:
        print(f"  [!] {os.path.basename(noct_path)} not found -- running in "
              f"WITH-CT-ONLY mode. Re-run the GUI pipeline (FIX 13 or later) "
              f"to get the without-Ct comparison and the gap-vs-tau curve.")

    trigger_ts = None
    if os.path.exists(trig_path):
        trigger_ts = pd.to_datetime(pd.read_csv(trig_path)["trigger_timestamp"])
    else:
        print(f"  [!] {os.path.basename(trig_path)} not found -- cannot compute "
              f"event proximity at all without real trigger timestamps. "
              f"Re-run the GUI pipeline (FIX 13 or later) to generate it.")

    return pred_with, pred_noct, trigger_ts


def observed_ranges(pred_with):
    """Range (max-min) of the ACTUAL series per target, over the whole
    test window -- used only to build the normalized (% of range) view;
    identical across models/variants since it's ground truth."""
    ranges = {}
    for t in ALL_TARGETS:
        sub = pred_with[pred_with.target == t]["actual"]
        if len(sub):
            ranges[t] = float(sub.max() - sub.min())
    return ranges


# ══════════════════════════════════════════════════════════════════════════
# MAIN COMPUTATION
# ══════════════════════════════════════════════════════════════════════════
def compute_curves(pred_with, pred_noct, trigger_ts, models, taus):
    ranges = observed_ranges(pred_with)
    raw_rows, norm_rows = [], []

    for m in models:
        w_all = pred_with[pred_with.model == m]
        n_all = pred_noct[pred_noct.model == m] if pred_noct is not None else None
        if len(w_all) == 0:
            print(f"  [!] no with-Ct rows for model '{m}' -- skipping.")
            continue

        for t in ALL_TARGETS:
            w_sub = w_all[w_all.target == t]
            if len(w_sub) == 0 or t not in ranges or ranges[t] <= 0:
                continue
            dist_w = (minutes_to_nearest_trigger(w_sub.future_timestamp, trigger_ts)
                      if trigger_ts is not None else np.zeros(len(w_sub)))

            n_sub = None
            dist_n = None
            if n_all is not None:
                n_sub = n_all[n_all.target == t]
                if len(n_sub):
                    dist_n = (minutes_to_nearest_trigger(n_sub.future_timestamp, trigger_ts)
                               if trigger_ts is not None else np.zeros(len(n_sub)))

            for tau in taus:
                rw = weighted_rmse(w_sub.actual.values, w_sub.predicted.values, dist_w, tau)
                ess_w = effective_sample_size(dist_w, tau)
                rn = ess_n = gap_abs = gap_pct = np.nan
                if n_sub is not None and len(n_sub):
                    rn = weighted_rmse(n_sub.actual.values, n_sub.predicted.values, dist_n, tau)
                    ess_n = effective_sample_size(dist_n, tau)
                    if rw and rw > 0:
                        gap_abs = rn - rw
                        gap_pct = 100.0 * gap_abs / rw

                raw_rows.append({
                    "model": m, "target": t, "tau_min": tau,
                    "rmse_with_Ct": round(rw, 4) if pd.notna(rw) else None,
                    "rmse_without_Ct": round(rn, 4) if pd.notna(rn) else None,
                    "gap_abs": round(gap_abs, 4) if pd.notna(gap_abs) else None,
                    "gap_pct": round(gap_pct, 2) if pd.notna(gap_pct) else None,
                    "eff_n_with_Ct": round(ess_w, 1),
                    "eff_n_without_Ct": round(ess_n, 1) if pd.notna(ess_n) else None,
                })
                norm_rows.append({
                    "model": m, "target": t, "tau_min": tau,
                    "norm_rmse_with_Ct": round(100 * rw / ranges[t], 2) if pd.notna(rw) else None,
                    "norm_rmse_without_Ct": round(100 * rn / ranges[t], 2) if pd.notna(rn) else None,
                    "gap_pts": round(100 * gap_abs / ranges[t], 2) if pd.notna(gap_abs) else None,
                })

    return pd.DataFrame(raw_rows), pd.DataFrame(norm_rows)


# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════
def make_plots(norm_df, outdir, lead, has_noct):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    })
    WITH_C, WITHOUT = "#1B5E7A", "#C1440E"

    # normalized RMSE (% of range), averaged across all 5 targets, per model
    agg = norm_df.groupby(["model", "tau_min"], as_index=False).mean(numeric_only=True)
    models = sorted(agg.model.unique())
    ncols = min(4, len(models)) or 1
    nrows = int(np.ceil(len(models) / ncols)) if models else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.9 * nrows), squeeze=False)
    axes = axes.ravel()
    for i, m in enumerate(models):
        ax = axes[i]
        sub = agg[agg.model == m].sort_values("tau_min")
        ax.plot(sub.tau_min, sub.norm_rmse_with_Ct, color=WITH_C, marker="o",
                markersize=3, lw=1.6, label="with C$_t$")
        if has_noct:
            ax.plot(sub.tau_min, sub.norm_rmse_without_Ct, color=WITHOUT, marker="o",
                    markersize=3, lw=1.6, label="without C$_t$")
        ax.set_xscale("log")
        ax.set_title(m, fontsize=9, fontweight="bold", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, which="both", color="#DDDDDD", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        if i % ncols == 0:
            ax.set_ylabel("Norm. RMSE (%)", fontsize=8)
        if i >= len(models) - ncols:
            ax.set_xlabel("τ (min, log scale)", fontsize=8)
    for j in range(len(models), len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("Event-proximity-weighted normalized RMSE vs. τ, per model\n"
                 "(smaller τ = weight concentrated near real trigger events)",
                 fontsize=10.5, y=1.02)
    plt.tight_layout()
    p1 = os.path.join(outdir, f"EVENT_PROXIMITY_WEIGHTED_RMSE_lead{lead}.png")
    plt.savefig(p1, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  saved {p1}")

    if not has_noct:
        return

    # pooled gap-vs-tau summary (single curve, averaged across models+targets)
    pooled = norm_df.groupby("tau_min", as_index=False)["gap_pts"].agg(["mean", "std", "count"])
    pooled = pooled.reset_index().sort_values("tau_min")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(pooled.tau_min, pooled["mean"], color=WITH_C, marker="o", markersize=4, lw=2)
    ax.fill_between(pooled.tau_min,
                     pooled["mean"] - pooled["std"], pooled["mean"] + pooled["std"],
                     color=WITH_C, alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("τ (minutes, log scale) — decay constant of the event-proximity weight")
    ax.set_ylabel("Mean normalized-RMSE gap (pts)\nwithout C$_t$ − with C$_t$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, which="both", color="#DDDDDD", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Pooled event-proximity-weighted C$_t$ gap vs. τ\n"
                 "(shaded band = ±1 s.d. across models × pollutants)", fontsize=10)
    plt.tight_layout()
    p2 = os.path.join(outdir, f"EVENT_PROXIMITY_GAP_SUMMARY_lead{lead}.png")
    plt.savefig(p2, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  saved {p2}")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="Folder with the pipeline's output CSVs")
    ap.add_argument("--lead", type=int, default=10, help="Lead time in minutes (default 10)")
    ap.add_argument("--tau", default=",".join(str(t) for t in DEFAULT_TAUS),
                     help="Comma-separated list of tau values in minutes")
    ap.add_argument("--models", default=None,
                     help="Comma-separated model names (default: all deep models present)")
    ap.add_argument("--outdir", default=None, help="Output folder (default: same as --dir)")
    args = ap.parse_args()

    outdir = args.outdir or args.dir
    os.makedirs(outdir, exist_ok=True)
    taus = [float(x) for x in args.tau.split(",")]

    print(f"Loading predictions from: {args.dir}  (lead={args.lead} min)")
    pred_with, pred_noct, trigger_ts = load_inputs(args.dir, args.lead)
    has_noct = pred_noct is not None and trigger_ts is not None

    models = (args.models.split(",") if args.models
              else sorted(set(pred_with.model.unique()) & set(DEEP_MODELS)))
    print(f"Models: {models}")
    print(f"Tau values (min): {taus}")
    if trigger_ts is not None:
        print(f"Trigger events: {len(trigger_ts)}")

    raw_df, norm_df = compute_curves(pred_with, pred_noct, trigger_ts, models, taus)

    raw_path = os.path.join(outdir, f"event_proximity_weighted_rmse_raw_lead{args.lead}.csv")
    norm_path = os.path.join(outdir, f"event_proximity_weighted_rmse_normalized_lead{args.lead}.csv")
    raw_df.to_csv(raw_path, index=False)
    norm_df.to_csv(norm_path, index=False)
    print(f"  saved {raw_path}")
    print(f"  saved {norm_path}")

    if not has_noct:
        print("\n  NOTE: without-Ct predictions and/or trigger events were "
              "missing, so only the with-Ct curve was computed. The gap-vs-"
              "tau summary figure was skipped. Re-run the GUI pipeline "
              "(with FIX 13 or later) to populate predictions_noct_lead"
              f"{args.lead}.csv and trigger_events_lead{args.lead}.csv, "
              "then re-run this script for the full comparison.")
    else:
        agg = norm_df.groupby("tau_min")["gap_pts"].mean()
        print("\nMean normalized-RMSE gap (pts) by tau, pooled across models & targets:")
        print(agg.round(2).to_string())
        if len(agg) >= 2:
            small_tau_gap = agg.iloc[agg.index.argmin()] if False else agg.loc[min(agg.index)]
            large_tau_gap = agg.loc[max(agg.index)]
            print(f"\n  At the smallest tau tested ({min(agg.index):.0f} min): gap = {small_tau_gap:.2f} pts")
            print(f"  At the largest tau tested ({max(agg.index):.0f} min):  gap = {large_tau_gap:.2f} pts")
            if small_tau_gap > large_tau_gap:
                print("  -> Gap is LARGEST close to real trigger events and shrinks as tau "
                      "grows, consistent with Ct being an anticipatory / transition signal "
                      "rather than a uniform accuracy booster.")
            else:
                print("  -> Gap does NOT shrink with distance from events in this run -- "
                      "worth inspecting per-model curves before drawing a conclusion.")

    make_plots(norm_df, outdir, args.lead, has_noct)
    print("\nDone.")


if __name__ == "__main__":
    main()
