"""
Persistence Baseline, T+1 to T+20 min — the bar any model must clear
=========================================================================
Computes the naive "predict no change" forecast (y_hat[t+lead] = y[t]) at
every lead time from 1 to 20 minutes, on the SAME test window (last 15%
of the session, matching TF=0.70/VF=0.15 used by train_bilstm_lead_sweep*
.py) that the real model sweeps are evaluated on. No training, no torch
-- this runs anywhere.

WHY THIS MATTERS MORE THAN IT LOOKS
The R^2 used everywhere else in this project compares a model to "always
predict the test window's own mean" -- a weak baseline for a TRENDING
series (CO2 and VOC drift steadily within a phase; see phase_wise_
calculation/'s trace plots). Persistence is usually a much tougher
baseline to beat for a trending or autocorrelated series, because the
most recent reading is often a good guess for what comes next. A model
that can't beat mean-baseline R^2 (as BiLSTM's first two attempts
couldn't) is failing a low bar; a model that can't beat PERSISTENCE is
failing an even more basic one. Comparing a future model sweep against
this curve, lead time by lead time, with a significance test, is how you
empirically find "the effective forecasting horizon" instead of assuming
it's 10 minutes.

Usage:
    python persistence_baseline_sweep.py --raw <sensor_data_merged_iaq_m2.csv> --outdir <analysis dir>
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

ALL_TGT = ["pm1", "pm2_5", "pm10", "co2", "voc"]
TF, VF = 0.70, 0.15  # matches train_bilstm_lead_sweep*.py's split
LEADS = list(range(1, 21))  # every minute, 1..20 -- finer than the 6-point model sweep


def r2_score(actual, pred):
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1 - ss_res / max(ss_tot, 1e-9)


def dm_test(sq_err_model, sq_err_baseline):
    """Diebold-Mariano test -- positive stat / small p means the model
    (first arg) has significantly LOWER error than the baseline."""
    d = np.asarray(sq_err_baseline, float) - np.asarray(sq_err_model, float)
    n = len(d)
    if n < 5 or d.var(ddof=1) <= 1e-12:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(d.var(ddof=1) / n)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    base = args.outdir or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    data_dir = os.path.join(base, "data")
    fig_dir = os.path.join(base, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(args.raw)
    df["timestamp_minute"] = pd.to_datetime(df["timestamp_minute"], format="%m-%d-%Y %H:%M")
    df = df.sort_values("timestamp_minute").reset_index(drop=True)
    n = len(df)
    va_end = int(n * (TF + VF))  # test window starts here, same convention as the model sweeps

    rows = []
    for lead in LEADS:
        # test-window origins: every minute from va_end to n-lead-1, so
        # (origin, origin+lead) both fall inside/after the test split
        origins = np.arange(va_end, n - lead)
        if len(origins) < 5:
            continue
        per_target_r2, per_target_rmse = {}, {}
        for t in ALL_TGT:
            y = df[t].values
            actual = y[origins + lead]
            pred = y[origins]  # persistence: predict no change from the last known value
            per_target_r2[t] = r2_score(actual, pred)
            per_target_rmse[t] = float(np.sqrt(np.mean((actual - pred) ** 2)))
        row = {"lead_min": lead, "n_test": len(origins),
               "overall_R2": float(np.mean(list(per_target_r2.values()))),
               "overall_RMSE": float(np.mean(list(per_target_rmse.values())))}
        for t in ALL_TGT:
            row[f"{t}_R2"] = per_target_r2[t]
            row[f"{t}_RMSE"] = per_target_rmse[t]
        rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = os.path.join(data_dir, "persistence_baseline_sweep.csv")
    out.round(4).to_csv(out_csv, index=False)

    print("=== Persistence baseline (predict no change), T+1 to T+20 min ===")
    show_cols = ["lead_min", "n_test", "overall_R2", "overall_RMSE"] + [f"{t}_R2" for t in ALL_TGT]
    with pd.option_context("display.width", 160):
        print(out[show_cols].to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print("\nRead this as the bar any real model needs to clear. A model with a LOWER R^2 than")
    print("this baseline at a given lead time is not doing useful forecasting work at that lead --")
    print("it would be more accurate to just repeat the last known reading.")

    # ---- chart ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                          "font.size": 9, "axes.linewidth": 0.8, "axes.edgecolor": "#333333"})
    SERIES_COLOR = {"pm1": "#2a78d6", "pm2_5": "#eb6834", "pm10": "#1baf7a",
                     "co2": "#eda100", "voc": "#e87ba4"}
    TARGET_LABEL = {"pm1": "PM1", "pm2_5": "PM2.5", "pm10": "PM10", "co2": "CO2", "voc": "VOC"}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(out["lead_min"], out["overall_R2"], color="#1a1a1a", lw=2.2, marker="o",
            markersize=4, label="Overall (pooled)", zorder=5)
    for t in ALL_TGT:
        ax.plot(out["lead_min"], out[f"{t}_R2"], color=SERIES_COLOR[t], lw=1.3,
                 marker="o", markersize=2.8, alpha=0.85, label=TARGET_LABEL[t], zorder=3)
    ax.axhline(0, color="#999999", lw=0.9, zorder=1)
    ax.axvline(10, color="#cccccc", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("Forecast lead time (minutes)")
    ax.set_ylabel("R$^2$ (persistence baseline)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#EEEEEE", linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="lower left")
    ax.set_title("Persistence baseline: R$^2$ of \"predict no change\" vs. lead time\n"
                  "any real model needs to sit ABOVE this line to be doing useful forecasting work",
                  fontsize=9.5, fontweight="bold", loc="left")
    plt.tight_layout()
    out_png = os.path.join(fig_dir, "PERSISTENCE_BASELINE.png")
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
