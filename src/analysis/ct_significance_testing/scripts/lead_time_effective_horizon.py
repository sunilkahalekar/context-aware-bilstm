"""
Effective Forecasting Horizon — Model vs. Persistence, with Bootstrap CI
=============================================================================
Answers "how many minutes ahead can this system actually forecast" as an
empirical crossing point, not an assumed one: plots the gap between a
real model's error and the persistence baseline's error at each lead
time, with a bootstrap confidence band (this dataset is only 386 rows --
a single point estimate per lead is not trustworthy on its own, as the
last several BiLSTM debugging rounds demonstrated directly). The lead
time where the CI band crosses zero is the answer.

TWO MODES
  1. No --model-predictions given: shows the persistence baseline ALONE,
     with a bootstrap CI on its own R^2 at each lead. This is fully
     computable right now (no trained model needed) and answers a
     narrower but still useful question: how uncertain is the "~8
     minutes" naive-forecasting-horizon estimate, given how few test-
     window origins each lead has (as few as 38 at T+20)?
  2. --model-predictions <csv> given: the real comparison. Expects a CSV
     with columns [lead_min, target, origin_idx, actual, predicted] --
     exactly what train_bilstm_lead_sweep_v2.py's --export-predictions
     output produces once a run succeeds. Computes, per lead, the paired
     gap (persistence squared error - model squared error) resampled at
     the ORIGIN level (not per-target independently, since all 5 targets
     at a given origin share the same moment in time and should be
     resampled together) to get a bootstrap CI on the mean gap. Reports
     the lead time where the CI's lower bound first drops below zero --
     the conservative, "still confident the model wins here" horizon --
     alongside the point-estimate crossing (least conservative).

Usage:
    # persistence-only (works today, no model needed):
    python lead_time_effective_horizon.py --raw <sensor_data_merged_iaq_m2.csv> --outdir <analysis dir>

    # full model-vs-persistence comparison (once a model export exists):
    python lead_time_effective_horizon.py --raw <sensor_data_merged_iaq_m2.csv> \\
        --model-predictions <bilstm_predictions_export.csv> --outdir <analysis dir>
"""

import argparse
import os

import numpy as np
import pandas as pd

ALL_TGT = ["pm1", "pm2_5", "pm10", "co2", "voc"]
TF, VF = 0.70, 0.15
LEADS = list(range(1, 21))
N_BOOT = 2000
CI = 95
SEED = 42


def r2_multi(actual_dict, pred_dict):
    """actual_dict/pred_dict: {target: array}. Returns mean R^2 across targets
    (matching this project's 'overall_R2' convention elsewhere)."""
    r2s = []
    for t in ALL_TGT:
        a, p = actual_dict[t], pred_dict[t]
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        r2s.append(1 - ss_res / max(ss_tot, 1e-9))
    return float(np.mean(r2s))


def persistence_only(df):
    n = len(df)
    va_end = int(n * (TF + VF))
    rng = np.random.default_rng(SEED)
    rows = []
    for lead in LEADS:
        origins = np.arange(va_end, n - lead)
        if len(origins) < 8:
            continue
        actual = {t: df[t].values[origins + lead] for t in ALL_TGT}
        pred = {t: df[t].values[origins] for t in ALL_TGT}
        point_r2 = r2_multi(actual, pred)

        boot = np.empty(N_BOOT)
        n_o = len(origins)
        for b in range(N_BOOT):
            idx = rng.integers(0, n_o, n_o)
            a_b = {t: actual[t][idx] for t in ALL_TGT}
            p_b = {t: pred[t][idx] for t in ALL_TGT}
            boot[b] = r2_multi(a_b, p_b)
        lo, hi = np.percentile(boot, [(100 - CI) / 2, 100 - (100 - CI) / 2])
        rows.append({"lead_min": lead, "n_test": n_o, "R2_point": point_r2,
                     "R2_ci_lo": lo, "R2_ci_hi": hi})
    return pd.DataFrame(rows)


def model_vs_persistence(df, model_csv):
    n = len(df)
    mp = pd.read_csv(model_csv)
    required = {"lead_min", "target", "origin_idx", "actual", "predicted"}
    missing = required - set(mp.columns)
    if missing:
        raise ValueError(f"{model_csv} is missing columns: {missing}")

    rng = np.random.default_rng(SEED)
    rows = []
    for lead in sorted(mp["lead_min"].unique()):
        sub = mp[mp["lead_min"] == lead]
        origins = np.sort(sub["origin_idx"].unique())
        if len(origins) < 8:
            continue
        model_actual = {t: sub[sub.target == t].set_index("origin_idx").loc[origins, "actual"].values for t in ALL_TGT}
        model_pred = {t: sub[sub.target == t].set_index("origin_idx").loc[origins, "predicted"].values for t in ALL_TGT}
        pers_pred = {t: df[t].values[origins] for t in ALL_TGT}  # persistence: value at the same origins

        def gap_stat(idx):
            model_r2 = r2_multi({t: model_actual[t][idx] for t in ALL_TGT},
                                 {t: model_pred[t][idx] for t in ALL_TGT})
            pers_r2 = r2_multi({t: model_actual[t][idx] for t in ALL_TGT},
                                {t: pers_pred[t][idx] for t in ALL_TGT})
            return model_r2 - pers_r2

        n_o = len(origins)
        point_gap = gap_stat(np.arange(n_o))
        boot = np.array([gap_stat(rng.integers(0, n_o, n_o)) for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [(100 - CI) / 2, 100 - (100 - CI) / 2])
        rows.append({"lead_min": lead, "n_test": n_o, "gap_point": point_gap,
                     "gap_ci_lo": lo, "gap_ci_hi": hi})
    return pd.DataFrame(rows)


def make_chart(out_df, outdir, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                          "font.size": 9, "axes.linewidth": 0.8, "axes.edgecolor": "#333333"})
    fig, ax = plt.subplots(figsize=(8, 5))

    if mode == "persistence":
        ax.fill_between(out_df["lead_min"], out_df["R2_ci_lo"], out_df["R2_ci_hi"],
                         color="#2a78d6", alpha=0.18, label=f"{CI}% bootstrap CI ({N_BOOT} resamples)")
        ax.plot(out_df["lead_min"], out_df["R2_point"], color="#2a78d6", lw=2, marker="o", markersize=4)
        ax.set_ylabel("R$^2$ (persistence baseline)")
        title = "Persistence baseline R$^2$ with bootstrap CI\nband width shows how uncertain the \"effective horizon\" estimate is at this sample size"
        # first lead where the CI upper bound drops below zero (fully outside the "positive" region)
        below = out_df[out_df["R2_ci_hi"] < 0]
    else:
        ax.fill_between(out_df["lead_min"], out_df["gap_ci_lo"], out_df["gap_ci_hi"],
                         color="#1baf7a", alpha=0.18, label=f"{CI}% bootstrap CI ({N_BOOT} resamples)")
        ax.plot(out_df["lead_min"], out_df["gap_point"], color="#1baf7a", lw=2, marker="o", markersize=4)
        ax.set_ylabel("R$^2$ gap (model $-$ persistence)")
        title = "Model vs. persistence gap, with bootstrap CI\ncrossing point = empirical effective forecasting horizon"
        below = out_df[out_df["gap_ci_lo"] < 0]

    ax.axhline(0, color="#999999", lw=1.0, zorder=1)
    if len(below):
        cross = below["lead_min"].iloc[0]
        ax.axvline(cross, color="#C1440E", lw=1.2, ls="--", zorder=1)
        ax.text(cross + 0.3, ax.get_ylim()[1] * 0.9, f"CI crosses 0\nat ~{cross} min",
                fontsize=8, color="#C1440E")
    ax.set_xlabel("Forecast lead time (minutes)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#EEEEEE", linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="upper right" if mode == "persistence" else "lower left")
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    plt.tight_layout()
    out_png = os.path.join(outdir, f"EFFECTIVE_HORIZON_{mode.upper()}.png")
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--model-predictions", default=None,
                     help="CSV with [lead_min,target,origin_idx,actual,predicted]. "
                          "Omit to run persistence-only mode.")
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

    if args.model_predictions:
        out_df = model_vs_persistence(df, args.model_predictions)
        mode = "model_vs_persistence"
        csv_name = "effective_horizon_model_vs_persistence.csv"
    else:
        out_df = persistence_only(df)
        mode = "persistence"
        csv_name = "effective_horizon_persistence_ci.csv"
        print("No --model-predictions given -- running persistence-only mode.")
        print("This shows how uncertain the naive-forecasting-horizon estimate is, not a real model comparison.")
        print("Pass --model-predictions <export.csv> (see train_bilstm_lead_sweep_v2.py --export-predictions)")
        print("once a BiLSTM run actually succeeds, to get the real answer.\n")

    out_csv = os.path.join(data_dir, csv_name)
    out_df.round(4).to_csv(out_csv, index=False)
    with pd.option_context("display.width", 160):
        print(out_df.to_string(index=False))

    png = make_chart(out_df, fig_dir, "persistence" if mode == "persistence" else "gap")
    print(f"\nSaved: {out_csv}\nSaved: {png}")


if __name__ == "__main__":
    main()
