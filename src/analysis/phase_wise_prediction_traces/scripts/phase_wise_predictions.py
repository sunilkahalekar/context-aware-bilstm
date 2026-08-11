"""
Phase-Wise Predictions: With-Ct vs Without-Ct, per Model
============================================================
Earlier analysis (analysis/v1/) answered "how big is Ct's effect,
per model, per phase" as a single summary number (RMSE gap). This
script answers the more direct visual question: "what do the actual
predicted curves look like, phase by phase, with vs without Ct?"

"With Ct" = the model trained on the full feature set, including the
vision-derived context vector C_t (door-state + human-motion
descriptors). "Without Ct" = the SAME model/architecture, SAME
training run, with only the C_t-derived columns removed from the
feature set -- every other feature (pollutant lags, temp/hum, rolling
stats, etc.) is identical. Both come straight out of the pipeline's
own two prediction exports for a run:
    predictions_lead{N}_v2.csv     (with-Ct)
    predictions_noct_lead{N}.csv   (without-Ct)

For each of the 7 sequence models, this produces ONE figure: a
5 (pollutant) x 4 (phase) grid of small multiples. Each cell plots
Actual vs Predicted-with-Ct vs Predicted-without-Ct as time series
over that phase's own time window, so you can see directly where the
without-Ct curve drifts away from Actual and the with-Ct curve
doesn't (or vice versa) -- the same information the phase-stratified
RMSE table summarizes into one number, but here you see the shape of
the disagreement, not just its size.

The underlying per-cell calculation (also written to
data/phase_wise_prediction_metrics.csv) is plain RMSE and R^2 in the
pollutant's own raw units, computed directly from the merged
prediction rows -- no normalization, no significance test. That's
intentional: this folder is the visual/diagnostic companion, not a
replacement for analysis/v1's normalized, significance-tested summary
(analysis/v1/figures/OVERALL_AND_PHASE_EFFECT_SIGNIF.png). Use that
one for magnitude claims in a paper; use this one to show what a
prediction curve actually looks like as supporting/appendix figures.

Usage:
    python phase_wise_predictions.py --dir <run_folder> --lead 10
"""

import argparse
import os

import numpy as np
import pandas as pd

MODEL_ORDER = ["BiLSTM", "Seq2Seq", "GRU", "LSTM_uni", "VanillaRNN", "CNN_LSTM", "BiGRU"]
TARGETS = ["pm1", "pm2_5", "pm10", "co2", "voc"]
TARGET_LABEL = {"pm1": "PM1 (µg/m³)", "pm2_5": "PM2.5 (µg/m³)",
                "pm10": "PM10 (µg/m³)", "co2": "CO2 (ppm)", "voc": "VOC (ppb)"}
PHASES = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
PHASE_LABEL = {"Phase 1": "P1 Baseline", "Phase 2": "P2 High-occ.",
               "Phase 3": "P3 Fabrication", "Phase 4": "P4 Decay"}


def assign_phase(ts):
    """Same fixed clock-time boundaries as the main pipeline's
    _phase_boundaries(): Phase 1 <=14:00, Phase 2 <=15:30,
    Phase 3 <=16:30, Phase 4 the rest."""
    ts = pd.to_datetime(ts)
    day = ts.min().normalize()
    b1400 = day + pd.Timedelta(hours=14)
    b1530 = day + pd.Timedelta(hours=15, minutes=30)
    b1630 = day + pd.Timedelta(hours=16, minutes=30)
    return np.where(ts <= b1400, "Phase 1",
           np.where(ts <= b1530, "Phase 2",
           np.where(ts <= b1630, "Phase 3", "Phase 4")))


def load_merged(run_dir, lead):
    with_df = pd.read_csv(os.path.join(run_dir, f"predictions_lead{lead}_v2.csv"))
    without_df = pd.read_csv(os.path.join(run_dir, f"predictions_noct_lead{lead}.csv"))

    keep = ["trigger_timestamp", "future_timestamp", "model", "target", "actual", "predicted"]
    with_df = with_df[keep].rename(columns={"predicted": "pred_with"})
    without_df = without_df[keep].rename(columns={"predicted": "pred_without"})
    m = pd.merge(with_df, without_df.drop(columns=["actual"]),
                 on=["trigger_timestamp", "future_timestamp", "model", "target"], how="inner")
    m = m[m["model"].isin(MODEL_ORDER)].copy()
    m["future_timestamp"] = pd.to_datetime(m["future_timestamp"])
    m["phase"] = assign_phase(m["future_timestamp"])
    return m


def r2_score(actual, pred):
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    if ss_tot <= 1e-12:
        return np.nan
    return 1 - ss_res / ss_tot


def compute_metrics(m):
    rows = []
    for model in MODEL_ORDER:
        for phase in PHASES:
            for target in TARGETS:
                sub = m[(m.model == model) & (m.phase == phase) & (m.target == target)]
                if len(sub) == 0:
                    continue
                rmse_with = float(np.sqrt(np.mean((sub["actual"] - sub["pred_with"]) ** 2)))
                rmse_without = float(np.sqrt(np.mean((sub["actual"] - sub["pred_without"]) ** 2)))
                rows.append({
                    "model": model, "phase": phase, "target": target, "n": len(sub),
                    "RMSE_with_Ct": rmse_with, "RMSE_without_Ct": rmse_without,
                    "R2_with_Ct": r2_score(sub["actual"].values, sub["pred_with"].values),
                    "R2_without_Ct": r2_score(sub["actual"].values, sub["pred_without"].values),
                    "RMSE_gap (without - with)": rmse_without - rmse_with,
                    "pct_RMSE_change_without_Ct": 100 * (rmse_without - rmse_with) / rmse_with if rmse_with > 1e-9 else np.nan,
                })
    return pd.DataFrame(rows)


def make_model_figure(m, model, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.linewidth": 0.7, "axes.edgecolor": "#333333",
    })
    ACTUAL, WITH_C, WITHOUT = "#222222", "#1B5E7A", "#C1440E"

    fig, axes = plt.subplots(len(TARGETS), len(PHASES), figsize=(16, 13), sharex=False)

    for i, target in enumerate(TARGETS):
        row_sub = m[(m.model == model) & (m.target == target)]
        for j, phase in enumerate(PHASES):
            ax = axes[i, j]
            sub = row_sub[row_sub.phase == phase].sort_values("future_timestamp")
            if len(sub) == 0:
                ax.axis("off")
                continue
            t = sub["future_timestamp"]
            ax.plot(t, sub["actual"], color=ACTUAL, lw=1.3, label="Actual", zorder=3)
            ax.plot(t, sub["pred_with"], color=WITH_C, lw=1.1, alpha=0.9, label="Predicted (with C$_t$)", zorder=2)
            ax.plot(t, sub["pred_without"], color=WITHOUT, lw=1.1, alpha=0.9, label="Predicted (without C$_t$)", zorder=2)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.tick_params(axis="x", rotation=45, labelsize=6.5)
            ax.tick_params(axis="y", labelsize=6.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(color="#EEEEEE", linewidth=0.5)
            if i == 0:
                ax.set_title(PHASE_LABEL[phase], fontsize=9, fontweight="bold")
            if j == 0:
                ax.set_ylabel(TARGET_LABEL[target], fontsize=8, fontweight="bold")

    handles = [plt.Line2D([0], [0], color=ACTUAL, lw=1.5),
               plt.Line2D([0], [0], color=WITH_C, lw=1.5),
               plt.Line2D([0], [0], color=WITHOUT, lw=1.5)]
    labels = ["Actual", "Predicted (with C$_t$)", "Predicted (without C$_t$)"]
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle(f"{model}: Phase-Wise Prediction Traces, With vs Without C$_t$", fontsize=13, fontweight="bold", y=1.045)
    plt.tight_layout()
    out_png = os.path.join(outdir, f"{model}_phase_wise_predictions.png")
    plt.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close()
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="Run folder containing predictions_lead{N}_v2.csv / predictions_noct_lead{N}.csv")
    ap.add_argument("--lead", type=int, default=10)
    ap.add_argument("--outdir", default=None, help="Default: this script's ../ (phase_wise_calculation/)")
    args = ap.parse_args()

    base = args.outdir or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    fig_dir = os.path.join(base, "figures")
    data_dir = os.path.join(base, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    m = load_merged(args.dir, args.lead)
    print(f"Loaded {len(m)} merged prediction rows across {m['model'].nunique()} models.")

    metrics = compute_metrics(m)
    metrics_fp = os.path.join(data_dir, f"phase_wise_prediction_metrics_lead{args.lead}.csv")
    metrics.round(4).to_csv(metrics_fp, index=False)
    print(f"Saved: {metrics_fp}")

    for model in MODEL_ORDER:
        png = make_model_figure(m, model, fig_dir)
        print(f"Saved: {png}")


if __name__ == "__main__":
    main()
