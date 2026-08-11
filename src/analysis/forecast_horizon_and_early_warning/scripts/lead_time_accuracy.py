"""
Prediction Accuracy (R^2) vs Forecast Lead Time, T+1 to T+20 min
=====================================================================
Answers: as the forecast horizon lengthens, how much accuracy is lost?
Uses the lead-time sweep the pipeline's own ablation study already
computed for the v1 run (ablation_results_lead10.csv, rows where
type == "lead_time") -- no retraining needed, this data already exists.

IMPORTANT CAVEAT -- read before citing these numbers as a model result
This sweep is NOT produced by any of the 7 production architectures
(BiLSTM, BiGRU, GRU, LSTM_uni, Seq2Seq, CNN_LSTM, VanillaRNN) discussed
everywhere else in this project. The pipeline's run_ablation_study()
trains a separate, lightweight single-layer GRU proxy (`_AblGRU`, hidden
size capped at 64, 100 epochs) purely for ablation speed -- see
iaq_early_detection_gui_v3.py:2338. The raw chart it produces
(ABLATION_STUDY_lead10.png) labels itself "Model=BiGRU" in the title,
which is misleading: that name comes from a label-preference list
(["BiGRU","GRU","BiLSTM"], pipeline line ~2303) used only to pick a
title string, not the architecture actually trained. Treat every number
here as "how does accuracy degrade with lead time, in general, for this
feature set" -- not as a BiLSTM- or BiGRU-specific result. Also: this
sweep only used the WITH-Ct (68-feature) set -- no without-Ct lead-time
comparison exists, because predictions were only exported with and
without Ct at lead=10 (predictions_lead10_v2.csv / predictions_noct_
lead10.csv), not at the other five horizons.

Usage:
    python lead_time_accuracy.py --raw <ablation_results_lead10.csv> --outdir <analysis dir>
"""

import argparse
import os

import numpy as np
import pandas as pd

TARGETS = ["pm1", "pm2_5", "pm10", "co2", "voc"]
TARGET_LABEL = {"pm1": "PM1", "pm2_5": "PM2.5", "pm10": "PM10", "co2": "CO2", "voc": "VOC"}
# Categorical palette (fixed order, validated set) -- pollutants get slots 1-5,
# Overall gets a neutral dark ink line so it reads as "the aggregate", not a 6th series.
SERIES_COLOR = {"pm1": "#2a78d6", "pm2_5": "#eb6834", "pm10": "#1baf7a",
                "co2": "#eda100", "voc": "#e87ba4"}
OVERALL_COLOR = "#1a1a1a"


def load_lead_sweep(raw_csv):
    """Accepts either schema:
    - the pipeline's ablation_results_lead10.csv (has 'type'/'variant' columns,
      mixed in with the modality-ablation rows -- filtered down to the
      lead_time rows here), or
    - a pre-filtered lead sweep like bilstm_lead_time_sweep.csv (already has
      a 'lead_min' column directly, one row per lead, nothing to filter)."""
    df = pd.read_csv(raw_csv)
    if "lead_min" in df.columns:
        lead = df.copy()
    elif "type" in df.columns:
        lead = df[df["type"] == "lead_time"].copy()
        lead["lead_min"] = lead["variant"].str.extract(r"T\+(\d+)min").astype(int)
    else:
        raise ValueError(
            f"{raw_csv} has neither a 'lead_min' column nor a 'type' column -- "
            "unrecognized schema for a lead-time sweep.")
    return lead.sort_values("lead_min").reset_index(drop=True)


def make_chart(lead, outdir, model_label, source_note, out_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    })

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ---- Panel A: R^2 vs lead time (one axis, directly comparable -- R^2 is
    # already a normalized 0-1 quantity, so no indexing trick is needed here) ----
    ax = axes[0]
    ax.plot(lead["lead_min"], lead["overall_R2"], color=OVERALL_COLOR, lw=2.2,
            marker="o", markersize=5, label="Overall (pooled)", zorder=5)
    for t in TARGETS:
        ax.plot(lead["lead_min"], lead[f"{t}_R2"], color=SERIES_COLOR[t], lw=1.4,
                marker="o", markersize=3.5, alpha=0.9, label=TARGET_LABEL[t], zorder=3)
    ax.axvline(10, color="#999999", lw=0.9, ls="--", zorder=1)
    ax.text(10.3, ax.get_ylim()[0] if False else 0.02, "lead=10min\n(used elsewhere\nin this project)",
            fontsize=6.8, color="#777777", va="bottom")
    ax.set_xlabel("Forecast lead time (minutes)")
    ax.set_ylabel("R$^2$")
    ax.set_ylim(0, 1.03)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#EEEEEE", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="lower left")
    ax.set_title("(a) R$^2$ vs. lead time", fontsize=10, fontweight="bold", loc="left")

    # ---- Panel B: RMSE vs lead time, indexed to lead=1 = 100 so pollutants of
    # very different raw scale (CO2 in ppm vs PM in ug/m3) are comparable on one
    # axis -- avoids a dual-axis chart and avoids a misleading shared raw-unit axis ----
    ax2 = axes[1]
    base_overall = lead["overall_RMSE"].iloc[0]
    ax2.plot(lead["lead_min"], 100 * lead["overall_RMSE"] / base_overall, color=OVERALL_COLOR,
             lw=2.2, marker="o", markersize=5, label="Overall (pooled)", zorder=5)
    for t in TARGETS:
        base_t = lead[f"{t}_RMSE"].iloc[0]
        ax2.plot(lead["lead_min"], 100 * lead[f"{t}_RMSE"] / base_t, color=SERIES_COLOR[t],
                 lw=1.4, marker="o", markersize=3.5, alpha=0.9, label=TARGET_LABEL[t], zorder=3)
    ax2.axhline(100, color="#cccccc", lw=0.8, zorder=1)
    ax2.axvline(10, color="#999999", lw=0.9, ls="--", zorder=1)
    ax2.set_xlabel("Forecast lead time (minutes)")
    ax2.set_ylabel("RMSE, indexed to T+1min = 100\n(each series to its own T+1 baseline)")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(color="#EEEEEE", linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left")
    ax2.set_title("(b) RMSE growth vs. lead time (indexed)", fontsize=10, fontweight="bold", loc="left")

    fig.suptitle(f"Prediction Accuracy vs. Forecast Lead Time, T+1 to T+20 min (v1) -- {model_label}\n"
                  f"{source_note}",
                  fontsize=10.5, fontweight="bold", y=1.08)
    plt.tight_layout()
    out_png = os.path.join(outdir, f"LEAD_TIME_ACCURACY_{out_stem}.png")
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="Path to ablation_results_lead10.csv OR a pre-filtered "
                    "lead sweep CSV (e.g. bilstm_lead_time_sweep.csv)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--model-label", default=None,
                     help="What produced this sweep, e.g. 'BiLSTM' or 'ablation-proxy GRU'. "
                          "Auto-detected from --raw's filename if omitted (best-effort -- verify it's right).")
    args = ap.parse_args()
    base = args.outdir or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    data_dir = os.path.join(base, "data")
    fig_dir = os.path.join(base, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    raw_name = os.path.basename(args.raw).lower()
    if args.model_label:
        model_label, source_note = args.model_label, "See script invocation for source file."
    elif "bilstm" in raw_name:
        model_label = "BiLSTM"
        source_note = "Real BiLSTM architecture (train_bilstm_lead_sweep.py), full 68-feature (with-C$_t$) set."
    elif "ablation" in raw_name:
        model_label = "ablation-proxy GRU"
        source_note = "Lightweight single-layer GRU proxy, full 68-feature (with-C$_t$) set only -- NOT a production architecture, see script docstring."
    else:
        model_label = "UNSPECIFIED MODEL -- pass --model-label"
        source_note = "Model source not identified from filename -- verify before citing."
    out_stem = model_label.replace(" ", "_").replace("(", "").replace(")", "")

    lead = load_lead_sweep(args.raw)
    out_csv = os.path.join(data_dir, f"lead_time_accuracy_{out_stem}.csv")
    lead.round(4).to_csv(out_csv, index=False)

    print(f"=== R^2 / RMSE vs lead time (v1, {model_label}) ===")
    show_cols = ["lead_min", "overall_R2", "overall_RMSE"] + [f"{t}_R2" for t in TARGETS]
    with pd.option_context("display.width", 160):
        print(lead[show_cols].to_string(index=False))

    r2_drop = lead["overall_R2"].iloc[0] - lead["overall_R2"].iloc[-1]
    rmse_ratio = lead["overall_RMSE"].iloc[-1] / lead["overall_RMSE"].iloc[0]
    print(f"\nT+1 -> T+20: overall R^2 drops by {r2_drop:.4f} ({lead['overall_R2'].iloc[0]:.4f} -> {lead['overall_R2'].iloc[-1]:.4f}), "
          f"while overall RMSE grows {rmse_ratio:.2f}x ({lead['overall_RMSE'].iloc[0]:.2f} -> {lead['overall_RMSE'].iloc[-1]:.2f}).")
    print("Small R^2 drop + large RMSE growth is the same ceiling-compression effect documented")
    print("elsewhere in this project for the Ct comparison -- R^2 alone understates how much harder")
    print("the forecasting problem gets at longer horizons.")

    png = make_chart(lead, fig_dir, model_label, source_note, out_stem)
    print(f"\nSaved: {out_csv}\nSaved: {png}")


if __name__ == "__main__":
    main()
