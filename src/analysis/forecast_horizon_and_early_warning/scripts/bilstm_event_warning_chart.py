"""
BiLSTM Event Early-Warning Chart — With vs Without Vision Context (C_t)
=============================================================================
Visualizes BiLSTM's rows from event_detection_lead10.csv: for each
pollutant's real alert threshold, how many minutes of advance warning did
BiLSTM give before an actual spike, with vs without C_t?

WHY THIS ISN'T A PLAIN BAR CHART
"0 minutes of lead time" and "never caught the event at all" are both
0-or-blank in the raw data, but they mean very different things -- one is
a late-but-correct alert, the other is a complete miss. A bar chart that
draws both as an empty/missing bar would erase that distinction. Cases
with zero true positives are drawn as an explicit hatched "no early
warning" marker instead of an absent bar, and every bar is annotated with
"caught k of n" so the underlying event count is never hidden behind a
single averaged number (each lead-time figure here rests on just 1-4
real events).

Usage:
    python bilstm_event_warning_chart.py --csv <event_detection_lead10.csv> --outdir <lead_time folder>
"""

import argparse
import os

import numpy as np
import pandas as pd

WITH_C, WITHOUT = "#1B5E7A", "#C1440E"
# Name/units are fixed; the THRESHOLD VALUE is read from the CSV itself
# (each row's own 'threshold' column), never hardcoded -- so this chart
# can't silently show a stale number if event_detection_lead10.csv is
# regenerated with a different alert threshold (e.g. VOC changed from
# 200 to 100 ppb in the GUI's "VOC alert (ppb)" field).
POLLUTANT_NAME = {"co2": "CO2", "pm2_5": "PM2.5", "voc": "VOC"}
POLLUTANT_UNITS = {"co2": "ppm", "pm2_5": "µg/m³", "voc": "ppb"}
ORDER = ["co2", "pm2_5", "voc"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Path to event_detection_lead10.csv")
    ap.add_argument("--model", default="BiLSTM")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--suffix", default=None,
                     help="Appended to the output filename so re-runs against a different "
                          "threshold/CSV don't silently overwrite a previous chart. "
                          "Default: derived from the input CSV's own filename.")
    ap.add_argument("--ymax", type=float, default=None,
                     help="Fixed y-axis max (minutes). Default: auto-scaled to the data's own "
                          "max lead time. A fixed value that's smaller than the tallest bar will "
                          "clip it -- check the output if you set this explicitly.")
    args = ap.parse_args()

    csv_stem = os.path.splitext(os.path.basename(args.csv))[0]
    suffix = args.suffix if args.suffix is not None else (
        "" if csv_stem == "event_detection_lead10" else f"_{csv_stem.replace('event_detection_lead10_', '')}")

    df = pd.read_csv(args.csv)
    sub = df[df.model == args.model].copy()
    if sub.empty:
        raise SystemExit(f"No rows for model={args.model} in {args.csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                          "font.size": 10, "axes.linewidth": 0.8, "axes.edgecolor": "#333333"})

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(ORDER))
    width = 0.32
    # Scale to the actual data -- a fixed cap silently clips bars/labels off
    # the canvas (seen firsthand: PM2.5 hit 27.5 min once CO2/VOC thresholds
    # changed, while this was hardcoded at 11 and blew up bbox_inches="tight").
    max_lead = sub.loc[sub["TP"] > 0, "mean_lead_min"].max()
    ymax = args.ymax if args.ymax is not None else float(np.ceil((max_lead if pd.notna(max_lead) else 10) * 1.15 / 2) * 2)
    if pd.notna(max_lead) and max_lead > ymax:
        print(f"WARNING: --ymax={ymax} is smaller than the tallest bar ({max_lead:.1f} min) -- it will be clipped.")

    for offset, variant, color, label in [(-width / 2, "with_Ct", WITH_C, "With vision context (C$_t$)"),
                                            (width / 2, "without_Ct", WITHOUT, "Without vision context")]:
        for i, target in enumerate(ORDER):
            row = sub[(sub.target == target) & (sub.variant == variant)]
            if row.empty:
                continue
            row = row.iloc[0]
            tp, n_events = int(row["TP"]), int(row["n_actual_events"])
            lead = row["mean_lead_min"]
            xi = x[i] + offset

            if tp == 0:
                # No true positives at all -- draw an explicit "no early warning"
                # marker, never an empty/zero bar (which would look identical to
                # "caught it with zero minutes of lead," a different situation).
                ax.bar(xi, 0.55, width, color="white", edgecolor=color, hatch="////",
                       linewidth=1.3, zorder=3)
                ax.text(xi, 0.9, "no early\nwarning", ha="center", va="bottom",
                        fontsize=7.8, color=color, fontweight="bold")
            else:
                lead_v = float(lead)
                ax.bar(xi, max(lead_v, 0.3), width, color=color, zorder=3,
                       alpha=1.0 if lead_v > 0 else 0.55)
                label_y = max(lead_v, 0.3) + 0.25
                if lead_v == 0:
                    ax.text(xi, label_y, "0 min\n(caught, no lead)", ha="center", va="bottom",
                            fontsize=7.8, color=color, fontweight="bold")
                else:
                    ax.text(xi, label_y, f"{lead_v:.1f} min", ha="center", va="bottom",
                            fontsize=9, color=color, fontweight="bold")
            ax.text(xi, -0.55, f"caught {tp} of {n_events}", ha="center", va="top",
                    fontsize=7.3, color="#555555")

    thr_by_target = sub.groupby("target")["threshold"].first().to_dict()
    xticklabels = [f"{POLLUTANT_NAME[t]}\n(alert > {thr_by_target.get(t, float('nan')):g} {POLLUTANT_UNITS[t]})"
                   for t in ORDER]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=9.5)
    ax.set_ylim(-1.3, ymax)
    ax.set_yticks(range(0, int(ymax) + 1, 2))
    ax.set_ylabel("Minutes of advance warning before the real spike")
    ax.axhline(0, color="#999999", lw=0.9, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=WITH_C), plt.Rectangle((0, 0), 1, 1, color=WITHOUT),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#333333", hatch="////")]
    labels = ["With vision context (C$_t$)", "Without vision context", "No true early warning at all"]
    ax.legend(handles, labels, fontsize=8.5, frameon=False, loc="upper right")

    ax.set_title(f"Event-Detection Lead Time With and Without Vision Context (C_t) — {args.model}\n",
                 fontsize=11.5, fontweight="bold", loc="left")
    plt.tight_layout()
    out_png = os.path.join(args.outdir, f"{args.model}_EVENT_WARNING{suffix}.png")
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
