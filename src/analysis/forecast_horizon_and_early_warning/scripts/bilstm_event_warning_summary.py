"""
BiLSTM Event Early-Warning — Overall Summary (Pooled Across Pollutants)
=============================================================================
Condenses the 3-pollutant event_detection_lead10*.csv breakdown into a
single pair of bars: with vs without C_t, pooled across CO2/PM2.5/VOC.

WHY POOL, AND WHY SHOW CATCH RATE TOO
Lead time alone can mislead when catch rates differ. Without C_t, BiLSTM's
one surviving pollutant (VOC) still averages 7.5 min of warning when it
fires -- which sounds almost competitive with the with-C_t number (9.8
min) until you see it only fired on 2 of 13 real events, versus 6 of 13
with C_t. Reporting lead time without catch rate would hide that. Both
numbers are shown on each bar rather than picking one.

Usage:
    python bilstm_event_warning_summary.py --csv <event_detection_lead10*.csv> --outdir <lead_time folder>
"""

import argparse
import os

import pandas as pd

WITH_C, WITHOUT = "#1B5E7A", "#C1440E"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="BiLSTM")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    sub = df[df.model == args.model]
    if sub.empty:
        raise SystemExit(f"No rows for model={args.model} in {args.csv}")

    stats = {}
    for variant in ["with_Ct", "without_Ct"]:
        v = sub[sub.variant == variant]
        n_events, tp = int(v["n_actual_events"].sum()), int(v["TP"].sum())
        avg_lead = (v["TP"] * v["mean_lead_min"].fillna(0)).sum() / tp if tp > 0 else float("nan")
        stats[variant] = {"n_events": n_events, "tp": tp, "catch_rate": tp / n_events, "avg_lead": avg_lead}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                          "font.size": 11, "axes.linewidth": 0.8, "axes.edgecolor": "#333333"})

    fig, ax = plt.subplots(figsize=(6.5, 6))
    labels = ["With Vision\nContext (C$_t$)", "Without Vision\nContext"]
    colors = [WITH_C, WITHOUT]
    leads = [stats["with_Ct"]["avg_lead"], stats["without_Ct"]["avg_lead"]]
    x = [0, 1]

    bars = ax.bar(x, leads, width=0.5, color=colors, zorder=3)
    ymax = max(leads) * 1.35
    for xi, variant, color in zip(x, ["with_Ct", "without_Ct"], colors):
        s = stats[variant]
        ax.text(xi, s["avg_lead"] + ymax * 0.03, f"{s['avg_lead']:.1f} min", ha="center", va="bottom",
                fontsize=15, color=color, fontweight="bold")
        ax.text(xi, s["avg_lead"] * 0.5, f"caught {s['tp']} of {s['n_events']}\nreal spikes ({s['catch_rate']:.0%})",
                ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Average advance-warning time (minutes)\nover successfully caught spikes")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"{args.model}: Does Vision Context Give Earlier, More Reliable Pollution Alerts?\n"
                 "Pooled across CO2, PM2.5, and VOC — lead time AND catch rate shown together",
                 fontsize=11.5, fontweight="bold", loc="left")
    plt.tight_layout()
    out_png = os.path.join(args.outdir, f"{args.model}_EVENT_WARNING_SUMMARY.png")
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
    for variant, s in stats.items():
        print(f"{variant}: {s}")


if __name__ == "__main__":
    main()
