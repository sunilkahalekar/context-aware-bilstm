"""
Cross-Run Ct Reliability Analysis
==================================
Single-run significance tests (the DM-test, phase-stratification, etc.)
tell you whether Ct mattered *in that one trained model instance* — they
do not tell you whether the effect would reproduce under a different
random seed. Across this project we now have several independent full
pipeline runs (different seeds / minor config changes), each producing
its own phase_stratified_metrics_lead{N}.csv. This script treats each
RUN as one independent replicate and asks the higher-bar question:
"how consistent is Ct's benefit for this architecture across runs?"

For each model, it computes the phase-averaged %-RMSE-increase-without-Ct
in every run, then reports the mean, standard deviation, and coefficient
of variation (CV = std/|mean|) across runs. Low CV + clearly positive
mean = a robust, reproducible finding. Near-zero mean regardless of run
= a genuine, reproducible null result (Ct doesn't help this architecture)
rather than noise. High CV = inconclusive; needs more replicates before
either direction is trustworthy.

Usage:
    python cross_run_ct_reliability.py --runs run1_dir,run2_dir,... [--lead 10]
    python cross_run_ct_reliability.py --runs-file runs.txt [--lead 10]
        (one directory per line in runs.txt; lines starting with # are labels,
         format "label: path" also accepted)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def load_run(path, lead):
    fp = os.path.join(path, f"phase_stratified_metrics_lead{lead}.csv")
    if not os.path.exists(fp):
        print(f"  [!] skipping {path}: no phase_stratified_metrics_lead{lead}.csv")
        return None
    df = pd.read_csv(fp)
    piv = df.pivot_table(index=["model", "phase"], columns="variant", values="overall_RMSE")
    piv["pct"] = 100 * (piv["without_Ct"] - piv["with_Ct"]) / piv["with_Ct"]
    return piv["pct"].groupby("model").mean()  # phase-averaged % per model, this run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="Comma-separated run directories")
    ap.add_argument("--runs-file", help="Text file, one run directory per line (optionally 'label: path')")
    ap.add_argument("--lead", type=int, default=10)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    entries = []  # (label, path)
    if args.runs:
        for p in args.runs.split(","):
            p = p.strip()
            entries.append((os.path.basename(p.rstrip("/\\")), p))
    if args.runs_file:
        with open(args.runs_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line and not line[1:3] == ":\\":  # crude Windows-drive-letter guard
                    label, p = line.split(":", 1)
                    entries.append((label.strip(), p.strip()))
                else:
                    entries.append((os.path.basename(line.rstrip("/\\")), line))
    if not entries:
        sys.exit("Provide --runs or --runs-file")

    series = {}
    for label, path in entries:
        s = load_run(path, args.lead)
        if s is not None:
            series[label] = s
            print(f"  loaded {label}: {len(s)} models")

    if len(series) < 2:
        sys.exit("Need at least 2 successfully-loaded runs to compute cross-run reliability.")

    mat = pd.DataFrame(series)
    mat["mean"] = mat.mean(axis=1, skipna=True)
    mat["std"] = mat.drop(columns=["mean"]).std(axis=1, ddof=1, skipna=True)
    mat["n_runs"] = mat.drop(columns=["mean", "std"]).notna().sum(axis=1)
    mat["cv"] = (mat["std"] / mat["mean"].abs()).round(2)

    def classify(row):
        if row["n_runs"] < 2:
            return "insufficient runs"
        if abs(row["mean"]) <= 15:
            return "NEAR-NULL (Ct ~doesn't matter, consistently)"
        if row["mean"] > 15 and (pd.isna(row["cv"]) or row["cv"] < 0.6):
            return "ROBUST benefit (consistent across runs)"
        if row["mean"] > 15:
            return "benefit present but SEED-SENSITIVE magnitude"
        return "mixed sign across runs — inconclusive"
    mat["verdict"] = mat.apply(classify, axis=1)

    mat_sorted = mat.sort_values("mean", ascending=False)
    print("\n" + "=" * 100)
    print("CROSS-RUN Ct RELIABILITY  (phase-averaged %% RMSE increase without Ct, per run)")
    print("=" * 100)
    cols = list(series.keys()) + ["mean", "std", "cv", "n_runs", "verdict"]
    with pd.option_context("display.width", 160, "display.max_colwidth", 40):
        print(mat_sorted[cols].round(1).to_string())

    out_csv = os.path.join(args.outdir, "cross_run_ct_reliability.csv")
    mat_sorted[cols].to_csv(out_csv)
    print(f"\nSaved: {out_csv}")

    # chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                          "font.size": 9, "axes.linewidth": 0.8, "axes.edgecolor": "#333333"})
    models = mat_sorted.index.tolist()
    run_labels = list(series.keys())
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(models))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, rl in enumerate(run_labels):
        ax.scatter(x, mat_sorted[rl], marker=markers[i % len(markers)], s=45,
                   label=rl, zorder=3, alpha=0.85)
    ax.errorbar(x, mat_sorted["mean"], yerr=mat_sorted["std"], fmt="none",
               ecolor="#333333", elinewidth=1.2, capsize=4, zorder=2)
    ax.plot(x, mat_sorted["mean"], "_", color="#1B5E7A", markersize=22, markeredgewidth=2.5, zorder=4)
    ax.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Phase-avg %% RMSE increase without C$_t$\n(per run, with mean $\\pm$ 1 s.d.)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
    ax.set_title("Cross-run Ct reliability: is the benefit reproducible, or one lucky seed?",
                fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_png = os.path.join(args.outdir, "CROSS_RUN_CT_RELIABILITY.png")
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
