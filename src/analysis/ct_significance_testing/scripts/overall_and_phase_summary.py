"""
Overall + Phase-Wise Ct Effect Summary
========================================
Answers two questions for a single pipeline run, from the CSVs it already
produces (regime_stratified_metrics_lead{N}.csv, phase_stratified_metrics_
lead{N}.csv) -- no retraining, no new predictions needed:

  1. OVERALL: pooled across the whole test window (baseline + onset
     minutes combined, correctly via MSE not by averaging RMSEs), how
     much does RMSE change when Ct is removed, per model?
  2. PHASE-WISE: same question, broken out by the four operational
     phases (Baseline / High-occupancy / Fabrication / Decay), so you can
     see *where* in the session each model's Ct-dependence shows up
     rather than only the session-average number.

This does NOT compare across separate pipeline runs (see
cross_run_ct_reliability.py for that) -- it summarizes ONE run in more
useful form than the raw per-regime / per-phase CSVs alone.

Usage:
    python overall_and_phase_summary.py --dir <run_folder> [--lead 10]
"""

import argparse
import os

import numpy as np
import pandas as pd


def pooled_overall(regime_df):
    """Combine baseline + onset into one whole-test-window RMSE per
    model/variant. RMSEs must be combined via their squared (MSE) form,
    weighted by sample count, then re-square-rooted -- averaging RMSEs
    directly would be wrong."""
    rows = []
    for m in regime_df.model.unique():
        sub = regime_df[regime_df.model == m]
        for variant in ["with_Ct", "without_Ct"]:
            vs = sub[sub.variant == variant]
            n_tot = vs["n"].sum()
            mse_pooled = (vs["n"] * vs["overall_RMSE"] ** 2).sum() / n_tot
            rows.append({"model": m, "variant": variant, "n": n_tot,
                        "rmse_pooled": np.sqrt(mse_pooled)})
    pooled = pd.DataFrame(rows)
    piv = pooled.pivot(index="model", columns="variant", values="rmse_pooled")
    piv["pct_change_without_ct"] = 100 * (piv["without_Ct"] - piv["with_Ct"]) / piv["with_Ct"]
    return piv.sort_values("pct_change_without_ct", ascending=False)


def phase_table(phase_df, model_order):
    piv = phase_df.pivot_table(index=["model", "phase"], columns="variant", values="overall_RMSE")
    piv["pct"] = 100 * (piv["without_Ct"] - piv["with_Ct"]) / piv["with_Ct"]
    table = piv["pct"].unstack().reindex(columns=["Phase 1", "Phase 2", "Phase 3", "Phase 4"])
    return table.reindex(model_order)


def make_chart(overall_piv, phase_tbl, outdir, run_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    })
    WITH_C, WITHOUT = "#1B5E7A", "#C1440E"
    order = overall_piv.index.tolist()
    overall_vals = overall_piv["pct_change_without_ct"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [0.85, 1.5]})

    ax = axes[0]
    y = np.arange(len(order))
    vals = overall_vals.values
    cols = [WITH_C if v >= 0 else WITHOUT for v in vals]
    ax.barh(y, vals, color=cols, height=0.6, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + (4 if v >= 0 else -4), yi, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, fontweight="bold")
    ax.axvline(0, color="#333333", lw=0.9, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9.5)
    ax.invert_yaxis()
    span = max(abs(vals.min()), abs(vals.max()))
    ax.set_xlim(-span * 1.35, span * 1.2)
    ax.set_xlabel("% RMSE change without C$_t$\n(whole test window, pooled)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5, zorder=0); ax.set_axisbelow(True)
    ax.set_title("(a) Overall effect, per model", fontsize=10, fontweight="bold", loc="left")

    ax2 = axes[1]
    phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
    mat = phase_tbl[phases].values
    vmax = np.nanmax(np.abs(mat))
    im = ax2.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if abs(v) > vmax * 0.55 else "#222222"
            ax2.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8.5,
                     color=txt_color, fontweight="bold")
    ax2.set_xticks(range(len(phases)))
    ax2.set_xticklabels(["P1\nBaseline", "P2\nHigh-occ.", "P3\nFabrication", "P4\nDecay"], fontsize=8.5)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order, fontsize=9.5)
    ax2.set_title("(b) Phase-wise breakdown — blue: C$_t$ helps · red: C$_t$ hurts",
                  fontsize=10, fontweight="bold", loc="left")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_xticks(np.arange(-0.5, len(phases), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=2)
    ax2.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(f"C$_t$ Effect — Overall vs. Phase-wise ({run_label})", fontsize=12.5, fontweight="bold", y=1.04)
    plt.tight_layout()
    out_png = os.path.join(outdir, "OVERALL_AND_PHASE_EFFECT.png")
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="Run folder containing the pipeline's CSV outputs")
    ap.add_argument("--lead", type=int, default=10)
    ap.add_argument("--outdir", default=None, help="Default: same as --dir")
    ap.add_argument("--label", default=None, help="Run label for the chart title (default: folder name)")
    args = ap.parse_args()
    outdir = args.outdir or args.dir
    label = args.label or os.path.basename(args.dir.rstrip("/\\"))

    regime = pd.read_csv(os.path.join(args.dir, f"regime_stratified_metrics_lead{args.lead}.csv"))
    phase = pd.read_csv(os.path.join(args.dir, f"phase_stratified_metrics_lead{args.lead}.csv"))

    overall_piv = pooled_overall(regime)
    phase_tbl = phase_table(phase, overall_piv.index.tolist())

    print(f"=== OVERALL (whole test window, pooled), {label} ===")
    print(overall_piv.round(2).to_string())
    print(f"\n=== PHASE-WISE %% RMSE change without Ct, {label} ===")
    print(phase_tbl.round(1).to_string())

    overall_piv.round(2).to_csv(os.path.join(outdir, "overall_ct_effect.csv"))
    phase_tbl.round(1).to_csv(os.path.join(outdir, "phase_wise_ct_effect.csv"))
    png = make_chart(overall_piv, phase_tbl, outdir, label)
    print(f"\nSaved: overall_ct_effect.csv, phase_wise_ct_effect.csv, {png}")


if __name__ == "__main__":
    main()
