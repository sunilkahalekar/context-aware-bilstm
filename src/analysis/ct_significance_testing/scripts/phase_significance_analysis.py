"""
Phase-Wise Ct Effect with Per-Phase Significance (revised panel b)
=====================================================================
Extends overall_and_phase_summary.py's panel (b). That panel showed
% RMSE change relative to the with-Ct RMSE, pooled across all 5
pollutants in RAW units -- which means CO2 (RMSE in the hundreds)
dominates the "overall_RMSE" almost completely and PM's contribution
is nearly invisible, and a relative-% ratio can look huge purely
because the with-Ct baseline is small (the same ceiling-compression
issue already documented for R^2 elsewhere in this project). Neither
problem means Ct doesn't matter -- it means the METRIC was misleading.

This script fixes both, and adds what panel (b) never had: a real
per-phase significance test.

  1. NORMALIZED, EQUAL-WEIGHTED GAP: for each pollutant, RMSE is
     expressed as % of that pollutant's own whole-session observed
     range (same fixed yardstick in every phase, so phases stay
     comparable). The with/without gap is then a PERCENTAGE-POINT
     difference (not a ratio), averaged with equal weight across the
     5 pollutants -- so CO2 can no longer drown out PM.

  2. PER-PHASE SIGNIFICANCE: Diebold-Mariano test (same test already
     used elsewhere in this project for the pooled onset window) on
     normalized squared errors, run separately per (model, phase,
     pollutant) -- 7 x 4 x 5 = 140 tests. Benjamini-Hochberg FDR
     correction is applied ACROSS ALL 140 TESTS AT ONCE (the correct
     granularity to correct at, since that's the actual family of
     simultaneous tests being run) to control the false-discovery rate.
     A (model, phase) cell is marked significant on the chart only if a
     MAJORITY of its 5 pollutants (>=3 of 5) are individually
     FDR-significant. This majority-rule was chosen deliberately over
     combining the 5 pollutants' p-values into one (e.g. Fisher's
     method, still reported in the CSV as a supplementary diagnostic):
     Fisher's method assumes the 5 underlying tests are independent,
     but PM1/PM2.5/PM10/CO2/VOC are measured at the same timestamps
     from the same physical process, so their prediction errors are
     almost certainly correlated -- which makes Fisher's combined
     p-value anti-conservative (it overstates significance; an early
     run of this exact analysis produced 27/28 "significant" cells via
     Fisher's method alone, including gaps as small as 0.3 percentage
     points, which is not credible). Majority-rule across independently
     FDR-corrected per-pollutant tests avoids that inflation. A
     Wilcoxon signed-rank test (distribution-free, doesn't assume
     normality) is also computed per pollutant as a further cross-check,
     reported in the detail CSV -- useful because Phase 1 has only
     n~33 minutes, where the DM-test's asymptotic-normality assumption
     is weakest.

Inputs (from the pipeline run folder):
    predictions_lead{N}_v2.csv     (with-Ct per-minute predictions)
    predictions_noct_lead{N}.csv   (without-Ct per-minute predictions)

Usage:
    python phase_significance_analysis.py --dir <run_folder> --lead 10
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overall_and_phase_summary import pooled_overall  # noqa: E402

MODEL_ORDER = ["BiLSTM", "Seq2Seq", "GRU", "LSTM_uni", "VanillaRNN", "CNN_LSTM", "BiGRU"]
TARGETS = ["pm1", "pm2_5", "pm10", "co2", "voc"]
PHASES = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
PHASE_SHORT = {"Phase 1": "P1\nBaseline", "Phase 2": "P2\nHigh-occ.",
               "Phase 3": "P3\nFabrication", "Phase 4": "P4\nDecay"}


def assign_phase(ts):
    """Same fixed clock-time boundaries as the pipeline's _phase_boundaries():
    Phase 1 <=14:00, Phase 2 <=15:30, Phase 3 <=16:30, Phase 4 rest."""
    ts = pd.to_datetime(ts)
    day = ts.min().normalize()
    b1400 = day + pd.Timedelta(hours=14)
    b1530 = day + pd.Timedelta(hours=15, minutes=30)
    b1630 = day + pd.Timedelta(hours=16, minutes=30)
    return np.where(ts <= b1400, "Phase 1",
           np.where(ts <= b1530, "Phase 2",
           np.where(ts <= b1630, "Phase 3", "Phase 4")))


def dm_test(sq_err_without, sq_err_with):
    """Diebold-Mariano test on squared errors. Positive dm_stat / small p
    means without-Ct has significantly larger error (Ct helps)."""
    d = np.asarray(sq_err_without, float) - np.asarray(sq_err_with, float)
    n = len(d)
    if n < 5:
        return np.nan, np.nan
    var_d = d.var(ddof=1)
    if var_d <= 1e-12:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var_d / n)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


def wilcoxon_test(sq_err_without, sq_err_with):
    d = np.asarray(sq_err_without, float) - np.asarray(sq_err_with, float)
    if len(d) < 5 or np.allclose(d, 0):
        return np.nan
    try:
        return float(stats.wilcoxon(d).pvalue)
    except ValueError:
        return np.nan


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction; NaNs pass through as NaN."""
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


def build_tables(run_dir, lead):
    with_df = pd.read_csv(os.path.join(run_dir, f"predictions_lead{lead}_v2.csv"))
    without_df = pd.read_csv(os.path.join(run_dir, f"predictions_noct_lead{lead}.csv"))

    keep = ["trigger_timestamp", "future_timestamp", "model", "target", "actual", "predicted"]
    with_df = with_df[keep].rename(columns={"predicted": "pred_with"})
    without_df = without_df[keep].rename(columns={"predicted": "pred_without"})
    m = pd.merge(with_df, without_df.drop(columns=["actual"]),
                 on=["trigger_timestamp", "future_timestamp", "model", "target"], how="inner")
    m = m[m["model"].isin(MODEL_ORDER)].copy()
    m["phase"] = assign_phase(m["future_timestamp"])

    # Fixed, whole-session, per-target range -- same denominator in every
    # phase so the normalized RMSE is comparable across phases.
    ranges = m.groupby("target")["actual"].agg(lambda s: s.max() - s.min())
    m["range"] = m["target"].map(ranges)
    m["sq_pct_with"] = (100 * (m["actual"] - m["pred_with"]) / m["range"]) ** 2
    m["sq_pct_without"] = (100 * (m["actual"] - m["pred_without"]) / m["range"]) ** 2

    rows = []
    for model in MODEL_ORDER:
        for phase in PHASES:
            for target in TARGETS:
                sub = m[(m.model == model) & (m.phase == phase) & (m.target == target)]
                if len(sub) == 0:
                    continue
                rmse_with = np.sqrt(sub["sq_pct_with"].mean())
                rmse_without = np.sqrt(sub["sq_pct_without"].mean())
                dm_stat, dm_p = dm_test(sub["sq_pct_without"], sub["sq_pct_with"])
                wx_p = wilcoxon_test(sub["sq_pct_without"], sub["sq_pct_with"])
                rows.append({
                    "model": model, "phase": phase, "target": target, "n": len(sub),
                    "normRMSE_with_pct": rmse_with, "normRMSE_without_pct": rmse_without,
                    "gap_pts": rmse_without - rmse_with,
                    "dm_stat": dm_stat, "dm_p": dm_p, "wilcoxon_p": wx_p,
                })
    detail = pd.DataFrame(rows)

    # Correct at the true test granularity: all 140 (model, phase, target)
    # DM-test p-values corrected together in one BH-FDR pass.
    detail["dm_q"] = bh_fdr(detail["dm_p"].values)
    detail["dm_sig_fdr"] = detail["dm_q"] < 0.05

    cell_rows = []
    for model in MODEL_ORDER:
        for phase in PHASES:
            sub = detail[(detail.model == model) & (detail.phase == phase)]
            if len(sub) == 0:
                continue
            dm_ps = np.clip(sub["dm_p"].dropna().values, 1e-300, 1)
            fisher_p = stats.combine_pvalues(dm_ps, method="fisher")[1] if len(dm_ps) >= 1 else np.nan
            n_sig_fdr = int(sub["dm_sig_fdr"].sum())
            n_tot = len(sub)
            cell_rows.append({
                "model": model, "phase": phase, "n": int(sub["n"].max()),
                "gap_pts": sub["gap_pts"].mean(),
                "n_targets_sig_fdr": n_sig_fdr, "n_targets_total": n_tot,
                "significant_majority": n_sig_fdr >= 3,
                "fisher_p_supplementary": fisher_p,
            })
    cells = pd.DataFrame(cell_rows)
    return detail, cells


def make_chart(overall_piv, cells, outdir, run_label):
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

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), gridspec_kw={"width_ratios": [0.8, 1.6]})

    # ---- panel (a): unchanged headline ----
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

    # ---- panel (b): normalized-range point-gap, n-annotated, FDR-marked ----
    ax2 = axes[1]
    gap_piv = cells.pivot(index="model", columns="phase", values="gap_pts").reindex(index=order, columns=PHASES)
    n_piv = cells.pivot(index="model", columns="phase", values="n").reindex(index=order, columns=PHASES)
    sig_piv = cells.pivot(index="model", columns="phase", values="significant_majority").reindex(index=order, columns=PHASES)
    ntsig_piv = cells.pivot(index="model", columns="phase", values="n_targets_sig_fdr").reindex(index=order, columns=PHASES)

    mat = gap_piv.values.astype(float)
    vmax = np.nanmax(np.abs(mat))
    cmap = plt.get_cmap("RdBu")
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    rgba = cmap(norm(mat))
    sig_mask = sig_piv.values.astype(bool)
    nan_mask = np.isnan(mat)
    rgba[..., 3] = np.where(sig_mask, 1.0, 0.30)
    rgba[nan_mask] = [0.85, 0.85, 0.85, 0.4]
    ax2.imshow(rgba, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            is_sig = bool(sig_mask[i, j])
            n_val = n_piv.values[i, j]
            nt_sig = int(ntsig_piv.values[i, j])
            txt_color = "white" if (abs(v) > vmax * 0.55 and is_sig) else "#222222"
            fw = "bold" if is_sig else "normal"
            main = f"{v:+.1f}pt" + (" *" if is_sig else "")
            ax2.text(j, i - 0.18, main, ha="center", va="center", fontsize=8.3,
                     color=txt_color, fontweight=fw)
            ax2.text(j, i + 0.16, f"n={int(n_val)}  ({nt_sig}/5 sig)", ha="center", va="center", fontsize=6.4,
                     color=txt_color, alpha=0.9 if is_sig else 0.75)

    ax2.set_xticks(range(len(PHASES)))
    ax2.set_xticklabels([PHASE_SHORT[p] for p in PHASES], fontsize=8.5)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order, fontsize=9.5)
    ax2.set_title("(b) Phase-wise gap, normalized-range pts — blue: C$_t$ helps · red: C$_t$ hurts\n"
                  "* = majority of pollutants (>=3/5) individually FDR-significant (BH, q<0.05, per-target Diebold-Mariano) · faded = not significant",
                  fontsize=8.5, fontweight="bold", loc="left")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_xticks(np.arange(-0.5, len(PHASES), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=2)
    ax2.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(f"C$_t$ Effect — Overall vs. Phase-wise, significance-tested ({run_label})",
                 fontsize=12.5, fontweight="bold", y=1.05)
    plt.tight_layout()
    out_png = os.path.join(outdir, "OVERALL_AND_PHASE_EFFECT_SIGNIF.png")
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--lead", type=int, default=10)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    outdir = args.outdir or args.dir
    label = args.label or os.path.basename(args.dir.rstrip("/\\"))

    regime = pd.read_csv(os.path.join(args.dir, f"regime_stratified_metrics_lead{args.lead}.csv"))
    overall_piv = pooled_overall(regime)

    detail, cells = build_tables(args.dir, args.lead)

    detail_fp = os.path.join(outdir, f"phase_significance_detail_lead{args.lead}.csv")
    cells_fp = os.path.join(outdir, f"phase_significance_cells_lead{args.lead}.csv")
    detail.round(4).to_csv(detail_fp, index=False)
    cells.round(4).to_csv(cells_fp, index=False)

    print(f"=== Per-(model,phase) significance summary, {label} ===")
    show = cells.copy()
    show["gap_pts"] = show["gap_pts"].round(1)
    show["fisher_p_supplementary"] = show["fisher_p_supplementary"].round(4)
    with pd.option_context("display.width", 160):
        print(show.to_string(index=False))

    n_sig = int(cells["significant_majority"].sum())
    n_tot = len(cells)
    print(f"\n{n_sig}/{n_tot} (model, phase) cells pass majority-rule significance "
          f"(>=3/5 pollutants individually FDR-significant, BH q<0.05).")

    png = make_chart(overall_piv, cells, outdir, label)
    print(f"\nSaved: {os.path.basename(detail_fp)}, {os.path.basename(cells_fp)}, {png}")


if __name__ == "__main__":
    main()
