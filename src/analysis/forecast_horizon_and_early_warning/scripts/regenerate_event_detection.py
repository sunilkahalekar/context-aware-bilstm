"""
Regenerate event_detection_lead10.csv with a Custom Alert Threshold
=========================================================================
Ports iaq_early_detection_gui_v3.py's _detect_events()/_event_scoring()
(lines ~2924-2970) verbatim, and applies them to the v1 run's EXISTING,
VALID T+10 predictions (predictions_lead10_v2.csv / predictions_noct_
lead10.csv) -- no model retraining needed. This is only possible because
those T+10 predictions are the original, working production run; the
lead-time SWEEP (T+1/3/5/15/20) is the part that's still broken (see
lead_time/CLAUDE.md) -- this script doesn't touch that at all.

Usage:
    python regenerate_event_detection.py --dir <v1 run folder> --outdir <lead_time folder> \\
        --voc-threshold 100
"""

import argparse
import os

import numpy as np
import pandas as pd

MODEL_ORDER = ["BiLSTM", "Seq2Seq", "GRU", "LSTM_uni", "VanillaRNN", "CNN_LSTM", "BiGRU"]
DEFAULT_THRESHOLDS = {"co2": 2000.0, "pm2_5": 35.0, "voc": 200.0}
EVENT_MATCH_HORIZON = 30  # GUI default (event_match_horizon_var)


def detect_events(arr, ts, thr, min_gap_min=5):
    arr = np.asarray(arr, dtype=float)
    above = arr > thr
    rise = above & (~np.r_[False, above[:-1]])
    idxs = np.where(rise)[0]
    times = pd.to_datetime(np.asarray(ts))[idxs]
    kept = []
    for t in times:
        if not kept or (t - kept[-1]).total_seconds() / 60.0 >= min_gap_min:
            kept.append(t)
    return kept


def event_scoring(actual_1d, ts_actual, pred_1d, ts_pred, thr, horizon_min):
    actual_events = detect_events(actual_1d, ts_actual, thr)
    pred_arr = np.asarray(pred_1d, dtype=float)
    above_p = pred_arr > thr
    rise_p = above_p & (~np.r_[False, above_p[:-1]])
    pidxs = np.where(rise_p)[0]
    ptimes = list(pd.to_datetime(np.asarray(ts_pred))[pidxs])
    kept_p = []
    for t in ptimes:
        if not kept_p or (t - kept_p[-1]).total_seconds() / 60.0 >= 5:
            kept_p.append(t)
    ptimes = kept_p

    used_p = set()
    tp, leads = 0, []
    for ae in actual_events:
        best, best_lead = None, None
        for j, pt in enumerate(ptimes):
            if j in used_p:
                continue
            lead = (ae - pt).total_seconds() / 60.0
            if 0 <= lead <= horizon_min:
                if best is None or lead > best_lead:
                    best, best_lead = j, lead
        if best is not None:
            used_p.add(best); tp += 1; leads.append(best_lead)
    fn = len(actual_events) - tp
    fp = len(ptimes) - len(used_p)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    mean_lead = float(np.mean(leads)) if leads else float("nan")
    return {"n_actual_events": len(actual_events), "n_pred_alerts": len(ptimes),
            "TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall,
            "mean_lead_min": mean_lead}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="v1 run folder with predictions_lead10_v2.csv / predictions_noct_lead10.csv")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--lead", type=int, default=10)
    ap.add_argument("--co2-threshold", type=float, default=DEFAULT_THRESHOLDS["co2"])
    ap.add_argument("--pm25-threshold", type=float, default=DEFAULT_THRESHOLDS["pm2_5"])
    ap.add_argument("--voc-threshold", type=float, default=DEFAULT_THRESHOLDS["voc"])
    ap.add_argument("--horizon", type=int, default=EVENT_MATCH_HORIZON)
    args = ap.parse_args()

    thresholds = {"co2": args.co2_threshold, "pm2_5": args.pm25_threshold, "voc": args.voc_threshold}

    with_df = pd.read_csv(os.path.join(args.dir, f"predictions_lead{args.lead}_v2.csv"))
    without_df = pd.read_csv(os.path.join(args.dir, f"predictions_noct_lead{args.lead}.csv"))
    for d in (with_df, without_df):
        d["trigger_timestamp"] = pd.to_datetime(d["trigger_timestamp"])
        d["future_timestamp"] = pd.to_datetime(d["future_timestamp"])

    rows = []
    for model in MODEL_ORDER:
        for target, thr in thresholds.items():
            for variant, df in [("with_Ct", with_df), ("without_Ct", without_df)]:
                sub = df[(df.model == model) & (df.target == target)].sort_values("future_timestamp")
                if sub.empty:
                    continue
                sc = event_scoring(sub["actual"].values, sub["future_timestamp"].values,
                                    sub["predicted"].values, sub["trigger_timestamp"].values,
                                    thr, args.horizon)
                row = {"model": model, "target": target, "threshold": thr, "variant": variant}
                row.update(sc)
                rows.append(row)

    out = pd.DataFrame(rows)
    # e.g. threshold=1000.0 -> "1000" (not "10000" -- naively stripping the
    # "." from "1000.0" loses the decimal point instead of the fractional
    # part, which is misleading in a filename; this formats each changed
    # threshold as a clean integer where possible, one decimal otherwise).
    def fmt_thr(v):
        return f"{v:g}"
    changed = [f"{k}-{fmt_thr(v)}" for k, v in thresholds.items() if v != DEFAULT_THRESHOLDS[k]]
    suffix = "_" + "_".join(changed) if changed else ""
    out_csv = os.path.join(args.outdir, f"event_detection_lead{args.lead}{suffix}.csv")
    out.round(4).to_csv(out_csv, index=False)

    print(f"Thresholds used: {thresholds}  (changed from default: {changed or 'none'})")
    print(f"\n=== BiLSTM rows ===")
    with pd.option_context("display.width", 160):
        print(out[out.model == "BiLSTM"].to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
