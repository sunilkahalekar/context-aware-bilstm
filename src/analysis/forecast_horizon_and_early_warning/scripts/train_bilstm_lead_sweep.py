"""
BiLSTM Lead-Time Sweep, T+1 to T+20 min — RUN THIS IN A PYTORCH ENVIRONMENT
================================================================================
Generates a genuine, model-specific answer to "how does BiLSTM's R^2 change
with forecast lead time" -- the earlier LEAD_TIME_ACCURACY.png in this
folder was NOT BiLSTM; it came from the pipeline's lightweight ablation-
proxy GRU (see that figure's caption and analysis/v1/CLAUDE.md). This
script trains the real BiLSTM architecture, six times, once per lead time,
using the exact feature engineering and architecture from
iaq_early_detection_gui_v3.py (lines ~980-1467) -- copied here verbatim,
not reimplemented from memory. Training hyperparameters are set to a
user-specified configuration (see the block below), NOT the GUI's own
defaults: lookback=15, epochs=500 (with early stopping), batch=24,
lr=0.0003, dropout=0.1, hidden=160, layers=1, early-stop patience=50,
70/15/15 train/val/test split.

WHY THIS COULDN'T BE RUN AUTOMATICALLY
This analysis environment has no PyTorch installed and no network access
to install it (pip install fails on SSL certificate verification). The
GUI pipeline was run in a different environment that does have it (it
already produced BiLSTM_ck.pt for the v1 run) -- run this script there.

INCIDENTAL FINDING WHILE EXTRACTING THIS CODE (worth fixing separately)
`sensor_data_merged_iaq_m2.csv` has no M{1,2,3}_op_state, M{1,2,3}_emission_
weight, or M{1,2,3}_consecutive_full_open columns. The pipeline silently
defaults all of these to 0.0 when absent (its own base_cols fallback,
v3.py line ~1024-1025) rather than erroring. That means 18 of the 68
"with-C_t" feature columns in every run so far (3 machines x
[emission_weight, consecutive_full_open, 4 op-state one-hots]) are
constant zero -- present in the feature count, contributing nothing. This
doesn't affect the with/without-C_t comparisons elsewhere in this project
(those 18 columns are zero in both configurations' relevant subsets), but
it does mean the true informative C_t feature count is smaller than 68
suggests, and it's worth generating M{m}_op_state / emission_weight /
consecutive_full_open in the upstream data pipeline if those signals
exist and simply aren't being exported into this CSV yet.

Usage (in an environment with torch installed):
    python train_bilstm_lead_sweep.py --raw <sensor_data_merged_iaq_m2.csv> --outdir <analysis dir>
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

ALL_TGT = ["pm1", "pm2_5", "pm10", "co2", "voc"]
MACHINES = [1, 2, 3]
OP_ST = ["IDLE", "CUTTING", "EXPOSURE", "MAINTENANCE"]
LEADS = [1, 3, 5, 10, 15, 20]

# User-specified training configuration (overrides the GUI's own defaults,
# which were: lookback=20, epochs=200, batch=16, lr=0.001, dropout=0.3,
# hidden=64, n_layers=1, early_pat=30, train/val/test=70/15/15).
LB = 15           # lookback (sequence length fed to the LSTM)
EPOCHS = 500       # max training epochs (early stopping usually cuts this short)
BATCH = 24         # batch size
LR = 0.0003        # AdamW learning rate
DROPOUT = 0.35     # RAISED from the originally-specified 0.1 -- see note below
HIDDEN = 160       # hidden units per LSTM direction (kept as specified)
N_LAYERS = 1       # single LSTM layer
EPAT = 50          # early-stop patience (epochs with no val-loss improvement)
TF = 0.70          # train fraction
VF = 0.15          # val fraction (remaining 0.15 is test)
WEIGHT_DECAY = 3e-3  # RAISED from the pipeline's own default of 1e-3
SEED = 42

# REGULARIZATION NOTE: the first run of this sweep, at dropout=0.1, produced
# deeply negative R^2 at every lead time (-9.6 at T+1 down to -24.0 at
# T+15), getting WORSE at longer leads -- the signature of a hidden=160,
# lightly-regularized model overfitting a training set that shrinks further
# as lookback+lead eat into the ~270 available training rows. Dropout was
# raised 0.1 -> 0.35 and AdamW weight_decay 1e-3 -> 3e-3 to compensate,
# per the user's explicit choice to keep hidden=160 and add regularization
# rather than shrink the model. Re-run and check whether R^2 turns positive
# before trusting these numbers -- if it's still deeply negative, the
# dataset (386 rows) may simply be too small for a 160-unit bidirectional
# LSTM regardless of regularization strength, and reducing HIDDEN is the
# more likely fix at that point.


def build_features(df):
    """Verbatim port of v3.py's feature engineering (lines ~1006-1150),
    assuming every feature-group checkbox is True (the config that
    produced 68 with-Ct features for the v1 run)."""
    for m in MACHINES:
        col = f"M{m}_op_state"
        for s in OP_ST:
            df[f"M{m}_is_{s}"] = (df[col] == s).astype(float) if col in df.columns else 0.0

    base_cols = (["temp", "hum", "pm1", "pm2_5", "pm10", "co2", "voc"] +
                 [f"M{m}_f_trans" for m in MACHINES] +
                 [f"M{m}_rho_open" for m in MACHINES] +
                 [f"M{m}_eps_max" for m in MACHINES] +
                 [f"M{m}_phi_open" for m in MACHINES] +
                 [f"M{m}_emission_weight" for m in MACHINES] +
                 [f"M{m}_effective_tau" for m in MACHINES] +
                 [f"M{m}_consecutive_full_open" for m in MACHINES] +
                 [f"M{m}_is_{s}" for m in MACHINES for s in OP_ST] +
                 ["n_person", "mu_motion", "sigma2_motion"])
    for c in base_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[base_cols] = df[base_cols].ffill().bfill().fillna(0.0)

    door_cols_avail = [f"M{m}_phi_open" for m in MACHINES if f"M{m}_phi_open" in df.columns]
    df["door_open_sum"] = df[door_cols_avail].sum(axis=1) if door_cols_avail else 0.0
    # FIX 15 orientation correction (see iaq_early_detection_gui_v3.py) --
    # keep this, it's the corrected convention used for the v1 checkpoints.
    lo, hi = df["door_open_sum"].min(), df["door_open_sum"].max()
    df["door_open_sum"] = hi + lo - df["door_open_sum"]

    pc, mc = "n_person", "mu_motion"

    df["pm1_diff1"] = df["pm1"].diff(1).bfill()
    df["pm25_diff1"] = df["pm2_5"].diff(1).bfill()
    df["pm10_diff1"] = df["pm10"].diff(1).bfill()
    df["pm1_lag1"] = df["pm1"].shift(1).bfill()
    df["pm1_lag2"] = df["pm1"].shift(2).bfill()
    df["pm1_lag3"] = df["pm1"].shift(3).bfill()
    df["pm25_lag1"] = df["pm2_5"].shift(1).bfill()
    df["pm25_lag2"] = df["pm2_5"].shift(2).bfill()
    df["pm10_lag1"] = df["pm10"].shift(1).bfill()
    df["pm10_lag2"] = df["pm10"].shift(2).bfill()
    df["pm_total"] = df["pm1"] + df["pm2_5"] + df["pm10"]
    df["pm1_roll5"] = df["pm1"].rolling(5, min_periods=1).mean()
    df["pm25_roll5"] = df["pm2_5"].rolling(5, min_periods=1).mean()
    df["pm10_roll5"] = df["pm10"].rolling(5, min_periods=1).mean()
    df["voc_diff1"] = df["voc"].diff(1).bfill()
    df["voc_diff2"] = df["voc"].diff(2).bfill()
    df["co2_lag1"] = df["co2"].shift(1).bfill()
    df["co2_lag2"] = df["co2"].shift(2).bfill()
    df["co2_lag3"] = df["co2"].shift(3).bfill()
    df["co2_roll5"] = df["co2"].rolling(5, min_periods=1).mean()
    df["person_diff"] = df[pc].diff(1).bfill()
    df["door_diff"] = df["door_open_sum"].diff(1).bfill()
    df["motion_roll10"] = df[mc].rolling(10, min_periods=1).mean()
    df["door_exposure"] = (df["door_open_sum"] > 0).astype(float).rolling(10, min_periods=1).sum()
    df["trigger_strength"] = df[pc].clip(0) * df[mc].clip(0)

    feat_cols = (
        ["temp", "hum"] + ["pm1", "pm2_5", "pm10", "co2", "voc"] +
        ["pm1_lag1", "pm1_lag2", "pm1_lag3", "pm25_lag1", "pm25_lag2", "pm10_lag1", "pm10_lag2"] +
        ["pm1_diff1", "pm25_diff1", "pm10_diff1", "pm_total"] +
        ["pm1_roll5", "pm25_roll5", "pm10_roll5"] +
        ["voc_diff1", "voc_diff2"] +
        [pc, "person_diff"] +
        ["door_open_sum", "door_exposure"] +
        ["door_diff"] +
        [mc, "motion_roll10", "trigger_strength"] +
        ["co2_lag1", "co2_lag2", "co2_lag3", "co2_roll5"] +
        [f"M{m}_emission_weight" for m in MACHINES] +
        [f"M{m}_phi_open" for m in MACHINES] +
        [f"M{m}_rho_open" for m in MACHINES] +
        [f"M{m}_eps_max" for m in MACHINES] +
        [f"M{m}_effective_tau" for m in MACHINES] +
        [f"M{m}_f_trans" for m in MACHINES] +
        [f"M{m}_consecutive_full_open" for m in MACHINES] +
        [f"M{m}_is_{s}" for m in MACHINES for s in OP_ST]
    )
    seen, fc = set(), []
    for c in feat_cols:
        if c in df.columns and c not in seen:
            seen.add(c); fc.append(c)

    df[fc] = df[fc].ffill().bfill().fillna(0.0)
    df[ALL_TGT] = df[ALL_TGT].clip(lower=0.0).ffill().bfill().fillna(0.0)
    return df, fc


class Attention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.W = nn.Linear(h, 1, bias=False)
    def forward(self, x):
        scores = self.W(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        return (x * weights.unsqueeze(-1)).sum(1)


class BiLSTM(nn.Module):
    """Verbatim port of iaq_early_detection_gui_v3.py's BiLSTM class."""
    def __init__(self, in_dim, n_out):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, HIDDEN, N_LAYERS, batch_first=True,
                             bidirectional=True, dropout=DROPOUT if N_LAYERS > 1 else 0.0)
        self.attn = Attention(HIDDEN * 2)
        self.drop = nn.Dropout(DROPOUT)
        pm_idx, gas_idx = {0, 1, 2}, {3, 4}
        self.heads = nn.ModuleList()
        for i in range(n_out):
            if i in pm_idx:
                self.heads.append(nn.Sequential(
                    nn.Linear(HIDDEN * 2, HIDDEN), nn.ReLU(),
                    nn.Dropout(DROPOUT * 0.5), nn.Linear(HIDDEN, 1)))
            elif i in gas_idx:
                self.heads.append(nn.Sequential(
                    nn.Linear(HIDDEN * 2, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1)))
            else:
                self.heads.append(nn.Linear(HIDDEN * 2, 1))
    def forward(self, x):
        o, _ = self.lstm(x)
        ctx = self.drop(self.attn(o))
        return torch.cat([hd(ctx) for hd in self.heads], 1)


class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_seqs(Xs, ys, lb, lead):
    Xo, yo = [], []
    for i in range(len(Xs) - lb - lead + 1):
        Xo.append(Xs[i:i + lb])
        yo.append(ys[i + lb + lead - 1])
    return np.array(Xo, dtype=np.float32), np.array(yo, dtype=np.float32)


def r2_score(actual, pred):
    ss_res = np.sum((actual - pred) ** 2, axis=0)
    ss_tot = np.sum((actual - actual.mean(axis=0)) ** 2, axis=0)
    return 1 - ss_res / np.clip(ss_tot, 1e-9, None)


def train_one_lead(df_raw, feat_cols, lead, device):
    df = df_raw.copy()
    n = len(df)
    tr_end, va_end = int(n * TF), int(n * (TF + VF))

    X_raw = df[feat_cols].values.astype(np.float32)
    y_raw = df[ALL_TGT].values.astype(np.float32)

    x_sc = StandardScaler(); x_sc.fit(X_raw[:tr_end])
    Xsc = x_sc.transform(X_raw).astype(np.float32)

    y_sc_dict, y_sc_cols = {}, []
    for i, t in enumerate(ALL_TGT):
        sc = StandardScaler(); sc.fit(y_raw[:tr_end, i].reshape(-1, 1))
        y_sc_dict[t] = sc
        y_sc_cols.append(sc.transform(y_raw[:, i].reshape(-1, 1)))
    y_sc = np.hstack(y_sc_cols).astype(np.float32)
    in_dim, n_out = Xsc.shape[1], len(ALL_TGT)

    X_seq, y_seq = make_seqs(Xsc, y_sc, LB, lead)
    tr_s = max(0, tr_end - LB - lead + 1)
    va_s = max(tr_s, va_end - LB - lead + 1)
    if tr_s < 10:
        raise RuntimeError(f"Too few training sequences at lead={lead}")

    tr_ld = DataLoader(DS(X_seq[:tr_s], y_seq[:tr_s]), BATCH, shuffle=True)
    va_ld = DataLoader(DS(X_seq[tr_s:va_s], y_seq[tr_s:va_s]), BATCH, shuffle=False)
    te_X, te_y_sc = X_seq[va_s:], y_seq[va_s:]
    te_y_actual = y_raw[[min(i + LB + lead - 1, n - 1) for i in range(va_s, len(X_seq))]]

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = BiLSTM(in_dim, n_out).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)
    w_t = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0], device=device)  # fixed loss weights default
    best, pat, best_state = float("inf"), 0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr_ld:
            opt.zero_grad()
            pr = model(xb.to(device)); yt = yb.to(device)
            loss = ((pr - yt) ** 2 * w_t).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([nn.MSELoss()(model(xb.to(device)), yb.to(device)).item() for xb, yb in va_ld]))
        if vl < best - 1e-6:
            best, pat, best_state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= EPAT:
            break
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_sc = model(torch.tensor(te_X, dtype=torch.float32).to(device)).cpu().numpy()
    pred = np.zeros_like(pred_sc)
    for i, t in enumerate(ALL_TGT):
        pred[:, i] = y_sc_dict[t].inverse_transform(pred_sc[:, i].reshape(-1, 1)).ravel()

    r2_per = r2_score(te_y_actual, pred)
    rmse_per = np.sqrt(np.mean((te_y_actual - pred) ** 2, axis=0))
    row = {"lead_min": lead, "n_test": len(te_y_actual),
           "overall_R2": float(np.mean(r2_per)), "overall_RMSE": float(np.mean(rmse_per))}
    for i, t in enumerate(ALL_TGT):
        row[f"{t}_R2"] = float(r2_per[i]); row[f"{t}_RMSE"] = float(rmse_per[i])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                           ("cpu" if args.device == "auto" else args.device))
    print(f"Device: {device}")

    df = pd.read_csv(args.raw)
    df["timestamp_minute"] = pd.to_datetime(df["timestamp_minute"], format="%m-%d-%Y %H:%M")
    df = df.sort_values("timestamp_minute").reset_index(drop=True)
    df, feat_cols = build_features(df)
    print(f"Feature set: {len(feat_cols)} columns (should be 68, matching the v1 with-Ct run)")

    rows = []
    for lead in LEADS:
        print(f"\n=== Training BiLSTM, lead={lead}min ===")
        row = train_one_lead(df, feat_cols, lead, device)
        print(f"  overall_R2={row['overall_R2']:.4f}  overall_RMSE={row['overall_RMSE']:.3f}  n_test={row['n_test']}")
        rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, "bilstm_lead_time_sweep.csv")
    out.round(4).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print("Feed this into lead_time_accuracy.py's charting logic (or rerun it pointed at this CSV)")
    print("to reproduce LEAD_TIME_ACCURACY.png as a genuine BiLSTM-specific version.")


if __name__ == "__main__":
    main()
