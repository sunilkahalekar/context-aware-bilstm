"""
IAQ Early Detection GUI v3 — Context-Aware Lead-Time Forecasting
================================================================
IMPROVEMENTS over v1 (based on output analysis):

  FIX 1 — PM R² was negative (model worse than mean prediction)
           ROOT CAUSE: PM features not included in feature set by default.
           PM1/PM2.5/PM10 are NOT predictable from VOC/CO2 triggers alone;
           they need their own history (PM lags) and machine-state signals.
           SOLUTION: always include PM lags + PM diff + PM rolling as features.

  FIX 2 — Model complexity too high for 386 rows (overfitting)
           Deep models: hidden 128→64, layers 2→1 for small datasets.
           Added L2 weight-decay + stronger dropout when n < 500.
           Added Cosine Annealing LR schedule (better than ReduceOnPlateau
           for small data — avoids premature LR collapse).

  FIX 3 — Loss weights were [1,1,1,2,2] — PM had equal weight to CO2/VOC
           but PM variance is 10-100× smaller in normalised space.
           SOLUTION: per-target adaptive loss weights computed from train
           variance — targets with smaller variance get higher weight.

  FIX 4 — RobustScaler on entire dataset (data leakage)
           SOLUTION: scaler fitted on train split only, then applied to all.

  FIX 5 — No PM-specific momentum features
           Added: pm1_diff, pm25_diff, pm10_diff, pm_roll5, pm_total
           PM spikes are faster (1-3 min) than CO2 (5-10 min), so short
           lags matter more for PM than for CO2.

  FIX 6 — XScaler was RobustScaler (clips outliers but hurts PM spikes)
           SOLUTION: StandardScaler for features — preserves spike magnitude.

  FIX 7 — BiLSTM had fusion layer 64 but only 25 input features.
           Fusion bottleneck is counter-productive for small in_dim.
           SOLUTION: removed fusion layer, direct LSTM input.

  FIX 8 — Attention mechanism added to BiGRU/BiLSTM.
           Attention over time steps helps model focus on the trigger
           moment within the lookback window (key for lead-time forecasting).

  FIX 9 — Pooled R²/RMSE hides whether the vision Ct vector (door-state +
           motion) matters, because ~80% of the session is quiescent and
           trivially predictable from self-lags alone — any model gets
           those minutes almost right, and they dominate the aggregate
           metric by sheer volume.
           SOLUTION: regime-stratified validation — split the test window
           into baseline (quiescent) vs onset (transition/emission) minutes
           using RAW door/motion/machine-state labels (never the model's
           own features, so the split is independent of what's being
           tested), then report R²/RMSE separately per stratum for the
           configured (with-Ct) model against an automatically-trained
           without-Ct counterpart of the identical architecture.

  FIX 10 — R²/RMSE is the wrong metric for an early-warning claim; a model
           that "only" loses 2 points of R² can silently lose all of its
           useful lead time. SOLUTION: reframed as causal-detection —
           Granger-causality / cross-correlation-at-lag between Ct features
           and each pollutant series; event-detection precision/recall/
           mean-lead-time-to-alert at configurable thresholds (with-Ct vs
           without-Ct); and a paired Diebold-Mariano / paired t-test on
           squared errors restricted to onset-window minutes only.

  FIX 11 — Granger causality p-values came back blank for every single
           feature x pollutant pair, with no explanation anywhere in the
           log. ROOT CAUSE: _granger_pvalue() caught every failure —
           including a plain "statsmodels is not installed" ImportError —
           and returned (None, None) silently, so a missing optional
           dependency looked identical to "the test ran and found nothing."
           SOLUTION: _granger_pvalue() now returns a third value, the
           failure reason, and the pipeline (a) reports statsmodels'
           install status before running a single pair, and (b) if any
           pairs still fail, logs the first few distinct reasons (missing
           package, too few usable rows, zero-variance series, or the
           underlying statsmodels exception) so a blank cell is always
           traceable instead of a silent no-op.

  FIX 12 — The Door-State Temporal Encoding Vector (Dt) was incomplete.
           ROOT CAUSE: of Dt's five descriptors — phi_open, rho_open,
           eps_max, effective_tau, and f_trans — only phi_open (plus the
           derived emission_weight) was ever added to feat_cols. rho_open,
           eps_max, effective_tau, and f_trans were loaded into the
           dataframe (base_cols) and used by the standalone causality-lag
           diagnostic, but never reached the model itself as an input
           feature — the with-Ct and without-Ct comparison was therefore
           never actually testing the complete Dt vector described in the
           manuscript.
           SOLUTION: the door-physics feature block (toggle: "Door-state
           descriptors + emission weight per machine") now adds all five
           Dt descriptors — phi_open, rho_open, eps_max, effective_tau,
           f_trans — plus emission_weight, for each machine. No change was
           needed to the without-Ct stripping logic: CT_KEYWORDS already
           listed "rho_open", "eps_max", "effective_tau", and "f_trans"
           defensively, so the without-Ct variant correctly excludes all
           five the moment they are added to feat_cols.

  FIX 13 — No way to compare with-Ct and without-Ct predictions minute-by-
           minute, or to condition on real event timestamps, outside this
           GUI. ROOT CAUSE: only aggregated with-Ct predictions were ever
           exported (predictions_lead{N}_v2.csv); the without-Ct model's
           per-minute predictions, and the real door/motion/cutting
           trigger timestamps used internally by the onset/baseline split,
           were computed but discarded.
           SOLUTION: the rigorous-validation module now also exports
           predictions_noct_lead{N}.csv (without-Ct, same schema as the
           existing with-Ct file) and trigger_events_lead{N}.csv (the
           actual trigger timestamps). Together with the existing with-Ct
           export, this is enough for an external script to build
           continuous, non-binary analyses — e.g. event-proximity-weighted
           accuracy as a function of a decay constant tau — without
           needing the raw sensor CSV or retraining anything.

  FIX 14 — The "Door open sum" panel on every prediction plot ran opposite
           to intuition: it read HIGH during quiet, unoccupied phases and
           LOW during the busy, high-occupancy phases — visible as a
           clean anti-correlation with the People-count panel directly
           above it. ROOT CAUSE: door_open_sum is built from phi_open
           ("opening-phase position"), which the manuscript itself
           describes as reading LOW while a door is early in an active
           opening event and HIGH once it has settled back to idle — the
           reverse of "amount of door openness" the panel label implies.
           SOLUTION: _get_ctx_for_widx() now reflects door_open_sum
           around its own session min/max for DISPLAY ONLY, so the chart
           reads high = more open, matching its label. This does not
           touch the underlying df["door_open_sum"] column, which still
           drives feature engineering, trigger/onset detection, and the
           causality analysis exactly as before — those still use
           phi_open's original convention and have not been re-verified
           against it (see the flagged concern about door_rise's edge
           direction in _build_regime_labels / _get_trigger_timestamps).

  FIX 15 — (tentative) FIX 14 only patched the chart; door_open_sum was
           still in phi_open's original (low=opening, high=idle)
           orientation everywhere else, so feat_cols, door_diff,
           door_exposure, the onset/trigger rising-edge detector
           (door_rise in _build_regime_labels / _get_trigger_timestamps),
           and the causality cross-correlation were all still reading it
           backwards — meaning door_rise was plausibly firing on doors
           settling back to idle, not on doors opening.
           SOLUTION: df["door_open_sum"] is now reversed once, immediately
           after it is built (reflected around its own session min/max),
           before anything derives from it. Every downstream consumer
           therefore inherits the corrected "high = more open" orientation
           automatically, with no changes needed at each call site. FIX
           14's display-only flip in _get_ctx_for_widx has been reverted
           in this version, since flipping an already-corrected value
           again would silently restore the original wrong orientation.
           Tentative: confirm against a real per-machine door sensor
           before treating results built on this as final.

Run:  python iaq_early_detection_gui_v3.py
"""

import os
import sys
import time
import math
import queue
import logging
import warnings
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

warnings.filterwarnings("ignore")

MISSING = []
for pkg in ["numpy", "pandas", "sklearn", "torch", "matplotlib", "seaborn"]:
    try: __import__(pkg if pkg != "sklearn" else "sklearn")
    except ImportError: MISSING.append(pkg)

# ══════════════════════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════════════════════
BG        = "#0D0F14"
SURFACE   = "#161921"
SURFACE2  = "#1E2230"
SURFACE3  = "#252A3A"
ACCENT    = "#F5A623"
ACCENT2   = "#E8522A"
SUCCESS   = "#39D98A"
CYAN      = "#38C8E0"
MAGENTA   = "#E040FB"
TEXT      = "#ECF0F1"
TEXT_MUTE = "#6C7A8C"
BORDER    = "#252A3A"
WARN_YEL  = "#F7C948"

FONT_LABEL = ("Courier New", 10)
FONT_ENTRY = ("Courier New", 10)
FONT_MONO  = ("Courier New", 9)
FONT_SEC   = ("Courier New", 11, "bold")


# ══════════════════════════════════════════════════════════════════════════════
# QUEUE HANDLER
# ══════════════════════════════════════════════════════════════════════════════
class QueueHandler(logging.Handler):
    def __init__(self, q):
        super().__init__(); self.q = q
    def emit(self, record): self.q.put(self.format(record))


# ══════════════════════════════════════════════════════════════════════════════
# WIDGET HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def section_card(parent, title, icon="▸"):
    frame = tk.Frame(parent, bg=SURFACE,
                     highlightbackground=BORDER, highlightthickness=1)
    hdr = tk.Frame(frame, bg=SURFACE3)
    hdr.pack(fill="x")
    tk.Label(hdr, text=f" {icon}  {title}", bg=SURFACE3, fg=ACCENT,
             font=FONT_SEC, anchor="w", padx=10, pady=7).pack(side="left")
    return frame

def row_entry(parent, label, default="", width=32, tooltip=""):
    r = tk.Frame(parent, bg=SURFACE); r.pack(fill="x", padx=14, pady=3)
    tk.Label(r, text=label, bg=SURFACE, fg=TEXT_MUTE,
             font=FONT_LABEL, width=22, anchor="w").pack(side="left")
    var = tk.StringVar(value=default)
    tk.Entry(r, textvariable=var, width=width, bg=SURFACE2, fg=TEXT,
             insertbackground=ACCENT, relief="flat", font=FONT_ENTRY,
             highlightbackground=BORDER, highlightthickness=1,
             highlightcolor=ACCENT).pack(side="left", padx=(4,0))
    if tooltip:
        tk.Label(r, text=f" {tooltip}", bg=SURFACE, fg=TEXT_MUTE,
                 font=("Courier New",8)).pack(side="left", padx=4)
    return var

def row_spin(parent, label, lo, hi, default, step=1, tooltip=""):
    r = tk.Frame(parent, bg=SURFACE); r.pack(fill="x", padx=14, pady=3)
    tk.Label(r, text=label, bg=SURFACE, fg=TEXT_MUTE,
             font=FONT_LABEL, width=22, anchor="w").pack(side="left")
    var = tk.StringVar(value=str(default))
    tk.Spinbox(r, textvariable=var, from_=lo, to=hi, increment=step,
               width=9, bg=SURFACE2, fg=TEXT, buttonbackground=SURFACE3,
               relief="flat", font=FONT_ENTRY,
               highlightbackground=BORDER, highlightthickness=1,
               highlightcolor=ACCENT).pack(side="left", padx=(4,0))
    if tooltip:
        tk.Label(r, text=f" {tooltip}", bg=SURFACE, fg=TEXT_MUTE,
                 font=("Courier New",8)).pack(side="left", padx=4)
    return var

def row_combo(parent, label, choices, default=None, tooltip=""):
    r = tk.Frame(parent, bg=SURFACE); r.pack(fill="x", padx=14, pady=3)
    tk.Label(r, text=label, bg=SURFACE, fg=TEXT_MUTE,
             font=FONT_LABEL, width=22, anchor="w").pack(side="left")
    var = tk.StringVar(value=default or choices[0])
    ttk.Combobox(r, textvariable=var, values=choices,
                 width=14, font=FONT_ENTRY, state="readonly").pack(
                     side="left", padx=(4,0))
    if tooltip:
        tk.Label(r, text=f" {tooltip}", bg=SURFACE, fg=TEXT_MUTE,
                 font=("Courier New",8)).pack(side="left", padx=4)
    return var

def divider(parent, pady=4):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=pady)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
class EarlyDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IAQ Early Detection v3 — Context-Aware Lead-Time Forecasting")
        self.configure(bg=BG)
        self.geometry("1200x940")
        self.minsize(980, 720)
        self._q         = queue.Queue()
        self._stop_flag = threading.Event()
        self._style_ttk()
        self._build_ui()
        self._poll_log()

    def _style_ttk(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure("TCombobox", fieldbackground=SURFACE2, background=SURFACE2,
                    foreground=TEXT, selectbackground=ACCENT,
                    selectforeground=BG, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER, arrowcolor=TEXT)
        s.configure("Amber.Horizontal.TProgressbar",
                    troughcolor=SURFACE2, background=ACCENT,
                    darkcolor=ACCENT, lightcolor=ACCENT, bordercolor=BORDER)

    def _build_ui(self):
        banner = tk.Frame(self, bg=SURFACE3,
                          highlightbackground=ACCENT, highlightthickness=1)
        banner.pack(fill="x")
        tk.Label(banner, text="⚠  IAQ EARLY DETECTION SYSTEM  v3",
                 bg=SURFACE3, fg=ACCENT,
                 font=("Courier New",17,"bold"), padx=18, pady=10).pack(side="left")
        tk.Label(banner,
                 text="PM + CO₂ + VOC Lead-Time Forecasting  ·  "
                      "Door States + Human Motion  ·  T+N Prediction",
                 bg=SURFACE3, fg=TEXT_MUTE,
                 font=("Courier New",9)).pack(side="left")
        tk.Label(banner, text="  IIT BOMBAY  ",
                 bg=ACCENT2, fg=TEXT, font=("Courier New",8,"bold"),
                 padx=6, pady=2).pack(side="right", padx=14)

        pane = tk.PanedWindow(self, orient="horizontal", bg=BG,
                              sashwidth=5, sashrelief="flat", bd=0)
        pane.pack(fill="both", expand=True, padx=10, pady=8)
        left = tk.Frame(pane, bg=BG)
        right = tk.Frame(pane, bg=BG)
        pane.add(left, minsize=480)
        pane.add(right, minsize=440)
        self._build_left(left)
        self._build_right(right)

    # ── LEFT ──────────────────────────────────────────────────────────────────
    def _build_left(self, parent):
        cv  = tk.Canvas(parent, bg=BG, highlightthickness=0)
        vsb = tk.Scrollbar(parent, orient="vertical", command=cv.yview,
                           bg=SURFACE2, troughcolor=BG)
        cv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(cv, bg=BG)
        win   = cv.create_window((0,0), window=inner, anchor="nw")
        cv.bind("<Configure>", lambda e: cv.itemconfig(win, width=e.width))
        inner.bind("<Configure>",
                   lambda e: cv.configure(scrollregion=cv.bbox("all")))
        inner.bind_all("<MouseWheel>",
            lambda e: cv.yview_scroll(-1*(e.delta//120),"units"))

        self._sec_data(inner)
        self._sec_window(inner)
        self._sec_lead(inner)
        self._sec_features(inner)
        self._sec_models(inner)
        self._sec_targets(inner)
        self._sec_training(inner)
        self._sec_output(inner)
        self._sec_validation(inner)
        self._sec_run(inner)

    # ── SECTIONS ──────────────────────────────────────────────────────────────
    def _sec_data(self, p):
        c = section_card(p, "DATA SOURCE", "①"); c.pack(fill="x",padx=6,pady=(6,3))
        r = tk.Frame(c, bg=SURFACE); r.pack(fill="x", padx=14, pady=5)
        tk.Label(r, text="CSV file", bg=SURFACE, fg=TEXT_MUTE,
                 font=FONT_LABEL, width=22, anchor="w").pack(side="left")
        self.csv_var = tk.StringVar()
        tk.Entry(r, textvariable=self.csv_var, width=26, bg=SURFACE2, fg=TEXT,
                 insertbackground=ACCENT, relief="flat", font=FONT_ENTRY,
                 highlightbackground=BORDER, highlightthickness=1,
                 highlightcolor=ACCENT).pack(side="left", padx=(4,4))
        tk.Button(r, text="Browse", command=self._browse_csv,
                  bg=ACCENT, fg=BG, relief="flat",
                  font=("Courier New",9,"bold"), cursor="hand2",
                  padx=8).pack(side="left")
        self.ts_col_var  = row_entry(c, "Timestamp column", "timestamp_minute")
        self.door_col_var   = row_entry(c, "Door open column", "",
                                        tooltip="e.g. M1_phi_open  (blank=auto)")
        self.person_col_var = row_entry(c, "Person count column", "n_person")
        self.motion_col_var = row_entry(c, "Motion column", "mu_motion")
        tk.Frame(c, bg=BG, height=4).pack()

    def _sec_window(self, p):
        c = section_card(p, "TIME WINDOW", "②"); c.pack(fill="x",padx=6,pady=3)
        tk.Label(c, text="  Leave blank → use full test split automatically",
                 bg=SURFACE, fg=TEXT_MUTE,
                 font=("Courier New",8), anchor="w").pack(fill="x", padx=14)
        # self.start_var = row_entry(c, "Start datetime", "2024-01-15 08:00:00")
        # self.end_var   = row_entry(c, "End datetime",   "2024-01-15 20:00:00",
        #                            tooltip="YYYY-MM-DD HH:MM:SS")
        self.start_var = row_entry(c, "Start datetime", "12-02-2026 12:58:00")
        self.end_var   = row_entry(c, "End datetime",   "12-02-2026 18:00:00",
                                   tooltip="DD-MM-YYYY HH:MM")
        tk.Frame(c, bg=BG, height=4).pack()

    def _sec_lead(self, p):
        c = section_card(p, "LEAD-TIME SETTINGS", "③"); c.pack(fill="x",padx=6,pady=3)

        expl = tk.Frame(c, bg=SURFACE3,
                        highlightbackground=ACCENT, highlightthickness=1)
        expl.pack(fill="x", padx=14, pady=(4,6))
        for ln in [
            "  NEW in v2: PM lags always included as features",
            "  X[t…t+lookback] → y[t+lookback+lead-1]",
            "  Attention mechanism focuses on trigger moment",
            "  Adaptive loss weights balance PM vs CO₂/VOC",
        ]:
            tk.Label(expl, text=ln, bg=SURFACE3, fg=WARN_YEL,
                     font=("Courier New",8), anchor="w").pack(
                         fill="x", padx=6, pady=1)
        tk.Frame(expl, bg=BG, height=3).pack()

        self.lead_var     = row_spin(c, "Lead steps (minutes)", 1, 60, 10,
                                     tooltip="T+N ahead to predict")
        self.lookback_var = row_spin(c, "Lookback window",      5, 120, 20,
                                     tooltip="history steps fed to model")

        r = tk.Frame(c, bg=SURFACE); r.pack(fill="x", padx=14, pady=4)
        self.align_trigger_var = tk.BooleanVar(value=True)
        tk.Checkbutton(r, variable=self.align_trigger_var,
                       text="Plot prediction at TRIGGER TIME "
                            "(magenta at door-open, not at future time)",
                       bg=SURFACE, fg=MAGENTA, selectcolor=SURFACE3,
                       activebackground=SURFACE, activeforeground=MAGENTA,
                       font=("Courier New",9), cursor="hand2",
                       wraplength=400, justify="left").pack(side="left")
        tk.Frame(c, bg=BG, height=4).pack()

    def _sec_features(self, p):
        c = section_card(p, "FEATURE ENGINEERING", "④"); c.pack(fill="x",padx=6,pady=3)

        info = tk.Frame(c, bg=SURFACE3,
                        highlightbackground=SUCCESS, highlightthickness=1)
        info.pack(fill="x", padx=14, pady=(4,6))
        for ln in [
            "  FIX v2: PM lags (1-3min) and PM diff now always ON",
            "  PM spikes are 1-3 min fast; CO₂ is 5-10 min slow.",
            "  'Raw sensors' adds current PM/CO₂/VOC readings too.",
        ]:
            tk.Label(info, text=ln, bg=SURFACE3, fg=SUCCESS,
                     font=("Courier New",8), anchor="w").pack(
                         fill="x", padx=6, pady=1)
        tk.Frame(info, bg=BG, height=2).pack()

        self._feat_vars = {}
        feats = [
            ("use_pm_lags",       True,  SUCCESS, "PM lags 1-3  (ALWAYS recommended for PM R²)"),
            ("use_pm_diff",       True,  SUCCESS, "PM momentum diff1  (PM spike detection)"),
            ("use_pm_roll",       True,  SUCCESS, "PM rolling mean 5-step"),
            ("use_voc_diff",      True,  MAGENTA, "VOC momentum diff1 + diff2"),
            ("use_person_diff",   True,  CYAN,    "Person count Δ  (entry/exit events)"),
            ("use_door_sum",      True,  ACCENT,  "Door open sum + exposure rolling"),
            ("use_door_diff",     True,  ACCENT,  "Door open Δ  (open-event pulse)"),
            ("use_motion_roll",   True,  SUCCESS, "Motion rolling mean + trigger strength"),
            ("use_co2_lags",      True,  TEXT,    "CO₂ lags 1-3 + rolling mean"),
            ("use_emission_wt",   True,  TEXT,    "Dt: phi_open+rho_open+eps_max+eff_tau+f_trans+emission_wt"),
            ("use_consecutive",   True,  TEXT,    "Consecutive full-open counter"),
            ("use_env",           True,  TEXT,    "Environment: temp, humidity"),
            ("use_raw_sensors",   True,  WARN_YEL,"Raw current sensor values (PM/CO₂/VOC)"),
            ("use_op_state_onehot",True, TEXT,    "Op-state one-hot (IDLE/CUT/EXP/MAINT)"),
        ]

        ff = tk.Frame(c, bg=SURFACE); ff.pack(fill="x", padx=14, pady=4)
        for i, (key, default, color, label) in enumerate(feats):
            var = tk.BooleanVar(value=default)
            self._feat_vars[key] = var
            fr = tk.Frame(ff, bg=SURFACE)
            fr.grid(row=i//2, column=i%2, sticky="w", padx=4, pady=1)
            tk.Checkbutton(fr, variable=var, text=f"  {label}",
                           bg=SURFACE, fg=color, selectcolor=SURFACE3,
                           activebackground=SURFACE, activeforeground=color,
                           font=("Courier New",9), cursor="hand2").pack(side="left")
        tk.Frame(c, bg=BG, height=4).pack()

    def _sec_models(self, p):
        c = section_card(p, "MODELS", "⑤"); c.pack(fill="x",padx=6,pady=3)

        qs = tk.Frame(c, bg=SURFACE); qs.pack(fill="x", padx=14, pady=(4,2))
        tk.Label(qs, text="Quick select:", bg=SURFACE, fg=TEXT_MUTE,
                 font=("Courier New",8)).pack(side="left", padx=(0,6))
        for lbl, cmd in [("All",self._sel_all),("Deep",self._sel_deep),
                          ("ML",self._sel_ml),("None",self._sel_none)]:
            tk.Button(qs, text=lbl, command=cmd, bg=SURFACE3, fg=TEXT,
                      relief="flat", font=("Courier New",8), cursor="hand2",
                      padx=6, pady=2).pack(side="left", padx=2)

        self._model_vars = {}
        deep = [
            ("BiGRU",            True,  "★ Best trigger model — BiGRU + Attention"),
            ("BiLSTM",  True,  "★ BiLSTM + Attention"),
            ("GRU",              True,  "GRU + Attention"),
            ("LSTM_uni",         False, "Unidirectional LSTM"),
            ("VanillaRNN",       False, "Vanilla RNN"),
            ("TCN",              True,  "Temporal Conv Net"),
            ("Seq2Seq",          False, "Encoder-Decoder"),
            ("CNN_LSTM",         False, "CNN + LSTM"),
            ("Transformer",      False, "Transformer"),
            ("PatchTST",         False, "PatchTST (ICLR23)"),
        ]
        ml = [
            ("RandomForest",     True,  "Random Forest (strong baseline)"),
            ("XGBoost",          True,  "XGBoost"),
            ("LinearRegression", False, "Linear Regression"),
            ("Ridge",            False, "Ridge"),
            ("SVR",              False, "SVR"),
        ]

        tk.Label(c, text="  Deep / Sequence models",
                 bg=SURFACE, fg=CYAN,
                 font=("Courier New",9,"bold"), anchor="w").pack(fill="x",padx=14)
        self._checkgrid(c, deep)
        divider(c, 2)
        tk.Label(c, text="  Traditional ML baselines",
                 bg=SURFACE, fg=ACCENT,
                 font=("Courier New",9,"bold"), anchor="w").pack(fill="x",padx=14)
        self._checkgrid(c, ml)
        tk.Frame(c, bg=BG, height=4).pack()

    def _checkgrid(self, parent, items):
        g = tk.Frame(parent, bg=SURFACE); g.pack(fill="x",padx=14,pady=2)
        for i, (name, default, desc) in enumerate(items):
            var = tk.BooleanVar(value=default)
            self._model_vars[name] = var
            fr = tk.Frame(g, bg=SURFACE)
            fr.grid(row=i//2, column=i%2, sticky="w", padx=4, pady=1)
            tk.Checkbutton(fr, variable=var, text=name, bg=SURFACE, fg=TEXT,
                           selectcolor=SURFACE3, activebackground=SURFACE,
                           activeforeground=TEXT,
                           font=("Courier New",9), cursor="hand2").pack(side="left")
            tk.Label(fr, text=f"  {desc}", bg=SURFACE, fg=TEXT_MUTE,
                     font=("Courier New",8)).pack(side="left")

    def _sec_targets(self, p):
        c = section_card(p, "POLLUTANT TARGETS", "⑥"); c.pack(fill="x",padx=6,pady=3)
        self._target_vars = {}
        r = tk.Frame(c, bg=SURFACE); r.pack(fill="x", padx=14, pady=6)
        for name, default, label, color in [
            ("pm1",   True, "PM1   μg/m³",  CYAN),
            ("pm2_5", True, "PM2.5 μg/m³",  CYAN),
            ("pm10",  True, "PM10  μg/m³",  CYAN),
            ("co2",   True, "CO₂   ppm",    MAGENTA),
            ("voc",   True, "VOC   ppb",    MAGENTA),
        ]:
            var = tk.BooleanVar(value=default)
            self._target_vars[name] = var
            f = tk.Frame(r, bg=SURFACE); f.pack(side="left", padx=10)
            tk.Checkbutton(f, variable=var, text=label, bg=SURFACE, fg=color,
                           selectcolor=SURFACE3, activebackground=SURFACE,
                           font=("Courier New",9), cursor="hand2").pack()
        tk.Frame(c, bg=BG, height=4).pack()

    def _sec_training(self, p):
        c = section_card(p, "TRAINING PARAMETERS", "⑦"); c.pack(fill="x",padx=6,pady=3)

        info2 = tk.Frame(c, bg=SURFACE3,
                         highlightbackground=CYAN, highlightthickness=1)
        info2.pack(fill="x", padx=14, pady=(4,6))
        for ln in [
            "  v2: Smaller models (hidden=64) for n<500 rows — less overfitting",
            "  v2: Cosine Annealing LR — better convergence on small data",
            "  v2: Adaptive loss weights from train-set variance",
        ]:
            tk.Label(info2, text=ln, bg=SURFACE3, fg=CYAN,
                     font=("Courier New",8), anchor="w").pack(
                         fill="x", padx=6, pady=1)
        tk.Frame(info2, bg=BG, height=2).pack()

        self.epochs_var   = row_spin(c, "Epochs",          1, 2000, 200)
        self.batch_var    = row_spin(c, "Batch size",       4,  512,  16,
                                     tooltip="smaller=better for 386 rows")
        self.lr_var       = row_entry(c, "Learning rate",  "0.001")
        self.dropout_var  = row_entry(c, "Dropout",        "0.3",
                                      tooltip="higher=less overfit for small data")
        self.hidden_var   = row_spin(c, "Hidden size",     16,  512,  64,
                                     tooltip="64 recommended for <500 rows")
        self.n_layers_var = row_spin(c, "RNN layers",       1,    4,   1,
                                     tooltip="1 layer avoids overfit on small data")
        self.early_var    = row_spin(c, "Early stop pat.", 1,  200,  30)
        self.seed_var     = row_spin(c, "Random seed",     0, 9999,  42)
        self.train_frac   = row_entry(c, "Train fraction", "0.70")
        self.val_frac     = row_entry(c, "Val fraction",   "0.15")
        self.device_var   = row_combo(c, "Device",
                                      ["auto","cpu","cuda","mps"])

        # Adaptive loss weight toggle
        r = tk.Frame(c, bg=SURFACE); r.pack(fill="x", padx=14, pady=4)
        self.adaptive_loss_var = tk.BooleanVar(value=True)
        tk.Checkbutton(r, variable=self.adaptive_loss_var,
                       text="Adaptive loss weights  "
                            "(per-target weight ∝ 1/train_variance  → balances PM vs CO₂/VOC)",
                       bg=SURFACE, fg=SUCCESS, selectcolor=SURFACE3,
                       activebackground=SURFACE, activeforeground=SUCCESS,
                       font=("Courier New",9), cursor="hand2",
                       wraplength=420, justify="left").pack(side="left")
        tk.Frame(c, bg=BG, height=4).pack()

    def _sec_output(self, p):
        c = section_card(p, "OUTPUT", "⑧"); c.pack(fill="x",padx=6,pady=3)
        r = tk.Frame(c, bg=SURFACE); r.pack(fill="x", padx=14, pady=5)
        tk.Label(r, text="Output directory", bg=SURFACE, fg=TEXT_MUTE,
                 font=FONT_LABEL, width=22, anchor="w").pack(side="left")
        self.outdir_var = tk.StringVar(value="results_early_v3")
        tk.Entry(r, textvariable=self.outdir_var, width=22, bg=SURFACE2,
                 fg=TEXT, insertbackground=ACCENT, relief="flat",
                 font=FONT_ENTRY, highlightbackground=BORDER,
                 highlightthickness=1, highlightcolor=ACCENT
                 ).pack(side="left", padx=(4,4))
        tk.Button(r, text="Browse", command=self._browse_outdir,
                  bg=ACCENT, fg=BG, relief="flat",
                  font=("Courier New",9,"bold"), cursor="hand2",
                  padx=8).pack(side="left")
        opts = tk.Frame(c, bg=SURFACE); opts.pack(fill="x", padx=14, pady=(0,4))
        self.save_csv_var     = tk.BooleanVar(value=True)
        self.save_plots_var   = tk.BooleanVar(value=True)
        self.save_leadgap_var = tk.BooleanVar(value=True)
        for var, txt, color in [
            (self.save_csv_var,     "Save prediction CSV",        TEXT),
            (self.save_plots_var,   "Save PNG plots",             TEXT),
            (self.save_leadgap_var, "Save lead-gap comparison",   MAGENTA),
        ]:
            tk.Checkbutton(opts, variable=var, text=txt, bg=SURFACE, fg=color,
                           selectcolor=SURFACE3, activebackground=SURFACE,
                           font=("Courier New",9), cursor="hand2"
                           ).pack(side="left", padx=6)
        tk.Frame(c, bg=BG, height=4).pack()

    def _sec_validation(self, p):
        c = section_card(p, "RIGOROUS VALIDATION (Regime + Causal + Lead-Time)", "⑨")
        c.pack(fill="x",padx=6,pady=3)

        info3 = tk.Frame(c, bg=SURFACE3,
                         highlightbackground=MAGENTA, highlightthickness=1)
        info3.pack(fill="x", padx=14, pady=(4,6))
        for ln in [
            "  Pooled R²/RMSE masks whether the Ct (door+motion) vector",
            "  matters, because ~80% of the session is quiescent baseline.",
            "  This section conditions the metric on regime instead, and",
            "  reframes accuracy as a causal-detection + lead-time claim.",
        ]:
            tk.Label(info3, text=ln, bg=SURFACE3, fg=MAGENTA,
                     font=("Courier New",8), anchor="w").pack(
                         fill="x", padx=6, pady=1)
        tk.Frame(info3, bg=BG, height=2).pack()

        r = tk.Frame(c, bg=SURFACE); r.pack(fill="x", padx=14, pady=2)
        self.val_regime_var = tk.BooleanVar(value=True)
        tk.Checkbutton(r, variable=self.val_regime_var,
                       text="Regime-stratified R²/RMSE (baseline vs onset),"
                            " with an auto-trained without-Ct counterpart",
                       bg=SURFACE, fg=SUCCESS, selectcolor=SURFACE3,
                       activebackground=SURFACE, activeforeground=SUCCESS,
                       font=("Courier New",9), cursor="hand2",
                       wraplength=420, justify="left").pack(side="left")

        r2_ = tk.Frame(c, bg=SURFACE); r2_.pack(fill="x", padx=14, pady=2)
        self.val_event_var = tk.BooleanVar(value=True)
        tk.Checkbutton(r2_, variable=self.val_event_var,
                       text="Event-detection precision / recall / mean lead-time"
                            " at alert thresholds (with-Ct vs without-Ct)",
                       bg=SURFACE, fg=CYAN, selectcolor=SURFACE3,
                       activebackground=SURFACE, activeforeground=CYAN,
                       font=("Courier New",9), cursor="hand2",
                       wraplength=420, justify="left").pack(side="left")

        r3_ = tk.Frame(c, bg=SURFACE); r3_.pack(fill="x", padx=14, pady=2)
        self.val_causal_var = tk.BooleanVar(value=True)
        tk.Checkbutton(r3_, variable=self.val_causal_var,
                       text="Granger causality + cross-correlation-at-lag"
                            " (Ct features → each pollutant)",
                       bg=SURFACE, fg=ACCENT, selectcolor=SURFACE3,
                       activebackground=SURFACE, activeforeground=ACCENT,
                       font=("Courier New",9), cursor="hand2",
                       wraplength=420, justify="left").pack(side="left")

        r4_ = tk.Frame(c, bg=SURFACE); r4_.pack(fill="x", padx=14, pady=2)
        self.val_dm_var = tk.BooleanVar(value=True)
        tk.Checkbutton(r4_, variable=self.val_dm_var,
                       text="Diebold-Mariano + paired t-test on onset-window"
                            " squared errors (with-Ct vs without-Ct)",
                       bg=SURFACE, fg=WARN_YEL, selectcolor=SURFACE3,
                       activebackground=SURFACE, activeforeground=WARN_YEL,
                       font=("Courier New",9), cursor="hand2",
                       wraplength=420, justify="left").pack(side="left")

        r5_ = tk.Frame(c, bg=SURFACE); r5_.pack(fill="x", padx=14, pady=2)
        self.val_phase_var = tk.BooleanVar(value=True)
        tk.Checkbutton(r5_, variable=self.val_phase_var,
                       text="Phase-wise R² (Phase 1-4), with-Ct vs without-Ct"
                            " per model — same cutoffs as the plot bands",
                       bg=SURFACE, fg=CYAN, selectcolor=SURFACE3,
                       activebackground=SURFACE, activeforeground=CYAN,
                       font=("Courier New",9), cursor="hand2",
                       wraplength=420, justify="left").pack(side="left")

        divider(c, 4)
        self.onset_horizon_var = row_spin(
            c, "Onset window (min)", 1, 60, 15,
            tooltip="minutes after a door/motion/cut trigger counted as 'onset'")
        self.motion_thr_var = row_entry(
            c, "Motion trigger thresh.", "0.5",
            tooltip="mu_motion rising above this = onset trigger")
        self.voc_alert_var  = row_entry(c, "VOC alert (ppb)",       "200")
        self.co2_alert_var  = row_entry(c, "CO2 alert (ppm)",       "2000")
        self.pm25_alert_var = row_entry(c, "PM2.5 alert (µg/m³)",   "35")
        self.pm10_alert_var = row_entry(c, "PM10 alert (µg/m³)",    "",
                                        tooltip="blank = skip event test for this target")
        self.pm1_alert_var  = row_entry(c, "PM1 alert (µg/m³)",     "",
                                        tooltip="blank = skip event test for this target")
        self.causal_maxlag_var = row_spin(c, "Causality max lag (min)", 1, 60, 20)
        self.event_match_horizon_var = row_spin(
            c, "Event match horizon (min)", 1, 90, 30,
            tooltip="max minutes a predicted alert may precede its matched actual event")
        tk.Frame(c, bg=BG, height=4).pack()

    def _sec_run(self, p):
        tk.Frame(p, bg=BG, height=6).pack()
        self.run_btn = tk.Button(
            p, text="▶  RUN EARLY DETECTION v3",
            command=self._start, bg=ACCENT, fg=BG, relief="flat",
            font=("Courier New",13,"bold"), cursor="hand2",
            padx=20, pady=13, activebackground=ACCENT2,
            activeforeground=TEXT)
        self.run_btn.pack(fill="x", padx=6)
        self.stop_btn = tk.Button(
            p, text="■  STOP", command=self._stop,
            bg=ACCENT2, fg=TEXT, relief="flat",
            font=("Courier New",10,"bold"), cursor="hand2",
            pady=6, state="disabled", activebackground="#C04040")
        self.stop_btn.pack(fill="x", padx=6, pady=(4,0))
        self.progress = ttk.Progressbar(p, mode="indeterminate",
                                        style="Amber.Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=6, pady=(5,0))
        self.status_var = tk.StringVar(value="Ready — configure and press Run")
        tk.Label(p, textvariable=self.status_var, bg=BG, fg=TEXT_MUTE,
                 font=("Courier New",8), anchor="w").pack(
                     fill="x", padx=6, pady=(3,0))

    # ── RIGHT ─────────────────────────────────────────────────────────────────
    def _build_right(self, parent):
        lhdr = tk.Frame(parent, bg=BG); lhdr.pack(fill="x", pady=(0,4))
        tk.Label(lhdr, text="▸  LIVE TRAINING LOG",
                 bg=BG, fg=ACCENT, font=FONT_SEC).pack(side="left")
        tk.Button(lhdr, text="Clear", command=self._clear_log,
                  bg=SURFACE2, fg=TEXT_MUTE, relief="flat",
                  font=("Courier New",8), cursor="hand2",
                  padx=5).pack(side="right")
        tk.Button(lhdr, text="Copy", command=self._copy_log,
                  bg=SURFACE2, fg=TEXT_MUTE, relief="flat",
                  font=("Courier New",8), cursor="hand2",
                  padx=5).pack(side="right", padx=3)

        lf = tk.Frame(parent, bg=SURFACE,
                      highlightbackground=ACCENT, highlightthickness=1)
        lf.pack(fill="both", expand=True)
        self.log_text = tk.Text(lf, bg="#080A10", fg="#A8FF78",
                                font=FONT_MONO, wrap="word", state="disabled",
                                relief="flat", padx=8, pady=8,
                                selectbackground=SURFACE3,
                                insertbackground=ACCENT)
        vsb = tk.Scrollbar(lf, command=self.log_text.yview,
                           bg=SURFACE2, troughcolor=BG)
        self.log_text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)
        for tag, fg in [("ok","#A8FF78"),("warn","#F7C948"),("error","#F76E6E"),
                        ("accent","#F5A623"),("cyan","#38C8E0"),
                        ("magenta","#E040FB"),("mute","#4A5568"),
                        ("success","#39D98A")]:
            self.log_text.tag_configure(tag, foreground=fg)

        divider(parent, 6)
        tk.Label(parent, text="▸  IMPROVEMENTS v3",
                 bg=BG, fg=ACCENT, font=FONT_SEC).pack(anchor="w")
        ec = tk.Frame(parent, bg=SURFACE3,
                      highlightbackground=ACCENT2, highlightthickness=1)
        ec.pack(fill="x", pady=3)
        for lbl, color, desc in [
            ("FIX 1 →", SUCCESS, "PM lags always included → fixes negative PM R²"),
            ("FIX 2 →", CYAN,   "Smaller models (hidden=64) → less overfit on 386 rows"),
            ("FIX 3 →", MAGENTA,"Adaptive loss weights → balances PM vs CO₂/VOC"),
            ("FIX 4 →", ACCENT, "StandardScaler fit on TRAIN only → no data leakage"),
            ("FIX 5 →", SUCCESS,"Attention in BiGRU/BiLSTM → focuses on trigger moment"),
            ("FIX 6 →", WARN_YEL,"Cosine Annealing LR → better small-data convergence"),
            ("FIX 9 →", SUCCESS,"Regime-stratified R²/RMSE unmasks Ct's real value"),
            ("FIX 10→", MAGENTA,"Causal-detection + lead-time replaces pooled R² claim"),
            ("FIX 12→", CYAN,   "Full 5-descriptor Dt vector now reaches the model"),
        ]:
            r = tk.Frame(ec, bg=SURFACE3); r.pack(fill="x", padx=10, pady=2)
            tk.Label(r, text=lbl, bg=SURFACE3, fg=color,
                     font=("Courier New",9,"bold"), width=10, anchor="w").pack(side="left")
            tk.Label(r, text=desc, bg=SURFACE3, fg=TEXT_MUTE,
                     font=("Courier New",9), anchor="w").pack(side="left")

        divider(parent, 6)
        tk.Label(parent, text="▸  METRICS SUMMARY",
                 bg=BG, fg=ACCENT, font=FONT_SEC).pack(anchor="w")
        self.metrics_text = tk.Text(parent, bg=SURFACE2, fg=TEXT,
                                    font=FONT_MONO, height=11, state="disabled",
                                    relief="flat", padx=8, pady=6,
                                    highlightbackground=BORDER,
                                    highlightthickness=1)
        self.metrics_text.pack(fill="x")

    # ── HELPERS ───────────────────────────────────────────────────────────────
    def _sel_all(self):
        for v in self._model_vars.values(): v.set(True)
    def _sel_none(self):
        for v in self._model_vars.values(): v.set(False)
    def _sel_deep(self):
        deep = {"BiGRU","BiLSTM","GRU","LSTM_uni","VanillaRNN",
                "TCN","Seq2Seq","CNN_LSTM","Transformer","PatchTST"}
        for k,v in self._model_vars.items(): v.set(k in deep)
    def _sel_ml(self):
        ml = {"RandomForest","XGBoost","LinearRegression","Ridge","SVR"}
        for k,v in self._model_vars.items(): v.set(k in ml)

    def _browse_csv(self):
        p = filedialog.askopenfilename(
            title="Select merged IAQ CSV",
            filetypes=[("CSV files","*.csv"),("All","*.*")])
        if p: self.csv_var.set(p)

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p: self.outdir_var.set(p)

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0","end")
        self.log_text.configure(state="disabled")

    def _copy_log(self):
        self.clipboard_clear()
        self.clipboard_append(self.log_text.get("1.0","end"))

    def _append_log(self, msg):
        self.log_text.configure(state="normal")
        ml = msg.lower()
        tag = "ok"
        if any(w in ml for w in ["error","fail","✗","exception"]): tag="error"
        elif any(w in ml for w in ["warn","⚠"]): tag="warn"
        elif any(w in ml for w in ["fix","r²","rmse","mae"]): tag="cyan"
        elif any(w in ml for w in ["done","✓","saved"]): tag="success"
        elif any(w in ml for w in ["model:","training","═","─"]): tag="accent"
        elif any(w in ml for w in ["pm","lead","trigger","attention"]): tag="magenta"
        self.log_text.insert("end", msg+"\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _poll_log(self):
        try:
            while True: self._append_log(self._q.get_nowait())
        except queue.Empty: pass
        self.after(100, self._poll_log)

    def _update_metrics(self, text):
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0","end")
        self.metrics_text.insert("1.0", text)
        self.metrics_text.configure(state="disabled")

    def _validate(self):
        errs = []
        if not self.csv_var.get().strip():
            errs.append("CSV file path is required.")
        elif not Path(self.csv_var.get().strip()).exists():
            errs.append(f"CSV not found:\n  {self.csv_var.get()}")
        if not any(v.get() for v in self._model_vars.values()):
            errs.append("Select at least one model.")
        if not any(v.get() for v in self._target_vars.values()):
            errs.append("Select at least one target.")
        for lbl, var in [("LR", self.lr_var), ("Dropout", self.dropout_var)]:
            try: float(var.get())
            except: errs.append(f"{lbl} must be a number.")
        for lbl, var in [("Motion trigger thresh.", self.motion_thr_var)]:
            try: float(var.get())
            except: errs.append(f"{lbl} must be a number.")
        if errs:
            messagebox.showerror("Config Errors", "\n\n".join(errs))
            return False
        return True

    def _start(self):
        if not self._validate(): return
        if MISSING:
            messagebox.showerror("Missing packages",
                "Install:\n  pip install " + " ".join(MISSING))
            return
        self.run_btn.configure(state="disabled", text="Running…")
        self.stop_btn.configure(state="normal")
        self.progress.start(10)
        self._stop_flag.clear()
        self._clear_log()
        cfg = self._collect()
        threading.Thread(target=self._pipeline, args=(cfg,), daemon=True).start()

    def _stop(self):
        self._stop_flag.set()
        self.status_var.set("Stopping after current model…")

    def _finish(self, ok, summary=""):
        self.run_btn.configure(state="normal",
                               text="▶  RUN EARLY DETECTION v3")
        self.stop_btn.configure(state="disabled")
        self.progress.stop()
        self.status_var.set("Done ✓" if ok else "Stopped / Error")
        if summary: self._update_metrics(summary)

    def _collect(self):
        return {
            "csv":            self.csv_var.get().strip(),
            "ts_col":         self.ts_col_var.get().strip() or "timestamp_minute",
            "door_col":       self.door_col_var.get().strip(),
            "person_col":     self.person_col_var.get().strip() or "n_person",
            "motion_col":     self.motion_col_var.get().strip() or "mu_motion",
            "start":          self.start_var.get().strip(),
            "end":            self.end_var.get().strip(),
            "lead_steps":     int(self.lead_var.get()),
            "lookback":       int(self.lookback_var.get()),
            "align_trigger":  self.align_trigger_var.get(),
            "features":       {k: v.get() for k,v in self._feat_vars.items()},
            "models":         [k for k,v in self._model_vars.items() if v.get()],
            "targets":        [k for k,v in self._target_vars.items() if v.get()],
            "epochs":         int(self.epochs_var.get()),
            "batch":          int(self.batch_var.get()),
            "lr":             float(self.lr_var.get()),
            "dropout":        float(self.dropout_var.get()),
            "hidden":         int(self.hidden_var.get()),
            "n_layers":       int(self.n_layers_var.get()),
            "early_pat":      int(self.early_var.get()),
            "seed":           int(self.seed_var.get()),
            "train_frac":     float(self.train_frac.get()),
            "val_frac":       float(self.val_frac.get()),
            "device":         self.device_var.get(),
            "adaptive_loss":  self.adaptive_loss_var.get(),
            "outdir":         self.outdir_var.get().strip() or "results_early_v3",
            "save_csv":       self.save_csv_var.get(),
            "save_plots":     self.save_plots_var.get(),
            "save_leadgap":   self.save_leadgap_var.get(),
            "val_regime":     self.val_regime_var.get(),
            "val_event":      self.val_event_var.get(),
            "val_causal":     self.val_causal_var.get(),
            "val_dm":         self.val_dm_var.get(),
            "val_phase":      self.val_phase_var.get(),
            "onset_horizon":  int(self.onset_horizon_var.get()),
            "motion_thr":     float(self.motion_thr_var.get()),
            "causal_maxlag":  int(self.causal_maxlag_var.get()),
            "event_match_horizon": int(self.event_match_horizon_var.get()),
            "alerts": {
                "voc":   self.voc_alert_var.get().strip(),
                "co2":   self.co2_alert_var.get().strip(),
                "pm2_5": self.pm25_alert_var.get().strip(),
                "pm10":  self.pm10_alert_var.get().strip(),
                "pm1":   self.pm1_alert_var.get().strip(),
            },
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PIPELINE
    # ══════════════════════════════════════════════════════════════════════════
    def _pipeline(self, cfg):
        import numpy  as np
        import pandas as pd
        import torch, torch.nn as nn
        import torch.nn.functional as F
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates  as mdates
        import seaborn as sns
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.linear_model  import LinearRegression, Ridge
        from sklearn.ensemble      import RandomForestRegressor
        from sklearn.svm           import SVR
        from sklearn.multioutput   import MultiOutputRegressor
        from sklearn.metrics       import (mean_squared_error,
                                           mean_absolute_error, r2_score)
        from torch.utils.data import Dataset, DataLoader

        emit = lambda msg: self._q.put(msg)
        out  = cfg["outdir"]
        os.makedirs(out, exist_ok=True)

        SEED     = cfg["seed"]
        LEAD     = cfg["lead_steps"]
        LB       = cfg["lookback"]
        EPOCHS   = cfg["epochs"]
        BATCH    = cfg["batch"]
        LR       = cfg["lr"]
        DROPOUT  = cfg["dropout"]
        HIDDEN   = cfg["hidden"]
        N_LAYERS = cfg["n_layers"]
        EPAT     = cfg["early_pat"]
        TF       = cfg["train_frac"]
        VF       = cfg["val_frac"]
        ALL_TGT  = ["pm1","pm2_5","pm10","co2","voc"]
        TARGETS  = cfg["targets"]
        MACHINES = [1,2,3]
        OP_ST    = ["IDLE","CUTTING","EXPOSURE","MAINTENANCE"]
        n_out    = len(ALL_TGT)

        torch.manual_seed(SEED); np.random.seed(SEED)
        dev_str = cfg["device"]
        device  = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            if dev_str == "auto" else dev_str)

        emit("═"*62)
        emit("  IAQ EARLY DETECTION PIPELINE  v3")
        emit(f"  Lead={LEAD}min  Lookback={LB}  Hidden={HIDDEN}  Layers={N_LAYERS}")
        emit(f"  Device={device}  Seed={SEED}")
        emit("═"*62)

        # ── LOAD ──────────────────────────────────────────────────────────────
        self.after(0, lambda: self.status_var.set("Loading CSV…"))
        emit(f"\n  Loading: {cfg['csv']}")
        df = pd.read_csv(cfg["csv"])
        ts_col = cfg["ts_col"]
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.sort_values(ts_col).reset_index(drop=True)
        else:
            emit(f"  ⚠  '{ts_col}' not found — using row index")
            df[ts_col] = pd.date_range("2024-01-01", periods=len(df), freq="1min")

        # op-state one-hot
        for m in MACHINES:
            col = f"M{m}_op_state"
            for s in OP_ST:
                df[f"M{m}_is_{s}"] = (
                    (df[col] == s).astype(float)
                    if col in df.columns else 0.0)

        base_cols = (["temp","hum","pm1","pm2_5","pm10","co2","voc"] +
                     [f"M{m}_f_trans"  for m in MACHINES] +
                     [f"M{m}_rho_open" for m in MACHINES] +
                     [f"M{m}_eps_max"  for m in MACHINES] +
                     [f"M{m}_phi_open" for m in MACHINES] +
                     [f"M{m}_emission_weight"      for m in MACHINES] +
                     [f"M{m}_effective_tau"         for m in MACHINES] +
                     [f"M{m}_consecutive_full_open" for m in MACHINES] +
                     [f"M{m}_is_{s}" for m in MACHINES for s in OP_ST] +
                     ["n_person","mu_motion","sigma2_motion"])
        for c in base_cols:
            if c not in df.columns: df[c] = 0.0
        df[base_cols] = df[base_cols].ffill().bfill().fillna(0.0)

        n      = len(df)
        tr_end = int(n * TF)
        va_end = int(n * (TF + VF))
        emit(f"  Rows={n}  Train={tr_end}  Val={va_end-tr_end}  Test={n-va_end}")
        if n < 100:
            emit("  ✗ Too few rows (<100). Aborting.")
            self.after(0, lambda: self._finish(False)); return

        # ── FEATURE ENGINEERING ────────────────────────────────────────────────
        emit("\n  Building features…")
        feat_cfg = cfg["features"]

        # door signal
        door_cols_avail = [f"M{m}_phi_open" for m in MACHINES
                           if f"M{m}_phi_open" in df.columns]
        user_door = cfg.get("door_col","")
        if user_door and user_door in df.columns:
            df["door_open_sum"] = df[user_door]
        else:
            df["door_open_sum"] = df[door_cols_avail].sum(axis=1) if door_cols_avail else 0.0

        # FIX 15 (tentative — see FIX 14): phi_open reads LOW while a door
        # is actively/freshly opening and HIGH once it has settled back to
        # idle, the opposite of "amount of door openness" every downstream
        # consumer assumes. FIX 14 patched only the chart's display value;
        # this instead reverses door_open_sum once, at the source, right
        # after it's built, so every consumer inherits the corrected
        # orientation consistently: door_diff, door_exposure, and the
        # door_open_sum feature itself (all computed below from this same
        # column), the onset/trigger rising-edge detector in
        # _build_regime_labels/_get_trigger_timestamps (door_rise now
        # correctly fires when a door is opening, not settling), and the
        # causality cross-correlation against each pollutant. Because the
        # value is now correct at the source, _get_ctx_for_widx's FIX-14
        # display flip has been reverted below (undoing it there — flipping
        # twice would silently restore the original, wrong orientation).
        # Marked tentative: this assumes the FIX-14 hypothesis about
        # phi_open's convention is correct. Verify against a ground-truth
        # per-machine door sensor before citing results built on this.
        if "door_open_sum" in df.columns:
            _door_lo0 = float(df["door_open_sum"].min())
            _door_hi0 = float(df["door_open_sum"].max())
            df["door_open_sum"] = _door_hi0 + _door_lo0 - df["door_open_sum"]

        pc = cfg["person_col"] if cfg["person_col"] in df.columns else "n_person"
        mc = cfg["motion_col"] if cfg["motion_col"] in df.columns else "mu_motion"

        # ── FIX 5: PM momentum features (always computed, selectively added) ──
        df["pm1_diff1"]   = df["pm1"].diff(1).bfill()
        df["pm25_diff1"]  = df["pm2_5"].diff(1).bfill()
        df["pm10_diff1"]  = df["pm10"].diff(1).bfill()
        df["pm1_lag1"]    = df["pm1"].shift(1).bfill()
        df["pm1_lag2"]    = df["pm1"].shift(2).bfill()
        df["pm1_lag3"]    = df["pm1"].shift(3).bfill()
        df["pm25_lag1"]   = df["pm2_5"].shift(1).bfill()
        df["pm25_lag2"]   = df["pm2_5"].shift(2).bfill()
        df["pm10_lag1"]   = df["pm10"].shift(1).bfill()
        df["pm10_lag2"]   = df["pm10"].shift(2).bfill()
        df["pm_total"]    = df["pm1"] + df["pm2_5"] + df["pm10"]
        df["pm1_roll5"]   = df["pm1"].rolling(5,  min_periods=1).mean()
        df["pm25_roll5"]  = df["pm2_5"].rolling(5, min_periods=1).mean()
        df["pm10_roll5"]  = df["pm10"].rolling(5, min_periods=1).mean()

        # VOC + CO2 momentum
        df["voc_diff1"]   = df["voc"].diff(1).bfill()
        df["voc_diff2"]   = df["voc"].diff(2).bfill()
        df["co2_lag1"]    = df["co2"].shift(1).bfill()
        df["co2_lag2"]    = df["co2"].shift(2).bfill()
        df["co2_lag3"]    = df["co2"].shift(3).bfill()
        df["co2_roll5"]   = df["co2"].rolling(5, min_periods=1).mean()

        # Trigger / door / motion
        df["person_diff"]   = df[pc].diff(1).bfill()
        df["door_diff"]     = df["door_open_sum"].diff(1).bfill()
        df["motion_roll10"] = df[mc].rolling(10, min_periods=1).mean()
        df["door_exposure"] = (df["door_open_sum"]>0).astype(float)\
                               .rolling(10,min_periods=1).sum()
        df["trigger_strength"] = df[pc].clip(0) * df[mc].clip(0)

        emit("  FIX 1: PM lags, diffs, rolling means computed")
        emit("  FIX 5: Trigger features (door/person/motion) computed")

        # ── SELECT FEATURES ────────────────────────────────────────────────────
        feat_cols = []
        if feat_cfg.get("use_env"):
            feat_cols += ["temp","hum"]
        if feat_cfg.get("use_raw_sensors"):
            feat_cols += ["pm1","pm2_5","pm10","co2","voc"]
        # FIX 1: PM features — always include when checkbox is on
        if feat_cfg.get("use_pm_lags"):
            feat_cols += ["pm1_lag1","pm1_lag2","pm1_lag3",
                          "pm25_lag1","pm25_lag2",
                          "pm10_lag1","pm10_lag2"]
        if feat_cfg.get("use_pm_diff"):
            feat_cols += ["pm1_diff1","pm25_diff1","pm10_diff1","pm_total"]
        if feat_cfg.get("use_pm_roll"):
            feat_cols += ["pm1_roll5","pm25_roll5","pm10_roll5"]
        if feat_cfg.get("use_voc_diff"):
            feat_cols += ["voc_diff1","voc_diff2"]
        if feat_cfg.get("use_person_diff"):
            feat_cols += [pc, "person_diff"]
        if feat_cfg.get("use_door_sum"):
            feat_cols += ["door_open_sum","door_exposure"]
        if feat_cfg.get("use_door_diff"):
            feat_cols += ["door_diff"]
        if feat_cfg.get("use_motion_roll"):
            feat_cols += [mc,"motion_roll10","trigger_strength"]
        if feat_cfg.get("use_co2_lags"):
            feat_cols += ["co2_lag1","co2_lag2","co2_lag3","co2_roll5"]
        if feat_cfg.get("use_emission_wt"):
            # FIX 12: all five Dt descriptors (phi_open, rho_open, eps_max,
            # effective_tau, f_trans), plus the derived emission_weight —
            # previously only phi_open + emission_weight were included.
            feat_cols += [f"M{m}_emission_weight" for m in MACHINES] + \
                         [f"M{m}_phi_open"        for m in MACHINES] + \
                         [f"M{m}_rho_open"        for m in MACHINES] + \
                         [f"M{m}_eps_max"         for m in MACHINES] + \
                         [f"M{m}_effective_tau"   for m in MACHINES] + \
                         [f"M{m}_f_trans"         for m in MACHINES]
        if feat_cfg.get("use_consecutive"):
            feat_cols += [f"M{m}_consecutive_full_open" for m in MACHINES]
        if feat_cfg.get("use_op_state_onehot"):
            feat_cols += [f"M{m}_is_{s}" for m in MACHINES for s in OP_ST]

        # deduplicate
        seen = set(); fc = []
        for c in feat_cols:
            if c in df.columns and c not in seen:
                seen.add(c); fc.append(c)
        feat_cols = fc or ["pm1_lag1","pm25_lag1","voc_diff1",
                           "door_open_sum","person_diff","motion_roll10"]

        df[feat_cols] = df[feat_cols].ffill().bfill().fillna(0.0)
        df[ALL_TGT]   = df[ALL_TGT].clip(lower=0.0).ffill().bfill().fillna(0.0)
        emit(f"\n  Feature set ({len(feat_cols)} cols): {feat_cols}")

        # ── FIX 4: Scale fit on TRAIN only ────────────────────────────────────
        emit("\n  FIX 4: Fitting StandardScaler on TRAIN split only…")
        X_raw = df[feat_cols].values.astype(np.float32)
        y_raw = df[ALL_TGT].values.astype(np.float32)

        x_sc  = StandardScaler()                        # FIX 6: StandardScaler
        x_sc.fit(X_raw[:tr_end])
        Xsc   = x_sc.transform(X_raw).astype(np.float32)

        y_sc_dict = {}
        y_sc_cols = []
        for i, t in enumerate(ALL_TGT):
            sc = MinMaxScaler()
            sc.fit(y_raw[:tr_end, i].reshape(-1,1))    # train only
            y_sc_dict[t] = sc
            y_sc_cols.append(sc.transform(y_raw[:,i].reshape(-1,1)))
        y_sc   = np.hstack(y_sc_cols).astype(np.float32)
        in_dim = Xsc.shape[1]

        # ── FIX 3: ADAPTIVE LOSS WEIGHTS ──────────────────────────────────────
        if cfg["adaptive_loss"]:
            train_var = y_sc[:tr_end].var(axis=0).clip(1e-6)
            # weight ∝ 1/variance → low-variance targets (PM) get more emphasis
            raw_w  = 1.0 / train_var
            LOSS_W = (raw_w / raw_w.mean()).tolist()
            emit(f"\n  FIX 3: Adaptive loss weights: "
                 f"{ {t: round(w,3) for t,w in zip(ALL_TGT,LOSS_W)} }")
        else:
            LOSS_W = [1.0,1.0,1.0,2.0,2.0]
            emit(f"\n  Loss weights (fixed): {LOSS_W}")

        # ── SEQUENCES ─────────────────────────────────────────────────────────
        emit(f"\n  Sequence: X[t…t+{LB}] → y[t+{LB+LEAD-1}]  (Lead={LEAD}min)")

        def make_seqs(Xs, ys):
            Xo, yo = [], []
            for i in range(len(Xs) - LB - LEAD + 1):
                Xo.append(Xs[i:i+LB])
                yo.append(ys[i+LB+LEAD-1])
            return np.array(Xo,dtype=np.float32), np.array(yo,dtype=np.float32)

        X_seq, y_seq = make_seqs(Xsc, y_sc)
        tr_s = max(0, tr_end - LB - LEAD + 1)
        va_s = max(tr_s, va_end - LB - LEAD + 1)
        emit(f"  Sequences: total={len(X_seq)}  train={tr_s}  val={va_s-tr_s}")
        if tr_s < 10:
            emit("  ✗ Too few training sequences. "
                 "Reduce lead + lookback or increase dataset size.")
            self.after(0, lambda: self._finish(False)); return

        class DS(Dataset):
            def __init__(self,X,y):
                self.X=torch.tensor(X,dtype=torch.float32)
                self.y=torch.tensor(y,dtype=torch.float32)
            def __len__(self): return len(self.X)
            def __getitem__(self,i): return self.X[i],self.y[i]

        def mk(X,y,sh):
            return DataLoader(DS(X,y),BATCH,shuffle=sh,drop_last=False)

        tr_ld = mk(X_seq[:tr_s],      y_seq[:tr_s],      True)
        va_ld = mk(X_seq[tr_s:va_s],  y_seq[tr_s:va_s],  False)

        # ── FIX 8: ATTENTION MODULE ────────────────────────────────────────────
        class Attention(nn.Module):
            """Additive attention over time-steps — focuses on trigger moment."""
            def __init__(self, h):
                super().__init__()
                self.W = nn.Linear(h, 1, bias=False)
            def forward(self, x):
                # x: (B, T, H)
                scores = self.W(x).squeeze(-1)          # (B, T)
                weights = F.softmax(scores, dim=1)       # (B, T)
                ctx = (x * weights.unsqueeze(-1)).sum(1) # (B, H)
                return ctx

        # ── FIX 2+7+8: SIMPLIFIED MODELS WITH ATTENTION ───────────────────────
        class BiGRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru  = nn.GRU(in_dim, HIDDEN, N_LAYERS, batch_first=True,
                                   bidirectional=True,
                                   dropout=DROPOUT if N_LAYERS>1 else 0.0)
                self.attn = Attention(HIDDEN*2)
                self.drop = nn.Dropout(DROPOUT)
                pm_idx  = {0,1,2}   # pm1, pm2_5, pm10 get deeper heads
                gas_idx = {3,4}     # co2, voc
                self.heads = nn.ModuleList()
                for i in range(n_out):
                    if i in pm_idx:
                        self.heads.append(nn.Sequential(
                            nn.Linear(HIDDEN*2, HIDDEN), nn.ReLU(),
                            nn.Dropout(DROPOUT*0.5), nn.Linear(HIDDEN,1)))
                    elif i in gas_idx:
                        self.heads.append(nn.Sequential(
                            nn.Linear(HIDDEN*2, HIDDEN), nn.GELU(),
                            nn.Linear(HIDDEN,1)))
                    else:
                        self.heads.append(nn.Linear(HIDDEN*2, 1))
            def forward(self, x):
                o,_ = self.gru(x)
                ctx = self.drop(self.attn(o))  # attention over all time steps
                return torch.cat([hd(ctx) for hd in self.heads], 1)

        class BiLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(in_dim, HIDDEN, N_LAYERS, batch_first=True,
                                    bidirectional=True,
                                    dropout=DROPOUT if N_LAYERS>1 else 0.0)
                self.attn = Attention(HIDDEN*2)
                self.drop = nn.Dropout(DROPOUT)
                pm_idx  = {0,1,2}
                gas_idx = {3,4}
                self.heads = nn.ModuleList()
                for i in range(n_out):
                    if i in pm_idx:
                        self.heads.append(nn.Sequential(
                            nn.Linear(HIDDEN*2,HIDDEN), nn.ReLU(),
                            nn.Dropout(DROPOUT*0.5), nn.Linear(HIDDEN,1)))
                    elif i in gas_idx:
                        self.heads.append(nn.Sequential(
                            nn.Linear(HIDDEN*2,HIDDEN), nn.GELU(),
                            nn.Linear(HIDDEN,1)))
                    else:
                        self.heads.append(nn.Linear(HIDDEN*2,1))
            def forward(self, x):
                o,_ = self.lstm(x)
                ctx = self.drop(self.attn(o))
                return torch.cat([hd(ctx) for hd in self.heads], 1)

        class GRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.g    = nn.GRU(in_dim, HIDDEN, N_LAYERS, batch_first=True,
                                   dropout=DROPOUT if N_LAYERS>1 else 0.0)
                self.attn = Attention(HIDDEN)
                self.drop = nn.Dropout(DROPOUT)
                self.fc   = nn.Linear(HIDDEN, n_out)
            def forward(self,x):
                o,_ = self.g(x)
                return self.fc(self.drop(self.attn(o)))

        class LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.l  = nn.LSTM(in_dim, HIDDEN, N_LAYERS, batch_first=True,
                                  dropout=DROPOUT if N_LAYERS>1 else 0.0)
                self.fc = nn.Linear(HIDDEN, n_out)
            def forward(self,x):
                o,_ = self.l(x); return self.fc(o[:,-1,:])

        class RNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.r  = nn.RNN(in_dim, HIDDEN, N_LAYERS, batch_first=True,
                                 dropout=DROPOUT if N_LAYERS>1 else 0.0)
                self.fc = nn.Linear(HIDDEN, n_out)
            def forward(self,x):
                o,_ = self.r(x); return self.fc(o[:,-1,:])

        class _Ch(nn.Module):
            def __init__(self,s): super().__init__(); self.s=s
            def forward(self,x): return x[:,:,:-self.s] if self.s else x

        class _RB(nn.Module):
            def __init__(self,ic,oc,k,d,dr):
                super().__init__(); p=(k-1)*d
                self.net=nn.Sequential(
                    nn.Conv1d(ic,oc,k,dilation=d,padding=p),
                    _Ch(p),nn.ReLU(),nn.Dropout(dr),
                    nn.Conv1d(oc,oc,k,dilation=d,padding=p),
                    _Ch(p),nn.ReLU(),nn.Dropout(dr))
                self.dw=nn.Conv1d(ic,oc,1) if ic!=oc else None
                self.act=nn.ReLU()
            def forward(self,x):
                return self.act(self.net(x)+(self.dw(x) if self.dw else x))

        class TCN(nn.Module):
            def __init__(self):
                super().__init__()
                ch = min(HIDDEN, 64)
                self.net=nn.Sequential(
                    *[_RB(in_dim if i==0 else ch,ch,3,2**i,DROPOUT)
                      for i in range(3)])     # 3 levels (was 4) for small data
                self.fc=nn.Linear(ch,n_out)
            def forward(self,x):
                return self.fc(self.net(x.permute(0,2,1))[:,:,-1])

        class S2S(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc=nn.LSTM(in_dim,HIDDEN,1,batch_first=True)
                self.dec=nn.LSTM(n_out, HIDDEN,1,batch_first=True)
                self.fc=nn.Linear(HIDDEN,n_out)
            def forward(self,x):
                _,(h,c)=self.enc(x)
                di=torch.zeros(x.size(0),1,n_out,device=x.device)
                o,_=self.dec(di,(h,c)); return self.fc(o[:,-1,:])

        class CNNLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                ch = min(HIDDEN, 64)
                self.c=nn.Sequential(
                    nn.Conv1d(in_dim,ch,3,padding=1),nn.ReLU(),
                    nn.Conv1d(ch,ch,3,padding=1),nn.ReLU())
                self.l=nn.LSTM(ch,HIDDEN,1,batch_first=True)
                self.fc=nn.Linear(HIDDEN,n_out)
            def forward(self,x):
                o,_=self.l(self.c(x.permute(0,2,1)).permute(0,2,1))
                return self.fc(o[:,-1,:])

        class _PE(nn.Module):
            def __init__(self,d):
                super().__init__()
                pe=torch.zeros(512,d)
                pos=torch.arange(0,512).unsqueeze(1).float()
                div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000)/d))
                pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
                self.register_buffer("pe",pe.unsqueeze(0))
            def forward(self,x): return x+self.pe[:,:x.size(1)]

        class Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                d=max(32, HIDDEN)
                nhead = 4 if d>=64 else 2
                self.proj=nn.Linear(in_dim,d); self.pe=_PE(d)
                self.enc=nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d,nhead,d*2,DROPOUT,
                                               batch_first=True),1)
                self.fc=nn.Linear(d,n_out)
            def forward(self,x):
                return self.fc(self.enc(self.pe(self.proj(x)))[:,-1,:])

        class PatchTST(nn.Module):
            def __init__(self):
                super().__init__()
                pl=min(4,LB); d=max(32,HIDDEN); self.pl=pl; self.id=in_dim
                nhead = 4 if d>=64 else 2
                self.pp=nn.Linear(pl,d); self.pe=_PE(d)
                self.enc=nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d,nhead,d*2,DROPOUT,
                                               batch_first=True),1)
                self.fc=nn.Linear(d*in_dim,n_out)
            def forward(self,x):
                B,T,F=x.shape; nb=max(1,T//self.pl)
                xp=x[:,:nb*self.pl,:].reshape(B,nb,self.pl,F)
                xp=xp.permute(0,3,1,2).reshape(B*F,nb,self.pl)
                h=self.enc(self.pe(self.pp(xp)))[:,-1,:]
                return self.fc(h.reshape(B,F*h.shape[-1]))

        MODEL_MAP = {
            "BiGRU":           BiGRU,
            "BiLSTM": BiLSTM,
            "GRU":             GRU,
            "LSTM_uni":        LSTM,
            "VanillaRNN":      RNN,
            "TCN":             TCN,
            "Seq2Seq":         S2S,
            "CNN_LSTM":        CNNLSTM,
            "Transformer":     Transformer,
            "PatchTST":        PatchTST,
        }

        # ── FIX 2+6: TRAINING WITH COSINE ANNEALING ───────────────────────────
        def train_model(model, name):
            model = model.to(device)
            opt   = torch.optim.AdamW(model.parameters(), lr=LR,
                                      weight_decay=1e-3)    # stronger L2
            # FIX 6: Cosine annealing with warm restarts
            sch   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        opt, T_0=30, T_mult=2)
            w_t   = torch.tensor(LOSS_W, device=device, dtype=torch.float32)
            ckpt  = f"{out}/{name}_ck.pt"
            best  = float("inf"); pat = 0
            for ep in range(1, EPOCHS+1):
                if self._stop_flag.is_set(): break
                model.train()
                for xb,yb in tr_ld:
                    opt.zero_grad()
                    pr = model(xb.to(device)); yt = yb.to(device)
                    loss = ((pr-yt)**2 * w_t).mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sch.step()
                model.eval()
                with torch.no_grad():
                    vl = float(np.mean([
                        nn.MSELoss()(model(xb.to(device)), yb.to(device)).item()
                        for xb,yb in va_ld]))
                if vl < best - 1e-6:
                    best=vl; pat=0
                    torch.save(model.state_dict(), ckpt)
                else:
                    pat+=1
                if pat >= EPAT: break
                if ep%50==0 or ep==1:
                    emit(f"    ep{ep:>4}  val={vl:.5f}  best={best:.5f}")
            model.load_state_dict(
                torch.load(ckpt, map_location=device, weights_only=True))
            return model

        # ── INVERSE TRANSFORM ─────────────────────────────────────────────────
        def inv_t(arr):
            out_ = np.zeros_like(arr)
            for i,t in enumerate(ALL_TGT):
                out_[:,i] = y_sc_dict[t].inverse_transform(
                    arr[:,i].reshape(-1,1)).ravel()
            return out_

        # ── WINDOW MASK ───────────────────────────────────────────────────────
        st = cfg["start"].strip(); en = cfg["end"].strip()
        if st and en:
            wmask = ((df[ts_col] >= pd.to_datetime(st)) &
                     (df[ts_col] <= pd.to_datetime(en)))
        else:
            wmask = pd.Series([False]*n)
            wmask.iloc[va_end:] = True
        emit(f"\n  Window rows: {wmask.sum()}")
        if wmask.sum() == 0:
            emit("  ✗ No rows in window."); self.after(0, lambda: self._finish(False)); return

        def predict_deep(model):
            model.eval()
            widx = [i for i in np.where(wmask.values)[0]
                    if i >= LB and (i+LEAD) < n]
            if not widx: return None,None,None,[]
            ps,ac = [],[]
            with torch.no_grad():
                for i in widx:
                    xb = torch.tensor(Xsc[i-LB:i][None],
                                      dtype=torch.float32).to(device)
                    ps.append(model(xb).cpu().numpy()[0])
                    ac.append(y_sc[min(i+LEAD, n-1)])
            p_inv = inv_t(np.array(ps))
            a_inv = inv_t(np.array(ac))
            ts_trig = df[ts_col].iloc[widx].values
            ts_fut  = df[ts_col].iloc[[min(i+LEAD,n-1) for i in widx]].values
            return a_inv, p_inv, ts_trig, ts_fut

        def predict_ml(clf):
            widx = [i for i in np.where(wmask.values)[0]
                    if i >= LB and (i+LEAD) < n]
            if not widx: return None,None,None,[]
            Xf = np.array([Xsc[i-LB:i].ravel() for i in widx])
            p  = clf.predict(Xf)
            a  = y_raw[np.array([min(i+LEAD,n-1) for i in widx])]
            ts_trig = df[ts_col].iloc[widx].values
            ts_fut  = df[ts_col].iloc[[min(i+LEAD,n-1) for i in widx]].values
            return a, p, ts_trig, ts_fut

        # ── METRICS ───────────────────────────────────────────────────────────
        def metrics(a, p):
            res = {}
            for i,t in enumerate(ALL_TGT):
                at=a[:,i]; pt=p[:,i]
                res[t] = {
                    "RMSE": float(np.sqrt(mean_squared_error(at,pt))),
                    "MAE":  float(mean_absolute_error(at,pt)),
                    "R2":   float(r2_score(at,pt)) if at.std()>1e-6 else 0.0}
            res["overall"] = {
                "RMSE": float(np.sqrt(mean_squared_error(a,p))),
                "MAE":  float(mean_absolute_error(a,p)),
                "R2":   float(r2_score(a.ravel(),p.ravel()))}
            return res

        UNITS = {"pm1":"μg/m³","pm2_5":"μg/m³","pm10":"μg/m³",
                 "co2":"ppm","voc":"ppb"}
        PAL   = {"BiGRU":"#1565C0","BiLSTM":"#B71C1C",
                 "GRU":"#1B5E20","LSTM_uni":"#0277BD","VanillaRNN":"#546E7A",
                 "TCN":"#4A148C","Seq2Seq":"#004D40","CNN_LSTM":"#BF360C",
                 "Transformer":"#E65100","PatchTST":"#006064",
                 "LinearRegression":"#78909C","Ridge":"#607D8B",
                 "RandomForest":"#263238","SVR":"#37474F","XGBoost":"#1A237E"}

        # ── LIGHT THEME TOKENS ────────────────────────────────────────────────
        BG_PLOT  = "#F7F5F0"   # warm off-white page
        AX_BG    = "#FFFFFF"   # pure white axes
        GR_C     = "#E5E2D8"   # subtle warm grid
        SP_C     = "#BCBAB0"   # spine colour
        TX_C     = "#0A0A00"   # near-black text
        TX_MUTE  = "#0E0E01"   # muted labels
        STRIP_BG = "#F0EDE5"   # alternating panel background

        # Machine colour palette — one warm tone per machine
        M_COLS = {1: "#D32F2F", 2: "#1565C0", 3: "#2E7D32"}  # red/blue/green

        def _style_light(ax, alt=False):
            ax.set_facecolor(STRIP_BG if alt else AX_BG)
            ax.tick_params(colors=TX_MUTE, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(SP_C); sp.set_linewidth(0.5)
            ax.grid(True, alpha=0.50, lw=0.4, color=GR_C, zorder=0)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        # ── BRIGHT / HIGH-RES THEME — used only by plot_early() ──────────────
        BRIGHT_BG   = "#FFFFFF"   # pure white page (was warm off-white BG_PLOT)
        BRIGHT_AXBG = "#FFFFFF"   # pure white axes
        BRIGHT_STRIP= "#F5F7FA"   # very light alternating strip (barely tinted)
        BRIGHT_GRID = "#D5D8DD"   # slightly stronger grid for print clarity
        BRIGHT_SP   = "#9AA0AA"   # slightly darker spine for definition
        BRIGHT_TX   = "#1A1A1A"   # near-black text, higher contrast
        BRIGHT_MUTE = "#55606B"   # darker muted label than TX_MUTE

        def _style_bright(ax, alt=False):
            ax.set_facecolor(BRIGHT_STRIP if alt else BRIGHT_AXBG)
            ax.tick_params(colors=BRIGHT_MUTE, labelsize=9)
            for sp in ax.spines.values():
                sp.set_edgecolor(BRIGHT_SP); sp.set_linewidth(0.8)
            ax.grid(True, alpha=0.55, lw=0.6, color=BRIGHT_GRID, zorder=0)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        # ── CONTEXT SIGNAL EXTRACTOR ──────────────────────────────────────────
        def _get_ctx_for_widx(widx_list):
            ctx_ts   = pd.to_datetime(df[ts_col].iloc[widx_list].values)
            ctx_per  = np.array([float(df[pc].iloc[i]) if pc in df.columns
                                  else 0.0 for i in widx_list])
            ctx_mot  = np.array([float(df[mc].iloc[i]) if mc in df.columns
                                  else 0.0 for i in widx_list])
            ctx_mvar = np.array([float(df[mc].iloc[max(0,i-4):i+1].std())
                                  if mc in df.columns else 0.0
                                  for i in widx_list])
            # FIX 14 flipped this value for display only, reflecting it
            # around the session min/max because df["door_open_sum"] was
            # still in phi_open's original (backwards) orientation at that
            # point. FIX 15 now reverses door_open_sum once at the source
            # (right after it's built, before any feature derives from it),
            # so this panel is already correctly oriented straight from the
            # column — flipping it again here would silently cancel FIX 15
            # out and restore the original wrong orientation. Read plainly.
            ctx_door = np.array([float(df["door_open_sum"].iloc[i])
                                  if "door_open_sum" in df.columns else 0.0
                                  for i in widx_list])
            return ctx_ts, ctx_per, ctx_mot, ctx_mvar, ctx_door

        def _get_machine_features(widx_list, m_num):
            """
            Extract door physics features for one machine (M1/M2/M3).
            Returns dict of {feature_short_name: np.array}.
            """
            feats = {}
            col_map = {
                "rho_open":   f"M{m_num}_rho_open",
                "eps_max":    f"M{m_num}_eps_max",
                "phi_open":   f"M{m_num}_phi_open",
                "em_weight":  f"M{m_num}_emission_weight",
                "eff_tau":    f"M{m_num}_effective_tau",
                "consec":     f"M{m_num}_consecutive_full_open",
            }
            for short, col in col_map.items():
                if col in df.columns:
                    feats[short] = np.array(
                        [float(df[col].iloc[i]) for i in widx_list])
                else:
                    feats[short] = np.zeros(len(widx_list))
            return feats

        # ── ANNOTATION HELPERS ────────────────────────────────────────────────
        def _draw_lead_bracket(ax, ts_x, p_, ts_a, a, LEAD, ylo, yhi):
            """
            Draw a horizontal bracket showing the LEAD-time gap between
            when the prediction fires (trigger time) and when the actual
            peak arrives.  Only drawn near the maximum predicted value.
            """
            if len(p_) < 5 or len(ts_x) == 0: return
            peak_idx = int(np.argmax(p_))
            if peak_idx >= len(ts_x): return
            t_trigger = ts_x[peak_idx]
            # find closest future ts
            if peak_idx < len(ts_a):
                t_future = ts_a[peak_idx]
            else:
                return
            yb = ylo + (yhi-ylo)*0.04
            try:
                ax.annotate(
                    "", xy=(t_future, yb),
                    xytext=(t_trigger, yb),
                    arrowprops=dict(arrowstyle="<->",
                                    color="#555550", lw=1.2,
                                    connectionstyle="arc3,rad=0"))
                mid_t = t_trigger + (t_future - t_trigger)/2
                ax.text(mid_t, yb+(yhi-ylo)*0.025,
                        f"+{LEAD} min lead",
                        ha="center", va="bottom", fontsize=7,
                        color="#555550",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="#FFFDE7", ec="#BCBAB0",
                                  lw=0.5, alpha=0.85))
            except Exception:
                pass

        # ── PHASE DEFINITIONS ─────────────────────────────────────────────────

        def _phase_boundaries(ts_ref, t_start=None, t_end=None):
            """
            Build the 4 operational-phase boundary datetimes using the
            calendar date present in ts_ref (the session's own date).
            If t_start/t_end are given (the axes' true x-limits), the
            first/last band is anchored to them instead of the data's
            min/max timestamp — guaranteeing Phase 1 and Phase 4 render
            as complete rectangles flush to the plot edge and the first
            /last divider line, even when the plotted data begins at or
            after a phase boundary.
            """
            ts_ref = pd.to_datetime(ts_ref)
            if len(ts_ref) == 0:
                return []
            day    = pd.Timestamp(ts_ref[0]).normalize()
            b_1258 = day + pd.Timedelta(hours=12, minutes=58)
            b_1400 = day + pd.Timedelta(hours=14, minutes=0)
            b_1530 = day + pd.Timedelta(hours=15, minutes=30)
            b_1630 = day + pd.Timedelta(hours=16, minutes=30)

            if t_start is None:
                t_start = pd.Timestamp(np.min(ts_ref))
            if t_end is None:
                t_end = pd.Timestamp(np.max(ts_ref))

            # Align timezone awareness between t_start/t_end and day/b_1400
            if day.tz is not None and getattr(t_start, 'tz', None) is None:
                t_start = t_start.tz_localize(day.tz)
                t_end = t_end.tz_localize(day.tz)
            elif day.tz is None and getattr(t_start, 'tz', None) is not None:
                t_start = t_start.tz_convert(None)
                t_end = t_end.tz_convert(None)

            return [
                ("Phase 1", "Baseline\n(Pre-occupancy)",
                 t_start, min(max(b_1400, t_start), t_end), "#646464"),
                ("Phase 2", "High-occupancy\n Demonstration",
                 max(min(b_1400, t_end), t_start), min(max(b_1530, t_start), t_end), "#D25F12"),
                ("Phase 3", "Independent fabrication\nsustained cutting",
                 max(min(b_1530, t_end), t_start), min(max(b_1630, t_start), t_end), "#AF1C1C"),
                ("Phase 4", "Post-occupancy decay",
                 max(min(b_1630, t_end), t_start), t_end, "#1C4CAF"),
            ]


        def _draw_phase_bands(ax, ts_ref, ylo, yhi):
            """
            Shade the 4 operational phases across `ax` and label each one
            along the top of the curve flush to canvas borders.
            """
            # 1. Get exact numeric frame limits
            x0, x1 = ax.get_xlim()

            # 2. Extract timezone info from reference data
            ts_ref = pd.to_datetime(ts_ref)
            ref_tz = ts_ref.dt.tz if hasattr(ts_ref, 'dt') else getattr(ts_ref[0], 'tz', None)

            # 3. Convert float x-limits to Timestamps matching ts_ref's timezone awareness
            t0_dt = mdates.num2date(x0)
            t1_dt = mdates.num2date(x1)

            if ref_tz is None:
                # Convert UTC num2date representation to naive local time
                axis_t_start = pd.Timestamp(t0_dt).tz_convert(None) if t0_dt.tzinfo else pd.Timestamp(t0_dt)
                axis_t_end   = pd.Timestamp(t1_dt).tz_convert(None) if t1_dt.tzinfo else pd.Timestamp(t1_dt)
            else:
                # Convert UTC num2date representation to reference timezone
                axis_t_start = pd.Timestamp(t0_dt).tz_convert(ref_tz)
                axis_t_end   = pd.Timestamp(t1_dt).tz_convert(ref_tz)

            phases = _phase_boundaries(ts_ref, t_start=axis_t_start, t_end=axis_t_end)

            label_y = yhi - (yhi - ylo) * 0.035
            for name, sub, t0, t1, col in phases:
                if t1 <= t0:
                    continue

                # Draw background patch flush to boundaries
                ax.axvspan(t0, t1, color=col, alpha=0.10, zorder=0, lw=0)

                # Draw boundary divider line
                if t0 > axis_t_start:
                    ax.axvline(t0, color="#302F2F", lw=1.0, ls="--",
                            alpha=0.55, zorder=6)

                mid = t0 + (t1 - t0) / 2
                ax.text(mid, label_y, f"{name}\n{sub}",
                        ha="center", va="top", fontsize=11,
                        fontweight="bold", color="white", linespacing=1.3,
                        zorder=7,
                        bbox=dict(boxstyle="round,pad=0.28",
                                fc=col, ec="none", alpha=0.88))

            # 4. Lock x-limits so drawing patches/text doesn't shift the canvas
            ax.set_xlim(x0, x1)

        # def _draw_phase_bands(ax, ts_ref, ylo, yhi):
        #     """
        #     Shade the 4 operational phases across `ax` and label each one
        #     along the top of the curve. Bands are anchored to the axes'
        #     actual x-limits (not just the data's min/max timestamp), so
        #     every phase — including Phase 1 — renders as a complete,
        #     fully visible rectangle reaching its bounding divider line.
        #     """
        #     x0, x1 = ax.get_xlim()
        #     axis_t_start = pd.Timestamp(mdates.num2date(x0)).tz_localize(None)
        #     axis_t_end   = pd.Timestamp(mdates.num2date(x1)).tz_localize(None)
        #     phases = _phase_boundaries(ts_ref, t_start=axis_t_start, t_end=axis_t_end)

        #     label_y = yhi - (yhi - ylo) * 0.035
        #     for name, sub, t0, t1, col in phases:
        #         if t1 <= t0:
        #             continue
        #         ax.axvspan(t0, t1, color=col, alpha=0.10, zorder=0, lw=0)
        #         if t0 > axis_t_start:
        #             ax.axvline(t0, color="#302F2F", lw=1.0, ls="--",
        #                        alpha=0.55, zorder=6)
        #         mid = t0 + (t1 - t0) / 2
        #         ax.text(mid, label_y, f"{name}\n{sub}",
        #                 ha="center", va="top", fontsize=6.3,
        #                 fontweight="bold", color="white", linespacing=1.3,
        #                 zorder=7,
        #                 bbox=dict(boxstyle="round,pad=0.28",
        #                           fc=col, ec="none", alpha=0.88))

        # ── PLOT 1: PER-MODEL PER-TARGET — full causal chain ──────────────────
        def plot_early(mname, res):
            """
            Light-theme 4-panel layout (per model × per target):
              Panel 1 (tall): Actual + Predicted + warning gap
                              + lead-time bracket annotation
              Panel 2: Person count (orange)
              Panel 3: Motion magnitude + variance (dual, teal/purple)
              Panel 4: Door open sum (green)

            """
            widx = [i for i in np.where(wmask.values)[0]
                    if i >= LB and (i+LEAD) < n]
            if not widx: return
            ctx_ts, ctx_per, ctx_mot, ctx_mvar, ctx_door = \
                _get_ctx_for_widx(widx)

            # pre-load machine features for all 3 machines
            mach_feats = {m: _get_machine_features(widx, m)
                          for m in MACHINES}

            for target in TARGETS:
                ti   = ALL_TGT.index(target)
                a    = res["actual"][:,ti]
                p_   = res["predicted"][:,ti]
                m    = res["metrics"].get(target, {})
                col  = PAL.get(mname, "#1565C0")
                align= cfg["align_trigger"]
                ts_x = pd.to_datetime(
                    res["ts_trigger"] if align else res["ts_future"])
                ts_a = pd.to_datetime(res["ts_future"])

                # ── Figure layout: 4 rows ────────────────────────────────────
                fig = plt.figure(figsize=(16, 9))          # was (16, 18)
                fig.patch.set_facecolor(BRIGHT_BG)          # was BG_PLOT
                gs  = fig.add_gridspec(
                    4, 1,
                    height_ratios=[7, 1.0, 1.0, 1.0],
                    hspace=0.08)

                ylo = min(float(a.min()), float(p_.min())) * 0.94
                yhi = max(float(a.max()), float(p_.max())) * 1.06

                # ── Panel 1: Prediction vs Actual ────────────────────────────
                ax1 = fig.add_subplot(gs[0])
                # _style_light(ax1)
                _style_bright(ax1)

                # shaded warning gap
                if align and len(ts_x) == len(ts_a):
                    ax1.fill_between(ts_x, p_, a,
                                     alpha=0.10, color=col,
                                     label="Warning gap", zorder=1)

                # faint fill under actual
                ax1.fill_between(ts_a, a, ylo,
                                 color="#F39C12", alpha=0.08, zorder=2)
                ax1.plot(ts_a, a, color="#E67E22", lw=2.0,
                         alpha=0.88, label="Actual", zorder=4)

                # # predicted
                ax1.plot(ts_x, p_, color=col, lw=2.2, ls="--",
                         alpha=0.90,
                         label=f"Predicted  (+{LEAD} min)", zorder=5)

                # ±error band
                if len(a) == len(p_):
                    roll = pd.Series(np.abs(a-p_)
                                     ).rolling(8, min_periods=1).mean().values
                    ax1.fill_between(ts_x, p_-roll, p_+roll,
                                     alpha=0.12, color=col, zorder=3,
                                     label="±error band")

                # lead-time bracket at prediction peak
                # _draw_lead_bracket(ax1, ts_x, p_, ts_a, a,
                #                    LEAD, ylo, yhi)

                # door-open event arrows
                # door_opens = np.where(
                #     np.diff(ctx_door.astype(float), prepend=0) > 0.05)[0]
                # for di in door_opens[:12]:
                #     if di < len(ctx_ts):
                #         ax1.annotate(
                #             "",
                #             xy=(ctx_ts[di], ylo+(yhi-ylo)*0.06),
                #             xytext=(ctx_ts[di], ylo),
                #             arrowprops=dict(arrowstyle="->",
                #                             color="#27AE60",
                #                             lw=1.4, alpha=0.80))
                # ax1.scatter([], [], marker="^", color="#27AE60",
                #             s=55, label="Door open event")

                # person count faint on main panel (twin)
                # ax1r = ax1.twinx()
                # ax1r.fill_between(ctx_ts, 0, ctx_per,
                #                   color="#E07020", alpha=0.12,
                #                   step="post", zorder=2)
                # ax1r.set_ylim(0, max(float(ctx_per.max()), 1)*5.5)
                # ax1r.set_yticks([])
                # ax1r.spines[["top","left","bottom"]].set_visible(False)

                ax1.set_ylim(ylo, yhi)
                _draw_phase_bands(ax1, ts_a, ylo, yhi)
                r2v   = m.get("R2", 0)
                r2col = ("#1B5E20" if r2v > 0.85 else
                         ("#E65100" if r2v > 0.5 else "#B71C1C"))
                ax1.set_title(
                    f"{mname}  ·  {target.upper()} ({UNITS.get(target,'')})"
                    f"   Lead = T+{LEAD} min   "
                    f"R² = {r2v:.4f}"
                    f"RMSE = {m.get('RMSE',0):.3f}   "
                    f"MAE = {m.get('MAE',0):.3f}   ",
                    fontsize=13, color=r2col, pad=15,
                    fontweight="semibold")
                ax1.set_ylabel(UNITS.get(target, ""),
                               color=TX_C, fontsize=10)
                # Explicit legend order: Warning gap → Predicted → ±error band
                # → Door open event → Actual, stacked top-to-bottom, upper-right.
                _handles, _labels = ax1.get_legend_handles_labels()
                _order_pref = ["Actual","Warning gap", "±error band",f"Predicted  (+{LEAD} min)" ]
                _idx = [_labels.index(l) for l in _order_pref if l in _labels]
                _handles = [_handles[i] for i in _idx]
                _labels  = [_labels[i] for i in _idx]
                ax1.legend(_handles, _labels, fontsize=11, loc="lower right",
                           framealpha=0.80, facecolor="#FFFDE7",
                           edgecolor=SP_C, labelcolor=TX_C, ncol=1,
                           borderpad=0.5, labelspacing=0.35)
                ax1.tick_params(axis="x", labelbottom=False)

                # ── Panel 2: Person count ─────────────────────────────────────
                ax2 = fig.add_subplot(gs[1],sharex=ax1)
                # _style_light(ax2, alt=True)
                _style_bright(ax2, alt=True)
                ax2.fill_between(ctx_ts, 0, ctx_per,
                                 color="#D84315", alpha=0.45, step="post")
                ax2.plot(ctx_ts, ctx_per, color="#BF360C",
                         lw=1.1, alpha=0.85, drawstyle="steps-post")
                ax2.set_ylabel("People\ncount", color="#421204",
                               fontsize=10, labelpad=2)
                ax2.tick_params(axis="y", colors="#BF360C", labelsize=6)
                ax2.yaxis.set_major_locator(
                    plt.MaxNLocator(3, integer=True))
                ax2.tick_params(axis="x", labelbottom=False)

                # ── Panel 3: Motion magnitude + variance ──────────────────────
                ax3 = fig.add_subplot(gs[2], sharex=ax1)
                # _style_light(ax3)
                _style_bright(ax3)
                ax3.fill_between(ctx_ts, 0, ctx_mot,
                                 color="#0277BD", alpha=0.30)
                ax3.plot(ctx_ts, ctx_mot, color="#01579B",
                         lw=1.1, alpha=0.80, label="Motion")
                ax3b = ax3.twinx()
                ax3b.fill_between(ctx_ts, 0, ctx_mvar,
                                  color="#6A1B9A", alpha=0.30)
                ax3b.plot(ctx_ts, ctx_mvar, color="#4A148C",
                          lw=0.9, alpha=0.75, ls=":", label="Var σ²")
                ax3.set_ylabel("Motion", color="#011727", fontsize=10)
                ax3.tick_params(axis="y", colors="#01579B", labelsize=9)
                ax3b.set_ylabel("Var σ²", color="#1F043F", fontsize=10)
                ax3b.tick_params(axis="y", colors="#4A148C", labelsize=9)
                ax3b.spines[["top","left","bottom"]].set_visible(False)
                ax3.tick_params(axis="x", labelbottom=False)

                # ── Panel 4: Door open sum ────────────────────────────────────
                ax4 = fig.add_subplot(gs[3],sharex=ax1)
                # _style_light(ax4, alt=True)
                _style_bright(ax4, alt=True)
                ax4.fill_between(ctx_ts, 0, ctx_door,
                                 color="#2E7D32", alpha=0.45, step="post")
                ax4.step(ctx_ts, ctx_door, color="#1B5E20",
                         lw=1.2, alpha=0.85, where="post")
                ax4.set_ylabel("Door\nopen sum", color="#022B05", fontsize=10)
                ax4.tick_params(axis="y", colors="#1B5E20", labelsize=9)
                ax4.yaxis.set_major_locator(plt.MaxNLocator(3))
                #ax4.tick_params(axis="x", labelbottom=False)

                #ax4.set_xlabel("Time", color=TX_MUTE, fontsize=8)
                #ax4.tick_params(axis="x", labelbottom=True,
                #                          colors=TX_MUTE, labelsize=7)

                # shared x-format for all non-last axes
                #fig.autofmt_xdate(rotation=30, ha="right")

                # Explicit X-Axis formatting on bottom panel
                ax4.set_xlabel("Time", color=TX_MUTE, fontsize=12)
                ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                ax4.tick_params(axis="x", labelbottom=True, colors=TX_MUTE, labelsize=11, rotation=30)

                plt.suptitle(
                    f"Context-Aware Lead-Time Forecasting  ·  "
                    f"Prediction rises at trigger",
                    fontsize=14, color="#424242",
                    y=1.002, fontweight="semibold")
                plt.tight_layout(rect=[0, 0, 0.97, 1])
                plt.savefig(f"{out}/{mname}_{target}_early.png",
                            dpi=160, bbox_inches="tight",
                            facecolor=BG_PLOT)
                plt.close()


        # ── PLOT 2: CONSOLIDATED — all models per pollutant ───────────────────
        def plot_consolidated_per_pollutant(rdict):
            """
            For EACH pollutant: one figure, 4 panels.
              Panel 1 (tall): all model predictions + actual on same axes
              Panel 2: Person count fill
              Panel 3: Motion magnitude + variance (dual axis)
              Panel 4: Door open sum
            Enables direct visual comparison across all models.
            """
            widx_c = [i for i in np.where(wmask.values)[0]
                      if i >= LB and (i+LEAD) < n]
            if not widx_c: return
            ctx_ts_c, ctx_per_c, ctx_mot_c, ctx_mvar_c, ctx_door_c = \
                _get_ctx_for_widx(widx_c)

            for target in TARGETS:
                ti   = ALL_TGT.index(target)
                unit = UNITS[target]

                fig = plt.figure(figsize=(16, 9))
                fig.patch.set_facecolor(BG_PLOT)
                gs = fig.add_gridspec(4, 1,
                                      height_ratios=[7, 1, 1, 1],
                                      hspace=0.08)
                # gs  = fig.add_gridspec(
                #                     4, 1,
                #                     height_ratios=[7, 1.0, 1.0, 1.0],
                #                     hspace=0.08)
                ax_main = fig.add_subplot(gs[0])
                _style_light(ax_main)

                # actual from first result (identical across models)
                first = next(iter(rdict.values()))
                ts_a  = pd.to_datetime(first["ts_future"])
                a_act = first["actual"][:, ti]
                ax_main.fill_between(ts_a, a_act, a_act.min()*0.97,
                                     color="#E67E22", alpha=0.07, zorder=1)
                ax_main.plot(ts_a, a_act, color="#E67E22", lw=2.2,
                             alpha=0.88, zorder=5, label="Actual (sensor)")

                # each model prediction
                for mn, col in [(mn, PAL.get(mn,"#888"))
                                 for mn in rdict]:
                    res  = rdict[mn]
                    ts_x = pd.to_datetime(
                        res["ts_trigger"] if cfg["align_trigger"]
                        else res["ts_future"])
                    p_   = res["predicted"][:, ti]
                    r2   = res["metrics"].get(target,{}).get("R2", 0)
                    rmse = res["metrics"].get(target,{}).get("RMSE", 0)
                    ax_main.plot(ts_x, p_, color=col,
                                 lw=1.4, ls="--", alpha=0.78, zorder=4,
                                 label=f"{mn}  R²={r2:.3f} RMSE={rmse:.2f}")

                ax_main.set_ylabel(unit, color="#2C2C2A", fontsize=9)
                ax_main.legend(fontsize=10, loc="upper left",
                               framealpha=0.35, facecolor="#FFFDE7",
                               labelcolor="#2C2C2A", ncol=3,
                               borderpad=0.4, labelspacing=0.20)
                ax_main.set_title(
                    f"Consolidated Comparison: All Models  ·  {target.upper()} ({unit})"
                    f"   Lead=T+{LEAD}min ",
                    fontsize=14, color="#2C2C2A", fontweight="semibold", pad=10)

                # Panel 2: Person count
                ax_p = fig.add_subplot(gs[1])
                _style_light(ax_p)
                ax_p.fill_between(ctx_ts_c, 0, ctx_per_c,
                                  color="#E07020", alpha=0.60, step="post")
                ax_p.plot(ctx_ts_c, ctx_per_c, color="#E07020",
                          lw=1.1, alpha=0.85, drawstyle="steps-post")
                ax_p.set_ylabel("People", color="#6C340D", fontsize=10)
                ax_p.tick_params(axis="y", colors="#E07020", labelsize=9)
                ax_p.yaxis.set_major_locator(
                    plt.MaxNLocator(3, integer=True))
                ax_p.tick_params(axis="x", labelbottom=False)

                # Panel 3: Motion + variance dual axis
                ax_v = fig.add_subplot(gs[2])
                _style_light(ax_v)
                ax_v.fill_between(ctx_ts_c, 0, ctx_mot_c,
                                  color="#38C8E0", alpha=0.38)
                ax_v.plot(ctx_ts_c, ctx_mot_c, color="#38C8E0",
                          lw=1.0, alpha=0.75, label="Motion")
                ax_v2 = ax_v.twinx()
                ax_v2.fill_between(ctx_ts_c, 0, ctx_mvar_c,
                                   color="#7F4596", alpha=0.35)
                ax_v2.plot(ctx_ts_c, ctx_mvar_c, color="#9B59B6",
                           lw=0.9, alpha=0.70, ls=":",
                           label="Variance σ²")
                ax_v.set_ylabel("Motion", color="#010D0F", fontsize=10)
                ax_v.tick_params(axis="y", colors="#31A3B8", labelsize=9)
                ax_v2.set_ylabel("Var σ²", color="#1C0226", fontsize=10)
                ax_v2.tick_params(axis="y", colors="#9B59B6", labelsize=9)
                ax_v2.spines[["top","left","bottom"]].set_visible(False)
                ax_v2.spines["right"].set_color("#9B59B6")
                ax_v2.spines["right"].set_alpha(0.40)
                ax_v.tick_params(axis="x", labelbottom=False)

                # Panel 4: Door open
                ax_d = fig.add_subplot(gs[3])
                _style_light(ax_d)
                ax_d.fill_between(ctx_ts_c, 0, ctx_door_c,
                                  color="#179053", alpha=0.55, step="post")
                ax_d.step(ctx_ts_c, ctx_door_c, color="#149454",
                          lw=1.2, alpha=0.85, where="post")
                ax_d.set_ylabel("Door open", color="#000D07", fontsize=10)
                ax_d.tick_params(axis="y", colors="#1E9F5E", labelsize=9)
                ax_d.yaxis.set_major_locator(plt.MaxNLocator(3))
                ax_d.set_xlabel("Time", color=TX_MUTE, fontsize=11)

                fig.autofmt_xdate(rotation=30, ha="right")
                plt.suptitle(
                    f"Consolidated: All Models  ·  "
                    f"{target.upper()} ({unit})  ·  Lead=T+{LEAD}min",
                    fontsize=14, color="#2B0404", fontweight="semibold", y=1.002)
                plt.tight_layout()
                plt.savefig(f"{out}/CONSOLIDATED_{target}.png",
                            dpi=150, bbox_inches="tight",
                            facecolor=BG_PLOT)
                plt.close()
                emit(f"    ✓ CONSOLIDATED_{target}.png")

        # ── PLOT 3: LEAD-GAP per pollutant — with context panels ──────────────
        def plot_lead_gap(rdict):
            widx_c = [i for i in np.where(wmask.values)[0]
                      if i >= LB and (i+LEAD) < n]
            ctx_ts_c, ctx_per_c, ctx_mot_c, ctx_mvar_c, ctx_door_c = \
                _get_ctx_for_widx(widx_c) if widx_c else \
                (np.array([]), np.zeros(0), np.zeros(0),
                 np.zeros(0), np.zeros(0))

            for target in TARGETS:
                ti   = ALL_TGT.index(target)
                unit = UNITS[target]

                fig = plt.figure(figsize=(16, 8))
                fig.patch.set_facecolor(BG_PLOT)
                gs  = fig.add_gridspec(3, 1,
                                       height_ratios=[4, 1.2, 1.2],
                                       hspace=0.09)
                ax = fig.add_subplot(gs[0])
                _style_light(ax)

                first = next(iter(rdict.values()))
                ts_a  = pd.to_datetime(first["ts_future"])
                ax.plot(ts_a, first["actual"][:, ti],
                        color="#E67E22", lw=2.3,
                        alpha=0.88, zorder=5, label="Actual (sensor)")

                for mn, col in [(mn, PAL.get(mn,"#888")) for mn in rdict]:
                    res  = rdict[mn]
                    ts_x = pd.to_datetime(
                        res["ts_trigger"] if cfg["align_trigger"]
                        else res["ts_future"])
                    r2   = res["metrics"].get(target,{}).get("R2", 0)
                    rmse = res["metrics"].get(target,{}).get("RMSE", 0)
                    ax.plot(ts_x, res["predicted"][:, ti],
                            color=col, lw=1.6, ls="--", alpha=0.80,
                            label=f"{mn}  R²={r2:.3f}  RMSE={rmse:.2f}")

                ax.set_ylabel(unit, color="#2C2C2A", fontsize=10)
                ax.legend(fontsize=6.5, loc="upper left",
                          framealpha=0.35, facecolor="#FFFDE7",
                          labelcolor="#2C2C2A", ncol=3)
                ax.set_title(
                    f"Lead-Time Comparison  ·  {target.upper()} ({unit})"
                    f"   Lead=T+{LEAD}min",
                    fontsize=11, color="#2C2C2A")

                if len(ctx_ts_c) > 0:
                    # person + variance panel
                    ax2 = fig.add_subplot(gs[1])
                    _style_light(ax2)
                    ax2.fill_between(ctx_ts_c, 0, ctx_per_c,
                                     color="#E07020", alpha=0.55,
                                     step="post", label="Person count")
                    ax2.plot(ctx_ts_c, ctx_per_c, color="#E07020",
                             lw=1.0, alpha=0.80, drawstyle="steps-post")
                    ax2b = ax2.twinx()
                    ax2b.fill_between(ctx_ts_c, 0, ctx_mvar_c,
                                      color="#9B59B6", alpha=0.35)
                    ax2b.plot(ctx_ts_c, ctx_mvar_c, color="#9B59B6",
                              lw=0.9, alpha=0.70, ls=":",
                              label="Motion var σ²")
                    ax2.set_ylabel("People", color="#4C2204", fontsize=10)
                    ax2.tick_params(axis="y", colors="#E07020", labelsize=9)
                    ax2b.set_ylabel("Var σ²", color="#2C033C", fontsize=10)
                    ax2b.tick_params(axis="y", colors="#9B59B6", labelsize=9)
                    ax2b.spines[["top","left","bottom"]].set_visible(False)
                    ax2.tick_params(axis="x", labelbottom=False)

                    # door panel
                    ax3 = fig.add_subplot(gs[2])
                    _style_light(ax3)
                    ax3.fill_between(ctx_ts_c, 0, ctx_door_c,
                                     color="#39D98A", alpha=0.55,
                                     step="post")
                    ax3.step(ctx_ts_c, ctx_door_c, color="#39D98A",
                             lw=1.2, alpha=0.85, where="post")
                    ax3.set_ylabel("Door open", color="#39D98A", fontsize=10)
                    ax3.tick_params(axis="y", colors="#39D98A", labelsize=9)
                    ax3.yaxis.set_major_locator(plt.MaxNLocator(3))
                    ax3.set_xlabel("Time", color=TX_MUTE, fontsize=12)

                fig.autofmt_xdate(rotation=30, ha="right")
                plt.suptitle(
                    f"Lead-Time Comparison — {target.upper()} ({unit})  "
                    f"[Orange=Actual  ·  Dashed=Each model prediction at trigger]",
                    fontsize=10, color="#424242", y=1.002)
                plt.tight_layout()
                plt.savefig(f"{out}/LEAD_GAP_{target}.png",
                            dpi=150, bbox_inches="tight",
                            facecolor=BG_PLOT)
                plt.close()
                emit(f"    ✓ LEAD_GAP_{target}.png")

        # ── PLOT 4: R² HEATMAP ────────────────────────────────────────────────
        def plot_heatmap(rdict):
            rows, idx = [], []
            for mn, res in rdict.items():
                rows.append([res["metrics"].get(t,{}).get("R2", float("nan"))
                             for t in ALL_TGT])
                idx.append(mn)
            if not rows: return
            fig, ax = plt.subplots(figsize=(10, max(4, len(idx)*0.65)))
            fig.patch.set_facecolor(BG_PLOT)
            ax.set_facecolor(AX_BG)
            cmap = sns.diverging_palette(10, 130, s=80, l=50, as_cmap=True)
            sns.heatmap(pd.DataFrame(rows, index=idx, columns=ALL_TGT),
                        annot=True, fmt=".3f", cmap=cmap,
                        vmin=0, vmax=1, center=0.7,
                        linewidths=0.5, linecolor="#1A1D27",
                        ax=ax, annot_kws={"size": 9, "color": "#2C2C2A"},
                        cbar_kws={"label": "R²", "shrink": 0.7})
            ax.set_title(f"R²  Heatmap  —  T+{LEAD}min Lead-Time  (v2)",
                         fontsize=12, color="#2C2C2A", pad=10)
            ax.set_xlabel(""); ax.set_ylabel("")
            plt.xticks(color="#2C2C2A", fontsize=9)
            plt.yticks(color="#2C2C2A", fontsize=8, rotation=0)
            plt.tight_layout()
            plt.savefig(f"{out}/HEATMAP_R2_lead{LEAD}.png",
                        dpi=150, bbox_inches="tight", facecolor=BG_PLOT)
            plt.close()

        # ── PLOT 5: RMSE GROUPED BAR ──────────────────────────────────────────
        def plot_rmse_bar(rdict):
            models_l = list(rdict.keys())
            n_m      = len(models_l)
            bw       = 0.8 / n_m
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            fig.patch.set_facecolor(BG_PLOT)
            for ax_i, (ax, title, tgts) in enumerate(zip(
                axes,
                [f"RMSE — All Targets  (Lead=T+{LEAD}min)",
                 "PM Targets RMSE — zoomed (μg/m³)"],
                [ALL_TGT, ["pm1","pm2_5","pm10"]]
            )):
                ax.set_facecolor(AX_BG)
                ax.spines[["top","right"]].set_visible(False)
                ax.spines[["left","bottom"]].set_color(SP_C)
                ax.tick_params(colors=TX_MUTE, labelsize=8)
                ax.grid(True, axis="y", alpha=0.20, lw=0.4, color=GR_C)
                x_t = np.arange(len(tgts))
                for i, mn in enumerate(models_l):
                    col  = PAL.get(mn, "#888")
                    vals = [rdict[mn]["metrics"].get(t,{}).get("RMSE", 0)
                            for t in tgts]
                    bars = ax.bar(x_t+(i-n_m/2+0.5)*bw, vals, bw*0.88,
                                  label=mn, color=col, alpha=0.82,
                                  edgecolor=BG_PLOT, linewidth=0.4)
                    for bar, v in zip(bars, vals):
                        if v > 0:
                            ax.text(bar.get_x()+bar.get_width()/2,
                                    bar.get_height()+0.003*max(vals+[1]),
                                    f"{v:.2f}", ha="center", va="bottom",
                                    fontsize=5.5, color=TX_MUTE)
                ax.set_xticks(x_t)
                ax.set_xticklabels(
                    [f"{t.upper()}\n({UNITS[t]})" for t in tgts],
                    fontsize=8, color="#2C2C2A")
                ax.set_ylabel("RMSE", color="#2C2C2A", fontsize=9)
                ax.set_title(title, fontsize=10, color="#2C2C2A")
                if ax_i == 0:
                    ax.legend(fontsize=6.5, ncol=3, loc="upper right",
                              framealpha=0.35, facecolor="#FFFDE7",
                              labelcolor="#2C2C2A")
            fig.suptitle(f"Model RMSE Comparison  ·  Lead=T+{LEAD}min",
                         fontsize=12, color="#2C2C2A", y=1.01)
            plt.tight_layout()
            plt.savefig(f"{out}/RMSE_BAR_lead{LEAD}.png",
                        dpi=150, bbox_inches="tight", facecolor=BG_PLOT)
            plt.close()
            emit(f"    ✓ RMSE_BAR_lead{LEAD}.png")

        # ── PLOT 6: DETAILED ABLATION STUDY ──────────────────────────────────
        def run_ablation_study(rdict, feat_cols_full, in_dim_full):
            """
            A. Feature-group (modality) ablation — which groups matter most?
               6 variants: full / no PM lags / no triggers /
                           no CO2 lags / sensor-only / trigger-only
            B. Lead-time ablation — R² and RMSE vs N-minute horizon
               Leads tested: 1, 3, 5, 10, 15, 20 min
            C. Per-target × per-model summary table rendered as heat-coloured
               matplotlib table.
            Uses a lightweight GRU for speed (capped at 100 epochs).
            """
            emit("\n" + "═"*62)
            emit("  ABLATION STUDY  —  Feature Groups + Lead-Time Horizon")
            emit("═"*62)

            best_abl = next(
                (mn for mn in ["BiGRU","GRU","BiLSTM"]
                 if mn in cfg["models"]), cfg["models"][0]
                if cfg["models"] else None)
            if best_abl is None:
                emit("  ⚠  No models — skipping ablation."); return

            # ── Variant definitions ──────────────────────────────────────────
            variants = {
                "Full (all features)":
                    feat_cols_full,
                "No PM lags/diff/roll":
                    [c for c in feat_cols_full if not any(
                        k in c for k in ["pm1_lag","pm25_lag","pm10_lag",
                                         "pm1_diff","pm25_diff","pm10_diff",
                                         "pm1_roll","pm25_roll","pm10_roll",
                                         "pm_total"])],
                "No trigger features":
                    [c for c in feat_cols_full if not any(
                        k in c for k in ["door","person","motion",
                                         "trigger","n_person"])],
                "No CO₂ lags/roll":
                    [c for c in feat_cols_full if not any(
                        k in c for k in ["co2_lag","co2_roll"])],
                "Sensor history only":
                    [c for c in feat_cols_full if c in [
                        "pm1","pm2_5","pm10","co2","voc","temp","hum"]],
                "Trigger signals only":
                    [c for c in feat_cols_full if any(
                        k in c for k in ["door","person","motion",
                                         "trigger","n_person",
                                         "mu_motion","sigma2_motion"])],
            }
            variants = {k: (v or ["pm1"]) for k,v in variants.items()}

            # ── lightweight ablation GRU ─────────────────────────────────────
            class _AblGRU(nn.Module):
                def __init__(self, ind):
                    super().__init__()
                    h = min(HIDDEN, 64)
                    self.g  = nn.GRU(ind, h, 1, batch_first=True)
                    self.fc = nn.Linear(h, n_out)
                def forward(self, x):
                    o, _ = self.g(x); return self.fc(o[:,-1,:])

            def _quick_train(model_a, trl, val, ck):
                opt_a = torch.optim.AdamW(model_a.parameters(),
                                          lr=LR, weight_decay=1e-3)
                sch_a = torch.optim.lr_scheduler.\
                    CosineAnnealingWarmRestarts(opt_a, T_0=20, T_mult=2)
                w_a   = torch.tensor(LOSS_W, device=device,
                                     dtype=torch.float32)
                best_a = float("inf"); pat_a = 0
                for ep in range(1, min(EPOCHS, 100)+1):
                    if self._stop_flag.is_set(): break
                    model_a.train()
                    for xb,yb in trl:
                        opt_a.zero_grad()
                        ((model_a(xb.to(device))-yb.to(device))**2
                         * w_a).mean().backward()
                        nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
                        opt_a.step()
                    sch_a.step()
                    model_a.eval()
                    with torch.no_grad():
                        vl_a = float(np.mean([
                            nn.MSELoss()(model_a(xb.to(device)),
                                         yb.to(device)).item()
                            for xb,yb in val]))
                    if vl_a < best_a-1e-6:
                        best_a=vl_a; pat_a=0
                        torch.save(model_a.state_dict(), ck)
                    else: pat_a+=1
                    if pat_a >= EPAT: break
                model_a.load_state_dict(
                    torch.load(ck, map_location=device, weights_only=True))
                return model_a

            def _pred_window(model_a, Xsc_a):
                model_a.eval()
                widx_a = [i for i in np.where(wmask.values)[0]
                          if i >= LB and (i+LEAD) < n]
                if not widx_a: return None, None
                ps_a, ac_a = [], []
                with torch.no_grad():
                    for i in widx_a:
                        xb = torch.tensor(Xsc_a[i-LB:i][None],
                                          dtype=torch.float32).to(device)
                        ps_a.append(model_a(xb).cpu().numpy()[0])
                        ac_a.append(y_sc[min(i+LEAD, n-1)])
                return inv_t(np.array(ps_a)), inv_t(np.array(ac_a))

            # ── A. Modality ablation ─────────────────────────────────────────
            abl_results = {}
            from sklearn.preprocessing import StandardScaler as _SS
            for vname, vcols in variants.items():
                if self._stop_flag.is_set(): break
                emit(f"\n  [{vname}]  {len(vcols)} features")
                self.after(0, lambda vn=vname:
                    self.status_var.set(f"Ablation: {vn}…"))
                try:
                    vc_ok = [c for c in vcols if c in df.columns] or ["pm1"]
                    Xv    = df[vc_ok].ffill().bfill().fillna(0
                             ).values.astype(np.float32)
                    xsc_v = _SS(); xsc_v.fit(Xv[:tr_end])
                    Xv_sc = xsc_v.transform(Xv).astype(np.float32)
                    ind_v = Xv_sc.shape[1]

                    Xv_s, yv_s = [], []
                    for i in range(len(Xv_sc)-LB-LEAD+1):
                        Xv_s.append(Xv_sc[i:i+LB])
                        yv_s.append(y_sc[i+LB+LEAD-1])
                    if not Xv_s:
                        emit(f"    ✗ No sequences"); continue
                    Xv_s = np.array(Xv_s, dtype=np.float32)
                    yv_s = np.array(yv_s, dtype=np.float32)
                    tr_sv = max(0, tr_end-LB-LEAD+1)
                    va_sv = max(tr_sv, va_end-LB-LEAD+1)

                    trl_v = mk(Xv_s[:tr_sv], yv_s[:tr_sv], True)
                    val_v = mk(Xv_s[tr_sv:va_sv], yv_s[tr_sv:va_sv], False)

                    ma = _AblGRU(ind_v).to(device)
                    ma = _quick_train(ma, trl_v, val_v,
                                      f"{out}/_abl_tmp.pt")
                    # patch predict with variant scaler
                    ma.eval()
                    widx_a = [i for i in np.where(wmask.values)[0]
                              if i >= LB and (i+LEAD) < n]
                    if not widx_a:
                        emit("    ✗ Empty window"); continue
                    ps_v, ac_v = [], []
                    with torch.no_grad():
                        for i in widx_a:
                            xb = torch.tensor(Xv_sc[i-LB:i][None],
                                              dtype=torch.float32).to(device)
                            ps_v.append(ma(xb).cpu().numpy()[0])
                            ac_v.append(y_sc[min(i+LEAD, n-1)])
                    p_v = inv_t(np.array(ps_v))
                    a_v = inv_t(np.array(ac_v))
                    m_v = metrics(a_v, p_v)
                    abl_results[vname] = {"metrics": m_v,
                                          "n_feats": len(vc_ok)}
                    ov_v = m_v["overall"]
                    emit(f"    R²={ov_v['R2']:.4f}  "
                         f"RMSE={ov_v['RMSE']:.4f}  "
                         f"feats={len(vc_ok)}")
                except Exception as e:
                    import traceback
                    emit(f"    ✗ {vname}: {e}")
                    emit(traceback.format_exc())

            # ── B. Lead-time ablation ─────────────────────────────────────────
            lead_vals    = [1, 3, 5, 10, 15, 20]
            lead_results = {}
            for lead_v in lead_vals:
                if self._stop_flag.is_set(): break
                emit(f"\n  Lead T+{lead_v}min…")
                self.after(0, lambda lv=lead_v:
                    self.status_var.set(f"Lead ablation T+{lv}min…"))
                try:
                    Xl_s, yl_s = [], []
                    for i in range(len(Xsc)-LB-lead_v+1):
                        Xl_s.append(Xsc[i:i+LB])
                        yl_s.append(y_sc[i+LB+lead_v-1])
                    if not Xl_s: continue
                    Xl_s = np.array(Xl_s, dtype=np.float32)
                    yl_s = np.array(yl_s, dtype=np.float32)
                    tr_sl = max(0, tr_end-LB-lead_v+1)
                    va_sl = max(tr_sl, va_end-LB-lead_v+1)
                    trl_l = mk(Xl_s[:tr_sl], yl_s[:tr_sl], True)
                    val_l = mk(Xl_s[tr_sl:va_sl], yl_s[tr_sl:va_sl], False)

                    ml = _AblGRU(in_dim_full).to(device)
                    ml = _quick_train(ml, trl_l, val_l,
                                      f"{out}/_lead_abl_tmp.pt")
                    ml.eval()
                    widx_l = [i for i in np.where(wmask.values)[0]
                              if i >= LB and (i+lead_v) < n]
                    if not widx_l: continue
                    ps_l, ac_l = [], []
                    with torch.no_grad():
                        for i in widx_l:
                            xb = torch.tensor(
                                Xsc[i-LB:i][None],
                                dtype=torch.float32).to(device)
                            ps_l.append(ml(xb).cpu().numpy()[0])
                            ac_l.append(y_sc[min(i+lead_v, n-1)])
                    p_l = inv_t(np.array(ps_l))
                    a_l = inv_t(np.array(ac_l))
                    m_l = metrics(a_l, p_l)
                    lead_results[lead_v] = m_l
                    emit(f"    T+{lead_v:2d}min  "
                         f"R²={m_l['overall']['R2']:.4f}  "
                         f"RMSE={m_l['overall']['RMSE']:.4f}")
                except Exception as e:
                    emit(f"    ✗ T+{lead_v}: {e}")

            if not abl_results and not lead_results:
                emit("  ⚠  No ablation results."); return

            # ── ABLATION FIGURE ────────────────────────────────────────────────
            fig = plt.figure(figsize=(20, 16))
            fig.patch.set_facecolor(BG_PLOT)
            gs_a = fig.add_gridspec(3, 2, hspace=0.48, wspace=0.32)

            # A1: modality R² horizontal bars
            if abl_results:
                ax_A1 = fig.add_subplot(gs_a[0, 0])
                ax_A1.set_facecolor("#FFFFFF")
                ax_A1.spines[["top","right"]].set_visible(False)
                ax_A1.spines[["left","bottom"]].set_color(SP_C)
                ax_A1.tick_params(colors=TX_MUTE, labelsize=8)
                ax_A1.grid(True, axis="x", alpha=0.20, lw=0.4, color=GR_C)
                names_a  = list(abl_results.keys())
                r2s_a    = [abl_results[k]["metrics"]["overall"]["R2"]
                            for k in names_a]
                nf_a     = [abl_results[k]["n_feats"] for k in names_a]
                bar_cols = ["#39D98A" if r>0.85 else
                            ("#F7C948" if r>0.5 else "#F76E6E")
                            for r in r2s_a]
                bars_a   = ax_A1.barh(names_a, r2s_a, color=bar_cols,
                                      alpha=0.82, edgecolor=BG_PLOT,
                                      linewidth=0.5, height=0.6)
                for bar, r2v, nf in zip(bars_a, r2s_a, nf_a):
                    ax_A1.text(r2v+0.004,
                               bar.get_y()+bar.get_height()/2,
                               f"{r2v:.4f}  [{nf} feats]",
                               va="center", fontsize=7.5, color="#2C2C2A")
                ax_A1.set_xlim(0, 1.14)
                ax_A1.set_xlabel("Overall R²", color="#2C2C2A", fontsize=9)
                ax_A1.set_title(
                    "A1. Modality Ablation — Overall R²\n"
                    "(GRU, 100 epochs, same lead + lookback)",
                    fontsize=9.5, color="#2C2C2A")
                ax_A1.tick_params(axis="y", colors=TX_C, labelsize=8)

            # A2: modality RMSE per-target grouped
            if abl_results:
                ax_A2 = fig.add_subplot(gs_a[0, 1])
                ax_A2.set_facecolor("#F7F5F0")
                ax_A2.spines[["top","right"]].set_visible(False)
                ax_A2.spines[["left","bottom"]].set_color(SP_C)
                ax_A2.tick_params(colors=TX_MUTE, labelsize=7)
                ax_A2.grid(True, axis="x", alpha=0.20, lw=0.4, color=GR_C)
                names_a = list(abl_results.keys())
                x_A2    = np.arange(len(names_a))
                bw_A2   = 0.75 / len(ALL_TGT)
                t_cols  = ["#4A90D9","#27AE60","#8E44AD",
                           "#E07020","#E040FB"]
                for ti2, (t, tc) in enumerate(zip(ALL_TGT, t_cols)):
                    rmses_t = [abl_results[k]["metrics"].get(
                                   t,{}).get("RMSE", 0) for k in names_a]
                    ax_A2.barh(
                        x_A2+(ti2-len(ALL_TGT)/2+0.5)*bw_A2,
                        rmses_t, bw_A2*0.85,
                        color=tc, alpha=0.78, edgecolor=BG_PLOT,
                        linewidth=0.4, label=t.upper())
                ax_A2.set_yticks(x_A2)
                ax_A2.set_yticklabels(names_a, fontsize=7, color="#2C2C2A")
                ax_A2.set_xlabel("RMSE (original units)",
                                 color="#2C2C2A", fontsize=9)
                ax_A2.set_title(
                    "A2. Modality Ablation — RMSE per Pollutant",
                    fontsize=9.5, color="#2C2C2A")
                ax_A2.legend(fontsize=7, loc="lower right",
                             framealpha=0.35, facecolor="#FFFDE7",
                             labelcolor="#2C2C2A", ncol=3)

            # B1: R² vs lead time (per pollutant lines)
            if lead_results:
                ax_B1 = fig.add_subplot(gs_a[1, 0])
                ax_B1.set_facecolor("#FFFFFF")
                ax_B1.spines[["top","right"]].set_visible(False)
                ax_B1.spines[["left","bottom"]].set_color(SP_C)
                ax_B1.tick_params(colors=TX_MUTE, labelsize=8)
                ax_B1.grid(True, alpha=0.18, lw=0.4, color=GR_C)
                lv_x   = sorted(lead_results.keys())
                t_cols = ["#4A90D9","#27AE60","#8E44AD",
                          "#E07020","#E040FB"]
                for t, tc in zip(ALL_TGT, t_cols):
                    r2_t = [lead_results[lv].get(t,{}).get("R2", float("nan"))
                            for lv in lv_x]
                    ax_B1.plot(lv_x, r2_t, color=tc, lw=1.8,
                               marker="o", markersize=5, alpha=0.85,
                               label=t.upper())
                ax_B1.axvline(LEAD, color="#E67E22", lw=1.2, ls="--",
                              alpha=0.65, label=f"Current lead={LEAD}min")
                ax_B1.set_xlabel("Lead time (minutes)",
                                 color="#2C2C2A", fontsize=9)
                ax_B1.set_ylabel("R²", color="#2C2C2A", fontsize=9)
                ax_B1.set_title(
                    "B1. Lead-Time Ablation — R² per Pollutant vs N-min Horizon",
                    fontsize=9.5, color="#2C2C2A")
                ax_B1.legend(fontsize=7, framealpha=0.35,
                             facecolor="#FFFDE7", labelcolor="#2C2C2A", ncol=3)
                ax_B1.set_ylim(-0.15, 1.05)

            # B2: Overall RMSE + R² vs lead (dual axis)
            if lead_results:
                ax_B2 = fig.add_subplot(gs_a[1, 1])
                ax_B2.set_facecolor("#F7F5F0")
                ax_B2.spines[["top","right"]].set_visible(False)
                ax_B2.spines[["left","bottom"]].set_color(SP_C)
                ax_B2.tick_params(colors=TX_MUTE, labelsize=8)
                ax_B2.grid(True, alpha=0.18, lw=0.4, color=GR_C)
                lv_x   = sorted(lead_results.keys())
                rmse_y = [lead_results[lv]["overall"]["RMSE"] for lv in lv_x]
                r2_y   = [lead_results[lv]["overall"]["R2"]   for lv in lv_x]
                ax_B2.fill_between(lv_x, rmse_y, alpha=0.18, color="#4A90D9")
                ax_B2.plot(lv_x, rmse_y, color="#4A90D9", lw=2.0,
                           marker="s", markersize=6, alpha=0.90,
                           label="Overall RMSE")
                ax_B22 = ax_B2.twinx()
                ax_B22.plot(lv_x, r2_y, color="#39D98A", lw=1.8,
                            marker="^", markersize=5, alpha=0.85,
                            ls="--", label="Overall R²")
                ax_B22.set_ylabel("Overall R²", color="#39D98A", fontsize=9)
                ax_B22.tick_params(axis="y", colors="#39D98A", labelsize=7)
                ax_B22.spines[["top","left","bottom"]].set_visible(False)
                ax_B22.spines["right"].set_color("#39D98A")
                ax_B22.set_ylim(0, 1.1)
                ax_B2.axvline(LEAD, color="#E67E22", lw=1.2, ls="--",
                              alpha=0.65)
                ax_B2.set_xlabel("Lead time (minutes)",
                                 color="#2C2C2A", fontsize=9)
                ax_B2.set_ylabel("Overall RMSE",
                                 color="#4A90D9", fontsize=9)
                ax_B2.tick_params(axis="y", colors="#4A90D9", labelsize=7)
                ax_B2.set_title(
                    "B2. Lead-Time Ablation — Overall RMSE & R²",
                    fontsize=9.5, color="#2C2C2A")

            # C: per-model × per-target metrics table
            ax_C = fig.add_subplot(gs_a[2, :])
            ax_C.set_facecolor("#FFFFFF"); ax_C.axis("off")
            ax_C.set_title(
                f"C. Per-Model × Per-Target Metrics  (Lead=T+{LEAD}min)  "
                f"—  sorted by Overall R²",
                fontsize=10, color="#2C2C2A", pad=8)

            hdrs = (["Model","Overall R²","Overall RMSE"] +
                    [f"{t.upper()} R²"   for t in ALL_TGT] +
                    [f"{t.upper()} RMSE" for t in ALL_TGT])
            rows_tbl = []
            for mn, res in sorted(
                    rdict.items(),
                    key=lambda x: x[1]["metrics"]["overall"]["R2"],
                    reverse=True):
                ov = res["metrics"]["overall"]
                row = [mn, f"{ov['R2']:.4f}", f"{ov['RMSE']:.3f}"]
                for t in ALL_TGT:
                    row.append(f"{res['metrics'].get(t,{}).get('R2',0):.3f}")
                for t in ALL_TGT:
                    row.append(f"{res['metrics'].get(t,{}).get('RMSE',0):.3f}")
                rows_tbl.append(row)

            tbl = ax_C.table(cellText=rows_tbl, colLabels=hdrs,
                             loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1.0, 1.55)
            r2_col_idxs = set([1] + list(range(3, 3+len(ALL_TGT))))
            for (ri, ci), cell in tbl.get_celld().items():
                cell.set_edgecolor("#1A1D27"); cell.set_linewidth(0.5)
                if ri == 0:
                    cell.set_facecolor("#E3F2FD")
                    cell.set_text_props(color="#2C2C2A", fontweight="bold",
                                        fontsize=7)
                else:
                    try:
                        v = float(cell.get_text().get_text())
                        if ci in r2_col_idxs and 0 <= v <= 1:
                            g = int(v * 140); r = int((1-v)*140)
                            cell.set_facecolor(f"#{r:02x}{g:02x}30")
                        else:
                            cell.set_facecolor("#FAFAFA")
                    except:
                        cell.set_facecolor("#FAFAFA")
                    cell.set_text_props(color="#2C2C2A", fontsize=7)

            plt.suptitle(
                f"Ablation Study  v3  ·  Model={best_abl}  "
                f"·  Lead=T+{LEAD}min  ·  {len(feat_cols_full)} input features",
                fontsize=12, color="#2C2C2A", y=1.005)
            plt.savefig(f"{out}/ABLATION_STUDY_lead{LEAD}.png",
                        dpi=150, bbox_inches="tight", facecolor=BG_PLOT)
            plt.close()
            emit(f"  ✓ ABLATION_STUDY_lead{LEAD}.png")

            # save CSV
            abl_rows_csv = []
            for vn, vr in abl_results.items():
                ov = vr["metrics"]["overall"]
                row = {"variant": vn, "type": "modality",
                       "n_feats": vr["n_feats"],
                       "overall_R2":   round(ov["R2"],   4),
                       "overall_RMSE": round(ov["RMSE"], 4)}
                for t in ALL_TGT:
                    row[f"{t}_R2"]   = round(vr["metrics"].get(t,{}).get("R2",  0), 4)
                    row[f"{t}_RMSE"] = round(vr["metrics"].get(t,{}).get("RMSE",0), 4)
                abl_rows_csv.append(row)
            for lv, lr in lead_results.items():
                ov = lr["overall"]
                row = {"variant": f"Lead_T+{lv}min",
                       "type":    "lead_time",
                       "n_feats": len(feat_cols_full),
                       "overall_R2":   round(ov["R2"],   4),
                       "overall_RMSE": round(ov["RMSE"], 4)}
                for t in ALL_TGT:
                    row[f"{t}_R2"]   = round(lr.get(t,{}).get("R2",  0), 4)
                    row[f"{t}_RMSE"] = round(lr.get(t,{}).get("RMSE",0), 4)
                abl_rows_csv.append(row)
            if abl_rows_csv:
                pd.DataFrame(abl_rows_csv).to_csv(
                    f"{out}/ablation_results_lead{LEAD}.csv", index=False)
                emit(f"  ✓ ablation_results_lead{LEAD}.csv")

        # ══════════════════════════════════════════════════════════════════════
        # RIGOROUS VALIDATION — regime-conditioned metrics, causal-detection
        # lead-time analysis, and paired significance testing.
        #
        # Addresses two reviewer findings on this run:
        #   (1) Pooled R²/RMSE is a statistical-masking artefact: ~80% of the
        #       session is quiescent (Phase 1/4, low motion, doors closed) and
        #       trivially predictable from self-lags alone, so it dominates the
        #       aggregate metric regardless of whether Ct (door+motion) is used.
        #       The real test is R²/RMSE inside the ~10-15% onset/transition
        #       minutes, computed against a regime label built from RAW
        #       door/motion/machine-state columns (never the model's own
        #       features), so the split is independent of what's being tested.
        #   (2) "Early warning" is a causal-detection + lead-time claim, not a
        #       lower-MSE claim. This block adds Granger-causality /
        #       cross-correlation-at-lag between Ct features and each
        #       pollutant, an event-detection precision/recall/lead-time
        #       comparison at alert thresholds, and a paired Diebold-Mariano /
        #       paired t-test on squared errors restricted to onset minutes.
        #
        # Purely additive: every existing function above (train_model,
        # predict_deep/ml, metrics, all plot_* and run_ablation_study) is
        # untouched; this block only reads their outputs (`results`) and
        # trains its own independent "without-Ct" model variants.
        # ══════════════════════════════════════════════════════════════════════
        CT_KEYWORDS = ["door","phi_open","rho_open","eps_max","emission_weight",
                       "effective_tau","consecutive_full_open","f_trans",
                       "motion","mu_motion","sigma2_motion","trigger_strength",
                       "person","n_person","is_IDLE","is_CUTTING",
                       "is_EXPOSURE","is_MAINTENANCE"]

        def _strip_ct(cols):
            kept = [c for c in cols if not any(k in c for k in CT_KEYWORDS)]
            if kept:
                return kept
            kept = [c for c in cols if any(k in c for k in
                    ["lag","diff","roll"]) and not any(
                        k in c for k in CT_KEYWORDS)]
            return kept or cols[:1]

        def _fit_variant_scaler(cols):
            cols_ok = [c for c in cols if c in df.columns] or feat_cols[:1]
            Xv = df[cols_ok].values.astype(np.float32)
            sc = StandardScaler(); sc.fit(Xv[:tr_end])
            return cols_ok, sc.transform(Xv).astype(np.float32)

        def _make_seqs_variant(Xs):
            Xo = []
            for i in range(len(Xs) - LB - LEAD + 1):
                Xo.append(Xs[i:i+LB])
            return np.array(Xo, dtype=np.float32)

        def _train_variant(model_name, ind_v, tr_ld_v, va_ld_v, ckpt_tag):
            """Train one MODEL_MAP architecture at an alternate input width.

            MODEL_MAP classes read `in_dim` as a free variable from this
            method's scope at __init__ time, so temporarily rebinding it here
            (restored immediately in `finally`) builds a model sized for the
            without-Ct feature set without touching the class definitions
            used by the main (with-Ct) training loop above.
            """
            nonlocal in_dim
            saved = in_dim
            in_dim = ind_v
            try:
                model = MODEL_MAP[model_name]().to(device)
            finally:
                in_dim = saved

            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
            sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=30, T_mult=2)
            w_t = torch.tensor(LOSS_W, device=device, dtype=torch.float32)
            ckpt = f"{out}/{ckpt_tag}_ck.pt"
            best = float("inf"); pat = 0
            for ep in range(1, EPOCHS+1):
                if self._stop_flag.is_set(): break
                model.train()
                for xb,yb in tr_ld_v:
                    opt.zero_grad()
                    pr = model(xb.to(device)); yt = yb.to(device)
                    loss = ((pr-yt)**2 * w_t).mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sch.step()
                model.eval()
                with torch.no_grad():
                    vl = float(np.mean([
                        nn.MSELoss()(model(xb.to(device)), yb.to(device)).item()
                        for xb,yb in va_ld_v]))
                if vl < best - 1e-6:
                    best = vl; pat = 0
                    torch.save(model.state_dict(), ckpt)
                else:
                    pat += 1
                if pat >= EPAT: break
            model.load_state_dict(
                torch.load(ckpt, map_location=device, weights_only=True))
            return model

        def _predict_variant(model, Xsc_v):
            model.eval()
            widx_v = [i for i in np.where(wmask.values)[0]
                      if i >= LB and (i+LEAD) < n]
            if not widx_v: return None, None, None, None
            ps, ac = [], []
            with torch.no_grad():
                for i in widx_v:
                    xb = torch.tensor(Xsc_v[i-LB:i][None],
                                      dtype=torch.float32).to(device)
                    ps.append(model(xb).cpu().numpy()[0])
                    ac.append(y_sc[min(i+LEAD, n-1)])
            p_inv = inv_t(np.array(ps))
            a_inv = inv_t(np.array(ac))
            fut_v = [min(i+LEAD, n-1) for i in widx_v]
            return a_inv, p_inv, widx_v, fut_v

        def _build_regime_labels():
            """
            Ground-truth regime label per row, from RAW state columns only
            (door/motion/machine op-state) — independent of any model
            output, per the requirement that the split not be derived from
            the thing being tested. Returns a boolean array `is_onset` of
            length n: True for rows inside an emission-onset window (a
            door/motion/cutting trigger occurred within `onset_horizon`
            minutes before-or-at this row).
            """
            oh   = cfg["onset_horizon"]
            mthr = cfg["motion_thr"]
            door_rise = (df["door_open_sum"].diff().fillna(0) > 0.05)
            cut_cols  = [f"M{m}_is_CUTTING" for m in MACHINES
                         if f"M{m}_is_CUTTING" in df.columns]
            cut_any   = (df[cut_cols].sum(axis=1) > 0) if cut_cols \
                        else pd.Series(False, index=df.index)
            cut_rise  = cut_any & (~cut_any.shift(1).fillna(False))
            motion_rise = (df[mc] > mthr) & (df[mc].shift(1).fillna(0) <= mthr)
            person_rise = (df[pc] > 0) & (df[pc].shift(1).fillna(0) <= 0)
            trigger = (door_rise | cut_rise | motion_rise | person_rise).values
            is_onset = np.zeros(n, dtype=bool)
            for ti in np.where(trigger)[0]:
                is_onset[ti: min(n, ti+oh+1)] = True
            return is_onset

        def _get_trigger_timestamps():
            """
            FIX 13: the actual timestamps of each real door/motion/cutting
            trigger (the same rising-edge definition used by
            _build_regime_labels above, duplicated here rather than
            changing that function's return signature) — exported so a
            standalone script can compute continuous distance-to-nearest-
            event without needing the raw sensor CSV at all.
            """
            door_rise = (df["door_open_sum"].diff().fillna(0) > 0.05)
            cut_cols  = [f"M{m}_is_CUTTING" for m in MACHINES
                         if f"M{m}_is_CUTTING" in df.columns]
            cut_any   = (df[cut_cols].sum(axis=1) > 0) if cut_cols \
                        else pd.Series(False, index=df.index)
            cut_rise  = cut_any & (~cut_any.shift(1).fillna(False))
            motion_rise = (df[mc] > cfg["motion_thr"]) & \
                          (df[mc].shift(1).fillna(0) <= cfg["motion_thr"])
            person_rise = (df[pc] > 0) & (df[pc].shift(1).fillna(0) <= 0)
            trigger = (door_rise | cut_rise | motion_rise | person_rise).values
            return df[ts_col].iloc[np.where(trigger)[0]].tolist()

        def _build_phase_labels():
            """
            Assigns every row to one of the four named operational phases —
            reuses `_phase_boundaries()` (the same cutoffs already drawn as
            the shaded bands on the per-model plots) so the phase-wise
            validation below lines up exactly with what the plots show,
            instead of a second, independently-defined set of boundaries.
            Returns a length-n array of strings: "Phase 1".."Phase 4".
            """
            ts_all = pd.to_datetime(df[ts_col])
            phases = _phase_boundaries(ts_all)
            labels = np.full(n, phases[0][0] if phases else "Phase 1", dtype=object)
            for name, _sub, t0, t1, _col in phases:
                mask = ((ts_all >= t0) & (ts_all <= t1)).values
                labels[mask] = name
            return labels

        def _phase_stratified_metrics(a_, p_, fut_idx, phase_labels):
            fut_arr = np.asarray(fut_idx)
            labels_at_fut = phase_labels[fut_arr]
            res = {}
            for ph in ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]:
                msk = labels_at_fut == ph
                res[ph] = {"n": int(msk.sum())}
                if msk.sum() >= 3:
                    res[ph].update(metrics(a_[msk], p_[msk]))
            return res

        def _stratified_metrics(a_, p_, fut_idx, is_onset):
            fut_arr = np.asarray(fut_idx)
            mask_on = is_onset[fut_arr]
            mask_bl = ~mask_on
            res = {}
            for label, msk in [("baseline", mask_bl), ("onset", mask_on)]:
                res[label] = {"n": int(msk.sum())}
                if msk.sum() >= 3:
                    res[label].update(metrics(a_[msk], p_[msk]))
            return res

        def _detect_events(actual_1d, ts_actual, thr, min_gap_min=5):
            arr = np.asarray(actual_1d, dtype=float)
            above = arr > thr
            rise = above & (~np.r_[False, above[:-1]])
            idxs = np.where(rise)[0]
            times = pd.to_datetime(np.asarray(ts_actual))[idxs]
            kept = []
            for t in times:
                if not kept or (t - kept[-1]).total_seconds()/60.0 >= min_gap_min:
                    kept.append(t)
            return kept

        def _event_scoring(actual_1d, ts_actual, pred_1d, ts_pred, thr, horizon_min):
            actual_events = _detect_events(actual_1d, ts_actual, thr)
            pred_arr = np.asarray(pred_1d, dtype=float)
            above_p  = pred_arr > thr
            rise_p   = above_p & (~np.r_[False, above_p[:-1]])
            pidxs    = np.where(rise_p)[0]
            ptimes   = list(pd.to_datetime(np.asarray(ts_pred))[pidxs])
            kept_p   = []
            for t in ptimes:
                if not kept_p or (t - kept_p[-1]).total_seconds()/60.0 >= 5:
                    kept_p.append(t)
            ptimes = kept_p

            used_p = set()
            tp = 0; leads = []
            for ae in actual_events:
                best = None; best_lead = None
                for j, pt in enumerate(ptimes):
                    if j in used_p: continue
                    lead = (ae - pt).total_seconds()/60.0
                    if 0 <= lead <= horizon_min:
                        if best is None or lead > best_lead:
                            best, best_lead = j, lead
                if best is not None:
                    used_p.add(best); tp += 1; leads.append(best_lead)
            fn = len(actual_events) - tp
            fp = len(ptimes) - len(used_p)
            precision = tp/(tp+fp) if (tp+fp) else float("nan")
            recall    = tp/(tp+fn) if (tp+fn) else float("nan")
            mean_lead = float(np.mean(leads)) if leads else float("nan")
            return {"n_actual_events": len(actual_events),
                    "n_pred_alerts": len(ptimes),
                    "TP": tp, "FP": fp, "FN": fn,
                    "precision": precision, "recall": recall,
                    "mean_lead_min": mean_lead}

        def _diebold_mariano(e1, e2):
            d = e1.astype(float)**2 - e2.astype(float)**2
            Tn = len(d)
            if Tn < 5: return float("nan"), float("nan")
            dbar = d.mean(); var_d = d.var(ddof=1)
            if var_d <= 1e-12: return float("nan"), float("nan")
            dm = dbar / np.sqrt(var_d / Tn)
            from scipy.stats import norm as _norm
            p = 2*(1-_norm.cdf(abs(dm)))
            return float(dm), float(p)

        def _cross_corr_lag(x, y, max_lag):
            """corr(x[t], y[t+lag]) for lag=0..max_lag; x leads y for lag>0."""
            x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
            best_lag, best_r = 0, 0.0
            for lag in range(0, max_lag+1):
                if lag == 0:
                    xs, ys = x, y
                else:
                    xs, ys = x[:-lag], y[lag:]
                if len(xs) < 10 or xs.std() < 1e-9 or ys.std() < 1e-9:
                    r = 0.0
                else:
                    r = float(np.corrcoef(xs, ys)[0,1])
                if abs(r) > abs(best_r):
                    best_lag, best_r = lag, r
            return best_lag, best_r

        def _granger_pvalue(x, y, max_lag):
            """Returns (best_lag, min_pvalue, error_reason). error_reason is
            None on success — it is always populated on failure so the caller
            can surface *why* a pair came back blank instead of silently
            leaving the CSV empty (this used to swallow every failure,
            including a plain missing-package ImportError, with no trace in
            the log — see FIX 11 below)."""
            try:
                from statsmodels.tsa.stattools import grangercausalitytests
            except ImportError as e:
                return None, None, f"statsmodels not installed ({e})"
            d = (pd.DataFrame({"y": y, "x": x})
                 .replace([np.inf, -np.inf], np.nan).dropna())
            need = max(20, max_lag*3)
            if len(d) < need:
                return None, None, f"only {len(d)} usable rows, need >= {need} for maxlag={max_lag}"
            if d["x"].std() < 1e-9 or d["y"].std() < 1e-9:
                return None, None, "near-constant series (zero variance) — correlation undefined"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = grangercausalitytests(d[["y","x"]].values,
                                                maxlag=max_lag, verbose=False)
                pvals = {lag: res[lag][0]["ssr_ftest"][1] for lag in res}
                best_lag = min(pvals, key=pvals.get)
                return best_lag, pvals[best_lag], None
            except Exception as e:
                return None, None, f"{type(e).__name__}: {e}"

        def _plot_regime_stratified(rows):
            dfR = pd.DataFrame(rows)
            models_v = sorted(dfR["model"].unique())
            fig, axes = plt.subplots(1, len(models_v),
                                     figsize=(6.5*len(models_v), 5), squeeze=False)
            fig.patch.set_facecolor(BG_PLOT)
            axes = axes[0]
            cols_v = {"with_Ct":"#39D98A","without_Ct":"#F76E6E"}
            for ax, mname in zip(axes, models_v):
                ax.set_facecolor(AX_BG)
                ax.spines[["top","right"]].set_visible(False)
                ax.spines[["left","bottom"]].set_color(SP_C)
                ax.tick_params(colors=TX_MUTE, labelsize=8)
                ax.grid(True, axis="y", alpha=0.20, lw=0.4, color=GR_C)
                sub = dfR[dfR["model"]==mname]
                regimes = ["baseline","onset"]
                x = np.arange(len(regimes)); bw = 0.35
                for i, var in enumerate(["with_Ct","without_Ct"]):
                    vals = [sub[(sub.regime==rg)&(sub.variant==var)]
                                ["overall_R2"].mean() for rg in regimes]
                    bars = ax.bar(x+(i-0.5)*bw, vals, bw*0.9,
                                  label=var.replace("_"," "),
                                  color=cols_v[var], alpha=0.85)
                    for b, v in zip(bars, vals):
                        if pd.notna(v):
                            ax.text(b.get_x()+b.get_width()/2, v+0.01,
                                    f"{v:.3f}", ha="center", fontsize=8,
                                    color=TX_C)
                ax.set_xticks(x); ax.set_xticklabels(
                    ["Baseline\n(quiescent)","Onset\n(transition)"],
                    fontsize=9, color=TX_C)
                lo = dfR["overall_R2"].min() if dfR["overall_R2"].notna().any() else 0
                ax.set_ylim(min(0, lo*1.1), 1.05)
                ax.set_title(mname, fontsize=10, color=TX_C)
                ax.legend(fontsize=7, framealpha=0.4, facecolor="#FFFDE7",
                          labelcolor=TX_C)
            fig.suptitle(
                "Regime-Stratified Overall R²: With-Ct vs Without-Ct\n"
                "(baseline = quiescent minutes, onset = transition/emission windows)",
                fontsize=12, color="#2C2C2A", y=1.03)
            plt.tight_layout()
            plt.savefig(f"{out}/REGIME_STRATIFIED_R2.png", dpi=150,
                        bbox_inches="tight", facecolor=BG_PLOT)
            plt.close()
            emit("  ✓ REGIME_STRATIFIED_R2.png")

        def _plot_phase_stratified(rows):
            """
            One panel per model, same visual language as
            _plot_regime_stratified above, but with the four named
            operational phases on the x-axis instead of the binary
            baseline/onset split — answers "in which phase specifically
            does dropping Ct hurt this model?"
            """
            dfP = pd.DataFrame(rows)
            models_v = sorted(dfP["model"].unique())
            fig, axes = plt.subplots(1, len(models_v),
                                     figsize=(6.5*len(models_v), 5), squeeze=False)
            fig.patch.set_facecolor(BG_PLOT)
            axes = axes[0]
            cols_v = {"with_Ct":"#39D98A","without_Ct":"#F76E6E"}
            phases = ["Phase 1","Phase 2","Phase 3","Phase 4"]
            phase_ticks = ["P1\nBaseline","P2\nHigh-occ.","P3\nFabrication","P4\nDecay"]
            for ax, mname in zip(axes, models_v):
                ax.set_facecolor(AX_BG)
                ax.spines[["top","right"]].set_visible(False)
                ax.spines[["left","bottom"]].set_color(SP_C)
                ax.tick_params(colors=TX_MUTE, labelsize=8)
                ax.grid(True, axis="y", alpha=0.20, lw=0.4, color=GR_C)
                sub = dfP[dfP["model"]==mname]
                x = np.arange(len(phases)); bw = 0.35
                for i, var in enumerate(["with_Ct","without_Ct"]):
                    vals, ns = [], []
                    for ph in phases:
                        row = sub[(sub.phase==ph)&(sub.variant==var)]
                        vals.append(row["overall_R2"].mean() if len(row) else float("nan"))
                        ns.append(int(row["n"].iloc[0]) if len(row) else 0)
                    bars = ax.bar(x+(i-0.5)*bw, vals, bw*0.9,
                                  label=var.replace("_"," "),
                                  color=cols_v[var], alpha=0.85)
                    for b, v, nn in zip(bars, vals, ns):
                        if pd.notna(v):
                            ax.text(b.get_x()+b.get_width()/2, v+0.01,
                                    f"{v:.2f}\n(n={nn})", ha="center", fontsize=6.5,
                                    color=TX_C, linespacing=1.1)
                ax.set_xticks(x); ax.set_xticklabels(phase_ticks, fontsize=7.5, color=TX_C)
                lo = dfP["overall_R2"].min() if dfP["overall_R2"].notna().any() else 0
                ax.set_ylim(min(0, lo*1.1), 1.08)
                ax.set_title(mname, fontsize=10, color=TX_C)
                ax.legend(fontsize=7, framealpha=0.4, facecolor="#FFFDE7",
                          labelcolor=TX_C)
            fig.suptitle(
                "Phase-Stratified Overall R²: With-Ct vs Without-Ct\n"
                "(Phase 1 Baseline · Phase 2 High-occupancy · Phase 3 Fabrication · Phase 4 Decay)",
                fontsize=12, color="#2C2C2A", y=1.03)
            plt.tight_layout()
            plt.savefig(f"{out}/PHASE_STRATIFIED_R2.png", dpi=150,
                        bbox_inches="tight", facecolor=BG_PLOT)
            plt.close()
            emit("  ✓ PHASE_STRATIFIED_R2.png")

        def _plot_event_detection(rows):
            dfE = pd.DataFrame(rows)
            targets_v = sorted(dfE["target"].unique())
            if not targets_v: return
            fig, axes = plt.subplots(1, len(targets_v),
                                     figsize=(6*len(targets_v), 5), squeeze=False)
            fig.patch.set_facecolor(BG_PLOT)
            axes = axes[0]
            cols_v = {"with_Ct":"#39D98A","without_Ct":"#F76E6E"}
            for ax, t in zip(axes, targets_v):
                ax.set_facecolor(AX_BG)
                ax.spines[["top","right"]].set_visible(False)
                ax.spines[["left","bottom"]].set_color(SP_C)
                ax.tick_params(colors=TX_MUTE, labelsize=8)
                ax.grid(True, axis="y", alpha=0.20, lw=0.4, color=GR_C)
                sub = dfE[dfE.target==t]
                models_v = sorted(sub["model"].unique())
                x = np.arange(len(models_v)); bw = 0.35
                for i, var in enumerate(["with_Ct","without_Ct"]):
                    vals = [sub[(sub.model==m)&(sub.variant==var)]
                                ["mean_lead_min"].mean() for m in models_v]
                    bars = ax.bar(x+(i-0.5)*bw, vals, bw*0.9,
                                  label=var.replace("_"," "),
                                  color=cols_v[var], alpha=0.85)
                    for b, v in zip(bars, vals):
                        if pd.notna(v):
                            ax.text(b.get_x()+b.get_width()/2, v+0.2,
                                    f"{v:.1f}m", ha="center", fontsize=8,
                                    color=TX_C)
                ax.set_xticks(x); ax.set_xticklabels(
                    models_v, rotation=20, fontsize=8, color=TX_C)
                ax.set_ylabel("Mean lead time (min)", fontsize=9, color=TX_C)
                thr = sub["threshold"].iloc[0] if len(sub) else "?"
                ax.set_title(f"{t.upper()} (alert>{thr})", fontsize=10, color=TX_C)
                ax.legend(fontsize=7, framealpha=0.4, facecolor="#FFFDE7",
                          labelcolor=TX_C)
            fig.suptitle("Event-Detection Lead-Time: With-Ct vs Without-Ct",
                         fontsize=12, color="#2C2C2A", y=1.03)
            plt.tight_layout()
            plt.savefig(f"{out}/EVENT_DETECTION_LEADTIME.png", dpi=150,
                        bbox_inches="tight", facecolor=BG_PLOT)
            plt.close()
            emit("  ✓ EVENT_DETECTION_LEADTIME.png")

        def run_rigorous_validation():
            emit("\n" + "═"*62)
            emit("  RIGOROUS VALIDATION — Regime-Conditioned + Causal + Lead-Time")
            emit("  (addresses: pooled-R² masking; early-warning ≠ low-MSE)")
            emit("═"*62)

            deep_selected = [mn for mn in cfg["models"]
                              if mn in DEEP_SET and mn in results]
            if not deep_selected:
                emit("  ⚠  No trained sequence models available — skipping.")
                return

            is_onset = _build_regime_labels()
            emit(f"  Onset-window minutes (session-wide): {int(is_onset.sum())}"
                 f" / {n}  ({100*is_onset.mean():.1f}%)")

            phase_labels = _build_phase_labels()
            if cfg.get("val_phase"):
                ph_counts = {ph: int((phase_labels==ph).sum())
                             for ph in ["Phase 1","Phase 2","Phase 3","Phase 4"]}
                emit(f"  Phase minutes (session-wide): {ph_counts}")

            noct_cols, Xsc_noct = _fit_variant_scaler(_strip_ct(feat_cols))
            emit(f"  Without-Ct feature set: {len(noct_cols)} cols "
                 f"(vs {len(feat_cols)} with-Ct)")
            Xseq_noct  = _make_seqs_variant(Xsc_noct)
            tr_ld_noct = mk(Xseq_noct[:tr_s], y_seq[:tr_s], True)
            va_ld_noct = mk(Xseq_noct[tr_s:va_s], y_seq[tr_s:va_s], False)

            regime_rows, event_rows, sig_rows, phase_rows = [], [], [], []
            noct_pred_rows = []  # FIX 13: per-minute without-Ct predictions

            for mname in deep_selected:
                if self._stop_flag.is_set(): break
                emit(f"\n  [{mname}]  training WITHOUT-Ct counterpart…")
                self.after(0, lambda mn=mname:
                    self.status_var.set(f"Validation: {mn} (no-Ct)…"))
                m_noct = _train_variant(mname, len(noct_cols),
                                        tr_ld_noct, va_ld_noct,
                                        f"_valnoct_{mname}")
                a_noct, p_noct, widx_v, fut_v = _predict_variant(m_noct, Xsc_noct)
                if a_noct is None:
                    emit("    ✗ No predictions in window — skipping."); continue

                res_wct = results[mname]
                a_wct, p_wct = res_wct["actual"], res_wct["predicted"]
                if len(a_wct) != len(a_noct):
                    emit("    ⚠  Window mismatch with-Ct vs without-Ct — skipping.")
                    continue

                # FIX 13: mirror predictions_lead{LEAD}_v2.csv's schema for
                # the without-Ct variant, so a standalone script can compare
                # both variants minute-by-minute without retraining anything.
                ts_trig_noct = df[ts_col].iloc[widx_v].values
                ts_fut_noct  = df[ts_col].iloc[fut_v].values
                for j in range(len(widx_v)):
                    for ti2, tgt in enumerate(ALL_TGT):
                        noct_pred_rows.append({
                            "trigger_timestamp": str(ts_trig_noct[j]),
                            "future_timestamp":  str(ts_fut_noct[j]),
                            "lead_minutes": LEAD,
                            "model": mname, "target": tgt,
                            "actual":    round(float(a_noct[j, ti2]), 4),
                            "predicted": round(float(p_noct[j, ti2]), 4),
                            "error":     round(float(a_noct[j, ti2] -
                                                     p_noct[j, ti2]), 4),
                        })

                # ── A. Regime-stratified metrics ─────────────────────────────
                if cfg["val_regime"]:
                    for tag, p_ in [("with_Ct", p_wct), ("without_Ct", p_noct)]:
                        strat = _stratified_metrics(a_wct, p_, fut_v, is_onset)
                        for regime in ("baseline","onset"):
                            rinfo = strat.get(regime, {})
                            row = {"model": mname, "variant": tag,
                                   "regime": regime, "n": rinfo.get("n",0)}
                            ov = rinfo.get("overall", {})
                            row["overall_R2"]   = ov.get("R2")
                            row["overall_RMSE"] = ov.get("RMSE")
                            for t in ALL_TGT:
                                tv = rinfo.get(t, {})
                                row[f"{t}_R2"]   = tv.get("R2")
                                row[f"{t}_RMSE"] = tv.get("RMSE")
                            regime_rows.append(row)

                    swct  = _stratified_metrics(a_wct, p_wct,  fut_v, is_onset)
                    snoct = _stratified_metrics(a_wct, p_noct, fut_v, is_onset)
                    r2_on_wct  = swct.get("onset",{}).get("overall",{}).get("R2")
                    r2_on_noct = snoct.get("onset",{}).get("overall",{}).get("R2")
                    r2_bl_wct  = swct.get("baseline",{}).get("overall",{}).get("R2")
                    r2_bl_noct = snoct.get("baseline",{}).get("overall",{}).get("R2")
                    emit(f"    ONSET    R²  with-Ct={r2_on_wct}  without-Ct={r2_on_noct}")
                    emit(f"    BASELINE R²  with-Ct={r2_bl_wct}  without-Ct={r2_bl_noct}")

                # ── A2. Phase-stratified metrics (Phase 1-4) ──────────────────
                if cfg.get("val_phase"):
                    for tag, p_ in [("with_Ct", p_wct), ("without_Ct", p_noct)]:
                        strat_ph = _phase_stratified_metrics(a_wct, p_, fut_v, phase_labels)
                        for ph in ["Phase 1","Phase 2","Phase 3","Phase 4"]:
                            rinfo = strat_ph.get(ph, {})
                            row = {"model": mname, "variant": tag,
                                   "phase": ph, "n": rinfo.get("n",0)}
                            ov = rinfo.get("overall", {})
                            row["overall_R2"]   = ov.get("R2")
                            row["overall_RMSE"] = ov.get("RMSE")
                            for t in ALL_TGT:
                                tv = rinfo.get(t, {})
                                row[f"{t}_R2"]   = tv.get("R2")
                                row[f"{t}_RMSE"] = tv.get("RMSE")
                            phase_rows.append(row)

                    pwct  = _phase_stratified_metrics(a_wct, p_wct,  fut_v, phase_labels)
                    pnoct = _phase_stratified_metrics(a_wct, p_noct, fut_v, phase_labels)
                    r2_line = "  ".join(
                        f"{ph.split()[1]}: with={pwct[ph]['overall'].get('R2') if pwct[ph]['n']>=3 else 'n/a'}"
                        f"/without={pnoct[ph]['overall'].get('R2') if pnoct[ph]['n']>=3 else 'n/a'}"
                        for ph in ["Phase 1","Phase 2","Phase 3","Phase 4"])
                    emit(f"    PHASE R²  {r2_line}")

                # ── B. Event detection / lead-time ───────────────────────────
                if cfg["val_event"]:
                    ts_fut_wct  = pd.to_datetime(res_wct["ts_future"])
                    ts_trig_wct = pd.to_datetime(res_wct["ts_trigger"])
                    thr_map = {
                        "voc":   cfg["alerts"]["voc"],
                        "co2":   cfg["alerts"]["co2"],
                        "pm2_5": cfg["alerts"]["pm2_5"],
                        "pm10":  cfg["alerts"]["pm10"],
                        "pm1":   cfg["alerts"]["pm1"],
                    }
                    for ti2, t in enumerate(ALL_TGT):
                        thr_s = thr_map.get(t, "")
                        if not thr_s: continue
                        try: thr = float(thr_s)
                        except ValueError: continue
                        horizon = cfg["event_match_horizon"]
                        sc_wct  = _event_scoring(a_wct[:,ti2], ts_fut_wct,
                                                 p_wct[:,ti2], ts_trig_wct,
                                                 thr, horizon)
                        sc_noct = _event_scoring(a_wct[:,ti2], ts_fut_wct,
                                                 p_noct[:,ti2], ts_trig_wct,
                                                 thr, horizon)
                        for tag, sc in [("with_Ct", sc_wct), ("without_Ct", sc_noct)]:
                            row = {"model": mname, "target": t, "threshold": thr,
                                   "variant": tag}
                            row.update(sc)
                            event_rows.append(row)
                        emit(f"    [{t.upper()}>{thr}]  lead-time  "
                             f"with-Ct={sc_wct['mean_lead_min']:.1f}min "
                             f"(P={sc_wct['precision']:.2f} R={sc_wct['recall']:.2f})   "
                             f"without-Ct={sc_noct['mean_lead_min']:.1f}min "
                             f"(P={sc_noct['precision']:.2f} R={sc_noct['recall']:.2f})")

                # ── C. Paired significance test on onset-window residuals ────
                if cfg["val_dm"]:
                    onset_mask = is_onset[np.asarray(fut_v)]
                    if onset_mask.sum() >= 8:
                        for ti2, t in enumerate(ALL_TGT):
                            e_wct  = a_wct[onset_mask, ti2]  - p_wct[onset_mask, ti2]
                            e_noct = a_wct[onset_mask, ti2]  - p_noct[onset_mask, ti2]
                            dm, dm_p = _diebold_mariano(e_wct, e_noct)
                            try:
                                from scipy.stats import ttest_rel
                                tt, tt_p = ttest_rel(e_wct**2, e_noct**2)
                                tt, tt_p = float(tt), float(tt_p)
                            except Exception:
                                tt, tt_p = float("nan"), float("nan")
                            sig_rows.append({
                                "model": mname, "target": t,
                                "n_onset": int(onset_mask.sum()),
                                "DM_stat": dm, "DM_pvalue": dm_p,
                                "ttest_stat": tt, "ttest_pvalue": tt_p,
                                "mean_sq_err_with_Ct":    float(np.mean(e_wct**2)),
                                "mean_sq_err_without_Ct": float(np.mean(e_noct**2)),
                            })
                        emit(f"    Paired DM/t-test computed on "
                             f"{int(onset_mask.sum())} onset-window minutes.")
                    else:
                        emit("    ⚠  Too few onset-window minutes for a paired test.")

            # ── D. Causality: Ct features → pollutants ────────────────────────
            if cfg["val_causal"]:
                emit("\n  Cross-correlation-at-lag + Granger causality "
                     "(Ct → pollutants)…")
                # FIX 11: report statsmodels availability up front instead of
                # discovering it 25 blank CSV cells later.
                try:
                    import statsmodels
                    emit(f"  statsmodels {statsmodels.__version__} detected "
                         f"— Granger tests enabled.")
                except ImportError:
                    emit("  ⚠  statsmodels is NOT installed in this Python "
                         "environment — every Granger p-value below will be "
                         "blank. Cross-correlation lags are unaffected and "
                         "still compute normally. Fix with:")
                    emit("      pip install statsmodels")

                causal_rows = []
                granger_fail_reasons = []
                ct_series = {"door_open_sum": df["door_open_sum"].values,
                             mc: df[mc].values}
                for m in MACHINES:
                    col = f"M{m}_rho_open"
                    if col in df.columns:
                        ct_series[col] = df[col].values
                for cname, xseries in ct_series.items():
                    for t in ALL_TGT:
                        yseries = df[t].values
                        lag, r = _cross_corr_lag(xseries, yseries,
                                                 cfg["causal_maxlag"])
                        g_lag, g_p, g_err = _granger_pvalue(
                            xseries, yseries,
                            min(cfg["causal_maxlag"], max(1, n//10)))
                        if g_err is not None:
                            granger_fail_reasons.append(f"{cname} → {t}: {g_err}")
                        causal_rows.append({
                            "ct_feature": cname, "pollutant": t,
                            "best_lag_min": lag, "max_abs_corr": round(r,4),
                            "granger_best_lag": g_lag,
                            "granger_min_pvalue":
                                round(g_p,5) if g_p is not None else None,
                        })
                if granger_fail_reasons:
                    n_fail = len(granger_fail_reasons)
                    n_total = len(causal_rows)
                    emit(f"  ⚠  Granger test returned no result for "
                         f"{n_fail}/{n_total} pairs. First reason(s):")
                    seen = []
                    for reason in granger_fail_reasons:
                        msg = reason.split(": ", 1)[-1]
                        if msg not in seen:
                            seen.append(msg)
                        if len(seen) >= 3:
                            break
                    for msg in seen:
                        emit(f"    - {msg}")
                if causal_rows:
                    pd.DataFrame(causal_rows).to_csv(
                        f"{out}/causality_lag_analysis.csv", index=False)
                    emit(f"  ✓ causality_lag_analysis.csv "
                         f"({len(causal_rows)} feature x pollutant pairs)")
                    top = sorted(causal_rows, key=lambda r: -abs(r["max_abs_corr"]))[:5]
                    for rrow in top:
                        emit(f"    {rrow['ct_feature']:>16s} → {rrow['pollutant']:<6s}"
                             f"  lag={rrow['best_lag_min']:>2d}min "
                             f"  r={rrow['max_abs_corr']:+.3f}"
                             f"  Granger p={rrow['granger_min_pvalue']}")

            # ── SAVE CSVs ──────────────────────────────────────────────────────
            if regime_rows:
                pd.DataFrame(regime_rows).to_csv(
                    f"{out}/regime_stratified_metrics_lead{LEAD}.csv", index=False)
                emit(f"  ✓ regime_stratified_metrics_lead{LEAD}.csv")
            if event_rows:
                pd.DataFrame(event_rows).to_csv(
                    f"{out}/event_detection_lead{LEAD}.csv", index=False)
                emit(f"  ✓ event_detection_lead{LEAD}.csv")
            if sig_rows:
                pd.DataFrame(sig_rows).to_csv(
                    f"{out}/paired_significance_onset_lead{LEAD}.csv", index=False)
                emit(f"  ✓ paired_significance_onset_lead{LEAD}.csv")
            if phase_rows:
                pd.DataFrame(phase_rows).to_csv(
                    f"{out}/phase_stratified_metrics_lead{LEAD}.csv", index=False)
                emit(f"  ✓ phase_stratified_metrics_lead{LEAD}.csv")
            if noct_pred_rows:
                pd.DataFrame(noct_pred_rows).to_csv(
                    f"{out}/predictions_noct_lead{LEAD}.csv", index=False)
                emit(f"  ✓ predictions_noct_lead{LEAD}.csv "
                     f"({len(noct_pred_rows)} rows — without-Ct per-minute predictions)")
            trigger_ts = _get_trigger_timestamps()
            if trigger_ts:
                pd.DataFrame({"trigger_timestamp": [str(t) for t in trigger_ts]}).to_csv(
                    f"{out}/trigger_events_lead{LEAD}.csv", index=False)
                emit(f"  ✓ trigger_events_lead{LEAD}.csv "
                     f"({len(trigger_ts)} real door/motion/cutting trigger events)")

            # ── PLOTS ─────────────────────────────────────────────────────────
            try:
                if regime_rows: _plot_regime_stratified(regime_rows)
                if event_rows:  _plot_event_detection(event_rows)
                if phase_rows:  _plot_phase_stratified(phase_rows)
            except Exception as e:
                emit(f"  ⚠  Validation plotting error: {e}")

            emit("\n  RIGOROUS VALIDATION done ✓")

        # ── ML SETUP ──────────────────────────────────────────────────────────
        Xf_tr = X_seq[:tr_s].reshape(tr_s, -1)
        yf_tr = y_raw[[min(i+LB+LEAD-1,n-1) for i in range(tr_s)]]

        ML_CLFS = {
            "LinearRegression": MultiOutputRegressor(LinearRegression()),
            "Ridge":            MultiOutputRegressor(Ridge(alpha=1.0)),
            "RandomForest":     MultiOutputRegressor(
                RandomForestRegressor(200,max_depth=12,
                                      min_samples_leaf=2,
                                      random_state=SEED,n_jobs=-1)),
            "SVR": MultiOutputRegressor(SVR(kernel="rbf",C=10,epsilon=0.1)),
        }
        try:
            import xgboost as xgb
            ML_CLFS["XGBoost"] = MultiOutputRegressor(
                xgb.XGBRegressor(n_estimators=300,max_depth=6,
                                  learning_rate=0.03,subsample=0.8,
                                  colsample_bytree=0.8,
                                  min_child_weight=3,
                                  random_state=SEED,verbosity=0,n_jobs=-1))
        except ImportError: pass

        ML_SET  = set(ML_CLFS.keys())
        DEEP_SET= set(MODEL_MAP.keys())

        # ── RUN ───────────────────────────────────────────────────────────────
        results = {}
        for mname in cfg["models"]:
            if self._stop_flag.is_set():
                emit("  ⚠  Stopped by user."); break

            emit(f"\n{'─'*55}")
            emit(f"  Model: {mname}  (Lead=T+{LEAD}min  Hidden={HIDDEN})")
            self.after(0,lambda mn=mname:
                self.status_var.set(f"Training {mn} (T+{LEAD})…"))

            try:
                if mname in DEEP_SET:
                    model = MODEL_MAP[mname]()
                    train_model(model, mname)
                    a,p_,ts_trig,ts_fut = predict_deep(model)
                elif mname in ML_CLFS:
                    emit("  Fitting ML…")
                    ML_CLFS[mname].fit(Xf_tr, yf_tr)
                    a,p_,ts_trig,ts_fut = predict_ml(ML_CLFS[mname])
                else:
                    emit(f"  ✗ Unknown: {mname}"); continue

                if a is None or len(a)==0:
                    emit("  ✗ No predictions in window."); continue

                m = metrics(a, p_)
                results[mname] = {"actual":a,"predicted":p_,
                                  "ts_trigger":ts_trig,"ts_future":ts_fut,
                                  "metrics":m}
                ov = m["overall"]
                pm_r2s = [m.get(t,{}).get("R2",0) for t in ["pm1","pm2_5","pm10"]]
                emit(f"  Overall  R²={ov['R2']:.4f}  RMSE={ov['RMSE']:.4f}")
                emit(f"  PM R²  pm1={pm_r2s[0]:.4f}  pm2.5={pm_r2s[1]:.4f}  pm10={pm_r2s[2]:.4f}")
                emit(f"  CO₂ R²={m.get('co2',{}).get('R2',0):.4f}  "
                     f"VOC R²={m.get('voc',{}).get('R2',0):.4f}")

                if cfg["save_plots"]:
                    plot_early(mname, results[mname])

            except Exception as e:
                import traceback
                emit(f"  ✗ Error: {e}")
                emit(traceback.format_exc())

        # ── SUMMARY PLOTS ─────────────────────────────────────────────────────
        if results:
            emit("\n  Generating summary plots…")
            if cfg["save_leadgap"]:
                plot_lead_gap(results)
            plot_consolidated_per_pollutant(results)
            plot_heatmap(results)
            plot_rmse_bar(results)
            emit("\n  Running ablation study…")
            run_ablation_study(results, feat_cols, in_dim)
            if (cfg["val_regime"] or cfg["val_event"] or cfg["val_causal"]
                    or cfg["val_dm"] or cfg.get("val_phase")):
                run_rigorous_validation()

        if results and cfg["save_csv"]:
            rows_out = []
            for mn,res in results.items():
                for j in range(len(res["ts_trigger"])):
                    for ti2,tgt in enumerate(ALL_TGT):
                        rows_out.append({
                            "trigger_timestamp": str(res["ts_trigger"][j]),
                            "future_timestamp":  str(res["ts_future"][j]),
                            "lead_minutes": LEAD,
                            "model":    mn, "target": tgt,
                            "actual":   round(float(res["actual"][j,ti2]),4),
                            "predicted":round(float(res["predicted"][j,ti2]),4),
                            "error":    round(float(res["actual"][j,ti2]-
                                                     res["predicted"][j,ti2]),4),
                        })
            pd.DataFrame(rows_out).to_csv(
                f"{out}/predictions_lead{LEAD}_v2.csv", index=False)
            emit(f"\n  CSV → {out}/predictions_lead{LEAD}_v2.csv")

        # ── FINAL TABLE ───────────────────────────────────────────────────────
        emit("\n" + "═"*70)
        emit(f"  FINAL METRICS  v3  —  T+{LEAD}min Lead-Time Forecast")
        emit("═"*70)
        lines = [
            f"  {'Model':22s} {'Overall R²':>10} {'RMSE':>8} "
            f"{'pm1 R²':>8} {'pm2.5':>7} {'pm10':>7} {'co2':>7} {'voc':>7}",
            "  " + "─"*70]
        for mn,res in sorted(results.items(),
                             key=lambda x:x[1]["metrics"]["overall"]["R2"],
                             reverse=True):
            ov = res["metrics"]["overall"]
            tgt_r2 = [res["metrics"].get(t,{}).get("R2",0) for t in ALL_TGT]
            pm_flag = " ⚠PM" if any(r<0 for r in tgt_r2[:3]) else "    "
            lines.append(
                f"  {mn:22s} {ov['R2']:>10.4f} {ov['RMSE']:>8.2f}"
                f"  {tgt_r2[0]:>6.3f}  {tgt_r2[1]:>5.3f}"
                f"  {tgt_r2[2]:>5.3f}  {tgt_r2[3]:>5.3f}"
                f"  {tgt_r2[4]:>5.3f}{pm_flag}")

        for l in lines: emit(l)
        emit("\n  Output files:")
        emit(f"  {{}}_{{target}}_early.png   5-panel: Pred+Person+Motion+Variance+Door")
        emit(f"  CONSOLIDATED_{{target}}.png  All models on one chart per pollutant")
        emit(f"  LEAD_GAP_{{target}}.png      Lead-gap comparison per pollutant")
        emit(f"  HEATMAP_R2_lead{LEAD}.png    R² heatmap")
        emit(f"  RMSE_BAR_lead{LEAD}.png      RMSE grouped bar (full + PM zoom)")
        emit(f"  ABLATION_STUDY_lead{LEAD}.png  Feature + Lead-time ablation")
        emit(f"  ablation_results_lead{LEAD}.csv")
        emit(f"  REGIME_STRATIFIED_R2.png     Onset vs baseline R², with-Ct vs without-Ct")
        emit(f"  regime_stratified_metrics_lead{LEAD}.csv")
        emit(f"  PHASE_STRATIFIED_R2.png      Phase 1-4 R², with-Ct vs without-Ct per model")
        emit(f"  phase_stratified_metrics_lead{LEAD}.csv")
        emit(f"  EVENT_DETECTION_LEADTIME.png  Alert precision/recall/lead-time")
        emit(f"  event_detection_lead{LEAD}.csv")
        emit(f"  causality_lag_analysis.csv    Granger + cross-corr-at-lag (Ct→pollutants)")
        emit(f"  paired_significance_onset_lead{LEAD}.csv  DM-test + paired t-test")
        emit(f"  predictions_noct_lead{LEAD}.csv  Without-Ct per-minute predictions (FIX 13)")
        emit(f"  trigger_events_lead{LEAD}.csv    Real door/motion/cutting event timestamps (FIX 13)")

        summary = "\n".join(lines)
        self.after(0, lambda: self._update_metrics(summary))
        self.after(0, lambda: self._finish(True, summary))
        emit(f"\n  Output → {out}/"); emit("  DONE ✓")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    EarlyDetectionApp().mainloop()

