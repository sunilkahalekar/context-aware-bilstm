# IAQ Early Detection System — Complete Function Reference
## Every Function Explained for Modification and Extension

> This document explains every function in `train_context_aware_bilstm_gui.py` with the actual code, what each line does,
> why it was written that way, and exactly what to change when you need to modify it.

---

## Table of Contents

### PART 1 — Global Setup
- Theme Constants
- Package Check

### PART 2 — GUI Building Blocks
- QueueHandler
- section_card()
- row_entry()
- row_spin()
- row_combo()
- divider()

### PART 3 — Application Class
- EarlyDetectionApp.__init__()
- _style_ttk()
- _build_ui() / _build_left() / _build_right()
- _sec_data() / _sec_window() / _sec_lead()
- _sec_features() / _sec_models() / _sec_training()
- _sec_output() / _sec_run()

### PART 4 — Control Flow
- _validate() / _start() / _stop() / _finish()
- _collect() / _append_log() / _poll_log() / _update_metrics()

### PART 5 — The Pipeline
- _pipeline() / make_seqs() / DS / mk()

### PART 6 — Model Architectures
- Attention / BiGRU / BiLSTM / GRU / LSTM / RNN
- _Ch / _RB / TCN / S2S / CNNLSTM / _PE / Transformer / PatchTST

### PART 7 — Training and Inference
- train_model() / inv_t() / predict_deep() / predict_ml() / metrics()

### PART 8 — Plotting
- _style_light() / _style_bright()
- _get_ctx_for_widx() / _get_machine_features()
- _draw_lead_bracket() / _phase_boundaries() / _draw_phase_bands()
- plot_early() / plot_consolidated_per_pollutant()
- plot_heatmap() / plot_rmse_bar() / run_ablation_study()

---

# PART 1 — Global Setup

## Theme Constants

```python
BG        = "#0D0F14"   # main window background — very dark navy
SURFACE   = "#161921"   # card surfaces — slightly lighter
SURFACE2  = "#1E2230"   # input field backgrounds
SURFACE3  = "#252A3A"   # section header backgrounds
ACCENT    = "#F5A623"   # primary amber — run button, active borders
ACCENT2   = "#E8522A"   # secondary orange — stop button
SUCCESS   = "#39D98A"   # green — recommended features, good metrics
CYAN      = "#38C8E0"   # blue-cyan — informational highlights
MAGENTA   = "#E040FB"   # magenta — trigger/lead-time elements
TEXT      = "#ECF0F1"   # near-white — primary text
TEXT_MUTE = "#6C7A8C"   # grey — secondary/hint text
BORDER    = "#252A3A"   # widget border colour
WARN_YEL  = "#F7C948"   # yellow — warnings and notes
```

All colours are module-level constants. Change any hex here and it propagates everywhere.

**To change the accent colour to blue:**
```python
ACCENT = "#2979FF"   # replaces amber everywhere instantly
```

**Font definitions:**
```python
FONT_LABEL = ("Courier New", 10)    # labels next to controls
FONT_ENTRY = ("Courier New", 10)    # text inside input boxes
FONT_MONO  = ("Courier New", 9)     # log window
FONT_SEC   = ("Courier New", 11, "bold")  # section headers
```

---

## Package Check

```python
MISSING = []
for pkg in ["numpy", "pandas", "sklearn", "torch", "matplotlib", "seaborn"]:
    try: __import__(pkg if pkg != "sklearn" else "sklearn")
    except ImportError: MISSING.append(pkg)
```

Checks all required packages before the GUI launches. Missing packages are stored in
`MISSING` and checked again in `_start()` to show a user-friendly error.

**To add a new required package:**
```python
for pkg in ["numpy", "pandas", "sklearn", "torch", "matplotlib", "seaborn", "scipy"]:
```

---

# PART 2 — GUI Building Blocks

## QueueHandler

```python
class QueueHandler(logging.Handler):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def emit(self, record):
        self.q.put(self.format(record))
```

Bridges Python's `logging` module to the GUI's text widget via a thread-safe queue.
Background training threads write log messages here. The main GUI thread reads them every
100ms via `_poll_log()`. This is necessary because tkinter widgets cannot be updated
from background threads — doing so causes crashes.

**Data flow:**
```
Background thread → logging.info("msg") → QueueHandler.emit() → queue.put()
Main thread (100ms) → _poll_log() → _append_log() → log_text widget updated
```

---

## section_card()

```python
def section_card(parent, title, icon="▸"):
    frame = tk.Frame(parent, bg=SURFACE,
                     highlightbackground=BORDER, highlightthickness=1)
    hdr = tk.Frame(frame, bg=SURFACE3)
    hdr.pack(fill="x")
    tk.Label(hdr, text=f" {icon}  {title}", bg=SURFACE3, fg=ACCENT,
             font=FONT_SEC, anchor="w", padx=10, pady=7).pack(side="left")
    return frame
```

Creates a styled card with a coloured header strip. Every settings section
(DATA SOURCE, TIME WINDOW, etc.) is built inside a card.

**Structure:**
```
┌──────────────────────────────────┐  ← frame (SURFACE bg, BORDER outline)
│ ▸  SECTION TITLE                │  ← hdr (SURFACE3 bg, ACCENT text)
├──────────────────────────────────┤
│  content packed here by caller   │
└──────────────────────────────────┘
```

**To add a new section:**
```python
def _sec_new(self, p):
    c = section_card(p, "MY NEW SECTION", "⑨")
    c.pack(fill="x", padx=6, pady=3)
    self.my_var = row_entry(c, "My setting", "default")
    tk.Frame(c, bg=BG, height=4).pack()  # bottom padding
```

---

## row_entry()

```python
def row_entry(parent, label, default="", width=32, tooltip=""):
    r = tk.Frame(parent, bg=SURFACE)
    r.pack(fill="x", padx=14, pady=3)

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
```

Creates a `[label] [text input] [optional hint]` row. The `width=22` on the label
ensures all labels align across sections.

**Reading the value:**
```python
self.ts_var = row_entry(c, "Timestamp column", "timestamp_minute")
col_name = self.ts_var.get()   # returns the user's typed string
```

**To make it wider:**
```python
row_entry(c, "Long label", "default", width=40)
```

---

## row_spin()

```python
def row_spin(parent, label, lo, hi, default, step=1, tooltip=""):
    r = tk.Frame(parent, bg=SURFACE)
    r.pack(fill="x", padx=14, pady=3)
    tk.Label(r, text=label, ..., width=22).pack(side="left")
    var = tk.StringVar(value=str(default))
    tk.Spinbox(r, textvariable=var, from_=lo, to=hi, increment=step,
               width=9, ...).pack(side="left", padx=(4,0))
    if tooltip: ...
    return var
```

Creates a numeric input with up/down arrow buttons.

**Always cast the value when reading:**
```python
self.lead_var = row_spin(c, "Lead steps", 1, 60, 10)
lead = int(self.lead_var.get())    # always int() or float()
```

**For decimal steps (e.g. learning rate):**
```python
self.lr_var = row_spin(c, "Learning rate", 0.0001, 0.1, 0.001, step=0.0001)
lr = float(self.lr_var.get())
```

---

## row_combo()

```python
def row_combo(parent, label, choices, default=None, tooltip=""):
    r = tk.Frame(parent, bg=SURFACE)
    r.pack(fill="x", padx=14, pady=3)
    tk.Label(r, text=label, ..., width=22).pack(side="left")
    var = tk.StringVar(value=default or choices[0])
    ttk.Combobox(r, textvariable=var, values=choices,
                 width=14, state="readonly").pack(side="left", padx=(4,0))
    if tooltip: ...
    return var
```

Creates a dropdown. `state="readonly"` prevents the user from typing arbitrary values.

**To allow free typing as well as dropdown selection:**
```python
ttk.Combobox(r, ..., state="normal")   # change from "readonly" to "normal"
```

---

## divider()

```python
def divider(parent, pady=4):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=pady)
```

One-pixel horizontal separator. Change `height=1` to `height=3` for a thicker line.

---

# PART 3 — Application Class

## EarlyDetectionApp.__init__()

```python
class EarlyDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IAQ Early Detection v3 — Context-Aware Lead-Time Forecasting")
        self.configure(bg=BG)
        self.geometry("1200x940")   # initial width × height
        self.minsize(980, 720)      # minimum allowed size
        self._q         = queue.Queue()        # thread-safe log message queue
        self._stop_flag = threading.Event()   # set when user clicks STOP
        self._style_ttk()   # apply dark theme to ttk widgets (must be first)
        self._build_ui()    # create all panels and controls
        self._poll_log()    # start the 100ms log reading loop
```

Creates the main window and wires up all infrastructure.
Order matters: `_style_ttk()` before `_build_ui()` so comboboxes render correctly.

**To change initial window size:**
```python
self.geometry("1400x1000")
self.minsize(1100, 800)
```

---

## _style_ttk()

```python
def _style_ttk(self):
    s = ttk.Style(self)
    s.theme_use("clam")   # "clam" allows full colour customisation on all platforms
    s.configure("TCombobox",
        fieldbackground=SURFACE2, background=SURFACE2, foreground=TEXT,
        selectbackground=ACCENT, selectforeground=BG,
        bordercolor=BORDER, arrowcolor=TEXT)
    s.configure("Amber.Horizontal.TProgressbar",
        troughcolor=SURFACE2, background=ACCENT,
        darkcolor=ACCENT, lightcolor=ACCENT, bordercolor=BORDER)
```

The default Windows/macOS themes ignore custom colours. "clam" is the most customisable
cross-platform theme. Must be called before any ttk widgets are created.

**To style a ttk.Button if you add one:**
```python
s.configure("TButton", background=ACCENT, foreground=BG, font=FONT_LABEL)
```

---

## _build_ui()

```python
def _build_ui(self):
    # Top banner — full width title strip
    banner = tk.Frame(self, bg=SURFACE3, highlightbackground=ACCENT, highlightthickness=1)
    banner.pack(fill="x")
    tk.Label(banner, text="⚠  IAQ EARLY DETECTION SYSTEM  v2", ...).pack(side="left")
    tk.Label(banner, text="  IIT BOMBAY  ", bg=ACCENT2, ...).pack(side="right", padx=14)

    # Two-panel split (draggable divider)
    pane = tk.PanedWindow(self, orient="horizontal", bg=BG, sashwidth=5)
    pane.pack(fill="both", expand=True, padx=10, pady=8)
    left  = tk.Frame(pane, bg=BG)
    right = tk.Frame(pane, bg=BG)
    pane.add(left,  minsize=480)
    pane.add(right, minsize=440)
    self._build_left(left)
    self._build_right(right)
```

Builds the top banner and two-panel split layout. The `PanedWindow` creates
the draggable sash between left (controls) and right (log) panels.

---

## _build_left()

```python
def _build_left(self, parent):
    # Scrollable canvas — needed when sections overflow the window height
    cv  = tk.Canvas(parent, bg=BG, highlightthickness=0)
    vsb = tk.Scrollbar(parent, orient="vertical", command=cv.yview, ...)
    cv.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    cv.pack(side="left", fill="both", expand=True)
    inner = tk.Frame(cv, bg=BG)
    win   = cv.create_window((0,0), window=inner, anchor="nw")

    # Keep inner frame width synced with canvas width
    cv.bind("<Configure>", lambda e: cv.itemconfig(win, width=e.width))
    # Update scroll region when content height changes
    inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
    # Mouse wheel scrolling (Windows/macOS)
    inner.bind_all("<MouseWheel>",
        lambda e: cv.yview_scroll(-1*(e.delta//120), "units"))

    # Pack all configuration sections in order
    self._sec_data(inner)
    self._sec_window(inner)
    self._sec_lead(inner)
    self._sec_features(inner)
    self._sec_models(inner)
    self._sec_targets(inner)
    self._sec_training(inner)
    self._sec_output(inner)
    self._sec_run(inner)
```

Creates a scrollable canvas container. Without this, sections below the window
bottom would be unreachable.

**To add Linux mouse wheel support:**
```python
inner.bind_all("<Button-4>", lambda e: cv.yview_scroll(-1, "units"))
inner.bind_all("<Button-5>", lambda e: cv.yview_scroll( 1, "units"))
```

**To add a new section, insert the call here:**
```python
self._sec_training(inner)
self._sec_mynewsection(inner)   # ← add here
self._sec_output(inner)
```

---

## _sec_window()

```python
def _sec_window(self, p):
    c = section_card(p, "TIME WINDOW", "②")
    c.pack(fill="x", padx=6, pady=3)
    tk.Label(c, text="  Leave blank → use full test split automatically", ...).pack(...)
    self.start_var = row_entry(c, "Start datetime", "12-02-2026 12:58:00")
    self.end_var   = row_entry(c, "End datetime",   "12-02-2026 18:00:00",
                               tooltip="DD-MM-YYYY HH:MM")
    tk.Frame(c, bg=BG, height=4).pack()
```

Sets the plotting/prediction window. Data outside this range still trains the model,
but predictions and plots are generated only for this time range.

**To change default dates:**
```python
self.start_var = row_entry(c, "Start datetime", "15-03-2026 09:00:00")
self.end_var   = row_entry(c, "End datetime",   "15-03-2026 17:00:00")
```

**How it's used in `_pipeline()`:**
```python
st = cfg["start"].strip(); en = cfg["end"].strip()
if st and en:
    wmask = (df[ts_col] >= pd.to_datetime(st)) & (df[ts_col] <= pd.to_datetime(en))
else:
    wmask = pd.Series([False]*n); wmask.iloc[va_end:] = True
```

---

## _sec_lead()

```python
def _sec_lead(self, p):
    ...
    self.lead_var     = row_spin(c, "Lead steps (minutes)", 1, 60, 10)
    self.lookback_var = row_spin(c, "Lookback window",      5, 120, 20)
    self.align_trigger_var = tk.BooleanVar(value=True)
    tk.Checkbutton(r, variable=self.align_trigger_var,
                   text="Plot prediction at TRIGGER TIME ...").pack(side="left")
```

Controls the core temporal parameters.

**Lead vs Lookback illustrated (LB=3, LEAD=2):**
```
Row index: 0    1    2    3    4    5
           [input window i=0 to 2]  → predicts row 4
                [input window i=1 to 3]  → predicts row 5
```

**align_trigger toggle:** When ON, the predicted value plots at trigger time so the
lead-time gap is visually obvious. When OFF, it plots at the future time for magnitude
comparison.

---

## _sec_features()

```python
def _sec_features(self, p):
    self._feat_vars = {}
    feats = [
        ("use_pm_lags",   True,  SUCCESS, "PM lags 1-3  (ALWAYS recommended)"),
        ("use_pm_diff",   True,  SUCCESS, "PM momentum diff1"),
        ("use_voc_diff",  True,  MAGENTA, "VOC momentum diff1 + diff2"),
        # ... more ...
    ]
    for key, default, color, label in feats:
        var = tk.BooleanVar(value=default)
        self._feat_vars[key] = var
        tk.Checkbutton(..., variable=var, text=label, fg=color).pack(side="left")
```

One checkbox per feature group. The pattern from checkbox to prediction:

```
GUI checkbox → self._feat_vars["use_pm_lags"] = BooleanVar(True)
  ↓
_collect() → cfg["features"] = {k: v.get() for k,v in self._feat_vars.items()}
  ↓
_pipeline() → if feat_cfg.get("use_pm_lags"): feat_cols += ["pm1_lag1", ...]
```

**To add a new feature group (3 steps):**

Step 1 — Add checkbox:
```python
("use_door_angle", False, CYAN, "Door angle rate of change"),
```

Step 2 — Compute column in `_pipeline()`:
```python
df["door_angle"] = df["M1_phi_open"].diff(1).abs().rolling(3).mean().bfill()
```

Step 3 — Add to feature selection:
```python
if feat_cfg.get("use_door_angle"):
    feat_cols += ["door_angle"]
```

---

## _sec_models()

Creates one checkbox per model in a 3-column grid. Each model name maps to a class
defined inside `_pipeline()`.

**To add a new model (4 steps):**

1. Add to the models list: `("MyModel", "#FF6B6B")`
2. Define the class inside `_pipeline()`
3. Add to `MODEL_MAP = {"MyModel": MyModel, ...}`
4. Add to `_sel_deep()` or `_sel_ml()` for quick-select buttons

---

## _sec_training()

```python
self.epochs_var   = row_spin(c, "Epochs",         10, 500, 120)
self.hidden_var   = row_spin(c, "Hidden units",   16, 512,  64)
self.n_layers_var = row_spin(c, "Layers",          1,   4,   1)
self.dropout_var  = row_entry(c, "Dropout",       "0.20")
self.lr_var       = row_entry(c, "Learning rate", "0.001")
self.batch_var    = row_spin(c, "Batch size",      4, 256,  32)
self.train_frac   = row_spin(c, "Train fraction %", 50, 90, 70)
self.val_frac     = row_spin(c, "Val fraction %",   5, 30,  15)
self.early_var    = row_spin(c, "Early stop patience", 5, 100, 20)
self.device_var   = row_combo(c, "Device", ["auto","cpu","cuda"], "auto")
```

**Split calculation:**
```python
n      = len(df)
tr_end = int(n * 0.70)   # 70% training
va_end = int(n * 0.85)   # next 15% validation
# test: remaining 15%
```

**Small dataset recommendations (<500 rows):**
- Hidden: 32-64, Layers: 1, Dropout: 0.25-0.35, Patience: 25

**Large dataset recommendations (>2000 rows):**
- Hidden: 128-256, Layers: 2, Dropout: 0.15, Patience: 15

---

# PART 4 — Control Flow

## _validate()

```python
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
    if errs:
        messagebox.showerror("Config Errors", "\n\n".join(errs))
        return False
    return True
```

Runs before training starts. Collects ALL errors into one dialog so users don't
have to fix issues one at a time.

**To add a new rule:**
```python
# Validate lead < lookback:
if int(self.lead_var.get()) >= int(self.lookback_var.get()):
    errs.append("Lead must be less than lookback window.")
```

---

## _start()

```python
def _start(self):
    if not self._validate(): return
    if MISSING:
        messagebox.showerror("Missing packages",
            "Install:\n  pip install " + " ".join(MISSING)); return

    self.run_btn.configure(state="disabled", text="Running…")
    self.stop_btn.configure(state="normal")
    self.progress.start(10)
    self._stop_flag.clear()
    self._clear_log()
    cfg = self._collect()
    threading.Thread(target=self._pipeline, args=(cfg,), daemon=True).start()
```

Validates, collects configuration snapshot, then launches `_pipeline()` in a
daemon thread. `daemon=True` means the thread dies with the window — no orphaned
training processes.

The `cfg = self._collect()` snapshot is critical: if the user changes settings
during training, the running pipeline is unaffected.

---

## _stop()

```python
def _stop(self):
    self._stop_flag.set()
    self.status_var.set("Stopping after current model…")
```

Sets a `threading.Event`. The pipeline checks it at the start of each model and
inside the epoch loop. The stop is not instantaneous — the current epoch finishes
first to avoid corrupted state.

---

## _finish()

```python
def _finish(self, ok, summary=""):
    self.run_btn.configure(state="normal", text="▶  RUN EARLY DETECTION v2")
    self.stop_btn.configure(state="disabled")
    self.progress.stop()
    self.status_var.set("Done ✓" if ok else "Stopped / Error")
    if summary: self._update_metrics(summary)
```

Always called via `self.after(0, lambda: self._finish(...))` from the background
thread, because only the main thread can update GUI widgets.

---

## _collect()

```python
def _collect(self):
    return {
        "csv":           self.csv_var.get().strip(),
        "ts_col":        self.ts_col_var.get().strip() or "timestamp_minute",
        "lead_steps":    int(self.lead_var.get()),
        "lookback":      int(self.lookback_var.get()),
        "align_trigger": self.align_trigger_var.get(),
        "features":      {k: v.get() for k, v in self._feat_vars.items()},
        "models":        [k for k, v in self._model_vars.items() if v.get()],
        "epochs":        int(self.epochs_var.get()),
        "train_frac":    float(self.train_frac.get()) / 100,   # 70 → 0.70
        "val_frac":      float(self.val_frac.get())   / 100,
        # ... all other settings ...
    }
```

Converts all GUI widget state into a plain dict. After this point the background
thread has no reference to GUI widgets.

**To add a new setting (3 steps):**
1. Create widget in `_sec_*()`: `self.my_var = row_spin(...)`
2. Add to `_collect()`: `"my_setting": int(self.my_var.get())`
3. Use in `_pipeline()`: `MY_SETTING = cfg["my_setting"]`

---

## _append_log()

```python
def _append_log(self, msg):
    self.log_text.configure(state="normal")
    ml = msg.lower()
    tag = "ok"
    if   any(w in ml for w in ["error","fail","✗","exception"]): tag = "error"
    elif any(w in ml for w in ["warn","⚠"]):                      tag = "warn"
    elif any(w in ml for w in ["fix","r²","rmse","mae"]):         tag = "cyan"
    elif any(w in ml for w in ["done","✓","saved"]):              tag = "success"
    elif any(w in ml for w in ["model:","training","═","─"]):     tag = "accent"
    elif any(w in ml for w in ["pm","lead","trigger"]):           tag = "magenta"
    self.log_text.insert("end", msg+"\n", tag)
    self.log_text.see("end")
    self.log_text.configure(state="disabled")
```

Colour-codes log lines based on keywords. The `state="normal"/"disabled"` toggle
is required — tkinter Text widgets cannot be modified while disabled.

**To add a new colour category:**
```python
# In _build_right(): add the tag
self.log_text.tag_configure("orange", foreground="#FF9800")

# In _append_log(): add the condition
elif any(w in ml for w in ["ablation","feature"]): tag = "orange"
```

---

## _poll_log()

```python
def _poll_log(self):
    try:
        while True: self._append_log(self._q.get_nowait())
    except queue.Empty: pass
    self.after(100, self._poll_log)
```

Drains all queued log messages every 100ms. `get_nowait()` returns immediately
when the queue is empty (raises `queue.Empty`) rather than blocking.

`self.after(100, self._poll_log)` schedules the next call via tkinter's event loop —
this is how to do "recurring background tasks" in tkinter without blocking.

---

# PART 5 — The Pipeline

## _pipeline()

The entire ML pipeline runs in this method on a background thread. It is 2000+ lines
but logically divided into numbered phases:

```
Phase 1: LOAD          — CSV, timestamps, missing column fill
Phase 2: FEATURE ENG   — all lag/diff/roll/trigger columns
Phase 3: SELECT FEATS  — apply GUI toggle choices
Phase 4: SCALE         — StandardScaler(X), per-target MinMaxScaler(y)
Phase 5: SEQUENCES     — sliding window → (X_seq, y_seq)
Phase 6: DATASETS      — PyTorch Dataset + DataLoader
Phase 7: MODEL DEFS    — all model classes defined inline
Phase 8: TRAIN/PREDICT — loop over selected models
Phase 9: PLOTS         — all output plot files
Phase 10: ABLATION     — feature + lead-time ablation study
Phase 11: SUMMARY      — final metrics table
```

**Why heavy imports are inside this method:**
```python
def _pipeline(self, cfg):
    import numpy as np
    import torch
    import matplotlib; matplotlib.use("Agg")   # file-only backend — no GUI windows
    ...
```

Keeps startup instant even if torch is slow to import. Also, if torch is missing,
the error is caught gracefully inside the pipeline.

---

## make_seqs()

```python
def make_seqs(Xs, ys):
    Xo, yo = [], []
    for i in range(len(Xs) - LB - LEAD + 1):
        Xo.append(Xs[i : i+LB])           # input: LB rows ending before i+LB
        yo.append(ys[i + LB + LEAD - 1])  # target: row at i+LB+LEAD-1
    return np.array(Xo, dtype=np.float32), np.array(yo, dtype=np.float32)
```

Converts flat time series into sliding windows.

**Concrete example (LB=3, LEAD=2):**
```
Sequence 0: X = rows[0,1,2]    →   y = row[4]   (3 + 2 - 1 = 4)
Sequence 1: X = rows[1,2,3]    →   y = row[5]
Sequence 2: X = rows[2,3,4]    →   y = row[6]
```

**Output shapes (386 rows, LB=20, LEAD=10):**
```python
X_seq.shape  # (357, 20, n_features)
y_seq.shape  # (357, 5)
```

---

## DS (Dataset) and mk()

```python
class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # convert once upfront
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def mk(X, y, sh):
    return DataLoader(DS(X, y), BATCH, shuffle=sh, drop_last=False)

tr_ld = mk(X_seq[:tr_s],     y_seq[:tr_s],     True)   # shuffle training
va_ld = mk(X_seq[tr_s:va_s], y_seq[tr_s:va_s], False)  # no shuffle validation
```

`drop_last=False` keeps the final incomplete batch. Critical for small datasets
where discarding even 14 sequences would be significant.

---

# PART 6 — Model Architectures

## Attention

```python
class Attention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.W = nn.Linear(h, 1, bias=False)

    def forward(self, x):
        # x: (Batch, Time, Hidden)
        scores  = self.W(x).squeeze(-1)           # (B, T) — score each timestep
        weights = F.softmax(scores, dim=1)         # (B, T) — sum to 1 across time
        ctx     = (x * weights.unsqueeze(-1)).sum(1)  # (B, H) — weighted average
        return ctx
```

Learns which timestep within the lookback window carries the most predictive information.
For IAQ lead-time forecasting, this focuses on the trigger moment (door opening, person
entering) rather than treating all timesteps equally.

**To make weights inspectable:**
```python
def forward(self, x):
    scores  = self.W(x).squeeze(-1)
    weights = F.softmax(scores, dim=1)
    ctx     = (x * weights.unsqueeze(-1)).sum(1)
    return ctx, weights   # export weights for analysis
```

---

## BiGRU

```python
class BiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(in_dim, HIDDEN, N_LAYERS,
                            batch_first=True, bidirectional=True,
                            dropout=DROPOUT if N_LAYERS>1 else 0.0)
        self.attn = Attention(HIDDEN*2)   # *2 because bidirectional
        self.drop = nn.Dropout(DROPOUT)
        pm_idx  = {0,1,2}   # pm1, pm2_5, pm10
        gas_idx = {3,4}     # co2, voc
        self.heads = nn.ModuleList()
        for i in range(n_out):
            if i in pm_idx:
                # Deeper head for PM (ReLU — physically bounded ≥ 0)
                self.heads.append(nn.Sequential(
                    nn.Linear(HIDDEN*2, HIDDEN), nn.ReLU(),
                    nn.Dropout(DROPOUT*0.5), nn.Linear(HIDDEN, 1)))
            elif i in gas_idx:
                # GELU for gas (smoother for continuous large-range values)
                self.heads.append(nn.Sequential(
                    nn.Linear(HIDDEN*2, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1)))
            else:
                self.heads.append(nn.Linear(HIDDEN*2, 1))

    def forward(self, x):
        o, _  = self.gru(x)                    # (B, T, HIDDEN*2)
        ctx   = self.drop(self.attn(o))         # (B, HIDDEN*2)
        return torch.cat([hd(ctx) for hd in self.heads], 1)  # (B, 5)
```

**Bidirectional mechanism:**
```
Forward:   t0 → t1 → ... → t19   (learns emission onset)
Backward:  t19 → t18 → ... → t0  (learns ventilation decay)
Combined:  [forward_h, backward_h] — twice the hidden size
```

**Per-target heads:** Rather than one shared `Linear(HIDDEN*2, 5)`, separate heads
let each pollutant learn its own non-linear mapping. PM uses ReLU (concentrations
are non-negative). Gas uses GELU (smoother gradient for large-range targets).

---

## BiLSTM

```python
class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, HIDDEN, N_LAYERS, batch_first=True,
                            bidirectional=True,
                            dropout=DROPOUT if N_LAYERS>1 else 0.0)
        self.attn = Attention(HIDDEN*2)
        self.drop = nn.Dropout(DROPOUT)
        # Same per-target heads as BiGRU ...
    def forward(self, x):
        o, _ = self.lstm(x)   # _ = (h_n, c_n) — discarded, use all timesteps
        ctx  = self.drop(self.attn(o))
        return torch.cat([hd(ctx) for hd in self.heads], 1)
```

LSTM adds a separate memory cell `c` for long-range retention (CO₂ accumulation over
15-20 minutes). GRU has no separate cell — updates its hidden state directly. For
slowly-varying CO₂, the LSTM cell memory gives a marginal advantage.

The cell state `(h_n, c_n)` is discarded — we pass the full sequence `o` to attention
so it can focus on any timestep, not just the last one.

---

## GRU / LSTM / RNN

```python
class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.g    = nn.GRU(in_dim, HIDDEN, N_LAYERS, batch_first=True, ...)
        self.attn = Attention(HIDDEN)   # HIDDEN (not HIDDEN*2 — unidirectional)
        self.drop = nn.Dropout(DROPOUT)
        self.fc   = nn.Linear(HIDDEN, n_out)   # single shared head
    def forward(self, x):
        o, _ = self.g(x)
        return self.fc(self.drop(self.attn(o)))

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.l  = nn.LSTM(in_dim, HIDDEN, N_LAYERS, batch_first=True, ...)
        self.fc = nn.Linear(HIDDEN, n_out)
    def forward(self, x):
        o, _ = self.l(x)
        return self.fc(o[:, -1, :])   # last timestep only — no attention

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.r  = nn.RNN(in_dim, HIDDEN, N_LAYERS, batch_first=True, ...)
        self.fc = nn.Linear(HIDDEN, n_out)
    def forward(self, x):
        o, _ = self.r(x)
        return self.fc(o[:, -1, :])   # last timestep only
```

**Key differences:**
- BiGRU/BiLSTM: bidirectional + attention + per-target heads = most capable
- GRU: unidirectional + attention + single head = good balance for small data
- LSTM: unidirectional + no attention + last-timestep only = strong for CO₂
- RNN: no gating = vanishing gradient, cannot hold info >10 timesteps

---

## TCN (_Ch, _RB helpers)

```python
class _Ch(nn.Module):
    """Remove right-side padding to make convolution causal (no future leakage)."""
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x): return x[:, :, :-self.s] if self.s else x

class _RB(nn.Module):
    """Residual block with two dilated causal convolutions."""
    def __init__(self, ic, oc, k, d, dr):
        super().__init__()
        p = (k-1) * d   # padding for causal conv
        self.net = nn.Sequential(
            nn.Conv1d(ic, oc, k, dilation=d, padding=p), _Ch(p), nn.ReLU(), nn.Dropout(dr),
            nn.Conv1d(oc, oc, k, dilation=d, padding=p), _Ch(p), nn.ReLU(), nn.Dropout(dr))
        self.dw  = nn.Conv1d(ic, oc, 1) if ic != oc else None  # 1×1 for residual
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.net(x) + (self.dw(x) if self.dw else x))

class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        ch = min(HIDDEN, 64)
        self.net = nn.Sequential(
            *[_RB(in_dim if i==0 else ch, ch, 3, 2**i, DROPOUT) for i in range(3)])
        # Dilation [1, 2, 4] → receptive field = 15 timesteps
        self.fc = nn.Linear(ch, n_out)
    def forward(self, x):
        return self.fc(self.net(x.permute(0,2,1))[:, :, -1])
```

Dilated causal convolutions achieve exponential receptive field growth without
proportional parameter growth. 3 levels cover 15 timesteps of history.

**To extend receptive field (more levels):**
```python
*[_RB(in_dim if i==0 else ch, ch, 3, 2**i, DROPOUT) for i in range(4)]
# 4 levels → receptive field = 31 timesteps
```

---

## S2S (Seq2Seq)

```python
class S2S(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.LSTM(in_dim, HIDDEN, 1, batch_first=True)
        self.dec = nn.LSTM(n_out,  HIDDEN, 1, batch_first=True)
        self.fc  = nn.Linear(HIDDEN, n_out)
    def forward(self, x):
        _, (h, c) = self.enc(x)                           # compress input to (h,c)
        di = torch.zeros(x.size(0), 1, n_out, device=x.device)  # zero start token
        o, _ = self.dec(di, (h, c))                       # decode one step
        return self.fc(o[:, -1, :])
```

The encoder must compress 20 timesteps into a single `(h, c)` vector. This
bottleneck loses long-range CO₂ accumulation history — explaining its poor
CO₂ R² = 0.488.

---

## CNNLSTM

```python
class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        ch = min(HIDDEN, 64)
        self.c = nn.Sequential(
            nn.Conv1d(in_dim, ch, 3, padding=1), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1),     nn.ReLU())
        self.l  = nn.LSTM(ch, HIDDEN, 1, batch_first=True)
        self.fc = nn.Linear(HIDDEN, n_out)
    def forward(self, x):
        # x: (B,T,F) → permute for Conv1D → (B,F,T)
        conv_out = self.c(x.permute(0,2,1))           # (B, ch, T)
        o, _ = self.l(conv_out.permute(0,2,1))        # back to (B,T,ch)
        return self.fc(o[:, -1, :])
```

CNN extracts local 3-timestep patterns, LSTM captures temporal dependencies.
The CNN front-end compresses local feature interactions but may discard the
long-range accumulation history needed for CO₂.

---

## Transformer and PatchTST

```python
class _PE(nn.Module):
    """Sinusoidal positional encoding — tells Transformer which position each token is."""
    def __init__(self, d):
        super().__init__()
        pe  = torch.zeros(512, d)
        pos = torch.arange(0, 512).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000)/d))
        pe[:, 0::2] = torch.sin(pos * div)   # even: sine
        pe[:, 1::2] = torch.cos(pos * div)   # odd: cosine
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = max(32, HIDDEN); nhead = 4 if d>=64 else 2
        self.proj = nn.Linear(in_dim, d)   # project input features to d
        self.pe   = _PE(d)
        self.enc  = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead, d*2, DROPOUT, batch_first=True), 1)
        self.fc   = nn.Linear(d, n_out)
    def forward(self, x):
        return self.fc(self.enc(self.pe(self.proj(x)))[:, -1, :])
```

Positional encoding is required because self-attention is order-agnostic — without it,
the Transformer cannot distinguish timestep 0 from timestep 19.

`nhead` must evenly divide `d`: `d=64, nhead=4` → 16 dimensions per head.

PatchTST divides the sequence into patches of `pl` timesteps and treats each patch
as a token, reducing sequence length for the Transformer from T to T/pl.

---

# PART 7 — Training and Inference

## train_model()

```python
def train_model(model, name):
    model  = model.to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sch    = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)
    w_t    = torch.tensor(LOSS_W, device=device)   # per-target loss weights
    ckpt   = f"{out}/{name}_ck.pt"
    best   = float("inf"); pat = 0

    for ep in range(1, EPOCHS+1):
        if self._stop_flag.is_set(): break

        model.train()
        for xb, yb in tr_ld:
            opt.zero_grad()
            pr   = model(xb.to(device))
            loss = ((pr - yb.to(device))**2 * w_t).mean()   # weighted MSE
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent explosion
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            vl = float(np.mean([nn.MSELoss()(model(xb.to(device)), yb.to(device)).item()
                                 for xb, yb in va_ld]))

        if vl < best - 1e-6:
            best = vl; pat = 0
            torch.save(model.state_dict(), ckpt)
        else:
            pat += 1
        if pat >= EPAT: break

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model
```

**Weighted MSE explained:**
```python
loss = ((pr - yt)**2 * w_t).mean()
```
- Per-element squared errors multiplied by target-specific weights
- Targets with low normalised variance (PM) get higher weights
- Forces the model to reduce PM errors despite CO₂/VOC dominating raw MSE

**Cosine Annealing schedule:**
```
LR  ↑max  ↑restart    ↑restart
     \   / \         /
      \_/   \_______/
    0  30   60      120  epoch
```

Smooth LR decay encourages fine-tuning; restarts escape local minima.
Each cycle doubles in length (T_mult=2), allowing progressively finer optimisation.

**To switch to OneCycleLR:**
```python
sch = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(tr_ld))
# Must step inside the batch loop:
for xb, yb in tr_ld:
    ...
    opt.step()
    sch.step()   # ← move here
```

---

## inv_t()

```python
def inv_t(arr):
    out_ = np.zeros_like(arr)
    for i, t in enumerate(ALL_TGT):
        out_[:, i] = y_sc_dict[t].inverse_transform(
            arr[:, i].reshape(-1,1)).ravel()
    return out_
```

Converts normalised [0,1] model outputs back to physical units. Each target has
its own MinMaxScaler fitted on training data — essential because:
- PM₁ range:  15–24 μg/m³
- CO₂ range:  919–3181 ppm
- VOC range:  33–499 ppb

A shared scaler would compress PM values to near-zero (CO₂ scale dominates).

---

## predict_deep() and predict_ml()

```python
def predict_deep(model):
    model.eval()
    widx = [i for i in np.where(wmask.values)[0]
            if i >= LB and (i+LEAD) < n]
    ps, ac = [], []
    with torch.no_grad():
        for i in widx:
            xb = torch.tensor(Xsc[i-LB:i][None], dtype=torch.float32).to(device)
            ps.append(model(xb).cpu().numpy()[0])
            ac.append(y_sc[min(i+LEAD, n-1)])
    p_inv   = inv_t(np.array(ps))
    a_inv   = inv_t(np.array(ac))
    ts_trig = df[ts_col].iloc[widx].values
    ts_fut  = df[ts_col].iloc[[min(i+LEAD,n-1) for i in widx]].values
    return a_inv, p_inv, ts_trig, ts_fut

def predict_ml(clf):
    widx = [i for i in np.where(wmask.values)[0]
            if i >= LB and (i+LEAD) < n]
    Xf = np.array([Xsc[i-LB:i].ravel() for i in widx])  # flatten for sklearn
    p  = clf.predict(Xf)   # sklearn predicts in original scale
    a  = y_raw[np.array([min(i+LEAD,n-1) for i in widx])]
    ts_trig = df[ts_col].iloc[widx].values
    ts_fut  = df[ts_col].iloc[[min(i+LEAD,n-1) for i in widx]].values
    return a, p, ts_trig, ts_fut   # no inv_t needed for sklearn
```

**For deep learning:** Input shape `(1, LB, n_features)` via `[None]` (adds batch dim).
Output is in scaled space → `inv_t()` converts to original units.

**For sklearn:** The lookback window is flattened to `(LB × n_features)`. Sklearn
was trained on original-scale `y_raw`, so its predictions need no inverse transform.

---

## metrics()

```python
def metrics(a, p):
    res = {}
    for i, t in enumerate(ALL_TGT):
        at = a[:,i]; pt = p[:,i]
        res[t] = {
            "RMSE": float(np.sqrt(mean_squared_error(at, pt))),
            "MAE":  float(mean_absolute_error(at, pt)),
            "R2":   float(r2_score(at, pt)) if at.std() > 1e-6 else 0.0
        }
    res["overall"] = {
        "RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "MAE":  float(mean_absolute_error(a, p)),
        "R2":   float(r2_score(a.ravel(), p.ravel()))
    }
    return res
```

`at.std() > 1e-6` guards against a stuck sensor producing constant readings
(which would make R² mathematically undefined: 0/0).

**Warning:** Overall R² uses `a.ravel()` which concatenates all 5 targets.
CO₂ values (900-3000 ppm) dominate PM values (15-50 μg/m³). A model that
predicts CO₂ perfectly but fails on PM can still show R²=0.99 overall.
Always check per-target R² values.

**To add MAPE:**
```python
from sklearn.metrics import mean_absolute_percentage_error
res[t]["MAPE"] = float(mean_absolute_percentage_error(at, pt))
```

---

# PART 8 — Plotting

## _style_light() / _style_bright()

```python
def _style_light(ax, alt=False):
    ax.set_facecolor(STRIP_BG if alt else AX_BG)
    ax.tick_params(colors=TX_MUTE, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(SP_C); sp.set_linewidth(0.5)
    ax.grid(True, alpha=0.50, lw=0.4, color=GR_C, zorder=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

def _style_bright(ax, alt=False):
    # Higher contrast — used only in plot_early() for print-quality output
    ax.set_facecolor(BRIGHT_STRIP if alt else BRIGHT_AXBG)
    ax.tick_params(colors=BRIGHT_MUTE, labelsize=9)   # larger for print
    for sp in ax.spines.values(): sp.set_edgecolor(BRIGHT_SP); sp.set_linewidth(0.8)
    ax.grid(True, alpha=0.55, lw=0.6, color=BRIGHT_GRID, zorder=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
```

Applied to every axes object for visual consistency.

**To change time label format:**
```python
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))  # "12/02 14:30"
```

**To control tick frequency:**
```python
ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))  # every 30min
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))                   # every hour
```

---

## _get_ctx_for_widx()

```python
def _get_ctx_for_widx(widx_list):
    ctx_ts   = pd.to_datetime(df[ts_col].iloc[widx_list].values)
    ctx_per  = np.array([float(df[pc].iloc[i]) if pc in df.columns else 0.0
                          for i in widx_list])
    ctx_mot  = np.array([float(df[mc].iloc[i]) if mc in df.columns else 0.0
                          for i in widx_list])
    ctx_mvar = np.array([float(df[mc].iloc[max(0,i-4):i+1].std())
                          if mc in df.columns else 0.0 for i in widx_list])
    ctx_door = np.array([float(df["door_open_sum"].iloc[i])
                          if "door_open_sum" in df.columns else 0.0
                          for i in widx_list])
    return ctx_ts, ctx_per, ctx_mot, ctx_mvar, ctx_door
```

Extracts the 5 context signals plotted in panels 2-4.

**Motion variance `ctx_mvar`:** Rolling std of 5 rows ending at index `i`.
High σ² = motion is spiking (turbulent activity). Low σ² with high μ = steady movement.

**To add a CO₂ context panel:**
```python
ctx_co2 = np.array([float(df["co2"].iloc[i]) for i in widx_list])
return ctx_ts, ctx_per, ctx_mot, ctx_mvar, ctx_door, ctx_co2
```

---

## _get_machine_features()

```python
def _get_machine_features(widx_list, m_num):
    feats = {}
    col_map = {
        "rho_open":  f"M{m_num}_rho_open",
        "eps_max":   f"M{m_num}_eps_max",
        "phi_open":  f"M{m_num}_phi_open",
        "em_weight": f"M{m_num}_emission_weight",
        "eff_tau":   f"M{m_num}_effective_tau",
        "consec":    f"M{m_num}_consecutive_full_open",
    }
    for short, col in col_map.items():
        feats[short] = (np.array([float(df[col].iloc[i]) for i in widx_list])
                        if col in df.columns else np.zeros(len(widx_list)))
    return feats
```

Extracts door physics features for one machine. Called once per machine (M1, M2, M3)
to populate the machine-specific subplots in `plot_early()`.

**To add a new machine feature:**
```python
col_map["f_trans"] = f"M{m_num}_f_trans"   # transition frequency
```

---

## _draw_lead_bracket()

```python
def _draw_lead_bracket(ax, ts_x, p_, ts_a, a, LEAD, ylo, yhi):
    if len(p_) < 5 or len(ts_x) == 0: return
    peak_idx  = int(np.argmax(p_))
    t_trigger = ts_x[peak_idx]
    if peak_idx < len(ts_a): t_future = ts_a[peak_idx]
    y_bracket = ylo + (yhi - ylo) * 0.10   # 10% from bottom
    ax.annotate("",
        xy=(t_future, y_bracket), xytext=(t_trigger, y_bracket),
        arrowprops=dict(arrowstyle="<->", color="#2C2C2A", lw=1.2))
    ax.text(t_trigger + (t_future-t_trigger)/2, y_bracket*1.05,
            f"+{LEAD} min lead", ha="center", fontsize=8)
```

Draws the `←→ +10 min lead` arrow annotation at the prediction peak.

**To move it higher:**
```python
y_bracket = ylo + (yhi - ylo) * 0.18   # 18% from bottom
```

---

## _phase_boundaries()

```python
def _phase_boundaries(ts_ref, t_start=None, t_end=None):
    ts_ref = pd.to_datetime(ts_ref)
    day    = pd.Timestamp(ts_ref[0]).normalize()   # midnight of session date

    # Fixed boundaries for the IIT Bombay session
    b_1400 = day + pd.Timedelta(hours=14, minutes=0)
    b_1530 = day + pd.Timedelta(hours=15, minutes=30)
    b_1630 = day + pd.Timedelta(hours=16, minutes=30)

    if t_start is None: t_start = pd.Timestamp(np.min(ts_ref))
    if t_end   is None: t_end   = pd.Timestamp(np.max(ts_ref))

    # Timezone alignment (handles UTC vs naive)
    if day.tz is not None and getattr(t_start, 'tz', None) is None:
        t_start = t_start.tz_localize(day.tz)
        t_end   = t_end.tz_localize(day.tz)
    elif day.tz is None and getattr(t_start, 'tz', None) is not None:
        t_start = t_start.tz_convert(None)
        t_end   = t_end.tz_convert(None)

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
```

**The `min(max(...))` clamping:** Ensures boundaries never extend outside the
data range. If data starts at 14:30, Phase 1 has zero width, Phase 2 starts at
the data start — all handled automatically.

**To modify for a different session:**
```python
b_0900 = day + pd.Timedelta(hours=9, minutes=0)
b_1200 = day + pd.Timedelta(hours=12, minutes=0)
b_1700 = day + pd.Timedelta(hours=17, minutes=0)

return [
    ("Warm-up",  "Equipment startup", t_start, min(max(b_0900,t_start),t_end), "#646464"),
    ("Active",   "Full fabrication",  max(min(b_0900,t_end),t_start),
                                      min(max(b_1700,t_start),t_end), "#D25F12"),
    ("Cooldown", "Post-session",      max(min(b_1700,t_end),t_start), t_end, "#1C4CAF"),
]
```

---

## _draw_phase_bands()

```python
def _draw_phase_bands(ax, ts_ref, ylo, yhi):
    x0, x1 = ax.get_xlim()   # read BEFORE drawing (axvspan can shift limits)

    # Convert matplotlib float x-limits to Timestamps with timezone handling
    t0_dt = mdates.num2date(x0); t1_dt = mdates.num2date(x1)
    ts_ref = pd.to_datetime(ts_ref)
    ref_tz = ts_ref.dt.tz if hasattr(ts_ref, 'dt') else getattr(ts_ref[0], 'tz', None)
    if ref_tz is None:
        axis_t_start = pd.Timestamp(t0_dt).tz_convert(None) if t0_dt.tzinfo else pd.Timestamp(t0_dt)
        axis_t_end   = pd.Timestamp(t1_dt).tz_convert(None) if t1_dt.tzinfo else pd.Timestamp(t1_dt)
    else:
        axis_t_start = pd.Timestamp(t0_dt).tz_convert(ref_tz)
        axis_t_end   = pd.Timestamp(t1_dt).tz_convert(ref_tz)

    phases   = _phase_boundaries(ts_ref, t_start=axis_t_start, t_end=axis_t_end)
    label_y  = yhi - (yhi - ylo) * 0.035

    for name, sub, t0, t1, col in phases:
        if t1 <= t0: continue
        ax.axvspan(t0, t1, color=col, alpha=0.10, zorder=0, lw=0)
        if t0 > axis_t_start:
            ax.axvline(t0, color="#302F2F", lw=1.0, ls="--", alpha=0.55, zorder=6)
        mid = t0 + (t1 - t0) / 2
        ax.text(mid, label_y, f"{name}\n{sub}",
                ha="center", va="top", fontsize=8, fontweight="bold",
                color="white", linespacing=1.3, zorder=7,
                bbox=dict(boxstyle="round,pad=0.28", fc=col, ec="none", alpha=0.88))

    ax.set_xlim(x0, x1)   # restore original limits — axvspan can shift them
```

**The timezone problem:** `mdates.num2date()` returns UTC-aware datetime objects.
If your data timestamps are timezone-naive, direct comparison fails. The tz alignment
block handles both cases.

**`ax.set_xlim(x0, x1)` at the end:** `axvspan()` can extend the axes range to
include the span boundaries. Restoring the original limits keeps the x-axis anchored
to the actual data range.

**To make labels span the full width (horizontal bars instead of boxes):**
```python
# Replace ax.text() with:
bar_h = (yhi - ylo) * 0.06
ax.axhspan(yhi - bar_h, yhi, xmin=..., xmax=..., color=col, alpha=0.88, zorder=7)
ax.text(mid, yhi - bar_h/2, f"{name} — {sub}", ha="center", va="center",
        fontsize=7, fontweight="bold", color="white", zorder=8)
```

---

## plot_early()

The main per-model-per-target causal chain plot. Creates a 4-panel stacked figure.

**Figure layout:**
```
height_ratios=[4, 1, 1, 1]   — tall main panel + 3 equal context strips
hspace=0.06                   — minimal vertical gap between panels
sharex=ax1                    — all panels share the same x-axis
```

**Panel 1 — Pollutant prediction:**
```python
ax1.fill_between(ts_x, a, p_, alpha=0.20, color="#E67E22", label="Warning gap")
ax1.plot(ts_a, a,  color="#E67E22", lw=2.0, label="Actual")
ax1.plot(ts_x, p_, color=col, lw=2.0, ls="--", label=f"Predicted (+{LEAD} min)")
# Rolling error band (±8-sample rolling MAE):
roll = pd.Series(np.abs(a - p_)).rolling(8, min_periods=1).mean().values
ax1.fill_between(ts_x, p_-roll, p_+roll, color=col, alpha=0.12, label="±error band")
```

**X-axis — only on bottom panel:**
```python
ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax4.tick_params(axis="x", labelbottom=True, colors=TX_MUTE, labelsize=9, rotation=30)
```

**To add a 5th panel (e.g. CO₂ reference):**
```python
gs = fig.add_gridspec(5, 1, height_ratios=[4,1,1,1,1], hspace=0.06)
ax5 = fig.add_subplot(gs[4], sharex=ax1)
ax5.plot(ctx_ts, ctx_co2, color="#6B1FA0", lw=1.1)
ax5.set_ylabel("CO₂ ref", fontsize=8)
# Move x-labels from ax4 to ax5:
ax4.tick_params(axis="x", labelbottom=False)
ax5.tick_params(axis="x", labelbottom=True, labelsize=9, rotation=30)
```

---

## run_ablation_study()

```python
VARIANTS = {
    "Full model":          feat_cols,
    "No PM lags/diff/roll": [f for f in feat_cols if "pm" not in f.lower()],
    "No trigger feats":    [f for f in feat_cols if f not in trigger_feature_list],
    "No CO2 lags/roll":    [f for f in feat_cols if "co2" not in f],
    "Sensor history only": [f for f in feat_cols if any(p in f for p in pollutant_names)],
    "Trigger signals only":["door_open_sum","n_person","mu_motion","sigma2_motion",
                             "door_exposure","trigger_strength","person_diff","door_diff"],
}
```

Trains the BiLSTM with each feature subset and reports the resulting R²/RMSE.

**The key interpretations:**
- "Trigger signals only" (8 features, no chemical sensors): proves behavioural
  precursors alone predict 86.4% of air quality variance
- "Sensor history only": establishes the best any prior single-sensor system can do
- Gap between them + full model: quantifies the multimodal fusion benefit (0.08-0.12 R²)

**To add a new ablation variant:**
```python
VARIANTS["No motion feats"] = [f for f in feat_cols
                                if "motion" not in f and "trigger" not in f]
```

Lead-time ablation also runs here, training the model at each lead time
(T+1, T+3, T+5, T+10, T+15, T+20) and recording the R² curve. The peak at T+10
(or T+3 in some sessions) corresponds to the ventilation exchange timescale.

---

*End of complete function reference.*

*Every function has a single, clear responsibility.*
*Identify which function owns the behaviour you need to change,*
*read its explanation above, and make the targeted modification.*
*Changes to one function rarely require changes to more than 2-3 others.*
