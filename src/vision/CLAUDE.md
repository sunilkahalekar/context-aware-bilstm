# CLAUDE.md — internals reference for src/vision/

Scope: both scripts in this folder. For what they do and how to run them,
see [README.md](README.md) first. This file is for anyone about to modify
the code.

---

## `train_door_person_detector.py` (was `train_yolo26_seg_4110.py`)

A **standalone, run-once training script** (not a library/package). No CLI
arguments — every setting lives in the `Config` class at the top. No test
suite; correctness is checked by running training and reading the printed
validation table.

### Mental model
```
Config (all tunables)
   │
   ▼
validate_config()               — sanity-checks paths & imgsz
   │
   ▼
audit_dataset_balance()         — scans train/labels/*.txt, counts instances per class
   │
   ▼
write_data_yaml_with_weights()  — writes data_4110_weighted.yaml (per-class loss weights)
   │
   ▼
inspect_dataset()               — prints classes/splits, sanity-checks M3 class names exist
   │
   ▼
model = YOLO(MODEL_WEIGHTS)     — loads yolo26n-seg.pt
suggest_batch_size()             — shrinks BATCH_SIZE to fit detected VRAM
   │
   ▼
model.train(**cfg fields)       — the actual Ultralytics training loop
   │
   ▼
validate_per_class()            — re-runs val split, compares P/R/AP50 vs hardcoded TARGETS
model.val()                     — overall box/mask mAP
   │
   ▼
prints best.pt path + export hints + "NEXT STEPS" checklist
```
Everything is orchestrated by `main()`. Read that first for execution
order; read `Config` first to change a hyperparameter.

### File layout (section markers in the code)
| Section | Purpose |
|---|---|
| Logging | stdout + `training_run.log` |
| `Config` | **All tunables live here** — don't hardcode into `model.train(...)` |
| `audit_dataset_balance()` | Counts class instances, prints imbalance warnings |
| `write_data_yaml_with_weights()` | `min(3.0, max_count/count_i)` per class, writes weighted yaml |
| `validate_config()` | Fails fast if `DATASET_YAML` missing or `IMAGE_SIZE % 32 != 0` |
| `log_environment()` | Python/torch/CUDA/VRAM info |
| `suggest_batch_size()` | VRAM → batch size heuristic, imgsz-aware |
| `inspect_dataset()` | Warns if M3 class names don't match exactly |
| `validate_per_class()` | PASS/FAIL vs hardcoded `TARGETS` dict |
| `main()` | Orchestrates all of the above |

A large docstring block at the bottom of the file (`FULL CHANGE TABLE`) is
historical/comment-only — documents the diff vs. a previous 1,686-image
dataset version, not executed.

### Conventions to preserve
- All hyperparameters go through `Config`, with a comment explaining *why*
  that value — this script's comments consistently explain reasoning, keep
  that pattern.
- `RF-##` comment tags refer to changes made scaling from 1,686→4,110
  images. Add new tags/comments rather than deleting this history.
- **Grayscale-awareness is load-bearing**: `HSV_H`/`HSV_S` are `0.0` because
  the source images have no color channel. Pointing this at a *color*
  dataset requires restoring these or color augmentation silently no-ops.
- **DFL is not a valid YOLO26 kwarg** — explicitly not passed; don't re-add
  it porting settings from an older Ultralytics version.
- Class names matched by exact string (`"machine_3_door_open"`) in both
  `validate_per_class()`'s `TARGETS` and `inspect_dataset()`'s check — a
  Roboflow class-name change (even capitalization) silently no-ops these.
- Weights are inverse-frequency, capped at 3.0. If the label scan finds
  fewer than `nc - 1` classes, it silently falls back to a **hardcoded**
  weight list — check the log for `"Weight source: HARDCODED fallback"`.

### Things that break if changed carelessly
- `DATASET_YAML` is an absolute, machine-specific path. Commented-out prior
  paths above it are normal churn across machines, not dead code.
- `IMAGE_SIZE` must be divisible by 32 (YOLO stride requirement).
- `audit_dataset_balance()` checks 4 candidate label-directory layouts; an
  unrecognized Roboflow export layout returns `{}` silently and everything
  downstream falls back to hardcoded weights.
- `validate_per_class()`'s `TARGETS` dict assumes 4 specific class names
  exist. Classes not in it (`person`, `*_closed` for M1/M2) print but are
  never PASS/FAIL checked — intentional, not a bug.

### Environment
```bash
pip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyyaml
```
GPU (CUDA) strongly expected. `multiprocessing.freeze_support()` in the
entry point is Windows-specific — keep it even though it's a no-op
elsewhere.

### Non-goals
No inference/deployment code here — training + validation only. Export
commands are printed as suggestions, not executed. Downstream consumers
mentioned in the final log lines are other files in the broader project,
not part of this script.

---

## `extract_context_vector_from_video.py` (was `iaq_feature_extraction_pipeline_v7.py`)

### Mental model: three nested time granularities
Most bugs in this kind of pipeline come from code at one granularity
accidentally running at another. Keep these straight:

1. **Frame** (every `cap.read()`, ~25-30/sec) — tracker ideally runs here,
   but `INFERENCE_STRIDE` skips some for speed. `tracking_telemetry_*.csv`
   has one row per frame.
2. **Keyframe/second** (exactly 1 specific frame/video-second, via
   `keyframe_at`) — door-state debouncing commits here, motion score is
   computed here, `per_second_analytics_*.csv` gets a row.
3. **Window** (60 keyframes = `WINDOW_SECONDS`) — `compute_Dt`/`compute_Ht`
   run here, `Ct_vectors_*.csv` gets a row. This is the actual feature
   vector the downstream model consumes.

**Invariant this version restores**: every keyframe second must also be an
inference frame. An earlier version broke this on odd-fps video — half the
keyframe seconds silently skipped debouncing and time accounting. If you
touch `INFERENCE_STRIDE`, `keyframe_at`, or `run_inference`, re-verify this
invariant holds for both even and odd source fps.

### Module layout (top to bottom)
```
CONFIGURATION constants
DoorDebouncer            — per-machine door-state state machine
AsyncCSVWriter / AsyncDictCSVWriter — non-blocking CSV writers (background thread)
TrackState / StableIDRegistry — cross-occlusion stable person IDs
compute_Dt                — window-level door-state features
compute_frame_motion / compute_Ht — window-level motion features
WindowBuffer               — accumulates one 60-second window, then flushes
CSV SCHEMAS (CT_COLUMNS, TELEM_COLS, SEC_COLS)
HELPERS (folder picker, filename timestamp parsing, labeling, color palette)
read_door_detections       — turns one YOLO result into per-machine Open/Closed/None
main()                     — orchestrates everything, one big function
```

### Function reference

**`DoorDebouncer`** (per machine, per session) — turns noisy per-second raw
readings into a stable committed state.
- `observe(raw)` — call once per keyframe with `"Open"`/`"Closed"`/`None`
  (occluded). `None` repeats the current committed state (occlusion holds
  last known state). Reading pushed into a fixed-size vote buffer
  (`deque(maxlen=DOOR_VOTE_WINDOW)`, default 3s). Majority vote decides the
  candidate; it must win `DOOR_HOLD_PER_MACHINE` consecutive calls before
  committing. Returns `(committed_state, did_transition_this_call)`.

**`AsyncCSVWriter`/`AsyncDictCSVWriter`** — `write()` pushes onto a
`queue.Queue` instead of writing synchronously; a daemon thread drains it.
Exists so a slow disk never stalls the video hot loop. Always call
`close()` (the `finally:` block guarantees this even on exception). Three
instances: `ct_writer`, `sec_writer`, `tel_writer`.

**`StableIDRegistry` + `TrackState` + `_extract_crop_feature` + `_iou`** —
YOLO's tracker assigns a `tracker_id` per video that can change on
re-detection after occlusion. This is a second layer trying to keep a
**global ID** stable: ages every active track, demotes to `_lost` after
`LOST_TRACK_TTL` frames unseen, forgets entirely after `LOST_TRACK_TTL*3`.
For each new detection: if `tracker_id` already active, refresh and keep
global ID; else try matching a `_lost` track by combined IOU + HSV
color-histogram cosine similarity (`_extract_crop_feature`, 96-bin, a cheap
appearance fingerprint, not a learned embedding); else mint a new global ID.

**`compute_Dt(seq)`** — window door-state features. Input: list of 0/1
ints per keyframe second (1=Open). Output: `tau_open` (count of 1s),
`f_trans` (value changes), `rho_open` (`tau_open/len`), `eps_max` (longest
consecutive-1 run), `phi_open` (normalized index of first 1, `1.0` if door
never opened). **This is the complete feature set produced — see README's
"Known gap" note; `emission_weight`/`effective_tau`/
`consecutive_full_open`/op-state are NOT computed here.**

**`compute_frame_motion`/`compute_Ht`** — dense Farneback optical flow
restricted to person bounding boxes (`compute_frame_motion`, returns 0.0 if
no person boxes); window reduction to `n_person`/`mu_motion`/
`sigma2_motion` (`compute_Ht`).

**`WindowBuffer`** — accumulates one window's `door_seq` per machine,
`motion_seq`, `person_counts`. `push()` per keyframe; `is_full()` after
`WINDOW_SECONDS` pushes; `flush()` builds the `Ct_vectors` row (calls
`compute_Dt` per machine + `compute_Ht` once), formats `window_start` as
`"%Y-%m-%d %H:%M:%S"` (a string, NOT a raw `datetime` — pandas/Excel can't
parse that), resets buffers. Returns `{}` if nothing pushed.

### Execution flow (`main()`, per frame)
1. Compute `offset_sec`/timestamps from the video's anchor timestamp.
2. `is_keyframe = frame_count in keyframe_at`.
3. `run_inference = (frame_count % INFERENCE_STRIDE == 0) or is_keyframe`
   — inference runs on schedule AND unconditionally on every keyframe.
4. If `run_inference`: track → build in-zone person mask → update
   `StableIDRegistry` → accumulate person-seconds → optional M3 crop pass →
   `read_door_detections()` → **only if also `is_keyframe`**: diagnostics,
   `DoorDebouncer.observe()` per machine, update toggle/time counters.
5. Else: reuse cached `last_*` values.
6. Write one telemetry row unconditionally.
7. If `is_keyframe`: optical flow vs. previous keyframe's grayscale frame →
   `per_second_analytics` row → `window_buf.push()` → flush if full.
8. Annotate frame, write to output video.

End of session: flush any partial trailing window (don't lose up to 59s at
the end), `finally:` closes all writers regardless of exceptions, print
session summary.

### Gotchas for future changes
- **Never gate `DoorDebouncer.observe()`/time accumulators behind anything
  that can be `False` on a keyframe.** That's exactly the historical bug
  this version fixes. New per-second accumulators belong in the
  `if is_keyframe:` block that also calls `window_buf.push()`, not behind
  `run_inference`'s stride timing.
- **Annotation colors must be indexed by class id, not detection
  position.** New classes: extend `build_class_palette` (already sizes
  itself from `len(model_names)`); don't rebuild a per-frame palette from
  `detections.class_id` in encounter order.
- `M3_CROP`/`PERSON_ROI` are camera-position-specific pixel coordinates —
  re-derive both after any camera move or resolution change.
- **Adding a 4th machine**: extend `MACHINES`, `DOOR_CONF_PER_MACHINE`,
  `DOOR_HOLD_PER_MACHINE`, the `CT_COLUMNS`/`SEC_COLS` schema loops
  (already loop over `MACHINES`, mostly automatic) — but `TELEM_COLS` and
  the `tel_writer.write([...])` call hardcode `M1/M2/M3` and are NOT built
  from `MACHINES`. You'd need to generalize both.
