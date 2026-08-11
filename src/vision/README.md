# Vision Stage — Door/Person Detection + Context-Vector Extraction

Two scripts, run in order, that turn raw CCTV footage into the vision half
of the C_t context vector:

1. **`train_door_person_detector.py`** (was `train_yolo26_seg_4110.py`) —
   trains the YOLO model that detects `person` and each machine's
   open/closed door state. Run this once (or whenever the camera setup /
   dataset changes) to produce a `.pt` weights file.
2. **`extract_context_vector_from_video.py`** (was
   `iaq_feature_extraction_pipeline_v7.py`) — uses that trained model to
   process actual session video, frame by frame, into the CSVs the rest of
   this project consumes.

For internals/architecture detail on either script, see
[CLAUDE.md](CLAUDE.md).

---

## Part 1 — `train_door_person_detector.py`

Trains an object detection + segmentation model that looks at camera
images and figures out whether a **door is open or closed** on each of
three machines (`machine_1`, `machine_2`, `machine_3`), and where a
**person** is in the frame. The primary target is **M3** — a
ceiling-mounted camera looking straight down at machine 3's door.

Output: a trained model file (`best.pt`) deployable on edge devices like a
Raspberry Pi 5 for real-time door-state detection with no cloud dependency.

### Training data

Roboflow-annotated and pre-processed (auto-orient, grayscale, histogram
equalization, ~1.2% noise injection, 3× augmentation):

| Split | Images | Purpose |
|---|---|---|
| Train | 3,636 (88%) | What the model learns from |
| Valid | 319 (8%) | Checked during training, not trained on |
| Test | 155 (4%) | Held back for a final honest check |
| **Total** | **4,110** | from ~1,370 original unique photos |

Because the images are grayscale, color-based augmentation (hue/saturation)
is deliberately disabled — see the config table below.

### Requirements

```bash
pip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyyaml
```
NVIDIA GPU strongly recommended (10–16GB+ VRAM ideal; the script
auto-reduces batch size for smaller GPUs). 16GB+ system RAM recommended for
dataset caching.

### Before running

Update the dataset path near the top of the script:
```python
DATASET_YAML: str = (r"<path to your Roboflow export>/data.yaml")
```

### Running it

```bash
python train_door_person_detector.py
```

Takes roughly 1–6+ hours depending on GPU. Progress prints to the terminal
and to `training_run.log` in the same folder. In order: environment info →
config validation → dataset balance audit → weighted `data.yaml` written
(`data_4110_weighted.yaml`) → training (epoch by epoch) → validation
results (per-class scorecard) → export hints.

### Key config (already tuned for this dataset — see `Config` class)

| Setting | Value | Meaning |
|---|---|---|
| `MODEL_WEIGHTS` | `yolo26n-seg.pt` | Smallest/fastest YOLO26 segmentation model, edge-deployable |
| `EPOCHS` / `PATIENCE` | 200 / 40 | Max training passes / early-stop if no improvement for 40 epochs |
| `IMAGE_SIZE` | 1280 | Must be divisible by 32 (YOLO stride requirement) |
| `BATCH_SIZE` | 8 (auto-adjusted) | Auto-lowered by `suggest_batch_size()` based on detected VRAM |
| `LEARNING_RATE` (`lr0`) | 0.002 | |
| `HSV_H`, `HSV_S` | 0.0, 0.0 | Color augmentation off — dataset is grayscale |
| `HSV_V` | 0.3 | Brightness augmentation stays on |
| `FLIPUD` | 0.0 | Vertical flip off — camera orientation is fixed |
| `CLS_LOSS_WEIGHT` (`cls`) | 1.5 | Raised — open/closed doors can look visually similar |

Full table with every setting and its reasoning: [CLAUDE.md](CLAUDE.md).

### Reading the results

```
Class                                   P      R      F1     AP50   Result
machine_3_door_open                    0.85   0.87   0.86   0.84   PASS  ★
```
P = precision ("when it says open, how often is it right"), R = recall
("of all real open doors, how many did it catch"), F1 = balance of both,
AP50 = standard detection accuracy. ★ marks the M3 classes (primary
target). PASS/FAIL compares each class against a size-appropriate minimum
target.

### Output location
```
local_train/seg_laser_4110_grayscale/
├── weights/best.pt   ← use this one
├── weights/last.pt
└── results.png       ← training curves
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `data.yaml NOT FOUND` | Fix `DATASET_YAML` path |
| No GPU in the log, training very slow | Fix PyTorch/CUDA install, or expect CPU-speed training |
| CUDA out of memory | Lower `IMAGE_SIZE` (e.g. 960), or set `cache=False` |
| Class shows `CRITICAL < 150` | Too few training examples — label more in Roboflow |

---

## Part 2 — `extract_context_vector_from_video.py`

Batch-processes CCTV/NVR video using the model trained above, and turns it
into three CSVs: per-frame tracking telemetry, per-second door/occupancy
state, and 60-second statistical feature windows (the actual C_t vectors
fed downstream).

For each video it detects **people** and the **open/closed door state** of
up to 3 machines, tracks people with a stable ID across the whole session
(survives brief occlusions), debounces door detections so a single bad
frame doesn't get counted as a toggle, and writes an annotated copy of the
video for visual QA.

### What changed from the original `_v7` script (kept for the record)

Two bugs fixed during code review, full writeup in the module docstring:
1. **Keyframe/inference desync (data-corrupting)** — on odd-fps source
   video, roughly half of all seconds were silently skipped by the
   door-debounce and time accounting. Fixed by forcing inference on every
   keyframe.
2. **Wrong annotation colors (cosmetic)** — the color palette was indexed
   incorrectly. Fixed with one static, correctly-indexed palette.

### Requirements
```bash
pip install opencv-python numpy supervision ultralytics
```
A trained `.pt` model (Part 1, above) with classes named exactly `person`,
`machine_1_door_open`, `machine_1_door_closed`, `machine_2_door_open`,
`machine_2_door_closed`, `machine_3_door_open`, `machine_3_door_closed`.
`tkinter` for the folder-picker dialogs.

### Running it
```bash
python extract_context_vector_from_video.py
```
Two folder-picker dialogs: input videos (`.mp4`/`.avi`/`.3gp`, processed in
sorted filename order as one continuous session), then output folder.

### Configuration (top-of-file constants, no CLI flags)

| Constant | Purpose |
|---|---|
| `MODEL_PATH` | Path to the `.pt` weights from Part 1 |
| `PERSON_ROI` | Polygon (pixel coords) defining the counted/tracked zone — re-draw if the camera moves |
| `DOOR_CONF_PER_MACHINE` | Per-machine minimum detection confidence |
| `DOOR_HOLD_PER_MACHINE` | Consecutive seconds required before a door state change commits |
| `M3_CROP` | Optional zoomed-in second inference pass for M3's door (see tuning below) |
| `DIAG_MODE` | Verbose per-keyframe diagnostics; turn off once tuned |
| `WINDOW_SECONDS` | Feature-extraction window size (default 60s) |
| `INFERENCE_STRIDE` | Run YOLO every Nth frame for speed (keyframes always run regardless) |

### Tuning an unreliable machine (e.g. M3)
1. Run with `DIAG_MODE = True` on a representative clip.
2. Watch the console `[KF ...]` lines and annotated video — is it a
   confidence issue (scale mismatch, expected when close to camera) or not
   firing at all?
3. If scale mismatch: read M3's pixel bounding box from the annotated
   video, set `M3_CROP = (x1, y1, x2, y2)` with ~60px padding. This crops
   just that region and re-infers at 640×640, restoring training-time scale.
4. Re-run, check `[KF ...]` — confidences should rise.
5. Set `DIAG_MODE = False` for production.

### Output files
All tagged with a session timestamp parsed from the first video's filename.

**`tracking_telemetry_<tag>.csv`** — one row per processed frame. Raw,
undebounced: timestamp, debounced + raw door states per machine, in-zone
person count, per-video and stable cross-occlusion global person IDs.

**`per_second_analytics_<tag>.csv`** — one row per second (keyframe).
Debounced door state per machine, cumulative toggle count, cumulative
seconds per state, max person count, running unique-people total, this
second's optical-flow motion score.

**`Ct_vectors_<tag>.csv`** — one row per 60-second window. The actual
feature vectors for downstream modeling:
- `M{n}_tau_open` — seconds the door was open this window
- `M{n}_f_trans` — open↔closed transitions this window
- `M{n}_rho_open` — fraction of window spent open (`tau_open / window`)
- `M{n}_eps_max` — longest continuous open run
- `M{n}_phi_open` — fraction of the way into the window before the door
  first opened (1.0 if it never opened)
- `n_person`, `mu_motion`, `sigma2_motion` — max in-zone people, mean/
  variance of optical-flow motion over people's bounding boxes

> **Known gap**: this is the complete set of features `compute_Dt()`
> produces. The downstream modeling stage's feature list also expects
> `M{n}_emission_weight`, `M{n}_effective_tau`,
> `M{n}_consecutive_full_open`, and per-machine operational-state
> one-hots — **none of which are computed here**. See
> [`../modeling/README.md`](../modeling/README.md)'s "Known Gap" section
> for the full explanation and what to do about it.

**`annotated_<video_file>.mp4`** — full video with ROI polygon,
boxes/labels, 3-line HUD, keyframe marker bar.

### Known limitations
- 1-fps keyframe sampling means a person occluding the door for exactly
  one keyframe can still cause a phantom transition if it persists long
  enough to win the debounce vote.
- `PERSON_ROI` and `M3_CROP` are hardcoded pixel coordinates tied to one
  camera framing — re-derive both after any camera move or resolution change.
- The re-identification heuristic (`StableIDRegistry`) is a lightweight
  IOU + color-histogram matcher, not a learned re-ID embedding — good
  enough for brief occlusions, not for someone leaving frame for a long
  time or changing appearance drastically.
