"""
YOLO26 Segmentation Training Script  —  4110-Image Roboflow Dataset Edition
=============================================================================
Target  : All machine door states, focus on M3 (ceiling top-down camera)
Model   : yolo26n-seg  (edge-deployable on Raspberry Pi 5)

ROBOFLOW DATASET FACTS  (from your screenshot)
───────────────────────────────────────────────
Total images     : 4110  (after Roboflow augmentation)
Train            : 3636  (88%)
Valid            : 319   (8%)
Test             : 155   (4%)
Original unique frames : ~1370  (4110 ÷ 3 augmentation outputs)

Roboflow preprocessing applied to ALL images:
  1. Auto-Orient          → camera rotation corrected
  2. Grayscale            → ALL IMAGES ARE GRAYSCALE  ← most critical change
  3. Histogram Equalization → contrast already normalised
  4. Noise (1.21% pixels) → random noise already baked in

Roboflow augmentation already applied (3× per image):
  Outputs per training example = 3
  This means Roboflow already generated 3 versions of each original frame.
  YOLO must not duplicate augmentations Roboflow already performed.

WHAT CHANGED FROM PREVIOUS 1686-IMAGE SCRIPT (tagged RF below)
───────────────────────────────────────────────────────────────
RF-01  EPOCHS: 300 → 200
       3636 train images (vs 1348 before) = 2.7× more data per epoch.
       More data per epoch → faster convergence → fewer epochs needed.
       200 × 455 steps = 91,000 total steps — sufficient for medium dataset.

RF-02  BATCH_SIZE: 4 → 8
       3636 images supports larger batches without gradient noise.
       Batch=8 gives more stable gradients and faster GPU utilisation.
       At imgsz=1280, batch=8 needs ~12 GB VRAM — auto-adjusted below.

RF-03  LEARNING_RATE: 0.001 → 0.002
       Medium dataset (3636 train) supports higher LR than small (1348).
       MuSGD momentum handles 0.002 stably. Cosine decay to 0.00002.

RF-04  DROPOUT: 0.1 → 0.0
       Overfitting risk drops with 3636 images. Dropout adds no benefit
       once the dataset is large enough to provide natural regularisation.

RF-05  LABEL_SMOOTHING: 0.1 → 0.05
       Less smoothing needed with more data. 0.05 still prevents
       extreme overconfidence without softening sharp class boundaries.

RF-06  MIXUP: 0.2 → 0.1
       Roboflow already produced 3× augmented versions per image.
       Aggressive YOLO mixup on top of Roboflow augmentation creates
       over-augmented training data. Reduce to 0.1.

RF-07  HSV_H: 0.015 → 0.0  (CRITICAL — grayscale has no hue channel)
       HSV_S: 0.8   → 0.0  (CRITICAL — grayscale has no saturation)
       Images are GRAYSCALE. Applying hue/saturation augmentation to
       grayscale images does nothing useful — it wastes the augmentation
       budget and adds unnecessary variation to single-channel images.
       Only brightness (hsv_v) affects grayscale images.

RF-08  HSV_V: 0.5 → 0.3
       Histogram equalization already normalised contrast in all images.
       Further brightness augmentation in YOLO should be conservative.
       0.3 adds mild variation without fighting the equalization.

RF-09  SAVE_PERIOD: 5 → 10
       Training at batch=8 on 3636 images is slower per epoch than
       batch=4 on 1348. Save every 10 epochs — each run is longer.

RF-10  PATIENCE: 50 → 40
       Medium dataset converges more reliably. Fewer non-improving
       epochs needed before early stop.

RF-11  cache comment updated: 3636 images ≈ 5.3 GB — still fits in RAM
       if machine has ≥ 16 GB. Set cache=False if RAM < 12 GB.

RF-12  RUN_NAME updated to reflect 4110-image dataset.

RF-13  DEGREES: 10.0 → 5.0
       Auto-Orient already corrected camera rotation in Roboflow.
       Large rotation augmentation on already-oriented images adds
       unrealistic orientations. 5.0° is sufficient for minor variation.

RF-14  Validation targets raised (more data → higher expected performance):
       M3_open:   P≥0.82  R≥0.84  AP50≥0.82  (was 0.75/0.78/0.75)
       M3_closed: P≥0.84  R≥0.84  AP50≥0.82  (was 0.80/0.80/0.78)

UNCHANGED FROM PREVIOUS SCRIPT
───────────────────────────────
MODEL_WEIGHTS=yolo26n-seg, optimizer=MuSGD, IMAGE_SIZE=1280, SCALE=0.9,
COPY_PASTE=0.3, SHEAR=5.0, WARMUP_EPOCHS=5, FLIPUD=0.0, FLIPLR=0.5,
close_mosaic=20, BOX=7.5, CLS=1.5, overlap_mask, mask_ratio=4, amp=True,
audit_dataset_balance(), write_data_yaml_with_weights(), validate_per_class()

Requirements:
    pip install ultralytics torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
    pip install pyyaml
"""

import os
import sys
import time
import glob
import logging
import yaml
from pathlib import Path
from collections import Counter


# ─── 1. Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_run.log"),
    ],
)
log = logging.getLogger(__name__)


# ─── 2. Config ────────────────────────────────────────────────────────────────
class Config:

    # DATASET_YAML: str = (
    #     r"D:\Sunil Work\Laser AI Study\roboflow_dataset"
    #     r"\laser_person_machine.v1-laser_person_machine_dataset.yolo26\data.yaml"
    # )

    DATASET_YAML: str = (r"H:\Sunil Work\roboflow_dataset\laser_person_machine.v2-laser_person_machine_dataset.yolo26\data.yaml")

    MODEL_WEIGHTS: str = "yolo26n-seg.pt"
    #               alt: "yolo26s-seg.pt"  — more accurate, 3× heavier

    # RF-01: 300 → 200 epochs.
    # 3636 train images gives 455 steps/epoch at batch=8.
    # 200 × 455 = 91,000 total gradient steps — sufficient for this size.
    EPOCHS: int   = 200

    # RF-10: 50 → 40. Medium dataset converges reliably.
    PATIENCE: int = 40

    # imgsz=1280 preserves M3 hinge-edge shadow detail at P3/P4 level.
    IMAGE_SIZE: int = 1280

    # RF-02: 4 → 8. 3636 images supports stable batch=8 gradients.
    # Auto-adjusted in main() based on actual VRAM.
    BATCH_SIZE: int = 8

    # RF-03: 0.001 → 0.002. Medium dataset supports higher LR.
    # MuSGD momentum stable at this value. Cosine decay → 0.00002.
    LEARNING_RATE: float   = 0.002
    LR_FINAL_FACTOR: float = 0.01

    WARMUP_EPOCHS: int  = 5          # standard for ≥100 epoch runs
    WEIGHT_DECAY: float = 0.0005

    # RF-04: 0.1 → 0.0 dropout.
    # 3636 images provides enough natural regularisation.
    DROPOUT: float = 0.0

    SCALE: float       = 0.9    # critical for M3 close-camera fill %
    MOSAIC: float      = 1.0    # essential — combines M3 close-up with M1/M2 far
    CLOSE_MOSAIC: int  = 20     # disable mosaic last 20 epochs for clean fine-tune

    # RF-06: 0.2 → 0.1 mixup.
    # Roboflow already applied 3× augmentation per image.
    # Stack-augmenting with heavy YOLO mixup over-augments the dataset.
    MIXUP: float  = 0.1

    COPY_PASTE: float = 0.3      # paste M3 close-up door into M1/M2 scenes

    # RF-13: 10.0 → 5.0 degrees.
    # Roboflow Auto-Orient already corrected all rotation issues.
    # 5° covers minor wobble without adding unrealistic orientations.
    DEGREES: float   = 5.0

    TRANSLATE: float = 0.2       # partial door at frame edge — keep
    SHEAR: float     = 5.0       # perspective distortion for close M3

    # RF-07 CRITICAL: hsv_h and hsv_s set to 0.0 for grayscale images.
    # All images were converted to grayscale in Roboflow.
    # Hue and saturation augmentation has ZERO effect on grayscale images —
    # it only adds meaningless channel variation that wastes augmentation
    # diversity and can confuse YOLO's internal normalisation.
    HSV_H: float = 0.0     # RF-07: was 0.015 — NO EFFECT on grayscale
    HSV_S: float = 0.0     # RF-07: was 0.8   — NO EFFECT on grayscale

    # RF-08: 0.5 → 0.3 brightness (hsv_v).
    # Histogram equalization in Roboflow already normalised contrast.
    # Conservative 0.3 adds useful brightness variation (day/night lab
    # lighting) without fighting the already-equalized contrast range.
    HSV_V: float = 0.3

    FLIPUD: float = 0.0    # no vertical flip (ceiling-mounted camera)
    FLIPLR: float = 0.5

    # CLS=1.5 — M3 state classification still needs emphasis.
    # More data helps but the class-imbalance between open/closed means
    # the model still tends to favour the majority (closed) state.
    CLS_LOSS_WEIGHT: float = 1.5
    BOX_LOSS_WEIGHT: float = 7.5
    # NOTE: do NOT pass dfl= — DFL removed in YOLO26

    # RF-05: 0.1 → 0.05 label_smoothing.
    # Less smoothing needed with more data. 0.05 prevents overconfidence
    # while preserving sharp open/closed class boundaries.
    LABEL_SMOOTHING: float = 0.05

    DEVICE: int  = 0
    WORKERS: int = 4

    # RF-09: 5 → 10 save_period.
    # Batch=8 on 3636 images means each epoch takes longer.
    SAVE_PERIOD: int = 10

    PROJECT_DIR: str = "local_train"
    # RF-12: updated run name
    RUN_NAME: str    = "seg_laser_4110_grayscale"


# ─── 3. Dataset balance audit ─────────────────────────────────────────────────
def audit_dataset_balance(cfg: Config) -> dict:
    """
    Scan label .txt files in the train split.
    Count instances per class ID.
    Print balance table with inverse-frequency weights.
    Warn about dangerous imbalances.

    NOTE on Roboflow 3× augmentation:
      Roboflow generates 3 augmented versions per original image.
      The label files will reflect 3× the original unique instance counts.
      This does NOT change the weight calculation — we still want inverse
      frequency weights because the RELATIVE class imbalance is the same.
      M1_open at 33 original → ~99 augmented → still the minority.
    """
    source_yaml = Path(cfg.DATASET_YAML)
    if not source_yaml.exists():
        log.warning(f"data.yaml not found: {source_yaml}")
        return {}

    with open(source_yaml) as f:
        data = yaml.safe_load(f)

    dataset_root = Path(data.get("path", source_yaml.parent))
    names = data.get("names", {})
    id_to_name = ({int(k): v for k, v in names.items()}
                  if isinstance(names, dict)
                  else {i: v for i, v in enumerate(names)})

    label_dir = None
    for candidate in [
        dataset_root / "train" / "labels",
        dataset_root / "labels" / "train",
        source_yaml.parent / "train" / "labels",
        source_yaml.parent / "labels",
    ]:
        if candidate.exists():
            label_dir = candidate; break

    if label_dir is None:
        log.warning("Labels directory not found — using hardcoded weights.")
        return {}

    counts: Counter = Counter()
    file_count = 0
    for lf in glob.glob(str(label_dir / "**" / "*.txt"), recursive=True):
        try:
            for line in open(lf):
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
            file_count += 1
        except Exception:
            pass

    log.info("=" * 72)
    log.info(f"DATASET BALANCE AUDIT  ({file_count} label files in train split)")
    log.info(f"  Total instances (includes 3× Roboflow augmentation): "
             f"{sum(counts.values())}")
    log.info(f"  Estimated original unique instances: "
             f"~{sum(counts.values()) // 3}")
    log.info("")
    log.info(f"  {'ID':>3}  {'Class':38s}  {'Count':>6}  "
             f"{'% total':>7}  {'Weight':>8}  Status")
    log.info("  " + "─" * 68)

    total   = sum(counts.values()) or 1
    max_cnt = max(counts.values()) if counts else 1
    issues  = []

    for cid in sorted(counts.keys()):
        cnt    = counts[cid]
        name   = id_to_name.get(cid, f"class_{cid}")
        pct    = cnt / total * 100
        w      = min(3.0, round(max_cnt / cnt, 2))
        status = ("CRITICAL < 150" if cnt < 150 else
                  "LOW < 450"      if cnt < 450 else "OK")
        if cnt < 150:
            issues.append(f"[{cid}] {name}: {cnt} instances (< 150)")
        log.info(f"  {cid:>3}  {name:38s}  {cnt:>6}  "
                 f"{pct:>6.1f}%  {w:>8.2f}  {status}")

    if issues:
        log.warning("")
        log.warning("  LOW INSTANCE COUNTS (add more images if possible):")
        for iss in issues:
            log.warning(f"    → {iss}")

    # M3 balance check
    m3o_id = next((k for k, v in id_to_name.items()
                   if v == "machine_3_door_open"),   None)
    m3c_id = next((k for k, v in id_to_name.items()
                   if v == "machine_3_door_closed"), None)
    if m3o_id is not None and m3c_id is not None:
        m3o = counts.get(m3o_id, 0)
        m3c = counts.get(m3c_id, 0)
        if m3o > 0 and m3c > 0:
            ratio = m3o / m3c
            log.info("")
            log.info(f"  M3 open:closed = {m3o}:{m3c}  ratio={ratio:.2f}")
            if ratio > 3.0:
                log.warning(
                    f"  M3_open dominates ({ratio:.1f}×). "
                    f"M3_closed weight boosted automatically.")
            elif ratio < 0.3:
                log.warning(
                    f"  M3_open rare (ratio={ratio:.2f}). "
                    f"M3_open weight boosted automatically.")
            else:
                log.info("  M3 ratio healthy (0.3–3.0).")

    log.info("=" * 72)
    return dict(counts)


# ─── 4. Write data.yaml with inverse-frequency weights ────────────────────────
def write_data_yaml_with_weights(cfg: Config, counts: dict) -> str:
    """
    Compute per-class loss weights from actual instance counts.
    Formula: w_i = min(3.0, max_count / count_i)
    Cap at 3.0 to prevent extreme gradients from very rare classes.

    With 3× Roboflow augmentation, all counts are 3× the original.
    The relative weights remain correct because the RATIO between
    classes does not change with uniform augmentation.

    Hardcoded fallback if label scan fails:
      Reflects expected distribution after M3_open expansion +
      3× Roboflow augmentation:
      [person, M1_open, M1_closed, M2_open, M2_closed, M3_open, M3_closed]
      [  1.0,    3.0,     2.0,      3.0,     2.5,       1.5,     2.5   ]
    """
    source_yaml = Path(cfg.DATASET_YAML)
    with open(source_yaml) as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    id_to_name = ({int(k): v for k, v in names.items()}
                  if isinstance(names, dict)
                  else {i: v for i, v in enumerate(names)})
    nc = data.get("nc", len(id_to_name))

    if counts and len(counts) >= nc - 1:
        max_c   = max(counts.values())
        weights = [min(3.0, round(max_c / counts.get(cid, 1), 2))
                   for cid in range(nc)]
        log.info("Weight source: MEASURED from label files.")
    else:
        # Fallback with 3× Roboflow augmentation already factored in
        weights = [1.0, 3.0, 2.0, 3.0, 2.5, 1.5, 2.5]
        while len(weights) < nc: weights.append(1.0)
        weights = weights[:nc]
        log.warning("Weight source: HARDCODED fallback. "
                    "Verify class order matches your data.yaml.")

    data["weights"] = weights

    out_path = source_yaml.parent / "data_4110_weighted.yaml"
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    log.info("─" * 58)
    log.info("Per-class loss weights:")
    log.info(f"  {'ID':>3}  {'Class':38s}  Weight")
    log.info("  " + "─" * 50)
    for cid in range(nc):
        name = id_to_name.get(cid, f"class_{cid}")
        w    = weights[cid]
        note = ("  ← majority class"  if w <= 1.1 else
                "  ← CAPPED (rare)"   if w >= 3.0 else "")
        log.info(f"  {cid:>3}  {name:38s}  {w:.2f}{note}")
    log.info(f"Written: {out_path}")
    log.info("─" * 58)
    return str(out_path)


# ─── 5. Config validation ─────────────────────────────────────────────────────
def validate_config(cfg: Config) -> None:
    yaml_path = Path(cfg.DATASET_YAML)
    if not yaml_path.exists():
        log.error(f"data.yaml NOT FOUND: {yaml_path}"); sys.exit(1)
    if cfg.IMAGE_SIZE % 32 != 0:
        log.error(f"IMAGE_SIZE {cfg.IMAGE_SIZE} must be divisible by 32.")
        sys.exit(1)

    log.info(f"  Dataset      : {yaml_path}")
    log.info(f"  Model        : {cfg.MODEL_WEIGHTS}")
    log.info(f"  imgsz        : {cfg.IMAGE_SIZE}")
    log.info(f"  epochs       : {cfg.EPOCHS}  patience={cfg.PATIENCE}")
    log.info(f"  batch        : {cfg.BATCH_SIZE}  workers={cfg.WORKERS}")
    log.info(f"  lr0          : {cfg.LEARNING_RATE}")
    log.info(f"  dropout      : {cfg.DROPOUT}  (0 = enough data for natural reg)")
    log.info(f"  lbl_smooth   : {cfg.LABEL_SMOOTHING}")
    log.info(f"  optimizer    : MuSGD  (YOLO26 native)")
    log.info("")
    log.info("  GRAYSCALE AUGMENTATION (from Roboflow preprocessing):")
    log.info(f"    hsv_h={cfg.HSV_H}  (0 = correct, no hue on grayscale)")
    log.info(f"    hsv_s={cfg.HSV_S}  (0 = correct, no saturation on grayscale)")
    log.info(f"    hsv_v={cfg.HSV_V}  (brightness DOES affect grayscale — kept)")


# ─── 6. Environment info ──────────────────────────────────────────────────────
def log_environment() -> None:
    import torch
    log.info("=" * 65)
    log.info("ENVIRONMENT")
    log.info(f"  Python  : {sys.version.split()[0]}")
    log.info(f"  PyTorch : {torch.__version__}")
    log.info(f"  CUDA    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        dev  = torch.cuda.current_device()
        vram = torch.cuda.get_device_properties(dev).total_memory / 1e9
        log.info(f"  GPU     : {torch.cuda.get_device_name(dev)}")
        log.info(f"  VRAM    : {vram:.1f} GB")
        # Specific warnings for batch=8 at imgsz=1280
        if vram < 10.0:
            log.warning(
                f"  VRAM {vram:.1f}GB may be insufficient for batch=8 imgsz=1280. "
                f"Will auto-adjust to batch=4 or lower.")
        elif vram < 8.0:
            log.warning(
                f"  VRAM {vram:.1f}GB: try imgsz=960 batch=4 if OOM.")
    else:
        log.warning("  No GPU detected — training will be extremely slow.")
    log.info("=" * 65)


# ─── 7. Adaptive batch size ───────────────────────────────────────────────────
def suggest_batch_size(vram_gb: float, imgsz: int) -> int:
    """
    At imgsz=1280, each image uses ~2.5 GB GPU memory per batch item
    for yolo26n-seg. Batch=8 needs ~10 GB, batch=4 needs ~5 GB.
    """
    if imgsz >= 1280:
        if vram_gb >= 24: return 16
        if vram_gb >= 16: return 8
        if vram_gb >= 10: return 4
        if vram_gb >= 6:  return 2
        return 1
    else:
        if vram_gb >= 24: return 32
        if vram_gb >= 16: return 16
        if vram_gb >= 10: return 8
        if vram_gb >= 6:  return 4
        return 2


# ─── 8. Dataset inspection ────────────────────────────────────────────────────
def inspect_dataset(yaml_path: str) -> None:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    names_list = ([names[k] for k in sorted(names.keys())]
                  if isinstance(names, dict) else list(names))

    log.info("DATASET YAML")
    log.info(f"  nc        : {data.get('nc', '?')}")
    log.info(f"  classes   : {names_list}")
    log.info(f"  weights   : {data.get('weights', 'not set')}")
    for split in ("train", "val", "test"):
        if split in data:
            log.info(f"  {split:5s}     : {data[split]}")

    # Verify M3 classes exist with exact spelling
    for cls in ("machine_3_door_open", "machine_3_door_closed"):
        if not any(cls == n for n in names_list):
            log.warning(f"  '{cls}' NOT FOUND in class names — check spelling!")
    log.info("=" * 65)


# ─── 9. Per-class validation ──────────────────────────────────────────────────
def validate_per_class(model, data_yaml: str, imgsz: int, device: int) -> None:
    """
    Run val split and print per-class P / R / F1 / AP50.

    Targets raised from previous script because:
      - 3636 train images (vs 1348 before) = more data per class
      - Roboflow 3× augmentation = better distribution coverage
      - Grayscale normalisation = consistent input representation

    RF-14 targets:
      M3_open:   P≥0.82  R≥0.84  AP50≥0.82
      M3_closed: P≥0.84  R≥0.84  AP50≥0.82
      M1_open:   P≥0.65  R≥0.68  AP50≥0.62  (still limited data)
      M2_open:   P≥0.70  R≥0.72  AP50≥0.68
    """
    log.info("Running per-class validation...")
    metrics = model.val(data=data_yaml, imgsz=imgsz,
                        device=device, split="val", conf=0.25)

    # RF-14: raised targets for 4110-image dataset
    TARGETS = {
        "machine_3_door_open":   {"P": 0.82, "R": 0.84, "AP50": 0.82},
        "machine_3_door_closed": {"P": 0.84, "R": 0.84, "AP50": 0.82},
        "machine_1_door_open":   {"P": 0.65, "R": 0.68, "AP50": 0.62},
        "machine_2_door_open":   {"P": 0.70, "R": 0.72, "AP50": 0.68},
    }

    log.info(f"\n{'─'*82}")
    log.info(f"  {'Class':38s}  {'P':>5}  {'R':>5}  {'F1':>5}  "
             f"{'AP50':>5}  Result")
    log.info(f"{'─'*82}")

    all_passed = True
    for i, name in enumerate(model.names.values()):
        p  = float(metrics.box.p[i])
        r  = float(metrics.box.r[i])
        f1 = 2 * p * r / (p + r + 1e-9)
        ap = float(metrics.box.ap50[i])

        result = ""
        if name in TARGETS:
            t      = TARGETS[name]
            issues = []
            if p  < t["P"]:    issues.append(f"P<{t['P']:.2f}")
            if r  < t["R"]:    issues.append(f"R<{t['R']:.2f}")
            if ap < t["AP50"]: issues.append(f"AP50<{t['AP50']:.2f}")
            if issues:
                result     = "FAIL: " + " | ".join(issues)
                all_passed = False
            else:
                result = "PASS"

        flag = " ★" if "machine_3" in name else ""
        log.info(f"  {name:38s}  {p:>5.3f}  {r:>5.3f}  {f1:>5.3f}  "
                 f"{ap:>5.3f}  {result}{flag}")

    log.info(f"{'─'*82}")
    log.info("  ★ = M3 (ceiling top-down camera — primary target)")
    if all_passed:
        log.info("  ✓ ALL TARGETS PASSED")
    else:
        log.warning("  Some targets failed — see NEXT STEPS below.")


# ─── 10. Main training ────────────────────────────────────────────────────────
def main() -> None:
    cfg = Config()
    log_environment()
    validate_config(cfg)

    counts        = audit_dataset_balance(cfg)
    weighted_yaml = write_data_yaml_with_weights(cfg, counts=counts)
    inspect_dataset(weighted_yaml)

    import torch
    from ultralytics import YOLO

    # Auto-adjust batch for available VRAM
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        auto_batch = suggest_batch_size(vram, cfg.IMAGE_SIZE)
        if auto_batch != cfg.BATCH_SIZE:
            log.warning(f"Batch: {cfg.BATCH_SIZE} → {auto_batch} "
                        f"(VRAM={vram:.1f}GB, imgsz={cfg.IMAGE_SIZE})")
            cfg.BATCH_SIZE = auto_batch

    log.info(f"Loading model: {cfg.MODEL_WEIGHTS}")
    model = YOLO(cfg.MODEL_WEIGHTS)

    # Print class names — verify M3 naming and distance ordering
    log.info("Model class names:")
    for cid, cname in sorted(model.names.items()):
        tag = ("  ← NEAR (top-down, shadow detection)" if "machine_3" in cname else
               "  ← FAR"                               if "machine_1" in cname else
               "  ← MID"                               if "machine_2" in cname else "")
        log.info(f"  [{cid:>2}] {cname}{tag}")

    # Summary of all key settings
    log.info("")
    log.info("=" * 65)
    log.info("TRAINING CONFIGURATION — 4110-image Roboflow dataset")
    log.info("=" * 65)
    log.info(f"  Total/Train/Val/Test: 4110 / 3636 / 319 / 155")
    log.info(f"  Roboflow augmentation: 3× per image (already applied)")
    log.info(f"  Preprocessing: GRAYSCALE + Histogram EQ + Auto-Orient")
    log.info("")
    log.info(f"  epochs       : {cfg.EPOCHS}  (RF-01: was 300)")
    log.info(f"  batch        : {cfg.BATCH_SIZE}  (RF-02: was 4)")
    log.info(f"  lr0          : {cfg.LEARNING_RATE}  (RF-03: was 0.001)")
    log.info(f"  dropout      : {cfg.DROPOUT}  (RF-04: was 0.1)")
    log.info(f"  label_smooth : {cfg.LABEL_SMOOTHING}  (RF-05: was 0.1)")
    log.info(f"  mixup        : {cfg.MIXUP}  (RF-06: was 0.2)")
    log.info(f"  hsv_h/s/v    : {cfg.HSV_H}/{cfg.HSV_S}/{cfg.HSV_V}  "
             f"(RF-07/08: h=0 s=0 for grayscale)")
    log.info(f"  degrees      : {cfg.DEGREES}  (RF-13: was 10.0)")
    log.info(f"  cache        : True  (~5.3GB, needs ≥16GB RAM)")
    log.info("=" * 65)
    log.info("")

    t0 = time.time()

    results = model.train(
        # ── Core ─────────────────────────────────────────────────────────
        data         = weighted_yaml,
        epochs       = cfg.EPOCHS,          # 200 (RF-01)
        imgsz        = cfg.IMAGE_SIZE,       # 1280
        batch        = cfg.BATCH_SIZE,       # 8 (RF-02)
        device       = cfg.DEVICE,
        workers      = cfg.WORKERS,

        # RF-11: cache=True. 3636 images ≈ 5.3 GB — fits in RAM ≥16 GB.
        # Eliminates disk I/O per batch → ~3× faster per epoch.
        # Set cache=False if machine has < 12 GB RAM.
        cache        = True,

        # ── YOLO26 native MuSGD optimizer ────────────────────────────────
        optimizer    = "MuSGD",
        lr0          = cfg.LEARNING_RATE,    # 0.002 (RF-03)
        lrf          = cfg.LR_FINAL_FACTOR,  # final = 0.00002
        warmup_epochs= cfg.WARMUP_EPOCHS,    # 5
        cos_lr       = True,
        weight_decay = cfg.WEIGHT_DECAY,
        momentum     = 0.937,
        dropout      = cfg.DROPOUT,          # 0.0 (RF-04)
        patience     = cfg.PATIENCE,         # 40 (RF-10)

        # ── Loss weights ─────────────────────────────────────────────────
        # Do NOT pass dfl= — removed in YOLO26
        box          = cfg.BOX_LOSS_WEIGHT,  # 7.5
        cls          = cfg.CLS_LOSS_WEIGHT,  # 1.5

        # ── Scale augmentation — critical for M3 ─────────────────────────
        scale        = cfg.SCALE,            # 0.9

        # ── Mosaic ───────────────────────────────────────────────────────
        mosaic       = cfg.MOSAIC,           # 1.0
        close_mosaic = cfg.CLOSE_MOSAIC,     # 20

        # ── Copy-paste ───────────────────────────────────────────────────
        copy_paste   = cfg.COPY_PASTE,       # 0.3

        # RF-06: reduced mixup — Roboflow already augmented 3× per image
        mixup        = cfg.MIXUP,            # 0.1

        # RF-13: reduced rotation — Auto-Orient already applied
        degrees      = cfg.DEGREES,          # 5.0

        translate    = cfg.TRANSLATE,        # 0.2
        shear        = cfg.SHEAR,            # 5.0

        # RF-07/08: GRAYSCALE-CORRECT colour augmentation
        # hsv_h=0: hue has NO EFFECT on single-channel grayscale images
        # hsv_s=0: saturation has NO EFFECT on grayscale images
        # hsv_v=0.3: brightness DOES affect grayscale — keep conservative
        #            (histogram eq already normalised contrast)
        hsv_h        = cfg.HSV_H,            # 0.0 (RF-07)
        hsv_s        = cfg.HSV_S,            # 0.0 (RF-07)
        hsv_v        = cfg.HSV_V,            # 0.3 (RF-08)

        flipud       = cfg.FLIPUD,           # 0.0
        fliplr       = cfg.FLIPLR,           # 0.5

        # ── Segmentation ─────────────────────────────────────────────────
        overlap_mask    = True,
        mask_ratio      = 4,

        # RF-05: reduced smoothing — more data, less overconfidence risk
        label_smoothing = cfg.LABEL_SMOOTHING,  # 0.05

        # ── Output ───────────────────────────────────────────────────────
        project      = cfg.PROJECT_DIR,
        name         = cfg.RUN_NAME,         # "seg_laser_4110_grayscale"
        save         = True,
        save_period  = cfg.SAVE_PERIOD,      # 10 (RF-09)
        val          = True,
        plots        = True,
        amp          = True,
        rect         = False,
        verbose      = True,
        pretrained   = True,
    )

    elapsed = time.time() - t0
    log.info("=" * 65)
    log.info(f"  Training complete in {elapsed / 60:.1f} min")
    log.info(f"  Results: {results.save_dir}")

    # RF-14: per-class validation with raised targets
    validate_per_class(model, weighted_yaml, cfg.IMAGE_SIZE, cfg.DEVICE)

    # Overall metrics
    val_m = model.val(data=weighted_yaml, imgsz=cfg.IMAGE_SIZE,
                      device=cfg.DEVICE, split="val")
    box = val_m.box; seg = val_m.seg
    log.info("OVERALL VALIDATION METRICS")
    log.info(f"  Box  mAP50    : {box.map50:.4f}")
    log.info(f"  Box  mAP50-95 : {box.map:.4f}")
    log.info(f"  Mask mAP50    : {seg.map50:.4f}")
    log.info(f"  Mask mAP50-95 : {seg.map:.4f}")
    log.info("=" * 65)

    # Export for edge deployment
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        log.info(f"  Best weights : {best}")
        log.info("  Pi 5 export  : model.export(format='onnx', imgsz=1280)")
        log.info("  Jetson export: model.export(format='engine', half=True)")
        log.info("  YOLO26 NMS-free: ONNX export has no NMS post-processing node")
    else:
        log.warning("  best.pt not found — check run directory.")

    log.info("")
    log.info("NEXT STEPS:")
    log.info("  1. Check per-class results.  ★ M3 is primary target.")
    log.info("  2. M3_open low recall → lower DOOR_CONF_PER_MACHINE[3] "
             "in iaq_v6_shadow_integrated.py")
    log.info("  3. M3_shadow still needed → run calibrate_shadow_detector() "
             "in m3_shadow_fix.py")
    log.info("  4. M1/M2 open failing → collect ≥150 more open images each")
    log.info("  5. Set MODEL_PATH in inference pipeline to best.pt above")


# ─── 11. Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # required on Windows
    main()


# ══════════════════════════════════════════════════════════════════════════════
# FULL CHANGE TABLE  vs  1686-IMAGE SCRIPT
# ══════════════════════════════════════════════════════════════════════════════
"""
Parameter        1686 script    This (4110)    Reason
─────────────────────────────────────────────────────────────────────────────
epochs           300            200            3636 train imgs = more data/epoch
batch            4              8              Medium dataset supports larger batch
lr0              0.001          0.002          More data = stable higher LR
dropout          0.1            0.0            More data = natural regularisation
label_smoothing  0.1            0.05           Less needed with more data
mixup            0.2            0.1            Roboflow already augmented 3×
degrees          10.0           5.0            Auto-Orient already applied
hsv_h            0.015          0.0   ★★       GRAYSCALE — hue has zero effect
hsv_s            0.8            0.0   ★★       GRAYSCALE — saturation zero effect
hsv_v            0.5            0.3            Histogram EQ applied — conservative
patience         50             40             Converges reliably on medium data
save_period      5              10             Longer per epoch with batch=8
run_name         seg_laser_1686 seg_laser_4110 Reflects actual dataset size
val targets      lower          raised         More data = higher expected perf
─────────────────────────────────────────────────────────────────────────────
★★ = Most critical change. Setting hsv_h/hsv_s on grayscale images wastes
     augmentation slots and adds meaningless channel variation.

UNCHANGED: yolo26n-seg, MuSGD, imgsz=1280, SCALE=0.9, COPY_PASTE=0.3,
           SHEAR=5.0, warmup=5, FLIPUD=0, FLIPLR=0.5, close_mosaic=20,
           BOX=7.5, CLS=1.5, overlap_mask, mask_ratio=4, amp=True,
           audit_dataset_balance(), write_data_yaml_with_weights(),
           validate_per_class(), cache=True
"""
