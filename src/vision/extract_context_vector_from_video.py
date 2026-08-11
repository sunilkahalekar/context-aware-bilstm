"""
IAQ Feature Extraction Pipeline  v7
=====================================

CHANGELOG vs v6
──────────────────────────────────────────────────────────────────
v7 fixes two bugs found in a code review of v6. Neither crashes —
both silently corrupt output, which is why they survived v6.

FIX 1 — Keyframe / inference-stride misalignment (data-corrupting)
  v6 computed:
      is_keyframe   = frame_count in keyframe_at              # 1 per real second
      run_inference = (frame_count % INFERENCE_STRIDE == 0)   # every 2nd frame
  and only ran DoorDebouncer.observe() / accumulated
  cum_setup_time / cum_swap_time when BOTH were true.

  keyframe_at maps `round(second * fps)` to a frame index. When fps is
  ODD (25fps, 15fps — common for NVR/CCTV footage), that frame index
  alternates even/odd every second (e.g. 25fps: 0, 25, 50, 75, 100 ->
  even, odd, even, odd, even...). Since run_inference only fires on
  even frame_count (stride=2), every odd-numbered second had
  is_keyframe=True but run_inference=False — the debouncer's
  .observe() was skipped for that entire second.

  Effect on 25fps source video: on ~50% of seconds,
    - DoorDebouncer never receives that second's vote, softening/
      delaying transition detection (the "3-second vote window" /
      "N consecutive keyframes" design assumed 1 observation per
      real second — it was really getting one every ~2 seconds).
    - cum_setup_time / cum_swap_time (Setup_Sec / Swap_Sec columns)
      under-count real elapsed time by roughly HALF.
    - person_boxes_kf was force-set to [] on the skipped branch, so
      optical-flow motion scoring silently contributed 0 for that
      second too (mu_motion / sigma2_motion diluted).

  This only manifests when fps does not stay an exact multiple of
  INFERENCE_STRIDE after rounding — so it can pass fine on a 30fps
  test clip and then quietly corrupt numbers on real 25fps site
  footage.

  FIX: a keyframe now always forces inference to run in that same
  iteration, regardless of stride parity:
      run_inference = (frame_count % INFERENCE_STRIDE == 0) or is_keyframe
  This costs at most one extra inference call per second versus the
  intended stride (negligible), and restores the "one real second =
  one debounce observation" invariant the whole DoorDebouncer design
  depends on.

FIX 2 — Per-frame ColorPalette was indexed by the wrong key (cosmetic)
  v6 rebuilt a ColorPalette every frame, one color per DETECTION in
  that frame, in detection order:
      colours = sv.ColorPalette(
          colors=[class_colour(model.names[int(cid)]) for cid in detections.class_id]
      )
  supervision's annotators resolve color via ColorLookup.CLASS by
  default, which indexes the palette by the detection's CLASS ID
  value (e.g. 7 for "machine_2_door_open"), not by its position in
  this per-frame list. Since the per-frame list is only as long as
  the number of detections in that frame, the annotator ends up doing
  colors[class_id % len(colors)] — effectively a near-random color,
  not the semantic open/closed/person color scheme intended. This
  doesn't crash, but it defeats the annotated video's visual QA
  purpose, including the documented M3_CROP calibration workflow
  ("pause annotated video ... read pixel bbox from the annotation
  overlay") where color-coding was meant to help.

  FIX: build ONE static palette after the model loads, indexed by
  class id (0..len(model.names)-1), and reuse it for every frame.
  The old sv.ColorPalette.default()/.DEFAULT version-safety concern
  is now moot since we never ask for a "default" palette at all —
  we always build our own explicit, correctly-indexed one. The
  _sv_default_palette() helper from v6 is removed as dead code.

FIXES CARRIED OVER FROM v6 (see original file history)
  • Door State Debouncing (f_trans accuracy) — DoorDebouncer class:
    3-second vote window + per-machine hold-off before a state
    change commits, per-machine confidence thresholds, and holding
    last committed state on full occlusion (no door class detected).
  • window_start written as a formatted string, not a raw datetime
    object (raw datetime repr is not parseable by pandas / Excel).
  • Partial window flush at end of session now guards correctly.
"""

import cv2
import csv
import queue
import threading
import numpy as np
import supervision as sv
from ultralytics import YOLO
from datetime import datetime, timedelta
from collections import deque
import os
import re
import tkinter as tk
from tkinter import filedialog


# ══════════════════════════════ CONFIGURATION ══════════════════════════════════

# MODEL_PATH = r"D:\Sunil Work\Laser AI Study\code_folder\dataset_training\runs\segment\local_train\seg_laser_v10_28_03_26\weights\best.pt"

MODEL_PATH =(r"D:\Sunil Work\Laser AI Study\code_folder\dataset_training\runs\segment\local_train\seg_laser_4110_grayscale\weights\best.pt")


PERSON_ROI = np.array([
    [440,269],[488,255],[556,239],[603,225],[648,222],[699,210],
    [768,196],[789,188],[851,207],[1092,243],[1326,365],[1624,516],
    [1674,542],[1470,819],[1113,1023],[856,1060],[768,1030],
    [520,612],[451,390],[428,321],[422,294]
])

MACHINES       = [1, 2, 3]
WINDOW_SECONDS = 60

INFERENCE_STRIDE = 2
USE_HALF         = True
TRACKER_CONFIG   = "botsort.yaml"

PERSON_CONF = 0.25
DOOR_CONF   = 0.40   # default fallback
DOOR_INFERENCE_CONF = 0.20   # floor for YOLO inference; per-machine post-filter applies above

# ── Per-machine door confidence thresholds ─────────────────────────────────
# M3 is very close to camera → door fills large frame fraction →
# YOLO scale mismatch → lower confidence scores are physically expected.
# Lower M3 threshold corrects for this. Debounce majority-vote still
# protects against false positives at the lower threshold.
DOOR_CONF_PER_MACHINE = {
    1: 0.40,   # M1 far — normal scale, normal threshold
    2: 0.40,   # M2 far — normal scale, normal threshold
    3: 0.25,   # M3 close — scale mismatch, lower threshold
}

# ── Per-machine debounce hold seconds ─────────────────────────────────────
# M3 is close to camera — operators cycle its door quickly.
# Requiring 2 consecutive seconds of majority-Open means short opens
# (1–2 seconds) are never committed. Reduce to 1 for M3.
DOOR_HOLD_PER_MACHINE = {
    1: 2,   # M1 — standard hold
    2: 2,   # M2 — standard hold
    3: 1,   # M3 — fast nearby door, 1 confirmed second is enough
}

# ── M3 crop-and-reinfer region ─────────────────────────────────────────────
# When M3's door fills 30-60% of the full frame, YOLO sees an out-of-scale
# object. Cropping the M3 region and resizing to 640×640 restores the
# scale YOLO was trained on, recovering normal confidence scores.
#
# Set to (x1, y1, x2, y2) pixel rectangle around M3's door.
# Add ~60px padding on all sides. Set None to disable (use full frame only).
# HOW TO SET: run with DIAG_MODE=True, pause annotated video at a frame
# where M3 door is visible, read its pixel bbox from the annotation overlay.
M3_CROP = None   # e.g. (30, 80, 520, 720) — fill in after checking video

# ── Diagnostic mode ────────────────────────────────────────────────────────
DIAG_MODE = True

DOOR_VOTE_WINDOW  = 3
DOOR_HOLD_SECONDS = 2

LOST_TRACK_TTL        = 90
REID_IOU_THRESHOLD    = 0.35
REID_COSINE_THRESHOLD = 0.60
REID_CROP_SIZE        = (64, 128)

FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
)

CLR_PERSON   = sv.Color(r=100, g=230, b=100)
CLR_DOOR_OPN = sv.Color(r=255, g=190, b=70)
CLR_DOOR_CLS = sv.Color(r=130, g=200, b=255)
CLR_OTHER    = sv.Color(r=200, g=200, b=200)
CLR_ROI      = sv.Color(r=255, g=60,  b=60)


# ══════════════════════════════ DOOR DEBOUNCER ════════════════════════════════

class DoorDebouncer:
    def __init__(self, initial_state: str = "Closed", hold_seconds: int = None):
        # hold_seconds defaults to DOOR_HOLD_SECONDS if not overridden.
        # Per-machine values come from DOOR_HOLD_PER_MACHINE at instantiation.
        self._hold         = hold_seconds if hold_seconds is not None else DOOR_HOLD_SECONDS
        self._committed     = initial_state
        self._pending       = None
        self._pending_count = 0
        self._vote_buf      = deque(maxlen=DOOR_VOTE_WINDOW)
        for _ in range(DOOR_VOTE_WINDOW):
            self._vote_buf.append(initial_state)

    def observe(self, raw):
        vote = raw if raw is not None else self._committed
        self._vote_buf.append(vote)
        open_count = sum(1 for v in self._vote_buf if v == "Open")
        majority   = "Open" if open_count > len(self._vote_buf) - open_count else "Closed"
        transition = False
        if majority != self._committed:
            if majority == self._pending:
                self._pending_count += 1
            else:
                self._pending       = majority
                self._pending_count = 1
            if self._pending_count >= self._hold:   # use per-machine hold
                self._committed     = majority
                self._pending       = None
                self._pending_count = 0
                transition          = True
        else:
            self._pending       = None
            self._pending_count = 0
        return self._committed, transition

    @property
    def state(self): return self._committed

    def reset(self, state: str = "Closed"):
        self.__init__(state, hold_seconds=self._hold)


# ══════════════════════════════ ASYNC CSV WRITERS ════════════════════════════

class AsyncCSVWriter:
    def __init__(self, path: str, columns: list):
        self._q    = queue.Queue(maxsize=4096)
        self._file = open(path, "w", newline="")
        self._w    = csv.writer(self._file)
        self._w.writerow(columns)
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _worker(self):
        while True:
            row = self._q.get()
            if row is None: break
            self._w.writerow(row)

    def write(self, row: list):  self._q.put(row)

    def close(self):
        self._q.put(None); self._t.join(); self._file.close()


class AsyncDictCSVWriter:
    def __init__(self, path: str, fieldnames: list):
        self._q    = queue.Queue(maxsize=512)
        self._file = open(path, "w", newline="")
        self._w    = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._w.writeheader()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _worker(self):
        while True:
            row = self._q.get()
            if row is None: break
            self._w.writerow(row)

    def write(self, row: dict): self._q.put(row)

    def close(self):
        self._q.put(None); self._t.join(); self._file.close()


# ══════════════════════════════ STABLE ReID REGISTRY ══════════════════════════

class TrackState:
    __slots__ = ("global_id", "last_xyxy", "last_crop_feat",
                 "frames_lost", "in_zone")

    def __init__(self, gid, xyxy, feat, in_zone=False):
        self.global_id      = gid
        self.last_xyxy      = xyxy
        self.last_crop_feat = feat
        self.frames_lost    = 0
        self.in_zone        = in_zone


def _extract_crop_feature(frame: np.ndarray, xyxy) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = max(0, int(xyxy[0])); y1 = max(0, int(xyxy[1]))
    x2 = min(w, int(xyxy[2])); y2 = min(h, int(xyxy[3]))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(96, dtype=np.float32)
    crop   = cv2.resize(frame[y1:y2, x1:x2], REID_CROP_SIZE)
    hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    feat   = np.concatenate([hist_h, hist_s, hist_v])
    norm   = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat


def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    inter  = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


class StableIDRegistry:
    def __init__(self):
        self._next_gid   = 1
        self._active     = {}
        self._lost       = {}
        self._tid_to_gid = {}

    def update(self, frame, tracker_ids, xyxys, in_zone_mask) -> dict:
        for st in list(self._active.values()):
            st.frames_lost += 1

        for tid in [t for t, st in self._active.items()
                    if st.frames_lost > LOST_TRACK_TTL]:
            self._lost[tid] = self._active.pop(tid)

        for tid in [t for t, st in self._lost.items()
                    if st.frames_lost > LOST_TRACK_TTL * 3]:
            del self._lost[tid]

        result = {}
        for i, tid in enumerate(tracker_ids):
            tid    = int(tid)
            xyxy   = xyxys[i]
            in_roi = bool(in_zone_mask[i])
            feat   = _extract_crop_feature(frame, xyxy)

            if tid in self._active:
                st = self._active[tid]
                st.last_xyxy = xyxy; st.last_crop_feat = feat
                st.frames_lost = 0;  st.in_zone = in_roi
                result[tid] = st.global_id
                continue

            best_tid, best_score = None, -1.0
            for lt_tid, lt_st in self._lost.items():
                iou = _iou(xyxy, lt_st.last_xyxy)
                if iou < REID_IOU_THRESHOLD: continue
                cos   = float(np.dot(feat, lt_st.last_crop_feat))
                score = 0.5 * iou + 0.5 * cos
                if score > best_score:
                    best_score = score; best_tid = lt_tid

            threshold = 0.5 * (REID_IOU_THRESHOLD + REID_COSINE_THRESHOLD)
            if best_tid is not None and best_score >= threshold:
                recovered = self._lost.pop(best_tid)
                recovered.last_xyxy = xyxy; recovered.last_crop_feat = feat
                recovered.frames_lost = 0;  recovered.in_zone = in_roi
                self._active[tid]     = recovered
                self._tid_to_gid[tid] = recovered.global_id
                result[tid]           = recovered.global_id
            else:
                gid = self._next_gid; self._next_gid += 1
                st  = TrackState(gid, xyxy, feat, in_roi)
                self._active[tid]     = st
                self._tid_to_gid[tid] = gid
                result[tid]           = gid

        return result


# ══════════════════════════════ D_t  COMPUTATION ══════════════════════════════

def compute_Dt(seq: list) -> dict:
    T        = len(seq) or 1
    tau_open = int(sum(seq))
    f_trans  = int(sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1]))
    rho_open = round(tau_open / T, 4)
    eps_max = run = 0
    for s in seq:
        run     = run + 1 if s == 1 else 0
        eps_max = max(eps_max, run)
    first_open = next((i for i, s in enumerate(seq) if s == 1), None)
    phi_open   = round(first_open / T, 4) if first_open is not None else 1.0
    return dict(tau_open=tau_open, f_trans=f_trans,
                rho_open=rho_open, eps_max=eps_max, phi_open=phi_open)


# ══════════════════════════════ H_t  COMPUTATION ══════════════════════════════

def compute_frame_motion(fu, fv, boxes: list) -> float:
    if not boxes: return 0.0
    h, w = fu.shape
    mag  = np.sqrt(fu**2 + fv**2)
    scores = []
    for (x1, y1, x2, y2) in boxes:
        r = mag[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if r.size: scores.append(float(r.mean()))
    return float(np.mean(scores)) if scores else 0.0


def compute_Ht(motion_seq: list, n_max: int) -> dict:
    arr = np.array(motion_seq, dtype=float)
    return dict(
        n_person      = n_max,
        mu_motion     = round(float(arr.mean()) if arr.size else 0.0, 6),
        sigma2_motion = round(float(arr.var())  if arr.size else 0.0, 6),
    )


# ══════════════════════════════ 60-SECOND WINDOW BUFFER ═══════════════════════

class WindowBuffer:
    def __init__(self, ws=WINDOW_SECONDS):
        self.ws = ws; self._reset()

    def _reset(self):
        self.door_seq        = {m: [] for m in MACHINES}
        self.motion_seq      = []
        self.person_counts   = []
        self.count           = 0
        self.window_start_ts = None

    def push(self, ts, door_binary, motion, n_person):
        if self.count == 0: self.window_start_ts = ts
        for m in MACHINES: self.door_seq[m].append(door_binary[m])
        self.motion_seq.append(motion)
        self.person_counts.append(n_person)
        self.count += 1

    def is_full(self): return self.count >= self.ws

    def flush(self) -> dict:
        if self.count == 0: return {}
        # window_start is formatted as a string, not a raw datetime object —
        # raw datetime repr is unparseable by pandas and Excel.
        row = {"window_start": self.window_start_ts.strftime("%Y-%m-%d %H:%M:%S")}
        for m in MACHINES:
            for k, v in compute_Dt(self.door_seq[m]).items():
                row[f"M{m}_{k}"] = v
        row.update(compute_Ht(self.motion_seq,
                              max(self.person_counts, default=0)))
        self._reset()
        return row


# ══════════════════════════════ CSV SCHEMAS ════════════════════════════════════

CT_COLUMNS = ["window_start"]
for _m in MACHINES:
    CT_COLUMNS += [f"M{_m}_tau_open", f"M{_m}_f_trans", f"M{_m}_rho_open",
                   f"M{_m}_eps_max",  f"M{_m}_phi_open"]
CT_COLUMNS += ["n_person", "mu_motion", "sigma2_motion"]

TELEM_COLS = [
    "Timestamp_ISO8601", "Unix_ms",
    "M1_State_Debounced", "M2_State_Debounced", "M3_State_Debounced",
    "M1_State_Raw",       "M2_State_Raw",       "M3_State_Raw",
    "Zone_Count", "Tracked_Person_IDs", "Global_Person_IDs",
]

SEC_COLS = [
    "Timestamp",
    "M1_State", "M1_Toggles", "M1_Setup_Sec", "M1_Swap_Sec", "M1_Raw_State",
    "M2_State", "M2_Toggles", "M2_Setup_Sec", "M2_Swap_Sec", "M2_Raw_State",
    "M3_State", "M3_Toggles", "M3_Setup_Sec", "M3_Swap_Sec", "M3_Raw_State",
    "Person_Count_Max", "Unique_Total_Persons", "Motion_Score_This_Second",
]


# ══════════════════════════════ HELPERS ════════════════════════════════════════

def select_folders():
    root = tk.Tk(); root.withdraw()
    vd = filedialog.askdirectory(title="Select Folder with Input Videos")
    sd = filedialog.askdirectory(title="Select Folder to Save Outputs")
    root.destroy()
    return vd, sd


def parse_start_time(filename: str) -> datetime:
    m = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2})[-_]?(\d{2})[-_]?(\d{2})', filename)
    if m:
        return datetime.strptime(
            f"{m.group(1)}_{m.group(2)}{m.group(3)}{m.group(4)}", "%Y-%m-%d_%H%M%S")
    m2 = re.search(r'_(\d{6})[\._]', filename)
    if m2:
        return datetime.strptime(
            f"{datetime.now().strftime('%Y-%m-%d')}_{m2.group(1)}", "%Y-%m-%d_%H%M%S")
    return datetime.now()


def session_tag(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H%M%S")


def class_colour(class_name: str) -> sv.Color:
    if class_name == "person":  return CLR_PERSON
    if "open"   in class_name:  return CLR_DOOR_OPN
    if "closed" in class_name:  return CLR_DOOR_CLS
    return CLR_OTHER


def build_class_palette(model_names: dict) -> sv.ColorPalette:
    """
    Build ONE static, class-id-indexed palette (v7 fix).

    supervision's annotators resolve color via ColorLookup.CLASS by
    default, i.e. colors[class_id]. A palette must therefore be sized
    to the full class list and indexed by class id — NOT rebuilt per
    frame from only the classes present in that frame (v6's approach),
    which caused colors[class_id % num_detections_this_frame] to pick
    an unrelated color.
    """
    n = len(model_names)
    return sv.ColorPalette(colors=[class_colour(model_names[i]) for i in range(n)])


def make_label(class_name: str, global_id) -> str:
    if class_name == "person":
        gid = f"#{int(global_id)}" if global_id is not None else "#?"
        return f"{gid}  Person"
    for m in MACHINES:
        if f"machine_{m}_door_open"   in class_name: return f"M{m}  Open"
        if f"machine_{m}_door_closed" in class_name: return f"M{m}  Closed"
    return class_name


def read_door_detections(results, model_names: dict,
                         extra_results=None) -> tuple:
    """
    Extract door detections using per-machine DOOR_CONF_PER_MACHINE thresholds.

    Parameters
    ----------
    results       : YOLO result from full-frame inference
    model_names   : model.names dict
    extra_results : YOLO result from M3 crop inference (or None)
                    When provided, M3 detections from the crop pass override
                    the full-frame M3 detection if the crop gives higher conf.

    Returns
    -------
    raw_state     : {m: "Open"|"Closed"|None}
    raw_conf      : {m: float|None}
    all_door_hits : [(label, conf)] — all door hits for DIAG_MODE
    """
    raw_state     = {m: None  for m in MACHINES}
    raw_conf      = {m: None  for m in MACHINES}
    all_door_hits = []

    # Helper to process one result object
    def _process(res):
        if res.boxes is None:
            return
        for box in res.boxes:
            conf = float(box.conf[0])
            lbl  = model_names[int(box.cls[0])]
            is_door = any(
                lbl == f"machine_{m}_door_open" or
                lbl == f"machine_{m}_door_closed"
                for m in MACHINES
            )
            if is_door:
                all_door_hits.append((lbl, round(conf, 3)))

            for m in MACHINES:
                per_m_conf = DOOR_CONF_PER_MACHINE.get(m, DOOR_CONF)
                if conf < per_m_conf:
                    continue
                if lbl == f"machine_{m}_door_open":
                    # Take highest-confidence detection if two passes both fire
                    if raw_conf[m] is None or conf > raw_conf[m]:
                        raw_state[m] = "Open";   raw_conf[m] = conf
                elif lbl == f"machine_{m}_door_closed":
                    if raw_conf[m] is None or conf > raw_conf[m]:
                        raw_state[m] = "Closed"; raw_conf[m] = conf

    _process(results)
    if extra_results is not None:
        _process(extra_results)   # M3 crop pass — may override full-frame M3

    return raw_state, raw_conf, all_door_hits


# ══════════════════════════════ MAIN ══════════════════════════════════════════

def main():
    video_dir, save_dir = select_folders()
    if not video_dir or not save_dir:
        print("No folders selected — exiting."); return

    model      = YOLO(MODEL_PATH)
    name_to_id = {v: k for k, v in model.names.items()}

    # ── Diagnostic: print ALL class names the model knows ─────────────────
    # This immediately reveals if M3 classes are named differently
    # (e.g. "machine3_door_open" vs "machine_3_door_open")
    print("\n  Model class names:")
    for cid, cname in sorted(model.names.items()):
        print(f"    [{cid:>3}] {cname}")

    # Build expected class names using exact format "machine_{m}_door_open"
    expected_classes = (
        ["person"]
        + [f"machine_{m}_door_open"   for m in MACHINES]
        + [f"machine_{m}_door_closed" for m in MACHINES]
    )

    # Warn about any expected class missing from the model
    missing = [c for c in expected_classes if c not in name_to_id]
    if missing:
        print(f"\n  WARNING — these expected class names were NOT found in model:")
        for c in missing:
            print(f"    MISSING: '{c}'")
        print("  Check spelling above — M3 may be named differently in your model.\n")

    # Only request classes the model actually has (exact match)
    target_ids = [name_to_id[c] for c in expected_classes if c in name_to_id]
    person_cid = name_to_id.get("person")

    # v7 fix: one static, class-id-indexed palette built once — see
    # build_class_palette() docstring for why this replaces the v6
    # per-frame ColorPalette rebuild.
    CLASS_PALETTE = build_class_palette(model.names)

    zone           = sv.PolygonZone(polygon=PERSON_ROI)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=CLR_ROI, thickness=2)

    global_door_state    = {m: "Closed" for m in MACHINES}
    toggle_counts        = {m: 0        for m in MACHINES}
    cum_setup_time       = {m: 0.0      for m in MACHINES}
    cum_swap_time        = {m: 0.0      for m in MACHINES}
    unique_global_ids    = set()
    total_person_seconds = 0.0

    debouncers = {
        m: DoorDebouncer("Closed", hold_seconds=DOOR_HOLD_PER_MACHINE.get(m, DOOR_HOLD_SECONDS))
        for m in MACHINES
    }
    reid_registry = StableIDRegistry()

    video_files = sorted(
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".avi", ".3gp"))
    )
    if not video_files:
        print("No video files found."); return

    tag = session_tag(parse_start_time(video_files[0]))

    ct_writer  = AsyncDictCSVWriter(
        os.path.join(save_dir, f"Ct_vectors_{tag}.csv"), CT_COLUMNS)
    sec_writer = AsyncCSVWriter(
        os.path.join(save_dir, f"per_second_analytics_{tag}.csv"), SEC_COLS)
    tel_writer = AsyncCSVWriter(
        os.path.join(save_dir, f"tracking_telemetry_{tag}.csv"), TELEM_COLS)

    window_buf = WindowBuffer(WINDOW_SECONDS)

    try:
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            base_ts    = parse_start_time(video_file)

            cap       = cv2.VideoCapture(video_path)
            src_fps   = max(1.0, cap.get(cv2.CAP_PROP_FPS) or 25.0)
            frame_dur = 1.0 / src_fps
            frame_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            tot_frm   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            tot_sec   = int(tot_frm / src_fps)

            out = cv2.VideoWriter(
                os.path.join(save_dir, f"annotated_{video_file}"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                src_fps, (frame_w, frame_h),
            )

            keyframe_at = {round(s * src_fps): s for s in range(tot_sec)}

            print(f"\n{'─'*60}")
            print(f"🚀  {video_file}")
            print(f"    fps={src_fps:.1f}  door_conf(per-machine)={DOOR_CONF_PER_MACHINE}  "
                  f"vote={DOOR_VOTE_WINDOW}s  hold(per-machine)={DOOR_HOLD_PER_MACHINE}")

            prev_gray            = None
            second_elapsed       = 0
            frame_count          = 0
            last_kf_motion       = 0.0
            last_detections      = sv.Detections.empty()
            last_debounced_state = {m: "Closed" for m in MACHINES}
            last_raw_state       = {m: "Closed" for m in MACHINES}
            last_count_zone      = 0
            last_global_map      = {}
            last_frame_ids       = []
            last_global_ids      = []

            while True:
                success, frame = cap.read()
                if not success: break

                offset_sec       = frame_count * frame_dur
                current_frame_dt = base_ts + timedelta(seconds=offset_sec)
                iso_ts           = current_frame_dt.isoformat(timespec='milliseconds')
                unix_ms          = int(current_frame_dt.timestamp() * 1000)
                is_keyframe      = frame_count in keyframe_at
                # v7 fix: a keyframe must ALWAYS run inference in the same
                # iteration. Previously this was `frame_count % INFERENCE_STRIDE
                # == 0` alone, which desynced from is_keyframe whenever fps was
                # odd (e.g. 25fps) — silently dropping ~50% of keyframe seconds
                # from door debouncing and setup/swap time accounting. See
                # module docstring "FIX 1" for the full analysis.
                run_inference    = (frame_count % INFERENCE_STRIDE == 0) or is_keyframe

                if run_inference:
                    # Use DOOR_INFERENCE_CONF (0.20) so low-confidence M3
                    # door detections are NOT dropped before read_door_detections.
                    # DOOR_CONF (0.40) post-filter then gates what gets committed.
                    results    = model.track(
                        frame, conf=DOOR_INFERENCE_CONF, verbose=False,
                        classes=target_ids, persist=True,
                        tracker=TRACKER_CONFIG, half=USE_HALF,
                    )
                    detections = sv.Detections.from_ultralytics(results[0])

                    # Person mask — apply PERSON_CONF gate because inference
                    # ran at DOOR_INFERENCE_CONF (0.20), which is lower than
                    # PERSON_CONF (0.25). Without this, ghost persons at conf
                    # 0.20–0.24 would inflate zone counts and motion scores.
                    p_mask = (
                        (detections.class_id == person_cid) &
                        (detections.confidence >= PERSON_CONF)
                        if person_cid is not None and detections.confidence is not None
                        else (detections.class_id == person_cid)
                        if person_cid is not None
                        else np.zeros(len(detections), bool)
                    )
                    p_dets   = detections[p_mask]
                    roi_mask = zone.trigger(detections=p_dets)
                    count_zone = int(roi_mask.sum())

                    global_map = {}
                    if p_dets.tracker_id is not None and len(p_dets.tracker_id) > 0:
                        global_map = reid_registry.update(
                            frame, p_dets.tracker_id, p_dets.xyxy, roi_mask)

                    frame_tracker_ids = []
                    frame_global_ids  = []
                    person_boxes_kf   = []

                    if p_dets.tracker_id is not None:
                        for i, (in_roi, tid) in enumerate(
                            zip(roi_mask, p_dets.tracker_id)
                        ):
                            if in_roi:
                                tid_i = int(tid)
                                gid_i = global_map.get(tid_i, tid_i)
                                frame_tracker_ids.append(tid_i)
                                frame_global_ids.append(gid_i)
                                unique_global_ids.add(gid_i)
                                if p_dets.xyxy is not None:
                                    x1,y1,x2,y2 = p_dets.xyxy[i].astype(int)
                                    person_boxes_kf.append((x1,y1,x2,y2))

                    total_person_seconds += count_zone * frame_dur

                    # ── M3 crop-and-reinfer pass ───────────────────────────
                    # If M3 is very close to camera, its door fills too large
                    # a fraction of the full frame → YOLO scale mismatch →
                    # low confidence. Crop the M3 region, resize to 640×640,
                    # run a separate inference pass. This restores the scale
                    # YOLO was trained on and recovers normal confidence.
                    m3_crop_results = None
                    if M3_CROP is not None:
                        cx1, cy1, cx2, cy2 = M3_CROP
                        h_f, w_f = frame.shape[:2]
                        cx1 = max(0, cx1); cy1 = max(0, cy1)
                        cx2 = min(w_f, cx2); cy2 = min(h_f, cy2)
                        if cx2 > cx1 and cy2 > cy1:
                            crop = cv2.resize(
                                frame[cy1:cy2, cx1:cx2], (640, 640))
                            m3_crop_r = model(
                                crop, conf=DOOR_INFERENCE_CONF,
                                verbose=False,
                                classes=[name_to_id[c] for c in [
                                    f"machine_3_door_open",
                                    f"machine_3_door_closed",
                                ] if c in name_to_id],
                                half=USE_HALF,
                            )
                            if m3_crop_r:
                                m3_crop_results = m3_crop_r[0]

                    raw_door, raw_conf, all_door_hits = read_door_detections(
                        results[0], model.names,
                        extra_results=m3_crop_results)

                    if is_keyframe:
                        if DIAG_MODE:
                            kf_ts = base_ts + timedelta(seconds=second_elapsed)
                            print(f"  [KF {kf_ts.strftime('%H:%M:%S')}] "
                                  f"raw={raw_door}  conf={raw_conf}")
                            if all_door_hits:
                                for lbl, c in sorted(all_door_hits,
                                                     key=lambda x: x[0]):
                                    # find which machine this hit belongs to
                                    m_hit = next(
                                        (m for m in MACHINES
                                         if lbl == f"machine_{m}_door_open"
                                         or lbl == f"machine_{m}_door_closed"),
                                        None)
                                    thr = DOOR_CONF_PER_MACHINE.get(
                                        m_hit, DOOR_CONF) if m_hit else DOOR_CONF
                                    gate = "PASS" if c >= thr else f"SKIP(need>={thr})"
                                    print(f"    {lbl:42s} conf={c:.3f}  {gate}")
                            else:
                                print(f"    NO door detections "
                                      f"(inference_conf={DOOR_INFERENCE_CONF})"
                                      + ("  M3_CROP active" if M3_CROP else
                                         "  set M3_CROP to enable crop pass"))

                        debounced_state = {}
                        for m in MACHINES:
                            committed, changed = debouncers[m].observe(raw_door[m])
                            debounced_state[m] = committed
                            if changed:
                                toggle_counts[m]    += 1
                                global_door_state[m] = committed
                            if global_door_state[m] == "Closed":
                                cum_setup_time[m] += 1.0
                            else:
                                cum_swap_time[m]  += 1.0
                    else:
                        debounced_state = {m: debouncers[m].state for m in MACHINES}

                    last_detections      = detections
                    last_debounced_state = debounced_state.copy()
                    last_raw_state       = {
                        m: (raw_door[m] if raw_door[m] else last_raw_state.get(m, "Closed"))
                        for m in MACHINES
                    }
                    last_count_zone  = count_zone
                    last_global_map  = global_map.copy()
                    last_frame_ids   = frame_tracker_ids.copy()
                    last_global_ids  = frame_global_ids.copy()

                else:
                    detections        = last_detections
                    debounced_state   = last_debounced_state
                    count_zone        = last_count_zone
                    global_map        = last_global_map
                    frame_tracker_ids = last_frame_ids
                    frame_global_ids  = last_global_ids
                    person_boxes_kf   = []
                    total_person_seconds += count_zone * frame_dur

                tel_writer.write([
                    iso_ts, unix_ms,
                    debounced_state[1], debounced_state[2], debounced_state[3],
                    last_raw_state[1],  last_raw_state[2],  last_raw_state[3],
                    count_zone, frame_tracker_ids, frame_global_ids,
                ])

                if is_keyframe:
                    current_ts = base_ts + timedelta(seconds=second_elapsed)
                    door_binary_kf = {
                        m: (1 if global_door_state[m] == "Open" else 0)
                        for m in MACHINES
                    }

                    gray_kf        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    last_kf_motion = 0.0
                    if prev_gray is not None and person_boxes_kf:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray, gray_kf, None, **FARNEBACK_PARAMS)
                        last_kf_motion = compute_frame_motion(
                            flow[..., 0], flow[..., 1], person_boxes_kf)
                    prev_gray = gray_kf

                    sec_writer.write([
                        current_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        *[v for m in MACHINES for v in [
                            global_door_state[m],
                            toggle_counts[m],
                            round(cum_setup_time[m], 2),
                            round(cum_swap_time[m],  2),
                            last_raw_state[m],
                        ]],
                        count_zone,
                        len(unique_global_ids),
                        round(last_kf_motion, 6),
                    ])

                    window_buf.push(
                        ts          = current_ts,
                        door_binary = door_binary_kf,
                        motion      = last_kf_motion,
                        n_person    = count_zone,
                    )

                    if window_buf.is_full():
                        ct_row = window_buf.flush()
                        if ct_row:
                            ct_writer.write(ct_row)
                            ws = ct_row["window_start"]
                            print(
                                f"  [C_t] {ws}  "
                                f"M1 τ={ct_row['M1_tau_open']:>2}s "
                                f"f={ct_row['M1_f_trans']:>2}  "
                                f"M2 τ={ct_row['M2_tau_open']:>2}s "
                                f"f={ct_row['M2_f_trans']:>2}  "
                                f"μ={ct_row['mu_motion']:.4f}  "
                                f"n={ct_row['n_person']}"
                            )

                    second_elapsed += 1

                # ── ANNOTATION ──────────────────────────────────────────────
                # v7 fix: reuse the static, class-id-indexed CLASS_PALETTE
                # built once above instead of rebuilding a per-frame,
                # detection-order palette (see build_class_palette() docstring).
                colours = CLASS_PALETTE

                labels = []
                if detections.tracker_id is not None:
                    for cid, tid in zip(detections.class_id, detections.tracker_id):
                        cname = model.names[int(cid)]
                        gid   = (global_map.get(int(tid), int(tid))
                                 if cname == "person" and tid is not None else None)
                        labels.append(make_label(cname, gid))
                else:
                    for cid in detections.class_id:
                        labels.append(model.names[int(cid)])

                annotated = zone_annotator.annotate(scene=frame.copy())
                box_ann   = sv.BoxAnnotator(color=colours)
                annotated = box_ann.annotate(scene=annotated, detections=detections)
                lbl_ann   = sv.LabelAnnotator(
                    color=colours, text_color=sv.Color.BLACK,
                    text_scale=0.52, text_thickness=1, text_padding=4)
                annotated = lbl_ann.annotate(
                    scene=annotated, detections=detections, labels=labels)

                hud1 = f"{iso_ts}  Zone:{count_zone}p  GIDs:{frame_global_ids}"
                hud2 = (f"DOOR debounced → "
                        f"M1:{debounced_state[1]}  "
                        f"M2:{debounced_state[2]}  "
                        f"M3:{debounced_state[3]}")
                hud3 = (f"DOOR raw (YOLO) → "
                        f"M1:{last_raw_state[1]}  "
                        f"M2:{last_raw_state[2]}  "
                        f"M3:{last_raw_state[3]}")

                for li, (hud, clr) in enumerate([
                    (hud1, (220, 235, 220)),
                    (hud2, (120, 220, 255)),
                    (hud3, (180, 180, 100)),
                ]):
                    (tw, th), _ = cv2.getTextSize(
                        hud, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
                    y_base = 10 + li * 24
                    ov = annotated.copy()
                    cv2.rectangle(ov, (4, y_base - 2),
                                  (min(tw+12, frame_w-4), y_base+th+4),
                                  (0,0,0), -1)
                    cv2.addWeighted(ov, 0.45, annotated, 0.55, 0, annotated)
                    cv2.putText(annotated, hud, (8, y_base+th),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, clr, 1, cv2.LINE_AA)

                if is_keyframe:
                    cv2.rectangle(annotated, (0,0), (frame_w, 3), (80,230,80), -1)

                out.write(annotated)
                frame_count += 1

            cap.release()
            out.release()
            print(f"  ✅ {video_file}  ({second_elapsed}kf / {frame_count}fr)")

        # Flush partial final window
        if window_buf.count > 0:
            ct_row = window_buf.flush()
            if ct_row:
                ct_writer.write(ct_row)

    finally:
        ct_writer.close()
        sec_writer.close()
        tel_writer.close()

    person_hours = total_person_seconds / 3600
    print(f"\n{'='*55}")
    print(f"📊  TRACKING & OCCUPANCY SUMMARY  v7")
    print(f"{'='*55}")
    print(f"  Session              : {tag}")
    print(f"  Unique people (ReID) : {len(unique_global_ids)}")
    print(f"  Person-hours in zone : {person_hours:.6f}")
    print(f"  Global IDs seen      : {sorted(unique_global_ids)}")
    print(f"  Door debounce        : vote={DOOR_VOTE_WINDOW}s  "
          f"hold(per-machine)={DOOR_HOLD_PER_MACHINE}  "
          f"conf(per-machine)={DOOR_CONF_PER_MACHINE}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
