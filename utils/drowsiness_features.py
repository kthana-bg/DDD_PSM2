"""
Drowsiness indicators and fusion for the Driver Drowsiness Detection System.

Replaces the eye-strain / posture logic of VisionMate. Pure NumPy/OpenCV so it is unit-testable
without a camera. MediaPipe Face Landmarker supplies the 468 landmarks (same as before).

Indicators
  EAR      eye aspect ratio                      -> geometric eye openness
  PERCLOS  % of time eyes are closed (window)    -> fed by EAR *or* by the trained CNN (P(closed))
  MAR      mouth aspect ratio                    -> yawning
  pitch    head-pose angle (solvePnP)            -> nodding / head drooping
"""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2

# ---- MediaPipe Face Mesh indices ------------------------------------------------------------
LEFT_EYE  = [362, 385, 387, 263, 373, 380]     # p1..p6 (same order as the original eye_detection.py)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_H   = (78, 308)                          # inner mouth corners
MOUTH_V   = ((82, 87), (13, 14), (312, 317))   # three inner-lip vertical pairs
POSE_IDX  = dict(nose=1, chin=152, r_eye=33, l_eye=263, r_mouth=61, l_mouth=291)

# Generic 3D face model (mm), camera convention: x right, y down, z away from the camera, origin at the nose tip.
# With this convention a frontal face gives pitch ~ 0 and a nod (head down) gives a positive pitch.
_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),            # nose tip
    (0.0, 63.6, 12.5),          # chin
    (-43.3, -32.7, 26.0),       # right eye outer corner (appears on the image left)
    (43.3, -32.7, 26.0),        # left eye outer corner
    (-28.9, 28.9, 24.1),        # right mouth corner
    (28.9, 28.9, 24.1),         # left mouth corner
], dtype=np.float64)


def landmarks_to_px(face_landmarks, w: int, h: int) -> np.ndarray:
    """MediaPipe landmark list (new Tasks API list, or legacy object with .landmark) -> (468,2) pixel array."""
    lms = getattr(face_landmarks, "landmark", face_landmarks)
    return np.array([(l.x * w, l.y * h) for l in lms], dtype=np.float64)


def _d(a, b): return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts6: np.ndarray) -> float:
    c = _d(pts6[0], pts6[3])
    return 0.0 if c == 0 else (_d(pts6[1], pts6[5]) + _d(pts6[2], pts6[4])) / (2.0 * c)


def mean_ear(lm_px: np.ndarray) -> float:
    return 0.5 * (eye_aspect_ratio(lm_px[LEFT_EYE]) + eye_aspect_ratio(lm_px[RIGHT_EYE]))


def mouth_aspect_ratio(lm_px: np.ndarray) -> float:
    h = _d(lm_px[MOUTH_H[0]], lm_px[MOUTH_H[1]])
    return 0.0 if h == 0 else sum(_d(lm_px[a], lm_px[b]) for a, b in MOUTH_V) / (3.0 * h)


def estimate_head_pose(lm_px: np.ndarray, w: int, h: int):
    """Return (pitch, yaw, roll) in degrees. Pitch > 0 = head tilted down (nodding)."""
    img = np.array([lm_px[POSE_IDX[k]] for k in ("nose", "chin", "r_eye", "l_eye", "r_mouth", "l_mouth")], dtype=np.float64)
    f = float(max(w, h))
    cam = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, img, cam, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = np.hypot(R[0, 0], R[1, 0])
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2])); yaw = np.degrees(np.arctan2(-R[2, 0], sy)); roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    wrap = lambda a: float((a + 180.0) % 360.0 - 180.0)
    return wrap(pitch), wrap(yaw), wrap(roll)


# ---- eye crop for the CNN ---------------------------------------------------------------------
def crop_eye(frame_bgr: np.ndarray, eye_pts6: np.ndarray, pad: float = 0.45) -> Optional[np.ndarray]:
    """Crop the eye with padding proportional to eye width (wider context than the landmark box)."""
    x0, y0 = eye_pts6.min(0); x1, y1 = eye_pts6.max(0)
    w, h = x1 - x0, y1 - y0
    x0, x1, y0, y1 = int(x0 - pad * w), int(x1 + pad * w), int(y0 - pad * w), int(y1 + pad * w)
    H, W = frame_bgr.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 8 or y1 - y0 < 8: return None
    return frame_bgr[y0:y1, x0:x1]


def prepare_eye_input(crop_bgr: np.ndarray, size: int = 96) -> np.ndarray:
    """Match the training pipeline: grayscale (MRL is near-infrared grey) -> 3 channels -> squash-resize
    -> float32 in 0..255, shape (1,size,size,3). The tflite models contain their own scaling."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return np.repeat(g[None, :, :, None], 3, axis=3).astype(np.float32)


def _load_interpreter_class():
    for mod, attr in (("ai_edge_litert.interpreter", "Interpreter"), ("tflite_runtime.interpreter", "Interpreter")):
        try:
            return getattr(__import__(mod, fromlist=[attr]), attr)
        except Exception:
            continue
    import tensorflow as tf
    return tf.lite.Interpreter


class TFLiteEyeClassifier:
    """Loads one of the exported .tflite models; returns P(closed) and latency."""
    def __init__(self, tflite_path: str, threshold: float = 0.5, size: int = 96, threads: int = 2):
        Interpreter = _load_interpreter_class()
        self.it = Interpreter(model_path=tflite_path, num_threads=threads); self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]["index"]; self.o = self.it.get_output_details()[0]["index"]
        self.threshold, self.size = threshold, size

    def predict(self, crop_bgr: np.ndarray):
        t0 = time.perf_counter()
        self.it.set_tensor(self.i, prepare_eye_input(crop_bgr, self.size)); self.it.invoke()
        p = float(self.it.get_tensor(self.o).ravel()[0])
        return p, (time.perf_counter() - t0) * 1000.0

    def predict_eyes(self, crops):
        """Mean P(closed) over the available eye crops (both eyes), and total latency."""
        res = [self.predict(c) for c in crops if c is not None]
        if not res: return None, 0.0
        return float(np.mean([p for p, _ in res])), float(sum(t for _, t in res))


# ---- temporal indicators ----------------------------------------------------------------------
class PerclosTracker:
    """PERCLOS = fraction of the last `window_s` seconds the eyes were closed (time-weighted)."""
    def __init__(self, window_s: float = 30.0, max_gap_s: float = 1.0):
        self.window_s, self.max_gap_s, self.buf = window_s, max_gap_s, deque()   # (t, closed)
        self.closed_since: Optional[float] = None

    def update(self, closed: bool, t: Optional[float] = None) -> float:
        t = time.time() if t is None else t
        self.buf.append((t, bool(closed)))
        while self.buf and t - self.buf[0][0] > self.window_s: self.buf.popleft()
        self.closed_since = (self.closed_since if self.closed_since is not None else t) if closed else None
        return self.perclos

    @property
    def perclos(self) -> float:
        if len(self.buf) < 2: return 0.0
        tot = closed = 0.0
        for (t0, c), (t1, _) in zip(self.buf, list(self.buf)[1:]):
            dt = min(t1 - t0, self.max_gap_s); tot += dt; closed += dt if c else 0.0
        return closed / max(tot, 0.5 * self.window_s) if tot > 0 else 0.0   # a short history must not inflate PERCLOS

    def closure_duration(self, t: Optional[float] = None) -> float:
        t = time.time() if t is None else t
        return 0.0 if self.closed_since is None else t - self.closed_since


class HoldTrigger:
    """Fires one event when `active` holds continuously for `min_s` (yawn: MAR high; nod: pitch low)."""
    def __init__(self, min_s: float, memory_s: float = 300.0):
        self.min_s, self.memory_s, self.t_on, self.fired, self.events = min_s, memory_s, None, False, deque()

    def update(self, active: bool, t: Optional[float] = None) -> bool:
        t = time.time() if t is None else t
        while self.events and t - self.events[0] > self.memory_s: self.events.popleft()
        if not active: self.t_on, self.fired = None, False; return False
        self.t_on = t if self.t_on is None else self.t_on
        if not self.fired and t - self.t_on >= self.min_s:
            self.fired = True; self.events.append(t); return True
        return False

    @property
    def count(self) -> int: return len(self.events)
    @property
    def holding(self) -> bool: return self.t_on is not None


@dataclass
class Thresholds:
    ear_closed: float = 0.20      # EAR below this = closed (replaced by per-driver calibration)
    cnn_closed: float = 0.50      # P(closed) threshold (val-tuned value from the results JSON)
    mar_yawn: float = 0.60
    pitch_nod: float = 15.0       # degrees below the driver's neutral pitch
    yawn_hold_s: float = 1.2
    nod_hold_s: float = 1.0
    perclos_alert: float = 0.15   # PERCLOS at which the eye component saturates (0.15-0.25 in the literature)
    microsleep_s: float = 1.5     # continuous closure treated as danger immediately
    warn_score: float = 30.0
    danger_score: float = 60.0
    pitch_neutral: float = 0.0


def calibrate(ear_open_samples, mar_closed_samples, pitch_samples, base: Thresholds = Thresholds()) -> Thresholds:
    """Per-driver thresholds from ~10 s of calm, eyes-open, mouth-closed, looking-ahead frames."""
    t = Thresholds(**vars(base))
    t.ear_closed = float(np.clip(0.70 * np.median(ear_open_samples), 0.12, 0.25))
    t.mar_yawn = float(max(0.45, np.median(mar_closed_samples) + 0.35))
    t.pitch_neutral = float(np.median(pitch_samples))
    return t


@dataclass
class DrowsinessState:
    score: float = 0.0
    level: str = "Alert"                 # Alert | Warning | Danger
    perclos: float = 0.0
    closure_s: float = 0.0
    yawns: int = 0
    nods: int = 0
    reasons: list = field(default_factory=list)


class DrowsinessFusion:
    """Transparent rule-based fusion of the four indicators -> score 0-100 and a graded alert level.
    Weights are design choices (eye 0.5, yawn 0.25, head 0.25); tune them on the validation videos."""
    def __init__(self, th: Thresholds = Thresholds(), window_s: float = 30.0):
        self.th, self.perclos = th, PerclosTracker(window_s)
        self.yawn, self.nod = HoldTrigger(th.yawn_hold_s), HoldTrigger(th.nod_hold_s)

    def set_thresholds(self, th: Thresholds):
        self.th = th; self.yawn.min_s = th.yawn_hold_s; self.nod.min_s = th.nod_hold_s

    def lost_face(self):
        """Face not visible: stop the running closure / yawn / nod timers (counts are kept)."""
        self.perclos.closed_since = None; self.yawn.t_on = self.nod.t_on = None
        self.yawn.fired = self.nod.fired = False

    def update(self, eye_closed: bool, mar: float, pitch: float, t: Optional[float] = None) -> DrowsinessState:
        t = time.time() if t is None else t; th = self.th
        perclos = self.perclos.update(eye_closed, t); closure = self.perclos.closure_duration(t)
        self.yawn.update(mar > th.mar_yawn, t); self.nod.update((pitch - th.pitch_neutral) > th.pitch_nod, t)
        eye = min(1.0, max(perclos / th.perclos_alert, closure / th.microsleep_s))
        yaw = 1.0 if self.yawn.holding and (t - self.yawn.t_on) >= th.yawn_hold_s else min(1.0, self.yawn.count / 2)
        head = 1.0 if self.nod.holding and (t - self.nod.t_on) >= th.nod_hold_s else min(1.0, self.nod.count / 2)
        score = 100.0 * (0.50 * eye + 0.25 * yaw + 0.25 * head)
        reasons = [r for r, on in (("eye closure (PERCLOS)", eye >= 0.6), ("yawning", yaw >= 0.5), ("head nodding", head >= 0.5)) if on]
        level = "Danger" if (score >= th.danger_score or closure >= th.microsleep_s) else "Warning" if score >= th.warn_score else "Alert"
        if closure >= th.microsleep_s: reasons.append("microsleep")
        return DrowsinessState(score, level, perclos, closure, self.yawn.count, self.nod.count, reasons)


SENSITIVITY = {
    "Low (fewer alerts)":     dict(perclos_alert=0.25, microsleep_s=2.0, warn_score=40.0, danger_score=70.0),
    "Normal":                 dict(perclos_alert=0.15, microsleep_s=1.5, warn_score=30.0, danger_score=60.0),
    "High (earlier alerts)":  dict(perclos_alert=0.10, microsleep_s=1.0, warn_score=25.0, danger_score=50.0),
}


def with_sensitivity(th: Thresholds, name: str) -> Thresholds:
    t = Thresholds(**vars(th))
    for k, v in SENSITIVITY.get(name, SENSITIVITY["Normal"]).items(): setattr(t, k, v)
    return t
