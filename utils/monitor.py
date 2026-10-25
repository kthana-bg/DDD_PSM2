"""Per-frame drowsiness pipeline: landmarks -> indicators -> fusion -> annotated frame."""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from utils.drowsiness_features import (
    LEFT_EYE, RIGHT_EYE, MOUTH_H, MOUTH_V, Thresholds, DrowsinessFusion, calibrate, crop_eye,
    estimate_head_pose, mean_ear, mouth_aspect_ratio, with_sensitivity,
)
from utils.model_loader import BASELINE

LEVEL_RANK = {"Alert": 0, "Warning": 1, "Danger": 2}
LEVEL_BGR = {"Alert": (80, 200, 60), "Warning": (30, 160, 245), "Danger": (60, 60, 235)}


@dataclass
class FrameResult:
    face_detected: bool = False
    status: str = "Starting"            # Starting | Calibrating | Monitoring | No face
    level: str = "Alert"                # Alert | Warning | Danger
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    eye_closed: bool = False
    p_closed: Optional[float] = None
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    perclos: float = 0.0
    closure_s: float = 0.0
    yawns: int = 0
    nods: int = 0
    calibrating: bool = False
    calib_progress: float = 0.0
    face_lost_s: float = 0.0
    model_latency_ms: float = 0.0
    proc_ms: float = 0.0
    fps: float = 0.0
    detector: str = BASELINE
    thresholds: Optional[Thresholds] = None
    timestamp: float = field(default_factory=time.time)


class DriverMonitor:
    FACE_LOST_WARN_S = 5.0

    def __init__(self, provider, detector: str = BASELINE, classifier=None, sensitivity: str = "Normal",
                 calib_seconds: float = 8.0, window_s: float = 30.0):
        self.provider, self.detector, self.classifier = provider, detector, classifier
        self.sensitivity, self.calib_seconds, self.window_s = sensitivity, calib_seconds, window_s
        self.events = deque()
        self.level_time = {"Alert": 0.0, "Warning": 0.0, "Danger": 0.0}
        self.max_score = 0.0
        self.max_closure_s = 0.0
        self._fps_ema = None
        self._last_lm = None
        self._last_result = FrameResult(detector=detector)
        self.recalibrate()

    def recalibrate(self):
        self._calibrating = True
        self._calib_elapsed = 0.0
        self._samples = dict(ear=[], mar=[], pitch=[])
        self._base_th = Thresholds()
        self._th = with_sensitivity(self._base_th, self.sensitivity)
        self.fusion = DrowsinessFusion(self._th, self.window_s)
        self._level = "Alert"
        self._last_t = None
        self._lost_since = None

    def set_detector(self, name: str, classifier):
        self.detector, self.classifier = name, classifier

    def set_sensitivity(self, name: str):
        if name == self.sensitivity:
            return
        self.sensitivity = name
        self._th = with_sensitivity(self._base_th, name)
        self.fusion.set_thresholds(self._th)

    def pop_events(self) -> list:
        out = []
        while self.events:
            out.append(self.events.popleft())
        return out

    def _bump_level(self, level, reasons, score, perclos, t):
        if LEVEL_RANK[level] > LEVEL_RANK[self._level]:
            self.events.append(dict(t=t, level=level, reasons=list(reasons), score=float(score), perclos=float(perclos)))
        self._level = level

    def process(self, frame_bgr: np.ndarray, t: Optional[float] = None, annotate: bool = True):
        """Returns (FrameResult, annotated_frame_bgr)."""
        t = time.time() if t is None else t
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), 1.0)
        self._last_t = t
        res = FrameResult(detector=self.detector, thresholds=self._th)

        lm = self.provider.detect(frame_bgr)
        self._last_lm = lm
        if lm is None:
            self._on_no_face(res, t)
        else:
            self._lost_since = None
            res.face_detected = True
            res.ear = mean_ear(lm)
            res.mar = mouth_aspect_ratio(lm)
            res.pitch, res.yaw, res.roll = estimate_head_pose(lm, w, h)
            if self._calibrating:
                self._calibrate_step(res, dt)
            else:
                self._monitor_step(frame_bgr, lm, res, t, dt)

        res.proc_ms = (time.perf_counter() - t0) * 1000.0
        inst = 1000.0 / max(res.proc_ms, 1e-3)
        self._fps_ema = inst if self._fps_ema is None else 0.9 * self._fps_ema + 0.1 * inst
        res.fps = self._fps_ema
        self._last_result = res
        out = self.annotate(frame_bgr, lm, res) if annotate else frame_bgr
        return res, out

    def _calibrate_step(self, res, dt):
        self._samples["ear"].append(res.ear)
        self._samples["mar"].append(res.mar)
        self._samples["pitch"].append(res.pitch)
        self._calib_elapsed += dt
        res.calibrating, res.status = True, "Calibrating"
        res.calib_progress = min(1.0, self._calib_elapsed / self.calib_seconds)
        if self._calib_elapsed >= self.calib_seconds and len(self._samples["ear"]) >= 8:
            self._base_th = calibrate(self._samples["ear"], self._samples["mar"], self._samples["pitch"])
            self._th = with_sensitivity(self._base_th, self.sensitivity)
            self.fusion = DrowsinessFusion(self._th, self.window_s)
            self._calibrating = False
            res.calibrating, res.calib_progress, res.status = False, 1.0, "Monitoring"
        res.thresholds = self._th

    def _monitor_step(self, frame, lm, res, t, dt):
        th = self._th
        closed = None
        if self.classifier is not None:
            crops = [crop_eye(frame, lm[LEFT_EYE]), crop_eye(frame, lm[RIGHT_EYE])]
            p, lat = self.classifier.predict_eyes(crops)
            if p is not None:
                res.p_closed, res.model_latency_ms = p, lat
                closed = p >= self.classifier.threshold
        if closed is None:
            closed = res.ear < th.ear_closed
        res.eye_closed = bool(closed)
        st = self.fusion.update(res.eye_closed, res.mar, res.pitch, t)
        res.status, res.level, res.score, res.reasons = "Monitoring", st.level, st.score, list(st.reasons)
        res.perclos, res.closure_s, res.yawns, res.nods = st.perclos, st.closure_s, st.yawns, st.nods
        self._bump_level(st.level, st.reasons, st.score, st.perclos, t)
        self.level_time[st.level] += dt
        self.max_score = max(self.max_score, st.score)
        self.max_closure_s = max(self.max_closure_s, st.closure_s)

    def _on_no_face(self, res, t):
        self._lost_since = self._lost_since if self._lost_since is not None else t
        lost = t - self._lost_since
        res.status, res.face_lost_s = "No face", lost
        if self._calibrating:
            return
        self.fusion.lost_face()
        if lost >= self.FACE_LOST_WARN_S:
            res.level, res.score, res.reasons = "Warning", self._th.warn_score, ["driver not visible"]
            self._bump_level("Warning", res.reasons, res.score, 0.0, t)
        else:
            self._level = "Alert" if self._level == "Alert" else self._level

    def annotate(self, frame, lm, res: FrameResult):
        out = frame.copy()
        h, w = out.shape[:2]
        color = LEVEL_BGR.get(res.level, (80, 200, 60))
        if lm is not None:
            eye_col = (60, 60, 235) if res.eye_closed else (80, 220, 80)
            for idx in (LEFT_EYE, RIGHT_EYE):
                cv2.polylines(out, [lm[idx].astype(np.int32)], True, eye_col, 2, cv2.LINE_AA)
            th = res.thresholds
            yawning = th is not None and res.mar > th.mar_yawn and not res.calibrating
            mouth = [MOUTH_H[0], MOUTH_V[0][0], MOUTH_V[1][0], MOUTH_V[2][0], MOUTH_H[1], MOUTH_V[2][1], MOUTH_V[1][1], MOUTH_V[0][1]]
            cv2.polylines(out, [lm[mouth].astype(np.int32)], True, (30, 160, 245) if yawning else (200, 200, 200), 2, cv2.LINE_AA)
        if res.status == "Calibrating":
            text, tcol = f"Calibrating {int(res.calib_progress * 100)}%  look ahead, eyes open", (230, 160, 60)
        elif res.status == "No face":
            text, tcol = "No face detected", (170, 170, 170)
        elif res.status == "Monitoring":
            text, tcol = f"{res.level.upper()}  {res.score:.0f}", color
        else:
            text, tcol = "Starting...", (200, 200, 200)
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), tcol if res.status != "Monitoring" or res.level != "Alert" else (60, 60, 60), 6 if res.level == "Danger" else 2)
        cv2.rectangle(out, (0, 0), (min(w, 12 + 14 * len(text)), 34), (20, 20, 20), -1)
        cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tcol, 2, cv2.LINE_AA)
        return out

    def draw_last(self, frame):
        """Overlay the most recent result on a frame that was not processed (frame skipping)."""
        return self.annotate(frame, self._last_lm, self._last_result)

    @property
    def last_result(self) -> FrameResult:
        return self._last_result
