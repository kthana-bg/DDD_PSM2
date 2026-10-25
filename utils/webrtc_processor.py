"""streamlit-webrtc video processor wrapping DriverMonitor (runs in the WebRTC worker thread)."""
import threading
import time

import av
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from utils.landmarks import create_provider
from utils.model_loader import BASELINE, build_classifier
from utils.monitor import DriverMonitor, FrameResult


class DrowsinessProcessor(VideoProcessorBase):
    FRAME_SKIP = 2

    def __init__(self):
        self._lock = threading.Lock()
        self._result = FrameResult()
        self._events = []
        self._monitor = None
        self._error = None
        self._n = 0
        self._cfg = dict(model=BASELINE, sensitivity="Normal", recal=0)
        self._applied = dict(model=None, sensitivity=None, recal=0)

    def configure(self, model: str, sensitivity: str, recal: int):
        with self._lock:
            self._cfg = dict(model=model, sensitivity=sensitivity, recal=recal)

    def _ensure_monitor(self):
        if self._monitor is None:
            try:
                self._monitor = DriverMonitor(create_provider())
            except Exception as e:
                self._error = str(e)
                return None
        with self._lock:
            cfg = dict(self._cfg)
        a = self._applied
        if cfg["model"] != a["model"]:
            try:
                clf = build_classifier(cfg["model"])
            except Exception as e:
                clf, self._error = None, f"Could not load {cfg['model']}: {e}"
            self._monitor.set_detector(cfg["model"] if clf is not None else BASELINE, clf)
            a["model"] = cfg["model"]
        if cfg["sensitivity"] != a["sensitivity"]:
            self._monitor.set_sensitivity(cfg["sensitivity"])
            a["sensitivity"] = cfg["sensitivity"]
        if cfg["recal"] != a["recal"]:
            self._monitor.recalibrate()
            a["recal"] = cfg["recal"]
        return self._monitor

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        img = frame.to_ndarray(format="bgr24")
        mon = self._ensure_monitor()
        if mon is None:
            return frame
        self._n += 1
        if self._n % self.FRAME_SKIP == 0:
            res, out = mon.process(img, time.time())
            ev = mon.pop_events()
            with self._lock:
                self._result = res
                self._events.extend(ev)
        else:
            out = mon.draw_last(img)
        return av.VideoFrame.from_ndarray(out, format="bgr24")

    def get_result(self) -> FrameResult:
        with self._lock:
            return self._result

    def pop_events(self) -> list:
        with self._lock:
            ev, self._events = self._events, []
        return ev

    @property
    def error(self):
        return self._error
