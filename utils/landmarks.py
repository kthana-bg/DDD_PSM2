"""Face landmark provider (MediaPipe Face Landmarker, Tasks API)."""
import os
import threading
import urllib.request
from typing import Optional

import cv2
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
FACE_MODEL_PATH = os.path.join(ASSETS_DIR, "face_landmarker.task")
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")


class LandmarkError(RuntimeError):
    pass


def ensure_face_model(path: str = FACE_MODEL_PATH, url: str = FACE_MODEL_URL) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        raise LandmarkError(
            f"Could not download the MediaPipe face model ({e}). Download face_landmarker.task from\n{url}\n"
            f"and place it in {os.path.dirname(path)}/"
        ) from e
    return path


class MediaPipeFaceProvider:
    """detect(frame_bgr) -> (N, 2) array of landmark pixel coordinates, or None when no face is found."""

    def __init__(self, model_path: Optional[str] = None):
        import mediapipe as mp
        path = ensure_face_model(model_path or FACE_MODEL_PATH)
        vision = mp.tasks.vision
        opts = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp = mp
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._lock = threading.Lock()

    def detect(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            result = self._lm.detect(image)
        if not result.face_landmarks:
            return None
        pts = result.face_landmarks[0]
        return np.array([(p.x * w, p.y * h) for p in pts], dtype=np.float64)


_PROVIDER_FACTORY = MediaPipeFaceProvider


def create_provider():
    """Single place to build the provider (tests replace _PROVIDER_FACTORY)."""
    return _PROVIDER_FACTORY()
