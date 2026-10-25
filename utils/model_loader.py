"""Eye-state models (.tflite) and their saved evaluation results."""
import json
import os
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models", "eye_state")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "assets", "results")

BASELINE = "EAR rule (no model)"
MODEL_FILES = {
    "MobileNetV2": "mobilenetv2.tflite",
    "Custom CNN": "custom_cnn.tflite",
    "EfficientNetB0": "efficientnetb0.tflite",
}
RESULT_FILES = {
    "MobileNetV2": "mobilenetv2_results.json",
    "Custom CNN": "custom_cnn_results.json",
    "EfficientNetB0": "efficientnetb0_results.json",
}
MODEL_ORDER = ["Custom CNN", "MobileNetV2", "EfficientNetB0"]
DESCRIPTIONS = {
    "Custom CNN": "Small residual CNN trained from random weights",
    "MobileNetV2": "ImageNet pre-trained, fine-tuned",
    "EfficientNetB0": "ImageNet pre-trained, fine-tuned",
    BASELINE: "Geometric eye aspect ratio (EAR) threshold, no trained model",
}


def model_path(name: str) -> Optional[str]:
    f = MODEL_FILES.get(name)
    if not f:
        return None
    p = os.path.join(MODELS_DIR, f)
    return p if os.path.exists(p) else None


def available_models() -> Dict[str, bool]:
    """name -> True if the .tflite file is present. The baseline is always available."""
    out = {n: model_path(n) is not None for n in MODEL_ORDER}
    out[BASELINE] = True
    return out


def load_results() -> Dict[str, dict]:
    res = {}
    for name, f in RESULT_FILES.items():
        p = os.path.join(RESULTS_DIR, f)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fh:
                res[name] = json.load(fh)
    return res


def suggested_model() -> Optional[str]:
    p = os.path.join(RESULTS_DIR, "summary.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh).get("suggested_model")
    return None


def default_model() -> str:
    """The model chosen by the comparison, if its file is present; otherwise the first available; else baseline."""
    avail = available_models()
    s = suggested_model()
    if s and avail.get(s):
        return s
    for n in MODEL_ORDER:
        if avail.get(n):
            return n
    return BASELINE


def model_threshold(name: str) -> float:
    r = load_results().get(name)
    return float(r["threshold"]) if r and "threshold" in r else 0.5


def build_classifier(name: str):
    """Returns a TFLiteEyeClassifier, or None for the baseline / a missing file."""
    p = model_path(name)
    if p is None:
        return None
    from utils.drowsiness_features import TFLiteEyeClassifier
    return TFLiteEyeClassifier(p, threshold=model_threshold(name))
