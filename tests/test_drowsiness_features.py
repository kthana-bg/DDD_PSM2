import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.drowsiness_features import *
from utils import drowsiness_features as F


def _eye(open_ratio):
    # p1..p6 around an eye of width 30 px
    h = 15 * open_ratio
    return np.array([(0, 0), (10, -h), (20, -h), (30, 0), (20, h), (10, h)], float)

def test_ear_open_vs_closed():
    assert eye_aspect_ratio(_eye(0.5)) > 0.25 > eye_aspect_ratio(_eye(0.05))

def test_mar_yawn():
    lm = np.zeros((468, 2)); lm[78], lm[308] = (0, 0), (40, 0)
    for (a, b), gap in zip(F.MOUTH_V, (30, 30, 30)): lm[a], lm[b] = (20, -gap / 2), (20, gap / 2)
    assert mouth_aspect_ratio(lm) == 0.75
    for (a, b) in F.MOUTH_V: lm[a], lm[b] = (20, -1), (20, 1)
    assert mouth_aspect_ratio(lm) < 0.1

def test_head_pose_roundtrip():
    w, h = 640, 480; cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1.0]])
    for pitch, yaw in ((-20, 0), (0, 0), (25, 0), (10, 15), (-10, -20)):
        rvec = np.radians([pitch, yaw, 0]).reshape(3, 1)
        pts, _ = cv2.projectPoints(F._MODEL_3D, rvec, np.array([[0], [0], [450.0]]), cam, np.zeros(4))
        lm = np.zeros((468, 2))
        for k, p in zip(("nose", "chin", "r_eye", "l_eye", "r_mouth", "l_mouth"), pts.reshape(-1, 2)): lm[F.POSE_IDX[k]] = p
        got_p, got_y, _ = estimate_head_pose(lm, w, h)
        assert abs(got_p - pitch) < 3.0 and abs(got_y - yaw) < 3.0, (pitch, yaw, got_p, got_y)

def test_nod_down_is_positive_pitch():
    # physically nod down: rotate the head about its centre (behind the nose) so the nose moves down in the image
    w, h = 640, 480; cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1.0]])
    centre = np.array([0, 0, 90.0])
    def project(deg):
        a = np.radians(deg); R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        pts3 = (F._MODEL_3D - centre) @ R.T + centre
        p, _ = cv2.projectPoints(pts3, np.zeros((3, 1)), np.array([[0], [0], [450.0]]), cam, np.zeros(4)); return p.reshape(-1, 2)
    assert project(20)[0][1] > project(0)[0][1]            # nose lower in the image when nodding down
    lm = np.zeros((468, 2))
    for k, p in zip(("nose", "chin", "r_eye", "l_eye", "r_mouth", "l_mouth"), project(20)): lm[F.POSE_IDX[k]] = p
    assert estimate_head_pose(lm, w, h)[0] > 10

def test_perclos_ignores_gaps():
    p = PerclosTracker(30.0)
    p.update(True, 0.0); p.update(False, 20.0)          # 20 s with no frames must not count as 20 s of closure
    assert abs(p.perclos - 1.0 / 15.0) < 1e-9            # only the capped 1 s counts, over the minimum 15 s span
    p.update(False, 21.0); p.update(False, 22.0)
    assert p.perclos < 0.1

def test_perclos_time_weighted():
    p = PerclosTracker(10.0)
    for i in range(0, 11): p.update(closed=(i % 2 == 0 and i >= 4), t=float(i))
    assert 0.0 < p.perclos < 0.5
    q = PerclosTracker(10.0)
    for i in range(0, 11): q.update(True, float(i))
    assert q.perclos == 1.0 and q.closure_duration(10.0) == 10.0

def test_fusion_levels():
    f = DrowsinessFusion(); t = 0.0
    for _ in range(40):                       # 4 s alert driver, 10 fps
        s = f.update(False, 0.3, 0.0, t); t += 0.1
    assert s.level == "Alert"
    for _ in range(20):                       # 2 s eyes closed -> microsleep danger
        s = f.update(True, 0.3, 0.0, t); t += 0.1
    assert s.level == "Danger" and "microsleep" in s.reasons
    g = DrowsinessFusion(); t = 0.0           # yawn + nod without eye closure -> warning, not danger
    for _ in range(30): s = g.update(False, 0.9, 20.0, t); t += 0.1
    assert s.level == "Warning", s

def test_prepare_input_shape_range():
    crop = (np.random.rand(40, 70, 3) * 255).astype(np.uint8)
    x = prepare_eye_input(crop); assert x.shape == (1, 96, 96, 3) and x.dtype == np.float32 and 0 <= x.min() and x.max() <= 255
    assert crop_eye(np.zeros((200, 200, 3), np.uint8), _eye(0.5) + 80) is not None

def test_calibration():
    th = calibrate([0.30] * 50, [0.05] * 50, [3.0] * 50)
    assert abs(th.ear_closed - 0.21) < 1e-9 and th.pitch_neutral == 3.0 and th.mar_yawn >= 0.45
