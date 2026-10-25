import os
import tempfile
import time

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from views.ui import CHART_THEME, button, download_button, image, level_color, plot, table
from utils.drowsiness_features import SENSITIVITY
from utils.landmarks import LandmarkError, create_provider
from utils.model_loader import BASELINE, MODEL_ORDER, available_models, build_classifier, default_model
from utils.monitor import DriverMonitor

MAX_KEYFRAMES = 6


def _mmss(t: float) -> str:
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def _analyse(path, detector, sensitivity, max_seconds, target_fps, calib_s):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error("Could not open this video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = min(total, int(max_seconds * fps)) if total else int(max_seconds * fps)
    step = max(1, round(fps / target_fps))
    try:
        provider = create_provider()
    except (LandmarkError, Exception) as e:
        st.error(f"Face detection could not start: {e}")
        return None
    clf = build_classifier(detector) if detector != BASELINE else None
    mon = DriverMonitor(provider, detector if clf is not None else BASELINE, clf, sensitivity, calib_seconds=calib_s)
    rows, key_frames, events = [], [], []
    bar, preview = st.progress(0.0, text="Analysing..."), st.empty()
    idx, t_wall = 0, time.perf_counter()
    while idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if max(frame.shape[:2]) > 800:
                s = 800 / max(frame.shape[:2]); frame = cv2.resize(frame, None, fx=s, fy=s)
            t = idx / fps
            res, out = mon.process(frame, t)
            for e in mon.pop_events():
                events.append(e)
                if len(key_frames) < MAX_KEYFRAMES:
                    key_frames.append((e["t"], e["level"], cv2.cvtColor(out, cv2.COLOR_BGR2RGB)))
            rows.append(dict(t=t, face=res.face_detected, status=res.status, level=res.level, score=res.score, eye_closed=res.eye_closed,
                             p_closed=res.p_closed, ear=res.ear, mar=res.mar, pitch=res.pitch, perclos=res.perclos,
                             yawns=res.yawns, nods=res.nods, proc_ms=res.proc_ms, model_ms=res.model_latency_ms))
            if len(rows) % 8 == 0:
                preview.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", width=420)
                bar.progress(min(1.0, idx / max(limit, 1)), text=f"Analysing {_mmss(t)} of {_mmss(limit / fps)}")
        idx += 1
    cap.release()
    bar.empty(); preview.empty()
    return dict(df=pd.DataFrame(rows), events=events, key_frames=key_frames, wall=time.perf_counter() - t_wall, fps=fps,
                detector=mon.detector, level_time=mon.level_time, max_closure=mon.max_closure_s, seconds=idx / fps)


def _timeline(df):
    fig = go.Figure()
    for y0, y1, c in ((0, 30, "rgba(46,204,113,0.08)"), (30, 60, "rgba(243,156,18,0.10)"), (60, 100, "rgba(231,76,60,0.12)")):
        fig.add_hrect(y0=y0, y1=y1, fillcolor=c, line_width=0)
    fig.add_trace(go.Scatter(x=df["t"], y=df["score"], name="Drowsiness score", line=dict(color="#60a5fa", width=2.5)))
    fig.add_trace(go.Scatter(x=df["t"], y=df["perclos"] * 100, name="PERCLOS %", line=dict(color="#a78bfa", width=1.5, dash="dot")))
    closed = df[df["eye_closed"]]
    if not closed.empty:
        fig.add_trace(go.Scatter(x=closed["t"], y=[2] * len(closed), mode="markers", name="Eyes closed", marker=dict(color="#e74c3c", size=5)))
    fig.update_layout(title="Drowsiness score over the video", xaxis_title="Video time (s)", yaxis=dict(range=[0, 100]),
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"), height=340, **CHART_THEME)
    return fig


def render_video_test_tab():
    st.header("Video Test")
    st.caption("Run the same detection pipeline on a recorded or simulated driving video (for example a phone clip or an NTHU-DDD video). "
               "Nothing is stored; results stay in this page.")
    up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"], key="vt_upload")
    avail = available_models()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        det = st.selectbox("Eye-closure detector", MODEL_ORDER + [BASELINE], key="vt_model",
                           index=(MODEL_ORDER + [BASELINE]).index(st.session_state.get("active_model", default_model())),
                           format_func=lambda n: n if avail.get(n) else f"{n} (file missing)")
    with c2:
        sens = st.selectbox("Sensitivity", list(SENSITIVITY), index=1, key="vt_sens")
    with c3:
        max_s = st.slider("Analyse up to (seconds)", 10, 600, 90, 10, key="vt_max")
    with c4:
        calib = st.slider("Calibration (seconds)", 3, 15, 6, key="vt_calib", help="The first seconds of the video should show the driver alert.")
    tgt = st.slider("Frames analysed per second", 2, 10, 5, key="vt_fps")

    if up is not None and button("Analyse video", key="vt_run", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as f:
            f.write(up.getbuffer()); path = f.name
        try:
            st.session_state["vt_result"] = _analyse(path, det, sens, max_s, tgt, calib)
        finally:
            os.unlink(path)

    r = st.session_state.get("vt_result")
    if not r or r["df"].empty:
        if up is None:
            st.info("Upload a clip showing the driver's face to begin. For a fair test, start the clip with a few seconds of normal driving.")
        return
    df = r["df"]
    mon_df = df[df["status"] == "Monitoring"]
    st.success(f"Analysed {r['seconds']:.0f} s of video in {r['wall']:.1f} s with **{r['detector']}**.")
    k = st.columns(5)
    k[0].metric("Frames analysed", len(df))
    k[1].metric("Face found", f"{df['face'].mean() * 100:.0f}%")
    k[2].metric("Pipeline speed", f"{len(df) / max(r['wall'], 1e-6):.1f} fps")
    k[3].metric("Warnings / Danger", f"{sum(e['level'] == 'Warning' for e in r['events'])} / {sum(e['level'] == 'Danger' for e in r['events'])}")
    k[4].metric("Peak score", f"{df['score'].max():.0f}")
    plot(_timeline(df), key="vt_timeline")
    if r["events"]:
        ev = pd.DataFrame([{"Video time": _mmss(e["t"]), "Level": e["level"], "Cause": ", ".join(e["reasons"]), "Score": round(e["score"])} for e in r["events"]])
        st.markdown("##### Events"); table(ev)
    else:
        st.info("No warnings or danger alerts were raised in this video.")
    if r["key_frames"]:
        st.markdown("##### Frames at alerts")
        cols = st.columns(min(3, len(r["key_frames"])))
        for i, (t, lvl, img) in enumerate(r["key_frames"]):
            with cols[i % len(cols)]:
                image(img, caption=f"{_mmss(t)} · {lvl}")
    tot = sum(r["level_time"].values()) or 1.0
    st.caption("Time at each level: " + " · ".join(f"{k} {v / tot * 100:.0f}%" for k, v in r["level_time"].items())
               + f" · longest eye closure {r['max_closure']:.1f} s. The first {calib} s are calibration and are not scored.")
    download_button("Download timeline (CSV)", df.to_csv(index=False), "video_test_timeline.csv", key="vt_dl")
