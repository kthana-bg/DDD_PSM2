import os
import time

import streamlit as st

from database.db_manager import end_session, save_event, save_metric, start_session
from views.trip_analytics_tab import render_trip_summary
from views.ui import banner, button, card, gauge, level_color, plot
from utils.alerts import AlertManager, play_alert
from utils.drowsiness_features import SENSITIVITY
from utils.mobile_support import keep_awake, notify_alert, render_notification_button
from utils.model_loader import BASELINE, DESCRIPTIONS, MODEL_ORDER, available_models, default_model
from utils.monitor import FrameResult
from utils.voice_alert import maybe_speak_alert, render_voice_controls

CAMERA_PRESETS = {
    "Standard (640x480)": (640, 480, 20),
    "Battery saver (480x360)": (480, 360, 15),
    "Low (320x240)": (320, 240, 10),
}
POLL_SECONDS = 1.5
METRIC_EVERY_S = 2.0


def _rtc_configuration():
    try:
        from twilio.rest import Client
        sid, token = os.environ.get("TWILIO_ACCOUNT_SID", "").strip(), os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
        if sid and token:
            return {"iceServers": Client(sid, token).tokens.create().ice_servers}
    except Exception as e:
        print(f"Twilio TURN fetch failed, falling back to STUN: {e}")
    return {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}, {"urls": "stun:stun1.l.google.com:19302"}]}


def set_active_model(name: str):
    st.session_state["active_model"] = name
    st.session_state["model_select_live"] = name


def _on_model_change():
    st.session_state["active_model"] = st.session_state["model_select_live"]


def _init_state():
    avail = available_models()
    if st.session_state.get("active_model") not in avail:
        st.session_state["active_model"] = default_model()
    st.session_state.setdefault("model_select_live", st.session_state["active_model"])
    for k, v in dict(monitoring_active=False, session_id=None, session_start=None, recal_id=0, sensitivity="Normal",
                     sound_on=True, volume=0.8, recent_events=[], last_metric_save=0.0, finished_session_id=None).items():
        st.session_state.setdefault(k, v)
    if "alert_mgr" not in st.session_state:
        st.session_state["alert_mgr"] = AlertManager()


def _model_label(name: str) -> str:
    avail = available_models()
    tag = "" if avail.get(name) else "  (model file missing)"
    return f"{name}{tag}"


def _settings():
    with st.expander("Detection and alert settings", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Eye-closure detector", MODEL_ORDER + [BASELINE], key="model_select_live", on_change=_on_model_change,
                         format_func=_model_label,
                         help="The model that decides if the eyes are closed. Compare them in the Model Comparison tab.")
            name = st.session_state["active_model"]
            st.caption(DESCRIPTIONS.get(name, ""))
            if not available_models().get(name):
                st.warning(f"models/eye_state has no file for {name}. The app falls back to the EAR rule until you add it.")
            st.radio("Sensitivity", list(SENSITIVITY), key="sensitivity", index=list(SENSITIVITY).index(st.session_state["sensitivity"]),
                     help="High warns earlier but may raise more false alarms.")
        with c2:
            st.toggle("Play alert sounds", key="sound_on")
            st.slider("Alert volume", 0.1, 1.0, step=0.05, key="volume")
            t1, t2 = st.columns(2)
            with t1:
                if button("Test soft beep", key="test_soft"):
                    play_alert("soft", st.session_state["volume"])
            with t2:
                if button("Test loud alarm", key="test_loud"):
                    play_alert("loud", st.session_state["volume"])
            st.selectbox("Camera quality", list(CAMERA_PRESETS), key="camera_quality",
                         help="Use Battery saver or Low on phones. Changing it restarts the camera.")
        render_voice_controls()
        st.toggle("Keep screen on while monitoring", value=True, key="keep_awake_enabled")
        st.toggle("Vibrate on danger alerts (Android phones)", value=True, key="vibrate_enabled")
        render_notification_button()
        st.caption("Browsers cannot use the camera while the screen is off. On a phone, keep this page open with the screen on "
                   "and mount the phone facing you. Camera access on a phone needs an HTTPS address.")


def _events_list():
    ev = st.session_state.get("recent_events", [])[:5]
    st.markdown("<p class='dg-label' style='margin-top:6px'>Recent events</p>", unsafe_allow_html=True)
    if not ev:
        st.caption("No warnings yet.")
    for e in ev:
        col = level_color(e["level"])
        st.markdown(f"<div class='dg-event' style='border-left:3px solid {col}'><b style='color:{col}'>{e['level']}</b> · "
                    f"{time.strftime('%H:%M:%S', time.localtime(e['t']))} · {', '.join(e['reasons']) or 'score threshold'}</div>",
                    unsafe_allow_html=True)


def _render_panel(res: FrameResult):
    pending = res.status in ("Starting",)
    if res.status == "Calibrating":
        banner("CALIBRATING", f"Look straight ahead, eyes open, mouth closed ({int(res.calib_progress * 100)}%)", level_color("Calibrating"))
    elif res.status == "No face" and res.level == "Alert":
        banner("NO FACE", "Move so your face is visible to the camera", level_color("No face"))
    elif pending:
        banner("WAITING FOR CAMERA", "Press START in the video box", level_color("Starting"))
    else:
        sub = ", ".join(res.reasons).capitalize() if res.reasons else "No drowsiness signs"
        banner(res.level.upper(), sub, level_color(res.level), pulse=res.level == "Danger")
    plot(gauge(res.score), key="live_gauge")

    c1, c2 = st.columns(2)
    with c1:
        eye_col = level_color("Danger") if res.eye_closed else level_color("Alert")
        p = f"P(closed) {res.p_closed:.2f}" if res.p_closed is not None else f"EAR {res.ear:.2f}"
        card("Eyes", "CLOSED" if res.eye_closed else "OPEN", eye_col, f"{p} · closed for {res.closure_s:.1f} s")
        card("Mouth", f"{res.yawns} yawn{'s' if res.yawns != 1 else ''}", level_color("Warning") if res.yawns else level_color("Alert"),
             f"MAR {res.mar:.2f}")
    with c2:
        pc = level_color("Danger") if res.perclos >= 0.15 else level_color("Warning") if res.perclos >= 0.08 else level_color("Alert")
        card("PERCLOS (30 s)", f"{res.perclos * 100:.0f}%", pc, "eyes-closed share", bar=min(1.0, res.perclos / 0.30))
        neutral = res.thresholds.pitch_neutral if res.thresholds else 0.0
        card("Head", f"{res.nods} nod{'s' if res.nods != 1 else ''}", level_color("Warning") if res.nods else level_color("Alert"),
             f"pitch {res.pitch - neutral:+.0f}° from neutral")
    st.caption(f"Detector: {res.detector} · {res.fps:.0f} frames/s processed · {res.proc_ms:.0f} ms/frame"
               + (f" (model {res.model_latency_ms:.1f} ms)" if res.p_closed is not None else ""))


def _handle_result(proc, res: FrameResult, user: dict):
    """Persist events/metrics and fire alerts. Runs on each poll."""
    sid, uid = st.session_state["session_id"], st.session_state["user_id"]
    for e in proc.pop_events():
        save_event(sid, uid, e["level"], e["reasons"], e["score"], e["perclos"])
        st.session_state["recent_events"].insert(0, e)
    now = time.time()
    if res.face_detected and not res.calibrating and res.status == "Monitoring" and now - st.session_state["last_metric_save"] >= METRIC_EVERY_S:
        save_metric(sid, uid, res.level, res.score, res.eye_closed, res.p_closed, res.ear, res.perclos, res.closure_s,
                    res.mar, res.pitch, res.yawns, res.nods, res.detector)
        st.session_state["last_metric_save"] = now
    level = res.level if res.status in ("Monitoring", "No face") else "Alert"
    kind = st.session_state["alert_mgr"].update(level)
    if kind:
        if st.session_state["sound_on"]:
            play_alert(kind, st.session_state["volume"])
        maybe_speak_alert(level, (user or {}).get("full_name"))
        if kind == "loud":
            notify_alert("DrowsyGuard: DANGER", "You look very drowsy. Pull over safely and rest.",
                         vibrate=st.session_state.get("vibrate_enabled", True))


def _live_panel(ctx):
    proc = ctx.video_processor if ctx else None
    res = proc.get_result() if proc else FrameResult()
    if proc is not None:
        if proc.error:
            st.error(f"Face detection could not start: {proc.error}")
        _handle_result(proc, res, st.session_state.get("user"))
    _render_panel(res)
    _events_list()
    start = st.session_state.get("session_start")
    if start:
        m, s = divmod(int(time.time() - start), 60)
        h, m = divmod(m, 60)
        st.caption(f"Trip duration: {h:d}:{m:02d}:{s:02d}" if h else f"Trip duration: {m:02d}:{s:02d}")


if hasattr(st, "fragment"):
    _live_panel_fragment = st.fragment(run_every=POLL_SECONDS)(_live_panel)
else:
    _live_panel_fragment = None


def render_live_tab(user_id: int):
    _init_state()
    st.header("Live Monitor")
    try:
        from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
        from utils.webrtc_processor import DrowsinessProcessor
    except ImportError as err:
        st.error(f"streamlit-webrtc is not installed: {err}")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        if button("Start trip", key="trip_start", type="primary", disabled=st.session_state["monitoring_active"]):
            st.session_state.update(monitoring_active=True, session_start=time.time(), recent_events=[], finished_session_id=None,
                                    last_metric_save=0.0, recal_id=st.session_state["recal_id"] + 1)
            st.session_state["session_id"] = start_session(user_id, st.session_state["active_model"], st.session_state["sensitivity"])
            st.session_state["alert_mgr"].reset()
            st.rerun()
    with c2:
        if button("End trip and show summary", key="trip_stop", disabled=not st.session_state["monitoring_active"]):
            end_session(st.session_state["session_id"])
            st.session_state.update(monitoring_active=False, finished_session_id=st.session_state["session_id"], session_start=None)
            st.session_state["alert_mgr"].reset()
            keep_awake(False)
            st.rerun()
    with c3:
        if button("Recalibrate", key="trip_recal", disabled=not st.session_state["monitoring_active"],
                  help="Re-learn your normal eye, mouth and head position (look ahead for 8 s)."):
            st.session_state["recal_id"] += 1

    _settings()
    st.divider()

    if not st.session_state["monitoring_active"]:
        keep_awake(False)
        if st.session_state.get("finished_session_id"):
            render_trip_summary(st.session_state["finished_session_id"], key="live_sum")
        return

    quality = st.session_state.get("camera_quality", "Standard (640x480)")
    w, h, fps = CAMERA_PRESETS.get(quality, CAMERA_PRESETS["Standard (640x480)"])
    left, right = st.columns([3, 2])
    with left:
        ctx = webrtc_streamer(
            key=f"drowsyguard-{w}x{h}", mode=WebRtcMode.SENDRECV, rtc_configuration=RTCConfiguration(_rtc_configuration()),
            video_processor_factory=DrowsinessProcessor,
            media_stream_constraints={"video": {"width": {"ideal": w}, "height": {"ideal": h}, "frameRate": {"ideal": fps},
                                                "facingMode": "user"}, "audio": False},
            async_processing=True)
        if ctx.video_processor:
            ctx.video_processor.configure(st.session_state["active_model"], st.session_state["sensitivity"], st.session_state["recal_id"])
        if not ctx.state.playing:
            st.caption("Click **START** in the video box above and allow camera access.")
        keep_awake(st.session_state.get("keep_awake_enabled", True))
    with right:
        if _live_panel_fragment is not None:
            _live_panel_fragment(ctx)
        else:
            _live_panel(ctx)
            time.sleep(POLL_SECONDS)
            st.rerun()
