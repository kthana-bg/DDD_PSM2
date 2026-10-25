import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from database.db_manager import delete_session, get_events, get_metrics, get_session, get_sessions
from views.ui import CARD_BG, CHART_THEME, banner, button, card, download_button, gauge, level_color, plot, table
from utils.summary import fmt_duration, summarize, to_frames

LEVEL_BAND = [(0, 30, "rgba(46,204,113,0.08)"), (30, 60, "rgba(243,156,18,0.10)"), (60, 100, "rgba(231,76,60,0.12)")]


def score_timeline(m: pd.DataFrame, e: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for y0, y1, col in LEVEL_BAND:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=col, line_width=0)
    fig.add_trace(go.Scatter(x=m["time"], y=m["score"], mode="lines", name="Drowsiness score",
                             line=dict(color="#60a5fa", width=2.5)))
    if e is not None and not e.empty:
        for lvl in ("Warning", "Danger"):
            d = e[e["level"] == lvl]
            if not d.empty:
                fig.add_trace(go.Scatter(x=d["time"], y=d["score"], mode="markers", name=f"{lvl} event",
                                         marker=dict(size=11, color=level_color(lvl), symbol="diamond",
                                                     line=dict(color="white", width=1)),
                                         hovertext=d["reasons"]))
    fig.update_layout(title="Drowsiness score over the trip", yaxis=dict(range=[0, 100], title="Score"),
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"), height=340, **CHART_THEME)
    return fig


def indicators_chart(m: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Eyes: PERCLOS (% of the last 30 s with eyes closed)",
                                        "Mouth: aspect ratio (MAR), higher = mouth open",
                                        "Head: pitch angle (deg), higher = head down"))
    fig.add_trace(go.Scatter(x=m["time"], y=m["perclos"] * 100, line=dict(color="#a78bfa"), name="PERCLOS %"), 1, 1)
    fig.add_trace(go.Scatter(x=m["time"], y=m["mar"], line=dict(color="#f59e0b"), name="MAR"), 2, 1)
    fig.add_trace(go.Scatter(x=m["time"], y=m["pitch"], line=dict(color="#34d399"), name="Pitch"), 3, 1)
    fig.update_layout(height=560, showlegend=False, **CHART_THEME)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=12, color="#9aa4b2")
    return fig


def level_share_chart(s: dict) -> go.Figure:
    vals = [s["pct_alert"], s["pct_warning"], s["pct_danger"]]
    fig = go.Figure(go.Pie(labels=["Alert", "Warning", "Danger"], values=vals, hole=0.6, sort=False,
                           marker=dict(colors=[level_color("Alert"), level_color("Warning"), level_color("Danger")]),
                           textinfo="percent"))
    fig.update_layout(title="Time at each level", height=300, legend=dict(orientation="h", y=-0.05),
                      paper_bgcolor=CARD_BG, font=dict(color="#e0e0e0"), margin=dict(t=50, b=10, l=10, r=10))
    return fig


def render_trip_summary(session_id: int, key: str = "sum"):
    session = get_session(session_id)
    if not session:
        st.info("Trip not found.")
        return
    metrics, events = get_metrics(session_id), get_events(session_id)
    s = summarize(session, metrics, events)
    st.markdown(f"#### Trip summary &nbsp;·&nbsp; {session['started_at'].replace('T', ' ')}")
    banner(s["verdict_level"].upper() if s["samples"] else "NO DATA", s["verdict"], level_color(s["verdict_level"]))
    if not s["samples"]:
        st.caption("No monitoring data was recorded for this trip (the face was not visible or the trip was very short).")
        return

    c = st.columns(5)
    c[0].metric("Duration", fmt_duration(s["duration_s"]))
    c[1].metric("Average score", f"{s['mean_score']:.0f}")
    c[2].metric("Peak score", f"{s['peak_score']:.0f}")
    c[3].metric("Warnings", s["warnings"])
    c[4].metric("Danger alerts", s["dangers"])
    c = st.columns(4)
    c[0].metric("Yawns", s["yawns"])
    c[1].metric("Head nods", s["nods"])
    c[2].metric("Peak PERCLOS", f"{s['max_perclos'] * 100:.0f}%")
    c[3].metric("Longest eye closure", f"{s['longest_closure_s']:.1f} s")

    m, e = to_frames(metrics, events)
    plot(score_timeline(m, e), key=f"{key}_tl")
    left, right = st.columns([2, 1])
    with left:
        plot(indicators_chart(m), key=f"{key}_ind")
    with right:
        plot(level_share_chart(s), key=f"{key}_share")
        st.caption(f"Based on {s['samples']} samples, one every ~2 s while the face was visible.")

    st.markdown("##### Drowsiness events")
    if e.empty:
        st.success("No warnings or danger alerts were raised on this trip.")
    else:
        d = e[["time", "level", "reasons", "score", "perclos"]].copy()
        d["time"] = d["time"].dt.strftime("%H:%M:%S")
        d["score"] = d["score"].round(0).astype(int)
        d["perclos"] = (d["perclos"] * 100).round(0).astype(int).astype(str) + "%"
        d.columns = ["Time", "Level", "Cause", "Score", "PERCLOS"]
        table(d)
    cols = st.columns(2)
    with cols[0]:
        download_button("Download samples (CSV)", m.drop(columns=["time"]).to_csv(index=False), f"trip_{session_id}_samples.csv", key=f"{key}_dl1")
    with cols[1]:
        if not e.empty:
            download_button("Download events (CSV)", e.drop(columns=["time"]).to_csv(index=False), f"trip_{session_id}_events.csv", key=f"{key}_dl2")


def render_analytics_tab(user_id: int):
    st.header("Trip History")
    sessions = get_sessions(user_id)
    if not sessions:
        st.info("No trips recorded yet. Start a trip in the Live Monitor tab; each trip is summarised here when it ends.")
        return

    rows = []
    for s in sessions:
        dur = 0.0
        if s["ended_at"]:
            from datetime import datetime
            dur = (datetime.fromisoformat(s["ended_at"]) - datetime.fromisoformat(s["started_at"])).total_seconds()
        rows.append({"Trip": s["id"], "Started": s["started_at"].replace("T", " "), "Duration": fmt_duration(dur) if s["ended_at"] else "in progress",
                     "Warnings": s["n_warning"], "Danger": s["n_danger"], "Peak score": round(s["peak_score"] or 0),
                     "Detector": s["detector"]})
    df = pd.DataFrame(rows)

    top = st.columns(3)
    top[0].metric("Trips", len(df))
    top[1].metric("Total warnings", int(df["Warnings"].sum()))
    top[2].metric("Total danger alerts", int(df["Danger"].sum()))
    if len(df) > 1:
        fig = go.Figure(go.Bar(x=df["Trip"].astype(str), y=df["Peak score"],
                               marker_color=[level_color("Danger" if p >= 60 else "Warning" if p >= 30 else "Alert") for p in df["Peak score"]]))
        fig.update_layout(title="Peak drowsiness score per trip", xaxis_title="Trip", yaxis=dict(range=[0, 100]), height=260, **CHART_THEME)
        plot(fig, key="hist_peak")
    table(df)

    ids = [s["id"] for s in sessions]
    choice = st.selectbox("Open a trip", ids, format_func=lambda i: f"Trip {i} · {next(s['started_at'] for s in sessions if s['id'] == i).replace('T', ' ')}", key="hist_select")
    render_trip_summary(choice, key="hist")
    with st.expander("Delete this trip"):
        if button("Delete trip permanently", key="hist_delete"):
            delete_session(choice, user_id)
            st.rerun()
