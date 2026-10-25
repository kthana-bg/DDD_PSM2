"""Shared look and feel: CSS, level colours, cards, gauge and version-safe widget helpers."""
import inspect

import plotly.graph_objects as go
import streamlit as st

APP_NAME = "DrowsyGuard"
APP_TAGLINE = "Real-Time Driver Drowsiness Detection"

LEVEL_COLORS = {"Alert": "#2ecc71", "Warning": "#f39c12", "Danger": "#e74c3c",
                "Calibrating": "#3498db", "No face": "#7f8c8d", "Starting": "#7f8c8d"}
CARD_BG = "#161b26"
CHART_THEME = dict(plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG, font=dict(color="#e0e0e0", size=12),
                   margin=dict(t=50, b=40, l=50, r=20))

CSS = """
<style>
  .stApp { background-color: #0e1117; }
  .block-container { padding-top: 3.2rem; max-width: 1280px; }
  .stTabs [data-baseweb="tab-list"] { background-color: #161b26; border-radius: 10px; padding: 4px; gap: 2px; }
  .stTabs [data-baseweb="tab"] { color: #9aa4b2; border-radius: 8px; padding: 8px 20px; }
  .stTabs [aria-selected="true"] { background-color: #2563eb !important; color: #fff !important; }
  .stButton > button { border-radius: 8px; font-weight: 600; }
  div[data-testid="metric-container"] { background-color: #161b26; border-radius: 10px; padding: 12px; }
  .dg-card { background:#161b26; border-radius:12px; padding:14px 16px; margin-bottom:10px; border-left:4px solid #3498db; }
  .dg-label { font-size:11px; color:#9aa4b2; margin:0 0 4px 0; text-transform:uppercase; letter-spacing:1px; }
  .dg-value { font-size:26px; font-weight:700; margin:0; line-height:1.15; }
  .dg-sub { font-size:12px; color:#9aa4b2; margin:3px 0 0 0; }
  .dg-bar { height:6px; background:#262d3b; border-radius:3px; margin-top:8px; overflow:hidden; }
  .dg-bar > div { height:100%; border-radius:3px; }
  .dg-banner { border-radius:14px; padding:18px 20px; margin-bottom:12px; text-align:center; }
  .dg-banner h2 { margin:0; font-size:34px; letter-spacing:2px; }
  .dg-banner p { margin:4px 0 0 0; font-size:14px; opacity:.9; }
  @keyframes dgpulse { 0%{box-shadow:0 0 0 0 rgba(231,76,60,.65);} 70%{box-shadow:0 0 0 18px rgba(231,76,60,0);} 100%{box-shadow:0 0 0 0 rgba(231,76,60,0);} }
  .dg-danger { animation: dgpulse 1.2s infinite; }
  .dg-chip { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; margin-right:6px; }
  .dg-event { font-size:13px; padding:6px 10px; border-radius:8px; background:#161b26; margin-bottom:6px; }
  @media (max-width: 768px) {
    .block-container { padding: 0.75rem 0.6rem 3rem 0.6rem !important; }
    h1, h2 { font-size: 1.4rem !important; }
    .stTabs [data-baseweb="tab-list"] { overflow-x: auto; }
    .stTabs [data-baseweb="tab"] { padding: 8px 12px; font-size: 13px; white-space: nowrap; }
    .stButton > button { min-height: 48px; width: 100%; }
    .dg-banner h2 { font-size: 26px; }
    .dg-value { font-size: 22px; }
    video { max-width: 100% !important; height: auto !important; }
  }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def _has_width(fn) -> bool:
    try:
        return "width" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


_BTN_WIDTH = dict(width="stretch") if _has_width(st.button) else dict(use_container_width=True)
_PLOT_WIDTH = dict(width="stretch") if _has_width(st.plotly_chart) else dict(use_container_width=True)
_DF_WIDTH = dict(width="stretch") if _has_width(st.dataframe) else dict(use_container_width=True)
_IMG_WIDTH = dict(width="stretch") if _has_width(st.image) else dict(use_container_width=True)


def button(label, **kw):
    return st.button(label, **_BTN_WIDTH, **kw)


def download_button(label, data, file_name, mime="text/csv", **kw):
    return st.download_button(label, data, file_name=file_name, mime=mime, **_BTN_WIDTH, **kw)


def plot(fig, key=None):
    st.plotly_chart(fig, **_PLOT_WIDTH, key=key)


def table(df, **kw):
    st.dataframe(df, hide_index=True, **_DF_WIDTH, **kw)


def image(img, **kw):
    st.image(img, **_IMG_WIDTH, **kw)


def level_color(name: str) -> str:
    return LEVEL_COLORS.get(name, "#7f8c8d")


def card(label: str, value: str, color: str, sub: str = "", bar: float = None):
    bar_html = (f"<div class='dg-bar'><div style='width:{max(0, min(1, bar)) * 100:.0f}%;background:{color};'></div></div>"
                if bar is not None else "")
    sub_html = f"<p class='dg-sub'>{sub}</p>" if sub else ""
    st.markdown(f"<div class='dg-card' style='border-left-color:{color};'><p class='dg-label'>{label}</p>"
                f"<p class='dg-value' style='color:{color};'>{value}</p>{sub_html}{bar_html}</div>", unsafe_allow_html=True)


def banner(title: str, subtitle: str, color: str, pulse: bool = False):
    st.markdown(f"<div class='dg-banner {'dg-danger' if pulse else ''}' style='background:{color}26;border:2px solid {color};'>"
                f"<h2 style='color:{color};'>{title}</h2><p>{subtitle}</p></div>", unsafe_allow_html=True)


def gauge(score: float, height: int = 210):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(score),
        number=dict(font=dict(size=40, color="#e0e0e0"), suffix=""),
        title=dict(text="Drowsiness score", font=dict(size=13, color="#9aa4b2")),
        gauge=dict(axis=dict(range=[0, 100], tickcolor="#9aa4b2"), bar=dict(color="#e0e0e0", thickness=0.22),
                   bgcolor=CARD_BG, borderwidth=0,
                   steps=[dict(range=[0, 30], color="#1e6b3c"), dict(range=[30, 60], color="#8a5a0a"),
                          dict(range=[60, 100], color="#8a2a22")])))
    fig.update_layout(height=height, paper_bgcolor=CARD_BG, margin=dict(t=50, b=10, l=25, r=25), font=dict(color="#e0e0e0"))
    return fig
