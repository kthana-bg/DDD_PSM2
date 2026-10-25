import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from views.live_monitor_tab import set_active_model
from views.ui import CARD_BG, CHART_THEME, button, card, image, plot, table
from utils.model_loader import BASELINE, DESCRIPTIONS, FIGURES_DIR, MODEL_ORDER, available_models, load_results

COLORS = {"Custom CNN": "#3498db", "MobileNetV2": "#2ecc71", "EfficientNetB0": "#f39c12"}


def _rows(res):
    rows = []
    for n in MODEL_ORDER:
        r = res.get(n)
        if not r:
            continue
        t = r["test"]
        lo, hi = r.get("test_f1_ci95", [None, None])
        rows.append({"Model": n, "Weights": "ImageNet pre-trained" if r.get("pretrained") else "From scratch", "Parameters": r["params"],
                     "Size (KB)": r["size_kb"], "Latency (ms)": r["latency_ms"], "Threshold": r["threshold"],
                     "Accuracy": t["accuracy"], "Precision": t["precision"], "Recall": t["recall"], "F1": t["f1_score"],
                     "F1 CI low": lo, "F1 CI high": hi, "False-alarm rate": t["false_alarm_rate"], "ROC-AUC": t["roc_auc"],
                     "Train time (min)": r["train_seconds"] / 60})
    return pd.DataFrame(rows)


def _metric_bars(df):
    fig = go.Figure()
    for metric, col in (("Precision", "#60a5fa"), ("Recall", "#f59e0b"), ("F1", "#34d399")):
        fig.add_trace(go.Bar(name=metric, x=df["Model"], y=df[metric] * 100, marker_color=col,
                             text=[f"{v * 100:.1f}" for v in df[metric]], textposition="outside"))
    fig.update_layout(title="Closed-eye detection on unseen test people (%)", barmode="group",
                      yaxis=dict(range=[0, 108], title="%"), legend=dict(orientation="h", y=1.12, x=1, xanchor="right"), height=360, **CHART_THEME)
    return fig


def _speed_scatter(df):
    fig = go.Figure()
    for _, r in df.iterrows():
        fig.add_trace(go.Scatter(x=[r["Latency (ms)"]], y=[r["F1"] * 100], mode="markers+text", name=r["Model"], text=[r["Model"]],
                                 textposition="top center", marker=dict(size=max(14, r["Size (KB)"] / 250), color=COLORS[r["Model"]]),
                                 error_y=dict(type="data", symmetric=False, array=[(r["F1 CI high"] - r["F1"]) * 100],
                                              arrayminus=[(r["F1"] - r["F1 CI low"]) * 100], color="#9aa4b2")))
    fig.update_layout(title="Accuracy vs speed (bubble = file size, bars = 95% CI)", xaxis_title="CPU latency per eye crop (ms)",
                      yaxis=dict(title="F1 (%)", range=[80, 100]), showlegend=False, height=360, **CHART_THEME)
    return fig


def _far_bar(df):
    fig = go.Figure(go.Bar(x=df["Model"], y=df["False-alarm rate"] * 100, marker_color=[COLORS[m] for m in df["Model"]],
                           text=[f"{v * 100:.2f}%" for v in df["False-alarm rate"]], textposition="outside"))
    fig.update_layout(title="False-alarm rate (open eyes judged closed, lower is better)", yaxis=dict(title="%", range=[0, max(6, df["False-alarm rate"].max() * 130)]),
                      height=300, **CHART_THEME)
    return fig


def _on_radio():
    set_active_model(st.session_state["model_select_cmp"])


def render_comparison_tab():
    st.header("Model Comparison")
    st.caption("Three eye-state models (open vs closed) were trained and tested on the MRL Eye Dataset. The test people were never seen "
               "in training, so the numbers are not inflated by the same face appearing on both sides.")
    res = load_results()
    df = _rows(res)
    if df.empty:
        st.warning("No result files found in results/. Run the Kaggle notebooks and copy the results here.")
        return
    a, b = st.columns(2)
    with a:
        plot(_metric_bars(df), key="cmp_bars")
    with b:
        plot(_speed_scatter(df), key="cmp_scatter")
    plot(_far_bar(df), key="cmp_far")

    st.markdown("##### Full results")
    view = df.copy()
    for c in ("Accuracy", "Precision", "Recall", "F1", "False-alarm rate", "ROC-AUC"):
        view[c] = view[c].map(lambda v: f"{v:.4f}")
    view["F1 95% CI"] = [f"{lo:.3f}-{hi:.3f}" for lo, hi in zip(df["F1 CI low"], df["F1 CI high"])]
    view["Latency (ms)"] = view["Latency (ms)"].map(lambda v: f"{v:.2f}")
    view["Train time (min)"] = view["Train time (min)"].map(lambda v: f"{v:.1f}")
    view["Parameters"] = view["Parameters"].map(lambda v: f"{v:,}")
    view["Size (KB)"] = view["Size (KB)"].map(lambda v: f"{v:,.0f}")
    view["Threshold"] = view["Threshold"].map(lambda v: f"{v:.2f}")
    table(view.drop(columns=["F1 CI low", "F1 CI high"]))

    figs = [("training_curves.png", "Training curves"), ("roc_test.png", "ROC curves"),
            ("confusion_matrices.png", "Confusion matrices"), ("accuracy_vs_latency.png", "Accuracy vs latency")]
    with st.expander("Charts from the training notebooks"):
        for fname, title in figs:
            p = os.path.join(FIGURES_DIR, fname)
            if os.path.exists(p):
                st.markdown(f"**{title}**")
                image(p)

    st.markdown("##### Active detector for live monitoring")
    avail = available_models()
    options = MODEL_ORDER + [BASELINE]
    st.session_state["model_select_cmp"] = st.session_state.get("active_model", options[0])
    st.radio("Detector", options, key="model_select_cmp", horizontal=True, label_visibility="collapsed",
             format_func=lambda n: n if avail.get(n) else f"{n} (file missing)", on_change=_on_radio)
    st.caption(DESCRIPTIONS.get(st.session_state["model_select_cmp"], ""))
    if not any(avail[n] for n in MODEL_ORDER):
        st.info("The .tflite model files are not in models/eye_state/ yet. Copy custom_cnn.tflite, mobilenetv2.tflite and "
                "efficientnetb0.tflite from your Kaggle outputs there. Until then the app uses the EAR rule.")
