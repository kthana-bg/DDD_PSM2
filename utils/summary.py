"""Trip summary computed from the stored metrics and events of one session."""
from datetime import datetime
from typing import List

import pandas as pd

SAMPLE_INTERVAL_S = 2.0   # the live tab stores one metric row per poll


def to_frames(metrics: List[dict], events: List[dict]):
    m = pd.DataFrame(metrics)
    e = pd.DataFrame(events)
    for df in (m, e):
        if not df.empty:
            df["time"] = pd.to_datetime(df["timestamp"])
    return m, e


def summarize(session: dict, metrics: List[dict], events: List[dict]) -> dict:
    m, e = to_frames(metrics, events)
    start = datetime.fromisoformat(session["started_at"])
    end = datetime.fromisoformat(session["ended_at"]) if session.get("ended_at") else datetime.now()
    out = dict(duration_s=max(0.0, (end - start).total_seconds()), samples=len(m),
               warnings=int((e["level"] == "Warning").sum()) if not e.empty else 0,
               dangers=int((e["level"] == "Danger").sum()) if not e.empty else 0,
               mean_score=0.0, peak_score=0.0, max_perclos=0.0, longest_closure_s=0.0, yawns=0, nods=0,
               pct_alert=100.0, pct_warning=0.0, pct_danger=0.0, verdict="No data", verdict_level="Alert")
    if m.empty:
        return out
    out.update(mean_score=float(m["score"].mean()), peak_score=float(m["score"].max()),
               max_perclos=float(m["perclos"].max()), longest_closure_s=float(m["closure_s"].max()),
               yawns=int(m["yawns"].max()), nods=int(m["nods"].max()))
    share = m["level"].value_counts(normalize=True) * 100
    out.update(pct_alert=float(share.get("Alert", 0.0)), pct_warning=float(share.get("Warning", 0.0)),
               pct_danger=float(share.get("Danger", 0.0)))
    if out["dangers"] > 0 or out["pct_danger"] >= 2:
        out["verdict"], out["verdict_level"] = "High drowsiness risk: take a proper rest before driving on.", "Danger"
    elif out["warnings"] > 0 or out["pct_warning"] >= 10:
        out["verdict"], out["verdict_level"] = "Some signs of fatigue. Plan a break soon.", "Warning"
    else:
        out["verdict"], out["verdict_level"] = "You stayed alert for this trip.", "Alert"
    return out


def fmt_duration(seconds: float) -> str:
    s = int(round(seconds)); h, rem = divmod(s, 3600); m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
