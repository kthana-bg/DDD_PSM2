import io
from datetime import datetime

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from utils.summary import fmt_duration

LEVEL_RGB = {"Alert": colors.HexColor("#1e8449"), "Warning": colors.HexColor("#b9770e"), "Danger": colors.HexColor("#b03a2e")}


def _score_chart(metrics, width, height=150):
    d = Drawing(width, height)
    if len(metrics) < 2:
        return d
    t0 = datetime.fromisoformat(metrics[0]["timestamp"]).timestamp()
    pts = [((datetime.fromisoformat(m["timestamp"]).timestamp() - t0) / 60.0, float(m["score"])) for m in metrics]
    if max(p[0] for p in pts) < 0.05:
        return d
    x0, y0, w, h = 38, 22, width - 50, height - 34
    for lo, hi, c in ((0, 30, "#e8f6ee"), (30, 60, "#fdf3e1"), (60, 100, "#fbe9e7")):
        d.add(Rect(x0, y0 + h * lo / 100, w, h * (hi - lo) / 100, fillColor=colors.HexColor(c), strokeColor=None))
    lp = LinePlot()
    lp.x, lp.y, lp.width, lp.height = x0, y0, w, h
    lp.data = [pts]
    lp.lines[0].strokeColor = colors.HexColor("#1f4e9c")
    lp.lines[0].strokeWidth = 1.2
    lp.yValueAxis.valueMin, lp.yValueAxis.valueMax, lp.yValueAxis.valueStep = 0, 100, 20
    lp.xValueAxis.valueMin = 0
    lp.xValueAxis.valueMax = max(p[0] for p in pts)
    lp.xValueAxis.labelTextFormat = "%.0f"
    lp.xValueAxis.labels.fontSize = lp.yValueAxis.labels.fontSize = 7
    d.add(lp)
    return d


def build_trip_pdf(session: dict, summary: dict, metrics: list, events: list, full_name: str = "") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"DrowsyGuard trip {session['id']}", author="DrowsyGuard")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=5)
    small = ParagraphStyle("sm", parent=ss["Normal"], fontSize=8.5, textColor=colors.HexColor("#555555"))
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=13)
    width = A4[0] - 36 * mm

    def tbl(rows, widths, header=True):
        t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        st = [("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9ced6")),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
        if header:
            st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                   ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t.setStyle(TableStyle(st))
        return t

    el = [Paragraph("DrowsyGuard Trip Report", h1),
          Paragraph("Real-Time Driver Drowsiness Detection", small), Spacer(1, 8)]
    started = session["started_at"].replace("T", " ")[:19]
    ended = (session.get("ended_at") or "in progress").replace("T", " ")[:19]
    el.append(tbl([["Driver", full_name or "-", "Trip number", str(session["id"])],
                   ["Started", started, "Ended", ended],
                   ["Duration", fmt_duration(summary["duration_s"]), "Detector", str(session.get("detector", "-"))],
                   ["Sensitivity", str(session.get("sensitivity", "-")), "Samples", str(summary["samples"])]],
                  [26 * mm, 52 * mm, 28 * mm, width - 106 * mm], header=False))

    vc = LEVEL_RGB.get(summary["verdict_level"], colors.black)
    el += [Paragraph("Overall result", h2),
           Paragraph(f"<font color='{vc.hexval()}'><b>{summary['verdict_level'].upper() if summary['samples'] else 'NO DATA'}</b></font>"
                     f" &nbsp; {summary['verdict']}", body)]

    el.append(Paragraph("Trip details", h2))
    el.append(tbl([["Measure", "Value", "Measure", "Value"],
                   ["Average drowsiness score", f"{summary['mean_score']:.0f} / 100", "Warnings", str(summary["warnings"])],
                   ["Peak drowsiness score", f"{summary['peak_score']:.0f} / 100", "Danger alerts", str(summary["dangers"])],
                   ["Peak PERCLOS", f"{summary['max_perclos'] * 100:.0f}%", "Yawns", str(summary["yawns"])],
                   ["Longest eye closure", f"{summary['longest_closure_s']:.1f} s", "Head nods", str(summary["nods"])],
                   ["Time in Alert", f"{summary['pct_alert']:.0f}%", "Time in Warning / Danger",
                    f"{summary['pct_warning']:.0f}% / {summary['pct_danger']:.0f}%"]],
                  [48 * mm, 30 * mm, 48 * mm, width - 126 * mm]))

    if len(metrics) >= 2:
        el += [Paragraph("Drowsiness score during the trip (minutes)", h2), _score_chart(metrics, width)]

    el.append(Paragraph(f"Incidents ({len(events)})", h2))
    if not events:
        el.append(Paragraph("No warnings or danger alerts were raised on this trip.", body))
    else:
        rows = [["Time", "Level", "Cause", "Score", "PERCLOS"]]
        for e in events:
            rows.append([str(e["timestamp"]).replace("T", " ")[11:19], e["level"], Paragraph(e.get("reasons") or "score threshold", body),
                         f"{e['score']:.0f}", f"{(e.get('perclos') or 0) * 100:.0f}%"])
        t = tbl(rows, [24 * mm, 24 * mm, width - 24 * mm - 24 * mm - 36 * mm, 18 * mm, 18 * mm])
        t.setStyle(TableStyle([("TEXTCOLOR", (1, i), (1, i), LEVEL_RGB.get(e["level"], colors.black)) for i, e in enumerate(events, 1)]
                              + [("FONTNAME", (1, 1), (1, -1), "Helvetica-Bold")]))
        el.append(t)

    el += [Spacer(1, 14), Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M}. Score: 0-100 from eye closure, yawns and head nods. "
                                    f"30+ is Warning, 60+ or an eye closure over 1.5 s is Danger.", small)]
    doc.build(el)
    return buf.getvalue()
