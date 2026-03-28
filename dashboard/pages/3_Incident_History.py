"""Streamlit page: Incident history (mission-control analytics)."""

from __future__ import annotations

import sys
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from database import crud
from database.models import AlertLog, Detection
from database.session import SessionLocal


PLOTLY_DARK_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#0D0F14",
        "plot_bgcolor": "#141720",
        "font": {"family": "JetBrains Mono", "color": "#8B92A5", "size": 11},
        "xaxis": {"gridcolor": "rgba(255,255,255,0.05)", "linecolor": "rgba(255,255,255,0.1)"},
        "yaxis": {"gridcolor": "rgba(255,255,255,0.05)", "linecolor": "rgba(255,255,255,0.1)"},
        "colorway": ["#FF4500", "#4A9EFF", "#00D46A", "#FFB800"],
    }
}


def _load_css() -> None:
    css_path = Path("dashboard/assets/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def main() -> None:
    try:
        st.set_page_config(page_title="PyroSense AI", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")
    except Exception:
        pass
    _load_css()

    st.markdown(
        """
    <div style="padding: 0 0 18px; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 18px;">
      <div style="font-family:'JetBrains Mono',monospace; font-size:22px; color:#E8EAF0; font-weight:700;">
        INCIDENT HISTORY
      </div>
      <div style="font-family:monospace; font-size:11px; color:#555C70; text-transform:uppercase; letter-spacing:0.1em; margin-top:6px;">
        Timeline, risk index, and operational reporting
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Horizontal filter panel
    with st.container():
        st.markdown("<div class='pyro-card'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
        dr = c1.date_input("Date Range (UTC)", value=(date.today(), date.today()))
        cls = c2.selectbox("Class", options=["All", "Fire", "Smoke"], index=0)
        min_conf = c3.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
        zone = c4.text_input("Zone/Location", value="")
        st.markdown("</div>", unsafe_allow_html=True)

    start_ts = None
    end_ts = None
    if isinstance(dr, tuple) and len(dr) == 2:
        start_ts = datetime.combine(dr[0], dtime.min).replace(tzinfo=timezone.utc)
        end_ts = datetime.combine(dr[1], dtime.max).replace(tzinfo=timezone.utc)

    class_filter = None
    if cls.lower() == "fire":
        class_filter = "fire"
    elif cls.lower() == "smoke":
        class_filter = "smoke"

    with SessionLocal() as db:
        rows = crud.list_detections(
            db,
            offset=0,
            limit=1000,
            class_name=class_filter,
            min_conf=min_conf if min_conf > 0 else None,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if zone.strip():
            z = zone.strip().lower()
            rows = [r for r in rows if z in (r.source or "").lower()]

        alerts = db.query(AlertLog).order_by(AlertLog.sent_at.desc()).limit(5000).all()

    if not rows:
        st.warning("No incidents match the current filters.")
        return

    df = pd.DataFrame(
        [
            {
                "ID": r.id,
                "Time": pd.to_datetime(r.timestamp, utc=True, errors="coerce"),
                "Type": r.class_name,
                "Confidence": float(r.confidence),
                "Risk": float(r.risk_score),
                "Location": r.source,
                "Frame": r.frame_path or "",
                "Heatmap": r.heatmap_path or "",
                "LLM Summary": (r.llm_summary or "")[:200],
            }
            for r in rows
        ]
    ).dropna(subset=["Time"])

    # Charts 2x2 grid
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    # 1) Line chart: detections over time
    ts = df.copy()
    ts["bucket"] = ts["Time"].dt.floor("15min")
    ts_counts = ts.groupby(["bucket", "Type"]).size().reset_index(name="count")
    fig1 = px.line(ts_counts, x="bucket", y="count", color="Type", markers=True, title="Detections over time")
    fig1.update_layout(template=PLOTLY_DARK_TEMPLATE)
    c1.plotly_chart(fig1, use_container_width=True)

    # 2) Bar chart: by hour of day
    by_hour = df.copy()
    by_hour["hour"] = by_hour["Time"].dt.hour
    bars = by_hour.groupby("hour").size().reset_index(name="count")
    fig2 = px.bar(bars, x="hour", y="count", title="Detections by hour (UTC)")
    fig2.update_layout(template=PLOTLY_DARK_TEMPLATE)
    c2.plotly_chart(fig2, use_container_width=True)

    # 3) Gauge chart: today's risk index
    risk_today = float(df["Risk"].mean()) if not df.empty else 0.0
    fig3 = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_today,
            title={"text": "Today's risk index (0-100)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#FF4500"},
                "steps": [
                    {"range": [0, 40], "color": "rgba(0,212,106,0.25)"},
                    {"range": [40, 70], "color": "rgba(255,184,0,0.25)"},
                    {"range": [70, 100], "color": "rgba(255,69,0,0.20)"},
                ],
            },
        )
    )
    fig3.update_layout(template=PLOTLY_DARK_TEMPLATE)
    c3.plotly_chart(fig3, use_container_width=True)

    # 4) Area chart: alert response time trend
    # Approximate response time: AlertLog.sent_at - Detection.timestamp (in seconds) for sent alerts.
    det_by_id = {r.id: r for r in rows}
    resp_points: List[Dict[str, Any]] = []
    for a in alerts:
        if a.status != "sent" or a.sent_at is None:
            continue
        det = det_by_id.get(a.detection_id)
        if det is None:
            continue
        dt_s = (a.sent_at - det.timestamp).total_seconds()
        resp_points.append({"Time": pd.to_datetime(det.timestamp, utc=True), "ResponseSeconds": max(0.0, float(dt_s))})
    if resp_points:
        rdf = pd.DataFrame(resp_points)
        rdf["bucket"] = rdf["Time"].dt.floor("30min")
        trend = rdf.groupby("bucket")["ResponseSeconds"].mean().reset_index()
        fig4 = px.area(trend, x="bucket", y="ResponseSeconds", title="Alert response time trend (avg seconds)")
    else:
        fig4 = px.area(pd.DataFrame({"bucket": [], "ResponseSeconds": []}), x="bucket", y="ResponseSeconds", title="Alert response time trend")
    fig4.update_layout(template=PLOTLY_DARK_TEMPLATE)
    c4.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Incidents")
    st.dataframe(df[["ID", "Time", "Type", "Confidence", "Risk", "Location", "LLM Summary"]], use_container_width=True, height=320)

    st.markdown("### Actions")
    for r in df.sort_values("Time", ascending=False).head(20).to_dict(orient="records"):
        with st.expander(f"#{r['ID']} • {r['Type']} • {r['Confidence']:.2f} • Risk {r['Risk']:.0f} • {str(r['Time'])}"):
            a1, a2, a3 = st.columns(3)
            if a1.button("View Frame", key=f"vf_{r['ID']}") and r.get("Frame"):
                fp = Path(str(r["Frame"]))
                if fp.exists():
                    st.image(str(fp))
            if a2.button("View Heatmap", key=f"vh_{r['ID']}") and r.get("Heatmap"):
                hp = Path(str(r["Heatmap"]))
                if hp.exists():
                    st.image(str(hp))
            if a3.button("Generate Report", key=f"gr_{r['ID']}"):
                st.markdown(
                    f"""
                <div class="pyro-card alert">
                  <div style="font-family:monospace; font-size:12px; color:#E8EAF0; font-weight:700;">Incident Report</div>
                  <div style="font-family:monospace; font-size:11px; color:#8B92A5; margin-top:8px; line-height:1.7;">
                    ID: {r['ID']}<br/>
                    Time: {r['Time']}<br/>
                    Type: {r['Type']} ({r['Confidence']*100:.0f}%)<br/>
                    Risk: {r['Risk']:.0f}/100<br/>
                    Location: {r['Location']}<br/><br/>
                    {r.get('LLM Summary','')}
                  </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()

