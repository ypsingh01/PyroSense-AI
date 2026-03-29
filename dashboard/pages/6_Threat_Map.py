"""Streamlit page: Zone-based threat map (mission-control UI)."""

from __future__ import annotations

import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import plotly.express as px
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from database.models import Detection
from database.session import SessionLocal


PLOTLY_DARK_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#F7F8FA",
        "plot_bgcolor": "#FFFFFF",
        "font": {"family": "JetBrains Mono", "color": "#4B5563", "size": 11},
        "xaxis": {"gridcolor": "rgba(0,0,0,0.06)", "linecolor": "rgba(0,0,0,0.08)"},
        "yaxis": {"gridcolor": "rgba(0,0,0,0.06)", "linecolor": "rgba(0,0,0,0.08)"},
        "colorway": ["#E53E3E", "#3B82F6", "#10B981", "#F59E0B"],
    }
}


ZONES_PATH = Path("data/processed/zones.json")


def _load_css() -> None:
    css_path = Path("dashboard/assets/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _load_zones() -> List[Dict[str, Any]]:
    if ZONES_PATH.exists():
        try:
            return json.loads(ZONES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    # default 16 zones
    zones = [{"name": f"Zone {i+1}", "camera_id": f"CAM-{i+1:02d}", "source": f"zone-{i+1}"} for i in range(16)]
    ZONES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ZONES_PATH.write_text(json.dumps(zones, indent=2), encoding="utf-8")
    return zones


def _save_zones(zones: List[Dict[str, Any]]) -> None:
    ZONES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ZONES_PATH.write_text(json.dumps(zones, indent=2), encoding="utf-8")


def _zone_status(z: Dict[str, Any], detections: List[Detection]) -> Dict[str, Any]:
    src = str(z.get("source", "")).lower()
    relevant = [d for d in detections if src and src in (d.source or "").lower()]
    last = relevant[0] if relevant else None
    if last is None:
        return {"status": "SAFE", "risk": 0.0, "class_name": "none", "time": "-"}
    cls = (last.class_name or "none").lower()
    risk = float(last.risk_score or 0.0)
    status = "SAFE"
    if "fire" in cls:
        status = "FIRE"
    elif "smoke" in cls:
        status = "SMOKE"
    return {"status": status, "risk": risk, "class_name": cls, "time": last.timestamp.isoformat()}


def main() -> None:
    try:
        st.set_page_config(page_title="PyroSense AI", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")
    except Exception:
        pass
    _load_css()

    # Auto-refresh every 10 seconds (JS click)
    st.markdown(
        """
<script>
setInterval(() => {
  const el = document.querySelector('[data-threat-refresh]');
  if (el) el.click();
}, 10000);
</script>
<button data-threat-refresh style="display:none" onclick="window.location.reload();">refresh</button>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="padding: 0 0 18px; border-bottom: 1px solid rgba(0,0,0,0.06); margin-bottom: 18px;">
      <div style="font-family:'JetBrains Mono',monospace; font-size:22px; color:#111827; font-weight:700;">
        THREAT MAP
      </div>
      <div style="font-family:monospace; font-size:11px; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.1em; margin-top:6px;">
        Zone-based risk overview (auto-refresh 10s)
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    zones = _load_zones()
    with st.expander("Add Zone", expanded=False):
        n = st.text_input("Zone name", value="")
        cam = st.text_input("Camera ID", value="")
        src = st.text_input("Source key (matches Detection.source)", value="")
        if st.button("Add Zone") and n and cam and src:
            zones.append({"name": n, "camera_id": cam, "source": src})
            zones = zones[:16] if len(zones) > 16 else zones
            _save_zones(zones)
            st.success("Zone added.")

    with SessionLocal() as db:
        dets = db.query(Detection).order_by(Detection.timestamp.desc()).limit(2000).all()

    # 4x4 grid of zone cards
    grid_cols = st.columns(4)
    statuses = []
    for i, z in enumerate(zones[:16]):
        s = _zone_status(z, dets)
        statuses.append(s)
        border = "#10B981"
        if s["status"] == "SMOKE":
            border = "#F59E0B"
        if s["status"] == "FIRE":
            border = "#E53E3E"
        with grid_cols[i % 4]:
            st.markdown(
                f"""
            <div class="pyro-card" style="border:1px solid rgba(0,0,0,0.06); border-left:3px solid {border};">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="font-family:monospace; font-size:12px; color:#111827; font-weight:700;">{z.get('name')}</div>
                <div style="font-family:monospace; font-size:10px; color:#9CA3AF;">{z.get('camera_id')}</div>
              </div>
              <div style="margin-top:10px; font-family:monospace; font-size:10px; color:#4B5563;">
                STATUS: <b>{s['status']}</b><br/>
                RISK: <b>{float(s['risk']):.0f}</b><br/>
                LAST: {str(s['time'])[-19:]}
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            if st.button("Details", key=f"zd_{i}"):
                st.session_state._zone_detail = {"zone": z, "status": s}

    # Inline detail panel
    detail = st.session_state.get("_zone_detail")
    if detail:
        z = detail["zone"]
        s = detail["status"]
        st.markdown(
            f"""
        <div class="pyro-card alert">
          <div style="font-family:monospace; font-size:12px; color:#111827; font-weight:800;">
            {z.get('name')} — DETAIL
          </div>
          <div style="font-family:monospace; font-size:11px; color:#4B5563; margin-top:10px; line-height:1.7;">
            Camera: {z.get('camera_id')}<br/>
            Source: {z.get('source')}<br/>
            Status: {s.get('status')}<br/>
            Risk: {float(s.get('risk',0)):.0f}/100<br/>
            Last event: {s.get('time')}
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Historical risk heatmap (zones x time bucket)
    st.markdown("### Historical risk heatmap (zones)")
    names = [z.get("name", f"Zone {i+1}") for i, z in enumerate(zones[:16])]
    matrix = np.zeros((4, 4), dtype=float)
    for i, s in enumerate(statuses[:16]):
        matrix[i // 4, i % 4] = float(s.get("risk", 0.0))
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=["#FFFFFF", "#E53E3E"], title="Zone risk matrix (current)")
    fig.update_layout(template=PLOTLY_DARK_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

