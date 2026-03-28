"""Streamlit dashboard entrypoint for PyroSense AI.

Run:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import streamlit as st

# Ensure project root is importable when run via Streamlit.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import get_settings
from database.models import Detection
from database.session import SessionLocal


st.set_page_config(
    page_title="PyroSense AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path("dashboard/assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)



@st.cache_resource
def get_engine():
    """Cached inference engine for Streamlit."""

    from inference.detector import InferenceEngine

    return InferenceEngine()


@st.cache_resource
def get_summarizer():
    """Cached LLM summarizer."""

    from llm.incident_summarizer import IncidentSummarizer

    return IncidentSummarizer()


@st.cache_resource
def get_faiss():
    """Cached FAISS history index."""

    from llm.faiss_history import FaissHistory

    return FaissHistory()


def _get_today_stats() -> Dict[str, int]:
    """Return today's summary without requiring extra CRUD helpers."""

    try:
        now = datetime.now(timezone.utc)
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        with SessionLocal() as db:
            rows = db.query(Detection).filter(Detection.timestamp >= start).all()
        fire = sum(1 for r in rows if "fire" in (r.class_name or "").lower())
        smoke = sum(1 for r in rows if "smoke" in (r.class_name or "").lower())
        return {"total": len(rows), "fire": fire, "smoke": smoke, "alerts_sent": 0}
    except Exception:
        return {"total": 0, "fire": 0, "smoke": 0, "alerts_sent": 0}


# Sidebar — brand + navigation
with st.sidebar:
    st.markdown(
        """
    <div style="padding: 20px 0 30px; border-bottom: 1px solid rgba(255,255,255,0.06);">
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 20px; 
                  font-weight: 700; color: #FF4500; letter-spacing: -0.02em;">
        PYROSENSE
      </div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px;
                  color: #555C70; text-transform: uppercase; letter-spacing: 0.2em;
                  margin-top: 2px;">
        AI FIRE DETECTION SYSTEM
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # System status indicator
    st.markdown(
        """
    <div style="padding: 16px 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
      <div style="display: flex; align-items: center; gap: 8px; 
                  font-family: monospace; font-size: 11px; color: #00D46A;">
        <span class="status-dot active"></span>
        SYSTEM OPERATIONAL
      </div>
      <div style="font-family: monospace; font-size: 10px; color: #555C70; margin-top: 6px;">
        MODEL: YOLOv8s-FireSmoke v1.2
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Quick stats in sidebar
    try:
        stats = _get_today_stats()
    except Exception:
        stats = {"total": 0, "fire": 0, "smoke": 0, "alerts_sent": 0}

    st.markdown(
        f"""
    <div style="padding: 16px 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
      <div style="font-family: monospace; font-size: 10px; color: #555C70;
                  text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
        TODAY'S SUMMARY
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
        <div style="text-align:center; padding: 8px; background: rgba(255,69,0,0.08); 
                    border-radius: 8px; border: 1px solid rgba(255,69,0,0.2);">
          <div style="font-family: monospace; font-size: 18px; color: #FF4500; font-weight: 700;">
            {stats['fire']}
          </div>
          <div style="font-family: monospace; font-size: 9px; color: #555C70;">FIRE</div>
        </div>
        <div style="text-align:center; padding: 8px; background: rgba(139,155,180,0.08); 
                    border-radius: 8px; border: 1px solid rgba(139,155,180,0.2);">
          <div style="font-family: monospace; font-size: 18px; color: #8B9BB4; font-weight: 700;">
            {stats['smoke']}
          </div>
          <div style="font-family: monospace; font-size: 9px; color: #555C70;">SMOKE</div>
        </div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Alert channel status
    settings = get_settings()
    channels = {
        "Email": bool(settings.email_enabled),
        "Telegram": bool(settings.telegram_enabled),
        "Webhook": bool(settings.webhook_enabled),
    }
    channel_html = ""
    for name, enabled in channels.items():
        color = "#00D46A" if enabled else "#555C70"
        label = "ON" if enabled else "OFF"
        channel_html += f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding: 4px 0; font-family: monospace; font-size: 11px;">
          <span style="color: #8B92A5;">{name}</span>
          <span style="color: {color}; font-size: 10px;">{label}</span>
        </div>
      """
    st.markdown(
        f"""
    <div style="padding: 16px 0;">
      <div style="font-family: monospace; font-size: 10px; color: #555C70;
                  text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">
        ALERT CHANNELS
      </div>
      {channel_html}
    </div>
    """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
<div style="padding: 18px 0 8px;">
  <div style="font-family:'JetBrains Mono',monospace; font-size: 18px; color:#E8EAF0; font-weight:700;">
    PYROSENSE AI
  </div>
  <div style="font-family: monospace; font-size: 11px; color:#555C70; text-transform:uppercase; letter-spacing:0.1em;">
    Real-time fire & smoke detection command center
  </div>
</div>
""",
    unsafe_allow_html=True,
)

