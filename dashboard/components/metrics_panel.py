"""Mission-control styled metrics panel component."""

from __future__ import annotations

import streamlit as st


def render_metrics_panel(fps, detections, confidence, inference_ms):
    """Renders a 4-column dark metrics row."""

    cols = st.columns(4)
    metrics = [
        ("FPS", f"{float(fps):.1f}", "frames/sec"),
        ("DETECTIONS", str(int(detections)), "this session"),
        ("PEAK CONF", f"{float(confidence):.0%}", "highest seen"),
        ("LATENCY", f"{float(inference_ms):.0f}ms", "inference time"),
    ]
    for col, (label, value, sub) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
        <div style="background:#1A1D26; border:1px solid rgba(255,255,255,0.06); 
                    border-radius:12px; padding:16px; text-align:center;">
          <div style="font-family:monospace; font-size:10px; color:#555C70; 
                      text-transform:uppercase; letter-spacing:0.1em;">{label}</div>
          <div style="font-family:monospace; font-size:24px; color:#E8EAF0; 
                      font-weight:700; margin:4px 0;">{value}</div>
          <div style="font-family:monospace; font-size:10px; color:#555C70;">{sub}</div>
        </div>
        """,
                unsafe_allow_html=True,
            )

