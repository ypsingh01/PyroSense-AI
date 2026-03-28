"""Mission-control styled detection card component."""

from __future__ import annotations

import streamlit as st


def render_detection_card(detection: dict, show_heatmap_btn: bool = True):
    """
    Renders a single detection result as a styled dark card.
    detection keys: class_name, confidence, risk_score, timestamp,
                    frame_path, llm_summary, bbox
    """

    cls = str(detection.get("class_name", "none")).lower()
    conf = float(detection.get("confidence", 0.0))
    risk = float(detection.get("risk_score", 0.0))

    # Risk level label
    if risk >= 90:
        risk_label, risk_color = "CRITICAL", "#FF0000"
    elif risk >= 70:
        risk_label, risk_color = "HIGH", "#FF4500"
    elif risk >= 40:
        risk_label, risk_color = "MEDIUM", "#FFB800"
    else:
        risk_label, risk_color = "LOW", "#00D46A"

    badge_color = "#FF4500" if cls == "fire" else "#8B9BB4"

    st.markdown(
        f"""
    <div class="pyro-card {'alert' if cls == 'fire' else ''}">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px;">
        <div style="display:flex; align-items:center; gap:12px;">
          <span class="detection-badge {cls}">{cls.upper()}</span>
          <span style="font-family:monospace; font-size:13px; color:#E8EAF0; font-weight:600;">
            {conf*100:.1f}% CONFIDENCE
          </span>
        </div>
        <div style="text-align:right;">
          <div style="font-family:monospace; font-size:18px; color:{risk_color}; font-weight:700;">
            {risk_label}
          </div>
          <div style="font-family:monospace; font-size:10px; color:#555C70;">
            RISK SCORE: {int(risk)}/100
          </div>
        </div>
      </div>

      <div class="risk-bar" style="margin-bottom:16px;">
        <div class="risk-fill {'critical' if risk>=90 else 'high' if risk>=70 else 'medium' if risk>=40 else 'low'}"
             style="width:{min(100.0, max(0.0, risk))}%;"></div>
      </div>

      {f'<div style="font-family:monospace; font-size:12px; color:#8B92A5; line-height:1.6; margin-bottom:12px; padding:12px; background:rgba(255,255,255,0.03); border-radius:8px; border-left:3px solid {badge_color};">{detection.get("llm_summary", "")}</div>' if detection.get('llm_summary') else ''}

      <div style="font-family:monospace; font-size:10px; color:#555C70;">
        {detection.get('timestamp', '')} | BBox: {detection.get('bbox', 'N/A')}
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

