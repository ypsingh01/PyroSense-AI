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
        risk_label, risk_color = "HIGH", "#E53E3E"
    elif risk >= 40:
        risk_label, risk_color = "MEDIUM", "#F59E0B"
    else:
        risk_label, risk_color = "LOW", "#10B981"

    badge_color = "#E53E3E" if cls == "fire" else "#4B5563"

    risk_fill_cls = 'critical' if risk >= 90 else 'high' if risk >= 70 else 'medium' if risk >= 40 else 'low'
    card_cls = 'pyro-card alert' if cls == 'fire' else 'pyro-card'
    summary_html = (
        f'<div style="font-family:monospace;font-size:12px;color:#4B5563;line-height:1.6;margin-bottom:12px;'
        f'padding:12px;background:rgba(0,0,0,0.06);border-radius:8px;border-left:3px solid {badge_color};">'
        f'{detection.get("llm_summary", "")}</div>'
    ) if detection.get('llm_summary') else ''

    st.markdown(
        f'<div class="{card_cls}">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;">'
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'<span class="detection-badge {cls}">{cls.upper()}</span>'
        f'<span style="font-family:monospace;font-size:13px;color:#111827;font-weight:600;">{conf*100:.1f}% CONFIDENCE</span>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="font-family:monospace;font-size:18px;color:{risk_color};font-weight:700;">{risk_label}</div>'
        f'<div style="font-family:monospace;font-size:10px;color:#9CA3AF;">RISK SCORE: {int(risk)}/100</div>'
        f'</div></div>'
        f'<div class="risk-bar" style="margin-bottom:16px;">'
        f'<div class="risk-fill {risk_fill_cls}" style="width:{min(100.0, max(0.0, risk))}%;"></div></div>'
        f'{summary_html}'
        f'<div style="font-family:monospace;font-size:10px;color:#9CA3AF;">'
        f'{detection.get("timestamp", "")} | BBox: {detection.get("bbox", "N/A")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

