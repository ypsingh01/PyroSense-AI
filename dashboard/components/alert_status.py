"""Alert channel status indicators for Streamlit."""

from __future__ import annotations

import streamlit as st

from config.settings import get_settings


def render_alert_status() -> None:
    """Render enabled/disabled badges for alert channels."""

    s = get_settings()
    st.sidebar.markdown("### Alert channels")
    st.sidebar.write(f"**Email**: {'✓' if s.email_enabled else '✗'}")
    st.sidebar.write(f"**Telegram**: {'✓' if s.telegram_enabled else '✗'}")
    st.sidebar.write(f"**Webhook**: {'✓' if s.webhook_enabled else '✗'}")

