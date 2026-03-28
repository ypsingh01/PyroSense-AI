"""Alert system tests with mocked channels."""

from __future__ import annotations

import asyncio

from alerts.alert_manager import AlertManager


def test_alert_manager_runs_without_enabled_channels(monkeypatch) -> None:
    """Ensure alert manager doesn't crash when channels disabled."""

    am = AlertManager()
    payload = {
        "primary_class": "fire",
        "ensemble_conf": 0.9,
        "risk": {"score": 80, "severity": "HIGH"},
        "timestamp": "2026-01-01T00:00:00Z",
        "llm_summary": "Test summary.",
        "detections": [],
        "inference_time_ms": 10.0,
    }

    asyncio.get_event_loop().run_until_complete(am.trigger_alert(payload, detection_id=1, location="Test"))

