"""API endpoint tests for PyroSense AI."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "db_ok" in data


def test_detect_endpoint_with_test_image() -> None:
    client = TestClient(app)
    img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    files = {"file": ("test.jpg", buf.getvalue(), "image/jpeg")}
    data = {"location": "UnitTest", "source": "test"}
    r = client.post("/api/v1/detect", files=files, data=data)
    assert r.status_code == 200
    payload = r.json()
    assert "detections" in payload
    assert "risk" in payload

