"""Unit tests for inference engine."""

from __future__ import annotations

import numpy as np
from PIL import Image

from inference.detector import InferenceEngine
from utils.image_utils import pil_to_bgr


def test_detect_image_with_pil_image() -> None:
    """Test detect_image with a sample PIL image."""

    eng = InferenceEngine()
    pil = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB")
    frame = pil_to_bgr(pil)
    payload = eng.detect_image(frame)
    assert "primary_class" in payload
    assert "detections" in payload
    assert "risk" in payload

