"""LLM incident summarizer for PyroSense AI.

Supports:
  - Groq (cloud) llama3-8b-8192 via `groq` Python SDK
  - Ollama (local) via `ollama` Python client
  - Rule-based fallback when LLM is unavailable

Example:
    >>> from models.yolo_detector import DetectionResult
    >>> import numpy as np
    >>> from llm.incident_summarizer import IncidentSummarizer
    >>> dr = DetectionResult([], [], [], [], 0.0, np.zeros((10,10,3), dtype=np.uint8), np.zeros((10,10,3), dtype=np.uint8))
    >>> s = IncidentSummarizer()
    >>> txt = s.summarize(dr, location="Test Lab")
    >>> isinstance(txt, str)
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from config.settings import get_settings
from llm.prompts import INCIDENT_PROMPT_TEMPLATE
from models.yolo_detector import DetectionResult
from utils.logger import logger


def _region_from_bbox(bbox_xyxy: tuple[float, float, float, float], frame_shape: tuple[int, int, int]) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    h, w = frame_shape[:2]
    horiz = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
    vert = "upper" if cy < h / 3 else ("middle" if cy < 2 * h / 3 else "lower")
    return f"{vert}-{horiz}"


class IncidentSummarizer:
    """Generate incident summaries via Groq/Ollama with graceful fallback."""

    def __init__(self) -> None:
        """Initialize summarizer from settings.

        Example:
            >>> from llm.incident_summarizer import IncidentSummarizer
            >>> _ = IncidentSummarizer()
        """

        self.settings = get_settings()

    def summarize(self, detection_result: DetectionResult, location: str) -> str:
        """Summarize a detection into a 3-sentence incident report.

        Example:
            >>> import numpy as np
            >>> from models.yolo_detector import DetectionResult
            >>> from llm.incident_summarizer import IncidentSummarizer
            >>> dr = DetectionResult([(0,0,5,5)], [0.9], [0], ["fire"], 1.0, np.zeros((10,10,3), dtype=np.uint8), np.zeros((10,10,3), dtype=np.uint8))
            >>> txt = IncidentSummarizer().summarize(dr, "Warehouse A")
            >>> txt.count('.') >= 2
            True
        """

        ts = datetime.utcnow().isoformat()
        class_name = "none"
        conf = 0.0
        region_hint = "unknown"
        if detection_result.scores:
            i = int(np.argmax(detection_result.scores))
            class_name = detection_result.class_names[i]
            conf = float(detection_result.scores[i])
            region_hint = _region_from_bbox(detection_result.boxes[i], detection_result.frame.shape)

        prompt = INCIDENT_PROMPT_TEMPLATE.format(
            timestamp=ts,
            location=location,
            class_name=class_name,
            confidence_pct=conf * 100.0,
            region_hint=region_hint,
        )

        try:
            if self.settings.llm_provider == "groq":
                return self._summarize_groq(prompt)
            return self._summarize_ollama(prompt)
        except Exception as e:
            logger.warning(f"LLM summary unavailable, using fallback: {e}")
            return self._fallback_summary(ts=ts, location=location, class_name=class_name, conf=conf, region_hint=region_hint)

    def _summarize_groq(self, prompt: str) -> str:
        try:
            from groq import Groq
        except Exception as e:
            raise RuntimeError(f"Groq SDK not available: {e}") from e

        if not self.settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

        client = Groq(api_key=self.settings.groq_api_key)
        resp = client.chat.completions.create(
            model=self.settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=160,
        )
        text = (resp.choices[0].message.content or "").strip()
        return self._normalize_three_sentences(text)

    def _summarize_ollama(self, prompt: str) -> str:
        try:
            import ollama
        except Exception as e:
            raise RuntimeError(f"Ollama client not available: {e}") from e

        resp = ollama.chat(
            model=self.settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        text = (resp.get("message", {}).get("content") or "").strip()
        return self._normalize_three_sentences(text)

    def _normalize_three_sentences(self, text: str) -> str:
        # Ensure exactly 3 sentences; best-effort split on period.
        t = " ".join(text.replace("\n", " ").split())
        parts = [p.strip() for p in t.split(".") if p.strip()]
        if len(parts) >= 3:
            return ". ".join(parts[:3]) + "."
        if len(parts) == 2:
            return ". ".join(parts) + ". Please follow your site's emergency protocol immediately."
        if len(parts) == 1 and parts[0]:
            return parts[0] + ". Please verify the area and notify responders. Evacuate if conditions worsen."
        return "Fire/smoke incident detected. Please verify the area immediately. Follow your site's emergency response protocol."

    def _fallback_summary(self, *, ts: str, location: str, class_name: str, conf: float, region_hint: str) -> str:
        conf_pct = int(round(conf * 100.0))
        s1 = f"{class_name.title()} detected at {location} at {ts}. Confidence: {conf_pct}%."
        s2 = f"The strongest activation appears in the {region_hint} region of the frame, suggesting localized hazard cues."
        s3 = "Recommend immediate site verification and escalation to evacuation/response if confirmed."
        return f"{s1} {s2} {s3}"

