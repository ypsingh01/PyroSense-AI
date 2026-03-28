"""Main inference engine for PyroSense AI (image/video/stream).

This module orchestrates:
  - YOLOv8 detection
  - EfficientNetV2 secondary confidence
  - Weighted ensemble scoring
  - Risk score calculation

Example:
    >>> import numpy as np
    >>> from inference.detector import InferenceEngine
    >>> eng = InferenceEngine()
    >>> res = eng.detect_image(np.zeros((128,128,3), dtype=np.uint8))
    >>> "risk" in res
    True
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from config.settings import get_settings
from models.efficientnet_classifier import EfficientNetV2Classifier
from models.ensemble import compute_risk_score, smoke_presence_from_classes, weighted_score
from models.yolo_detector import DetectionResult, YOLODetector
from utils.image_utils import ensure_bgr_uint8
from utils.logger import logger


@dataclass
class EnsembleResult:
    """Enriched inference result with ensemble score and risk."""

    detection: DetectionResult
    primary_class: str
    yolo_conf: float
    clf_conf: float
    ensemble_conf: float
    risk_score: float
    risk_severity: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "primary_class": self.primary_class,
            "yolo_conf": float(self.yolo_conf),
            "clf_conf": float(self.clf_conf),
            "ensemble_conf": float(self.ensemble_conf),
            "risk": {"score": float(self.risk_score), "severity": self.risk_severity},
            "detections": [
                {"bbox_xyxy": list(b), "score": float(s), "class_id": int(ci), "class_name": cn}
                for b, s, ci, cn in zip(self.detection.boxes, self.detection.scores, self.detection.class_ids, self.detection.class_names)
            ],
            "inference_time_ms": float(self.detection.inference_time_ms),
        }


class InferenceEngine:
    """High-level inference engine used by API and dashboard."""

    VALID_DETECTION_CLASSES = frozenset(["fire", "smoke", "flame"])

    def __init__(self) -> None:
        """Initialize models from settings.

        Example:
            >>> from inference.detector import InferenceEngine
            >>> _ = InferenceEngine()
        """

        s = get_settings()
        self.settings = s
        self.yolo = YOLODetector(
            model_path=s.yolo_model_path,
            device=s.device,
            conf_threshold=s.conf_threshold,
            iou_threshold=s.iou_threshold,
        )
        self.clf = EfficientNetV2Classifier(device=s.device)
        self._prev_area_ratio: Optional[float] = None

    def _filter_fire_smoke_only(self, results: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Hard filter: discard any detection not related to fire or smoke."""

        filtered: List[Dict[str, object]] = []
        for det in results:
            cls_name = str(det.get("class_name", "")).lower()
            if any(v in cls_name for v in self.VALID_DETECTION_CLASSES):
                filtered.append(det)
        if len(filtered) < len(results):
            dropped = len(results) - len(filtered)
            logger.warning(f"Filtered out {dropped} non-fire/smoke detections")
        return filtered

    def detect_image(self, image_bgr: np.ndarray) -> Dict[str, object]:
        """Detect fire/smoke on an image and compute ensemble + risk."""

        image_bgr = ensure_bgr_uint8(image_bgr)
        det = self.yolo.detect_image(image_bgr)
        if not det.scores:
            result = EnsembleResult(
                detection=det,
                primary_class="none",
                yolo_conf=0.0,
                clf_conf=0.0,
                ensemble_conf=0.0,
                risk_score=0.0,
                risk_severity="LOW",
                timestamp=datetime.utcnow(),
            ).to_dict()
            result["_annotated_frame"] = det.annotated_frame
            return result

        # Primary = highest YOLO confidence
        idx = int(np.argmax(det.scores))
        primary_class = det.class_names[idx]
        yolo_conf = float(det.scores[idx])

        # Secondary classifier: estimate probability for the same class
        probs = self.clf.predict_proba(det.frame)
        clf_conf = float(probs.get("fire" if "fire" in primary_class.lower() else "smoke", 0.0))

        ens = weighted_score(yolo_conf, clf_conf)

        bbox = det.boxes[idx]
        area_ratio = float(np.clip((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / max(1.0, det.frame.shape[0] * det.frame.shape[1]), 0.0, 1.0))
        growth = 0.0
        if self._prev_area_ratio is not None:
            growth = float(np.clip(area_ratio - self._prev_area_ratio, 0.0, 1.0))
        self._prev_area_ratio = area_ratio
        smoke_presence = smoke_presence_from_classes(det.class_names)

        risk = compute_risk_score(
            confidence=float(np.clip(ens, 0.0, 1.0)),
            bbox_xyxy=bbox,
            frame_shape=det.frame.shape,
            growth_rate=growth,
            smoke_presence=smoke_presence,
        )

        out = EnsembleResult(
            detection=det,
            primary_class=primary_class,
            yolo_conf=yolo_conf,
            clf_conf=clf_conf,
            ensemble_conf=ens,
            risk_score=risk.score,
            risk_severity=risk.severity,
            timestamp=datetime.utcnow(),
        ).to_dict()

        # Hard filter: fire/smoke/flame only
        dets = list(out.get("detections", []))  # type: ignore[assignment]
        if isinstance(dets, list):
            filtered = self._filter_fire_smoke_only([d for d in dets if isinstance(d, dict)])
            out["detections"] = filtered
            if not filtered:
                out["primary_class"] = "none"
                out["yolo_conf"] = 0.0
                out["clf_conf"] = 0.0
                out["ensemble_conf"] = 0.0
                out["risk"] = {"score": 0.0, "severity": "LOW"}
        out["_annotated_frame"] = det.annotated_frame
        return out

    def stream(self, source: Union[int, str], *, demo_mode: bool = False) -> Generator[Dict[str, object], None, None]:
        """Yield inference dicts from a stream source.

        Example:
            >>> from inference.detector import InferenceEngine
            >>> eng = InferenceEngine()
            >>> gen = eng.stream(0, demo_mode=True)
            >>> hasattr(gen, "__iter__")
            True
        """

        from inference.stream_processor import StreamProcessor

        sp = StreamProcessor(samples_dir=str(self.settings.data_dir / "samples"))
        for sf in sp.frames(source, demo_mode=demo_mode):
            yield self.detect_image(sf.frame_bgr)

    def detect_stream(self, source: Union[int, str], *, demo_mode: bool = False) -> Generator[Dict[str, object], None, None]:
        """Alias for stream() with fire/smoke filtering."""

        yield from self.stream(source, demo_mode=demo_mode)

