"""ONNX Runtime inference wrapper for edge deployment.

Example:
    >>> from models.onnx_inference import OnnxYoloRunner
    >>> _ = OnnxYoloRunner("models/weights/yolo.onnx")  # doctest: +ELLIPSIS
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from utils.image_utils import ensure_bgr_uint8
from utils.logger import logger


@dataclass
class OnnxDetection:
    """Simplified ONNX detection output."""

    boxes: List[Tuple[float, float, float, float]]
    scores: List[float]
    class_ids: List[int]


class OnnxYoloRunner:
    """Run an exported YOLO ONNX model via ONNX Runtime.

    Notes:
      - ONNX export formats can vary; this runner supports common YOLOv8 exports
        where output is Nx(4+num_classes) or similar. For demo robustness, if
        parsing fails we return empty detections instead of crashing.
    """

    def __init__(self, onnx_path: str, providers: List[str] | None = None) -> None:
        """Initialize ONNX session.

        Example:
            >>> from models.onnx_inference import OnnxYoloRunner
            >>> _ = OnnxYoloRunner("models/weights/yolo.onnx")  # doctest: +ELLIPSIS
        """

        self.onnx_path = str(onnx_path)
        p = Path(self.onnx_path)
        if not p.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError(f"onnxruntime is required: {e}") from e

        self.providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        logger.info(f"Loaded ONNX model: {self.onnx_path} providers={self.providers}")

    def _preprocess(self, image_bgr: np.ndarray, imgsz: int = 640) -> np.ndarray:
        try:
            import cv2
        except Exception as e:
            raise RuntimeError(f"OpenCV required for ONNX preprocessing: {e}") from e

        img = ensure_bgr_uint8(image_bgr)
        img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
        img = img[:, :, ::-1]  # BGR->RGB
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    def predict(self, image_bgr: np.ndarray, conf_threshold: float = 0.5) -> OnnxDetection:
        """Run ONNX inference and parse detections (best-effort).

        Example:
            >>> import numpy as np
            >>> from models.onnx_inference import OnnxDetection
            >>> isinstance(OnnxDetection([], [], []), OnnxDetection)
            True
        """

        x = self._preprocess(image_bgr)
        try:
            outputs = self.session.run(None, {self.input_name: x})
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            return OnnxDetection([], [], [])

        arr = None
        for o in outputs:
            if isinstance(o, np.ndarray):
                arr = o
                break
        if arr is None:
            return OnnxDetection([], [], [])

        # Try common shapes: (1, N, D) or (N, D)
        pred = arr.squeeze()
        if pred.ndim != 2 or pred.shape[1] < 6:
            logger.warning(f"Unexpected ONNX output shape: {arr.shape}")
            return OnnxDetection([], [], [])

        boxes: List[Tuple[float, float, float, float]] = []
        scores: List[float] = []
        class_ids: List[int] = []

        try:
            # Heuristic: if format is [x1,y1,x2,y2,score,cls]
            if pred.shape[1] == 6:
                for x1, y1, x2, y2, s, c in pred:
                    if float(s) < conf_threshold:
                        continue
                    boxes.append((float(x1), float(y1), float(x2), float(y2)))
                    scores.append(float(s))
                    class_ids.append(int(c))
            else:
                # [x1,y1,x2,y2,obj,cls1,cls2,...]
                for row in pred:
                    x1, y1, x2, y2 = row[:4]
                    obj = float(row[4])
                    cls_scores = row[5:]
                    ci = int(np.argmax(cls_scores))
                    s = float(obj * float(cls_scores[ci]))
                    if s < conf_threshold:
                        continue
                    boxes.append((float(x1), float(y1), float(x2), float(y2)))
                    scores.append(s)
                    class_ids.append(ci)
        except Exception as e:
            logger.warning(f"Failed to parse ONNX output: {e}")
            return OnnxDetection([], [], [])

        return OnnxDetection(boxes=boxes, scores=scores, class_ids=class_ids)

