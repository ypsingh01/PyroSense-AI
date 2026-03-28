"""Visualization helpers: bounding boxes, overlays, and color coding.

Example:
    >>> import numpy as np
    >>> from utils.visualization import draw_boxes
    >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
    >>> out = draw_boxes(img, [(10,10,30,30)], [0.9], ["fire"])
    >>> out.shape
    (100, 100, 3)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from utils.image_utils import ensure_bgr_uint8
from utils.logger import logger


CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "fire": (0, 69, 255),  # BGR OrangeRed
    "smoke": (169, 169, 169),  # BGR DarkGray
    "flame": (0, 100, 255),  # BGR Orange
}


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """Return BGR color for a detection class.

    Example:
        >>> get_class_color("fire")
        (0, 69, 255)
    """

    key = class_name.lower()
    return CLASS_COLORS.get(key, (0, 255, 0))  # Green fallback


def draw_detections(
    image_bgr: np.ndarray,
    boxes_xyxy: Sequence[Tuple[float, float, float, float]],
    scores: Sequence[float],
    class_names: Sequence[str],
) -> np.ndarray:
    """Draw bounding boxes and labels on an image with fire/smoke styling.

    Example:
        >>> import numpy as np
        >>> img = np.zeros((64,64,3), dtype=np.uint8)
        >>> draw_detections(img, [(1,1,10,10)], [0.5], ["smoke"]).shape
        (64, 64, 3)
    """

    image_bgr = ensure_bgr_uint8(image_bgr).copy()
    try:
        import cv2
    except Exception as e:
        logger.error(f"OpenCV unavailable for drawing: {e}")
        return image_bgr

    for (x1, y1, x2, y2), score, cname in zip(boxes_xyxy, scores, class_names):
        c = get_class_color(cname)
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        thickness = 3 if "fire" in cname.lower() else 2
        cv2.rectangle(image_bgr, p1, p2, c, thickness)
        label = f"{cname} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_bgr, (p1[0], max(0, p1[1] - th - 6)), (p1[0] + tw + 6, p1[1]), c, -1)
        cv2.putText(image_bgr, label, (p1[0] + 3, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Pulsing red border overlay when FIRE (not smoke) is detected
        if "fire" in cname.lower() and float(score) > 0.7:
            cv2.rectangle(
                image_bgr,
                (max(0, p1[0] - 4), max(0, p1[1] - 4)),
                (min(image_bgr.shape[1] - 1, p2[0] + 4), min(image_bgr.shape[0] - 1, p2[1] + 4)),
                (0, 0, 255),
                2,
            )
    return image_bgr


def draw_boxes(
    image_bgr: np.ndarray,
    boxes_xyxy: Sequence[Tuple[float, float, float, float]],
    scores: Sequence[float],
    class_names: Sequence[str],
) -> np.ndarray:
    """Backward-compatible alias for `draw_detections`."""

    return draw_detections(image_bgr, boxes_xyxy, scores, class_names)


def concat_horiz(images: Iterable[np.ndarray]) -> np.ndarray:
    """Concatenate images horizontally (auto-resize to same height).

    Example:
        >>> import numpy as np
        >>> a = np.zeros((10,20,3), dtype=np.uint8)
        >>> b = np.zeros((20,20,3), dtype=np.uint8)
        >>> out = concat_horiz([a,b])
        >>> out.shape[0]
        10
    """

    imgs: List[np.ndarray] = [ensure_bgr_uint8(i) for i in images]
    if not imgs:
        raise ValueError("No images provided")
    try:
        import cv2
    except Exception as e:
        logger.error(f"OpenCV unavailable for concat: {e}")
        return np.concatenate(imgs, axis=1)

    target_h = min(i.shape[0] for i in imgs)
    resized = []
    for im in imgs:
        if im.shape[0] == target_h:
            resized.append(im)
        else:
            scale = target_h / float(im.shape[0])
            new_w = int(im.shape[1] * scale)
            resized.append(cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_AREA))
    return np.concatenate(resized, axis=1)


def severity_from_risk(risk_score: float) -> str:
    """Convert risk score into severity band.

    Example:
        >>> severity_from_risk(90)
        'CRITICAL'
    """

    if risk_score >= 90:
        return "CRITICAL"
    if risk_score >= 70:
        return "HIGH"
    if risk_score >= 40:
        return "MEDIUM"
    return "LOW"

