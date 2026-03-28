"""Albumentations augmentation pipeline for PyroSense AI.

Includes a custom smoke simulation augmentation that overlays semi-transparent
smoke-like blobs to improve robustness.

Example:
    >>> import numpy as np
    >>> from training.augmentation import build_augmentation
    >>> aug = build_augmentation()
    >>> img = np.zeros((128,128,3), dtype=np.uint8)
    >>> out = aug(image=img)["image"]
    >>> out.shape
    (128, 128, 3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from utils.logger import logger


def _smoke_overlay(image: np.ndarray, intensity: float = 0.35, seed: int | None = None) -> np.ndarray:
    """Synthetic smoke overlay using blurred noise and alpha blending."""

    try:
        import cv2
    except Exception as e:
        logger.warning(f"OpenCV unavailable for smoke overlay: {e}")
        return image

    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    noise = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
    smoke = (noise * 255).astype(np.uint8)
    smoke_bgr = cv2.cvtColor(smoke, cv2.COLOR_GRAY2BGR)

    alpha = float(np.clip(intensity, 0.0, 0.9))
    out = cv2.addWeighted(smoke_bgr, alpha, image, 1.0 - alpha, 0.0)
    # Slight desaturation/whitening in smoky regions
    out = cv2.addWeighted(out, 0.85, np.full_like(out, 220), 0.15, 0.0)
    return out


def build_augmentation():
    """Return an Albumentations augmentation Compose."""

    try:
        import albumentations as A
    except Exception as e:
        raise RuntimeError(f"albumentations required: {e}") from e

    class SmokeSim(A.ImageOnlyTransform):
        def __init__(self, p: float = 0.35, intensity_range: Tuple[float, float] = (0.15, 0.5)) -> None:
            super().__init__(p=p)
            self.intensity_range = intensity_range

        def apply(self, img: np.ndarray, **params) -> np.ndarray:
            intensity = float(np.random.uniform(self.intensity_range[0], self.intensity_range[1]))
            return _smoke_overlay(img, intensity=intensity)

    return A.Compose(
        [
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.HorizontalFlip(p=0.5),
            A.ISONoise(p=0.25),
            SmokeSim(p=0.35),
        ]
    )

