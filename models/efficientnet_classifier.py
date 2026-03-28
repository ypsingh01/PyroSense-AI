"""EfficientNetV2 secondary classifier for PyroSense AI.

This classifier acts as a secondary confidence model to verify YOLO detections.
For a hackathon-friendly demo, it can run with ImageNet-pretrained weights
and treat "fire/smoke" as an out-of-distribution proxy using a small head.

In production, train `training/train_efficientnet.py` and place weights at
`models/weights/efficientnetv2.pt`.

Example:
    >>> import numpy as np
    >>> from models.efficientnet_classifier import EfficientNetV2Classifier
    >>> clf = EfficientNetV2Classifier(device="cpu")
    >>> img = np.zeros((224,224,3), dtype=np.uint8)
    >>> out = clf.predict_proba(img)
    >>> sorted(out.keys())
    ['fire', 'smoke']
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from utils.image_utils import bgr_to_pil, ensure_bgr_uint8
from utils.logger import logger


@dataclass
class ClassifierOutput:
    """Classifier output probabilities for fire and smoke."""

    fire: float
    smoke: float

    def as_dict(self) -> Dict[str, float]:
        return {"fire": float(self.fire), "smoke": float(self.smoke)}


class EfficientNetV2Classifier:
    """PyTorch EfficientNetV2 classifier with a 2-class head."""

    def __init__(self, weights_path: str = "models/weights/efficientnetv2.pt", device: str = "auto") -> None:
        """Initialize classifier.

        Example:
            >>> from models.efficientnet_classifier import EfficientNetV2Classifier
            >>> _ = EfficientNetV2Classifier(device="cpu")
        """

        self.weights_path = str(weights_path)
        self.device = device
        self._model, self._preprocess, self._device = self._load()

    def _resolve_device(self):
        import torch

        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.device)

    def _load(self):
        try:
            import torch
            import torch.nn as nn
            from torchvision import transforms
            from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        except Exception as e:
            logger.error(f"PyTorch/torchvision required for EfficientNetV2: {e}")
            raise

        device = self._resolve_device()
        base = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

        # Replace classifier head to 2 classes (fire, smoke)
        in_features = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_features, 2)

        weights = Path(self.weights_path)
        if weights.exists():
            try:
                sd = torch.load(weights, map_location="cpu")
                base.load_state_dict(sd, strict=False)
                logger.info(f"Loaded EfficientNetV2 weights from {self.weights_path}")
            except Exception as e:
                logger.warning(f"Failed to load EfficientNetV2 weights, using demo head: {e}")
        else:
            logger.warning(f"EfficientNetV2 weights not found at '{self.weights_path}'. Using demo (untrained) head.")

        base.eval()
        base.to(device)

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return base, preprocess, device

    def predict_proba(self, image_bgr: np.ndarray) -> Dict[str, float]:
        """Return probabilities for fire and smoke.

        Example:
            >>> import numpy as np
            >>> from models.efficientnet_classifier import EfficientNetV2Classifier
            >>> c = EfficientNetV2Classifier(device="cpu")
            >>> c.predict_proba(np.zeros((224,224,3), dtype=np.uint8))["fire"] >= 0.0
            True
        """

        image_bgr = ensure_bgr_uint8(image_bgr)
        pil = bgr_to_pil(image_bgr)
        try:
            import torch
        except Exception as e:
            raise RuntimeError(f"PyTorch is required: {e}") from e

        x = self._preprocess(pil).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().reshape(-1)
        fire_p = float(probs[0])
        smoke_p = float(probs[1])
        return ClassifierOutput(fire=fire_p, smoke=smoke_p).as_dict()

    def get_info(self) -> Dict[str, object]:
        """Return model metadata.

        Example:
            >>> from models.efficientnet_classifier import EfficientNetV2Classifier
            >>> EfficientNetV2Classifier(device="cpu").get_info()["classes"]
            ['fire', 'smoke']
        """

        return {"weights": self.weights_path, "device": str(self._device), "classes": ["fire", "smoke"]}

