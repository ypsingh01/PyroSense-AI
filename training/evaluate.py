"""Evaluation utilities for PyroSense AI models.

This script provides quick evaluation helpers (precision/recall summary and
confusion matrix) for the classifier dataset. YOLO evaluation is typically
handled by Ultralytics during training.

Run:
  python training/evaluate.py --data_dir data/processed/classifier/val

Example:
    >>> from training.evaluate import compute_confusion_matrix
    >>> compute_confusion_matrix([0,1,1],[0,1,0])
    [[1, 0], [1, 1]]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from utils.logger import logger


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> List[List[int]]:
    """Compute 2x2 confusion matrix for binary classification."""

    tn = fp = fn = tp = 0
    for t, p in zip(y_true, y_pred):
        if t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tp += 1
    return [[tn, fp], [fn, tp]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed/classifier/val")
    parser.add_argument("--weights", type=str, default="models/weights/efficientnetv2.pt")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    except Exception as e:
        raise RuntimeError(f"PyTorch/torchvision required: {e}") from e

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = datasets.ImageFolder(args.data_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    w = Path(args.weights)
    if w.exists():
        model.load_state_dict(torch.load(str(w), map_location="cpu"), strict=False)
    model.to(device).eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y.cpu().tolist())

    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))

    logger.info(f"Confusion matrix [[tn,fp],[fn,tp]] = {cm}")
    logger.info(f"precision={precision:.3f} recall={recall:.3f} acc={acc:.3f}")


if __name__ == "__main__":
    main()

