"""Train EfficientNetV2 classifier for PyroSense AI.

Expected dataset layout (image folders):
  data/processed/classifier/train/fire/
  data/processed/classifier/train/smoke/
  data/processed/classifier/val/fire/
  data/processed/classifier/val/smoke/

Weights are saved to `models/weights/efficientnetv2.pt`.

Run:
  python training/train_efficientnet.py

Example:
    >>> from training.train_efficientnet import DEFAULT_CLASSES
    >>> DEFAULT_CLASSES
    ['fire', 'smoke']
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from utils.logger import logger

DEFAULT_CLASSES: List[str] = ["fire", "smoke"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed/classifier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
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
    data_dir = Path(args.data_dir)
    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=tfm)
    val_ds = datasets.ImageFolder(str(data_dir / "val"), transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch), shuffle=False, num_workers=0)

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=float(args.lr))

    best_acc = 0.0
    for epoch in range(int(args.epochs)):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        acc = float(correct / max(1, total))
        logger.info(f"epoch={epoch+1} train_loss={total_loss/max(1,len(train_loader)):.4f} val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            out = Path("models/weights")
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(out / "efficientnetv2.pt"))
            logger.info("Saved best EfficientNetV2 weights.")


if __name__ == "__main__":
    main()

