"""Dataset downloader for PyroSense AI (real fire/smoke datasets).

Provides:
  - D-Fire (YOLO format, 0=fire, 1=smoke)
  - Kaggle fire dataset (requires Kaggle CLI credentials)
  - A `--dataset dfire-mini` mode that prepares a ~2000-image subset for quick training.

Run:
  python data/download_datasets.py --dataset dfire
  python data/download_datasets.py --dataset kaggle
  python data/download_datasets.py --dataset dfire-mini

Example:
    >>> from data.download_datasets import download_dfire
    >>> isinstance(download_dfire.__name__, str)
    True
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import tempfile
import zipfile
import urllib.request
from pathlib import Path
from typing import Iterable, List, Tuple


def ensure_demo_samples(output_dir: str = "data/samples") -> None:
    """Generate synthetic demo samples for Streamlit demo mode."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = list(out.glob("*.jpg")) + list(out.glob("*.png"))
    if len(existing) >= 5:
        return

    try:
        from PIL import Image, ImageDraw, ImageFilter
    except Exception:
        return

    for i in range(5):
        img = Image.new("RGB", (640, 480), (10, 12, 18))
        d = ImageDraw.Draw(img)
        # smoke gradients
        for k in range(10):
            x0 = 60 + i * 20 + k * 18
            y0 = 80 + k * 10
            x1 = x0 + 260
            y1 = y0 + 120
            shade = 150 + k * 5
            d.ellipse((x0, y0, x1, y1), fill=(shade, shade, shade))
        # flame blobs
        for k in range(6):
            x = 280 + (i * 25) + k * 18
            y = 260 - k * 25
            d.ellipse((x - 40, y - 60, x + 40, y + 60), fill=(255, 90 + k * 10, 0))
            d.ellipse((x - 25, y - 35, x + 25, y + 35), fill=(255, 200, 40))
        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
        img.save(out / f"sample_{i+1}.jpg", quality=92)


def download_dfire(output_dir: str = "data/raw/dfire") -> None:
    """
    Downloads the D-Fire dataset — 21,527 images of fire/smoke.
    Source: https://github.com/gaiasd/DFireDataset
    Labels: 0=fire, 1=smoke (already in YOLO format)
    """

    os.makedirs(output_dir, exist_ok=True)
    # Clone repository metadata (paper/utilities/links)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth=1",
            "https://github.com/gaiasd/DFireDataset.git",
            output_dir,
        ],
        check=False,
    )

    # If the repo clone does not contain the dataset files (common), download the official archive.
    # Source link from the official README (OneDrive): "D-Fire dataset (only images and labels)".
    # We then create train/test splits locally.
    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        dataset_url = "https://1drv.ms/u/c/c0bd25b6b048b01d/EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw?download=1"
        print(f"[PyroSense] Downloading D-Fire images/labels from {dataset_url} ...")
        with tempfile.TemporaryDirectory() as td:
            zpath = Path(td) / "dfire.zip"
            try:
                urllib.request.urlretrieve(dataset_url, str(zpath))
            except Exception as e:
                print(f"[PyroSense] Download failed (may require manual download): {e}")
            if zpath.exists() and zpath.stat().st_size > 0:
                try:
                    with zipfile.ZipFile(str(zpath), "r") as zf:
                        zf.extractall(output_dir)
                except Exception as e:
                    print(f"[PyroSense] Failed to extract D-Fire zip: {e}")

    # Attempt to locate extracted images/labels directories (some archives nest paths).
    if not images_dir.exists():
        candidates = list(Path(output_dir).rglob("images"))
        for c in candidates:
            if (c / "..").resolve() == Path(output_dir).resolve():
                continue
        if candidates:
            images_dir = candidates[0]
    if not labels_dir.exists():
        candidates = list(Path(output_dir).rglob("labels"))
        if candidates:
            labels_dir = candidates[0]

    # Create a train/test split if necessary.
    if images_dir.exists() and labels_dir.exists() and not (Path(output_dir) / "train").exists():
        pairs = _iter_pairs(images_dir, labels_dir)
        if pairs:
            random.seed(1337)
            random.shuffle(pairs)
            n_train = int(len(pairs) * 0.8)
            train_pairs = pairs[:n_train]
            test_pairs = pairs[n_train:]

            for split, sel in [("train", train_pairs), ("test", test_pairs)]:
                (Path(output_dir) / split / "images").mkdir(parents=True, exist_ok=True)
                (Path(output_dir) / split / "labels").mkdir(parents=True, exist_ok=True)
                for img, lab in sel:
                    shutil.copy2(img, Path(output_dir) / split / "images" / img.name)
                    shutil.copy2(lab, Path(output_dir) / split / "labels" / lab.name)

    # Create data.yaml for training
    yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: train/images
val: test/images
nc: 2
names: ['fire', 'smoke']
"""
    with open(f"{output_dir}/data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[PyroSense] D-Fire dataset ready at {output_dir}")


def download_kaggle_fire(output_dir: str = "data/raw/kaggle_fire") -> None:
    """
    Downloads fire detection dataset from Kaggle.
    Dataset: phylake1337/fire-dataset
    Run: pip install kaggle && kaggle datasets download phylake1337/fire-dataset
    """

    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "phylake1337/fire-dataset", "-p", output_dir, "--unzip"],
        check=False,
    )
    print(f"[PyroSense] Kaggle dataset extracted at {output_dir}")


def _iter_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    imgs: List[Path] = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    pairs: List[Tuple[Path, Path]] = []
    for img in imgs:
        lab = labels_dir / (img.stem + ".txt")
        if lab.exists():
            pairs.append((img, lab))
    return pairs


def _generate_synthetic_fire_smoke_yolo(out_dir: str, n_images: int = 80) -> None:
    """Create a small YOLO-format synthetic dataset (fire/smoke) for offline fallback."""

    out = Path(out_dir)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image, ImageDraw, ImageFilter
    except Exception:
        print("[PyroSense] WARNING: Pillow not available; cannot generate synthetic dataset.")
        return

    import math

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    # 80/20 train/val
    n_train = int(n_images * 0.8)
    for i in range(n_images):
        is_train = i < n_train
        split = "train" if is_train else "val"
        w, h = 640, 480
        img = Image.new("RGB", (w, h), (10, 12, 18))
        d = ImageDraw.Draw(img)

        # Fire blob region
        fx = 320 + int(80 * math.sin(i))
        fy = 330
        fw = 140
        fh = 180
        for k in range(7):
            ox = int((k - 3) * 10)
            oy = int((3 - k) * 14)
            d.ellipse((fx - fw // 2 + ox, fy - fh // 2 + oy, fx + fw // 2 + ox, fy + fh // 2 + oy), fill=(255, 70 + k * 15, 0))
        d.ellipse((fx - 45, fy - 75, fx + 45, fy + 25), fill=(255, 210, 60))

        # Smoke region
        sx = 180 + (i % 8) * 35
        sy = 150
        sw = 260
        sh = 140
        for k in range(10):
            shade = 130 + k * 7
            d.ellipse((sx + k * 10, sy + k * 6, sx + sw + k * 6, sy + sh + k * 4), fill=(shade, shade, shade))

        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

        stem = f"synth_{i:04d}"
        (out / "images" / split / f"{stem}.jpg").parent.mkdir(parents=True, exist_ok=True)
        img.save(out / "images" / split / f"{stem}.jpg", quality=90)

        # Labels: class x_center y_center width height (normalized)
        # Fire bbox
        fx1, fy1 = fx - fw // 2, fy - fh // 2
        fx2, fy2 = fx + fw // 2, fy + fh // 2
        # Smoke bbox
        sx1, sy1 = sx, sy
        sx2, sy2 = sx + sw, sy + sh

        def yolo_line(cls: int, x1: float, y1: float, x2: float, y2: float) -> str:
            cx = clamp01(((x1 + x2) / 2) / w)
            cy = clamp01(((y1 + y2) / 2) / h)
            bw = clamp01((x2 - x1) / w)
            bh = clamp01((y2 - y1) / h)
            return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

        label_lines = [
            yolo_line(0, fx1, fy1, fx2, fy2),  # fire
            yolo_line(1, sx1, sy1, sx2, sy2),  # smoke
        ]
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    yaml_content = f"""
path: {os.path.abspath(str(out))}
train: images/train
val: images/val
nc: 2
names: ['fire', 'smoke']
"""
    (out / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"[PyroSense] Synthetic YOLO dataset ready at {out_dir} (n={n_images})")


def prepare_dfire_mini(raw_dir: str = "data/raw/dfire", out_dir: str = "data/processed/dfire", n_images: int = 2000) -> None:
    """Prepare a ~2000-image subset (train/val) for quick YOLO training."""

    raw = Path(raw_dir)
    if not raw.exists():
        download_dfire(raw_dir)

    # Ensure full dfire has splits; if not, try to create via download_dfire()
    if not (raw / "train" / "images").exists():
        download_dfire(str(raw))

    train_pairs = _iter_pairs(raw / "train" / "images", raw / "train" / "labels")
    val_pairs = _iter_pairs(raw / "test" / "images", raw / "test" / "labels")

    if not train_pairs or not val_pairs:
        print(
            f"[PyroSense] WARNING: D-Fire structure not ready in {raw_dir}. "
            "Download the official D-Fire archive from the project README and place images/labels under data/raw/dfire/, "
            "then re-run with `python data/download_datasets.py --dataset dfire-mini`."
        )
        # Offline fallback: synthesize a small labeled dataset so the app can self-train and become functional.
        _generate_synthetic_fire_smoke_yolo(out_dir, n_images=80)
        return

    random.seed(1337)
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    n_train = int(n_images * 0.8)
    n_val = int(n_images - n_train)
    train_sel = train_pairs[:n_train]
    val_sel = val_pairs[: min(n_val, len(val_pairs))]

    out = Path(out_dir)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    def copy_pairs(pairs: List[Tuple[Path, Path]], split: str) -> None:
        for img, lab in pairs:
            shutil.copy2(img, out / "images" / split / img.name)
            shutil.copy2(lab, out / "labels" / split / lab.name)

    copy_pairs(train_sel, "train")
    copy_pairs(val_sel, "val")

    yaml_content = f"""
path: {os.path.abspath(str(out))}
train: images/train
val: images/val
nc: 2
names: ['fire', 'smoke']
"""
    (out / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"[PyroSense] D-Fire mini dataset ready at {out_dir} (train={len(train_sel)} val={len(val_sel)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dfire", choices=["dfire", "kaggle", "dfire-mini", "all"])
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    ensure_demo_samples()

    if args.dataset in {"dfire", "all"}:
        download_dfire(args.output_dir or "data/raw/dfire")
    if args.dataset in {"kaggle", "all"}:
        download_kaggle_fire(args.output_dir or "data/raw/kaggle_fire")
    if args.dataset == "dfire-mini":
        try:
            prepare_dfire_mini(raw_dir="data/raw/dfire", out_dir="data/processed/dfire", n_images=2000)
        except Exception as e:
            print(f"[PyroSense] WARNING: dfire-mini preparation failed: {e}")


if __name__ == "__main__":
    main()

