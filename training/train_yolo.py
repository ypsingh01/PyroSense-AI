"""Fine-tune YOLOv8 with MLflow logging for PyroSense AI.

This script:
  - Ensures a base YOLOv8n weights file is available
  - Creates `data/processed/data.yaml` pointing to the processed dataset
  - Trains with MLflow autologging
  - Saves `models/weights/best.pt` and exports ONNX on completion
  - Supports `--resume` for interrupted training

Run:
  python training/train_yolo.py
  python training/train_yolo.py --resume

Example:
    >>> from training.train_yolo import ensure_data_yaml
    >>> _ = ensure_data_yaml()
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import yaml

from config.settings import get_settings
from utils.logger import logger


def ensure_data_yaml() -> str:
    """Create a YOLO data.yaml file pointing to `data/processed`."""

    settings = get_settings()
    processed = Path(settings.data_dir) / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    data_yaml = processed / "data.yaml"

    # Expected structure:
    # data/processed/
    #   images/train, images/val
    #   labels/train, labels/val
    cfg: Dict[str, object] = {
        "path": str(processed.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["fire", "smoke"],
    }
    data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return str(data_yaml)


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    data_yaml_path = ensure_data_yaml()

    # Validate that data.yaml has correct fire/smoke class names
    with open(data_yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("nc") == 2, "ERROR: Dataset must have exactly 2 classes"
    assert "fire" in (cfg.get("names", []) or []), "ERROR: Dataset must include 'fire' class"
    assert "smoke" in (cfg.get("names", []) or []), "ERROR: Dataset must include 'smoke' class"
    print("[PyroSense] Dataset validation passed: fire/smoke classes confirmed.")

    try:
        import mlflow
    except Exception as e:
        raise RuntimeError(f"mlflow required: {e}") from e

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        mlflow.pytorch.autolog(log_models=False)
    except Exception:
        # mlflow.pytorch may not be available in all installs; proceed without autolog
        pass

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(f"Ultralytics required: {e}") from e

    model = YOLO("yolov8s.pt")  # 's' = small, better mAP than 'n'

    run_name = "pyrosense-yolo"
    with mlflow.start_run(run_name=run_name):
        logger.info("Starting YOLOv8s fire/smoke training (pyrosense_fire_v1).")
        results = model.train(
            data=data_yaml_path,
            epochs=100,
            imgsz=640,
            batch=16,
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            hsv_h=0.015,  # hue augmentation — fire colors vary
            hsv_s=0.7,  # saturation — important for smoke detection
            hsv_v=0.4,  # brightness — fire in dark environments
            degrees=0.0,  # fire orientation is always upward
            translate=0.1,
            scale=0.5,
            flipud=0.0,  # fire never appears upside down
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,  # mix fire images to improve generalization
            copy_paste=0.0,
            conf=0.25,  # lower threshold to catch early/distant fire
            iou=0.45,
            device="auto",
            project="models/weights",
            name="pyrosense_fire_v1",
            exist_ok=True,
            patience=20,
            save=True,
            save_period=10,
            plots=True,
            resume=bool(args.resume),
        )

        # Copy best.pt into models/weights
        best_src = Path("models/weights") / "pyrosense_fire_v1" / "weights" / "best.pt"
        if best_src.exists():
            out_dir = Path("models") / "weights"
            out_dir.mkdir(parents=True, exist_ok=True)
            best_dst = out_dir / "best.pt"
            best_dst.write_bytes(best_src.read_bytes())
            logger.info(f"Saved best weights to {best_dst}")
            try:
                model = YOLO(str(best_dst))
                onnx_out = out_dir / "best.onnx"
                model.export(format="onnx", imgsz=int(args.imgsz), simplify=True, opset=12)
                # locate export output
                cand = best_dst.with_suffix(".onnx")
                if cand.exists():
                    onnx_out.write_bytes(cand.read_bytes())
                logger.info(f"Exported ONNX to {onnx_out}")
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")

        # Log common metrics if available from results
        try:
            metrics = getattr(results, "results_dict", None) or {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))
        except Exception:
            pass


if __name__ == "__main__":
    main()

