"""Raspberry Pi 4 edge deployment script for PyroSense AI (ONNX Runtime).

Features:
  - Picamera2 capture (if available) with OpenCV fallback
  - ONNX Runtime inference via `models/onnx_inference.py`
  - Sends detections to central server via HTTP (webhook-style)
  - Optional local display overlay

Run (Pi):
  python edge_deploy/raspberry_pi.py --onnx models/weights/best.onnx --server http://<server-ip>:8000 --source picamera

Example:
    >>> from edge_deploy.raspberry_pi import parse_args
    >>> a = parse_args([])
    >>> hasattr(a, "server")
    True
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests

from models.onnx_inference import OnnxYoloRunner
from utils.logger import logger


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="models/weights/best.onnx")
    parser.add_argument("--server", type=str, required=True, help="Central server base URL (e.g., http://192.168.1.10:8000)")
    parser.add_argument("--source", type=str, default="picamera", choices=["picamera", "opencv"])
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--location", type=str, default="Edge-RPi")
    return parser.parse_args(argv)


def _read_frame_picamera() -> Optional[np.ndarray]:
    try:
        from picamera2 import Picamera2
    except Exception as e:
        logger.warning(f"Picamera2 not available: {e}")
        return None

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={"size": (640, 480)}))
    cam.start()
    time.sleep(0.5)
    try:
        frame = cam.capture_array()
        # frame is RGB; convert to BGR
        return frame[:, :, ::-1].copy()
    finally:
        cam.stop()


def _loop_opencv(index: int):
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open OpenCV camera.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    runner = OnnxYoloRunner(args.onnx)
    endpoint = args.server.rstrip("/") + "/api/v1/detect"

    if args.source == "picamera":
        # Continuous capture loop with picamera2 is environment-specific; we do per-iteration init for robustness.
        def frames():
            while True:
                f = _read_frame_picamera()
                if f is None:
                    logger.warning("Falling back to OpenCV capture.")
                    yield from _loop_opencv(args.camera_index)
                    return
                yield f

        frame_iter = frames()
    else:
        frame_iter = _loop_opencv(args.camera_index)

    try:
        import cv2
    except Exception:
        cv2 = None

    for frame in frame_iter:
        det = runner.predict(frame, conf_threshold=float(args.conf))
        if det.scores:
            top_i = int(np.argmax(det.scores))
            payload = {
                "location": args.location,
                "primary_class_id": int(det.class_ids[top_i]),
                "confidence": float(det.scores[top_i]),
                "bbox_xyxy": list(det.boxes[top_i]),
                "timestamp": time.time(),
            }
            try:
                # Send as webhook-style JSON to central server (users can adapt endpoint).
                requests.post(args.server.rstrip("/") + "/api/v1/health", timeout=2)  # keepalive check
                requests.post(args.server.rstrip("/") + "/webhook", json=payload, timeout=4)
            except Exception as e:
                logger.warning(f"Failed to send to server: {e}")

            if args.display and cv2 is not None:
                x1, y1, x2, y2 = map(int, det.boxes[top_i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ALERT {det.scores[top_i]:.2f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("PyroSense Edge", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            if args.display and cv2 is not None:
                cv2.imshow("PyroSense Edge", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        time.sleep(0.05)

    if args.display:
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()

