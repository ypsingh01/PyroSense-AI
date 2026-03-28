"""Launch the MLflow UI for PyroSense AI.

Run:
  python training/launch_mlflow.py

Example:
    >>> from training.launch_mlflow import mlflow_cmd
    >>> "mlflow ui" in mlflow_cmd("./mlruns")
    True
"""

from __future__ import annotations

import os
import subprocess
import sys

from config.settings import get_settings
from utils.logger import logger


def mlflow_cmd(tracking_uri: str) -> str:
    """Build the command string for MLflow UI."""

    return f"mlflow ui --backend-store-uri \"{tracking_uri}\" --host 0.0.0.0 --port 5000"


def main() -> None:
    settings = get_settings()
    cmd = mlflow_cmd(settings.mlflow_tracking_uri)
    logger.info(f"Starting MLflow UI with: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=False)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

