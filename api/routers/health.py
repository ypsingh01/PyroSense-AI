"""Health and metrics endpoints for PyroSense AI API."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.schemas import HealthResponse
from config.settings import get_settings
from database.session import init_engine
from utils.logger import logger

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    """Return model status, DB status, and GPU info."""

    settings = get_settings()
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        logger.warning(f"DB health check failed: {e}")
        db_ok = False

    model_ok = True
    device = settings.device
    try:
        from inference.detector import InferenceEngine

        _ = InferenceEngine()
    except Exception as e:
        logger.warning(f"Model health check failed: {e}")
        model_ok = False

    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            }
            device = "cuda"
    except Exception:
        pass

    return HealthResponse(status="ok" if (db_ok and model_ok) else "degraded", db_ok=db_ok, model_ok=model_ok, device=device, gpu=gpu)


@router.get("/metrics")
def metrics() -> dict:
    """Basic metrics endpoint (extendable)."""

    return {"service": "pyrosense-api", "version": "1.0.0"}

