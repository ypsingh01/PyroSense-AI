"""Pydantic request/response schemas for PyroSense AI API.

Example:
    >>> from api.schemas import HealthResponse
    >>> HealthResponse(status="ok", db_ok=True, model_ok=True, device="cpu", gpu=None).status
    'ok'
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    bbox_xyxy: List[float] = Field(..., min_length=4, max_length=4)
    score: float
    class_id: int
    class_name: str


class RiskInfo(BaseModel):
    score: float
    severity: str


class DetectResponse(BaseModel):
    timestamp: str
    primary_class: str
    yolo_conf: float
    clf_conf: float
    ensemble_conf: float
    risk: RiskInfo
    detections: List[DetectionItem]
    inference_time_ms: float
    snapshot_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    llm_summary: Optional[str] = None
    faiss_similar: List[Dict[str, Any]] = Field(default_factory=list)
    detection_id: Optional[int] = None


class StreamFrameRequest(BaseModel):
    frame_b64: str
    location: str = "Unknown"


class StreamFrameResponse(BaseModel):
    detections: DetectResponse


class HistoryItem(BaseModel):
    id: int
    timestamp: datetime
    class_name: str
    confidence: float
    bbox_json: str
    frame_path: Optional[str]
    heatmap_path: Optional[str]
    llm_summary: Optional[str]
    source: str
    risk_score: float


class HistoryResponse(BaseModel):
    offset: int
    limit: int
    items: List[HistoryItem]


class HealthResponse(BaseModel):
    status: str
    db_ok: bool
    model_ok: bool
    device: str
    gpu: Optional[Dict[str, Any]] = None

