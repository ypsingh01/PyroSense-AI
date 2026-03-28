"""CRUD helpers for PyroSense AI database.

Example:
    >>> from database.session import init_engine, SessionLocal
    >>> from database.migrations.init_db import init_db
    >>> _ = init_engine(); init_db()
    >>> db = SessionLocal()
    >>> db.close()
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from database.models import AlertLog, Detection, ModelRun


def create_detection(
    db: Session,
    *,
    timestamp: datetime,
    class_name: str,
    confidence: float,
    boxes_xyxy: List[Tuple[float, float, float, float]],
    frame_path: Optional[str],
    heatmap_path: Optional[str],
    llm_summary: Optional[str],
    source: str,
    risk_score: float,
) -> Detection:
    """Insert a detection record.

    Example:
        >>> from datetime import datetime
        >>> from database.session import init_engine, SessionLocal
        >>> from database.migrations.init_db import init_db
        >>> _ = init_engine(); init_db()
        >>> db = SessionLocal()
        >>> det = create_detection(db, timestamp=datetime.utcnow(), class_name="fire", confidence=0.9, boxes_xyxy=[(0,0,1,1)], frame_path=None, heatmap_path=None, llm_summary=None, source="test", risk_score=10.0)
        >>> det.id > 0
        True
        >>> db.close()
    """

    det = Detection(
        timestamp=timestamp,
        class_name=class_name,
        confidence=float(confidence),
        bbox_json=json.dumps(boxes_xyxy),
        frame_path=frame_path,
        heatmap_path=heatmap_path,
        llm_summary=llm_summary,
        source=source,
        risk_score=float(risk_score),
    )
    db.add(det)
    db.commit()
    db.refresh(det)
    return det


def list_detections(
    db: Session,
    *,
    offset: int = 0,
    limit: int = 50,
    class_name: Optional[str] = None,
    min_conf: Optional[float] = None,
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
) -> List[Detection]:
    """List detections with simple filtering and pagination."""

    stmt: Select[Detection] = select(Detection).order_by(Detection.timestamp.desc()).offset(int(offset)).limit(int(limit))
    if class_name:
        stmt = stmt.where(Detection.class_name == class_name)
    if min_conf is not None:
        stmt = stmt.where(Detection.confidence >= float(min_conf))
    if start_ts is not None:
        stmt = stmt.where(Detection.timestamp >= start_ts)
    if end_ts is not None:
        stmt = stmt.where(Detection.timestamp <= end_ts)
    return list(db.execute(stmt).scalars().all())


def get_detection(db: Session, detection_id: int) -> Optional[Detection]:
    """Get a detection by ID."""

    return db.get(Detection, int(detection_id))


def create_alert_log(
    db: Session,
    *,
    detection_id: int,
    channel: str,
    status: str,
    sent_at: Optional[datetime],
    error_msg: Optional[str],
) -> AlertLog:
    """Insert an alert delivery log."""

    log = AlertLog(
        detection_id=int(detection_id),
        channel=channel,
        status=status,
        sent_at=sent_at,
        error_msg=error_msg,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def create_model_run(db: Session, *, run_id: str, model_name: str, mAP50: Optional[float]) -> ModelRun:
    """Insert a model run record."""

    r = ModelRun(run_id=run_id, model_name=model_name, mAP50=mAP50)
    db.add(r)
    db.commit()
    db.refresh(r)
    return r


def latest_detection(db: Session) -> Optional[Detection]:
    """Return the most recent detection."""

    stmt = select(Detection).order_by(Detection.timestamp.desc()).limit(1)
    return db.execute(stmt).scalars().first()


def last_n_detections(db: Session, n: int = 5) -> List[Detection]:
    """Return last N detections."""

    stmt = select(Detection).order_by(Detection.timestamp.desc()).limit(int(n))
    return list(db.execute(stmt).scalars().all())

