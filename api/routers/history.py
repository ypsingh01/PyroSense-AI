"""Detection history endpoints for PyroSense AI API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.schemas import HistoryItem, HistoryResponse
from database import crud

router = APIRouter(tags=["history"])


@router.get("/history", response_model=HistoryResponse)
def history(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    """Paginated detection history."""

    rows = crud.list_detections(db, offset=offset, limit=limit)
    items = [
        HistoryItem(
            id=r.id,
            timestamp=r.timestamp,
            class_name=r.class_name,
            confidence=r.confidence,
            bbox_json=r.bbox_json,
            frame_path=r.frame_path,
            heatmap_path=r.heatmap_path,
            llm_summary=r.llm_summary,
            source=r.source,
            risk_score=r.risk_score,
        )
        for r in rows
    ]
    return HistoryResponse(offset=offset, limit=limit, items=items)


@router.get("/history/{detection_id}", response_model=HistoryItem)
def history_item(detection_id: int, db: Session = Depends(get_db)) -> HistoryItem:
    """Return a single detection by ID."""

    r = crud.get_detection(db, detection_id)
    if r is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Detection not found")
    return HistoryItem(
        id=r.id,
        timestamp=r.timestamp,
        class_name=r.class_name,
        confidence=r.confidence,
        bbox_json=r.bbox_json,
        frame_path=r.frame_path,
        heatmap_path=r.heatmap_path,
        llm_summary=r.llm_summary,
        source=r.source,
        risk_score=r.risk_score,
    )

