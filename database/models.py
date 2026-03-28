"""SQLAlchemy ORM models for PyroSense AI.

Tables:
  - Detection: each incident with paths to snapshot/heatmap and LLM summary
  - AlertLog: delivery status per channel for a detection
  - ModelRun: MLflow run metadata for trained models

Example:
    >>> from database.models import Base
    >>> Base.metadata.tables.keys()  # doctest: +ELLIPSIS
    dict_keys([...])
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for ORM models."""


class Detection(Base):
    """Detection event with persisted artifacts."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    class_name: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    bbox_json: Mapped[str] = mapped_column(Text)  # JSON string for list of boxes
    frame_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    heatmap_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    llm_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="unknown", index=True)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)

    alerts: Mapped[list["AlertLog"]] = relationship(back_populates="detection", cascade="all, delete-orphan")


class AlertLog(Base):
    """Alert delivery log for a detection."""

    __tablename__ = "alert_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    detection_id: Mapped[int] = mapped_column(ForeignKey("detections.id"), index=True)
    channel: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending/sent/failed/skipped
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_msg: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    detection: Mapped["Detection"] = relationship(back_populates="alerts")


class ModelRun(Base):
    """MLflow model training run metadata."""

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    mAP50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)

