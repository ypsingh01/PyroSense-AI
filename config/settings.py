"""Typed application settings for PyroSense AI.

This module centralizes configuration for the API, dashboard, models, alerts,
LLM provider, and MLflow. Values are loaded from environment variables (e.g.
via a `.env` file) using `pydantic-settings`.

Example:
    >>> from config.settings import get_settings
    >>> s = get_settings()
    >>> s.conf_threshold
    0.5
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, HttpUrl, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Model
    # Default to the pretrained fire/smoke weights (auto-downloaded if missing).
    yolo_model_path: str = Field(default="models/weights/fire_smoke_yolov8.pt", alias="YOLO_MODEL_PATH")
    conf_threshold: float = Field(default=0.25, alias="CONF_THRESHOLD", ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, alias="IOU_THRESHOLD", ge=0.0, le=1.0)
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(default="auto", alias="DEVICE")

    # Database
    database_url: str = Field(default="sqlite:///./pyrosense.db", alias="DATABASE_URL")

    # LLM
    llm_provider: Literal["groq", "ollama"] = Field(default="groq", alias="LLM_PROVIDER")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama3-8b-8192", alias="GROQ_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3", alias="OLLAMA_MODEL")

    # Alerts
    alert_cooldown_seconds: PositiveInt = Field(default=60, alias="ALERT_COOLDOWN_SECONDS")
    email_enabled: bool = Field(default=False, alias="EMAIL_ENABLED")
    email_smtp_host: str = Field(default="smtp.gmail.com", alias="EMAIL_SMTP_HOST")
    email_smtp_port: int = Field(default=587, alias="EMAIL_SMTP_PORT", ge=1, le=65535)
    email_user: Optional[str] = Field(default=None, alias="EMAIL_USER")
    email_password: Optional[str] = Field(default=None, alias="EMAIL_PASSWORD")
    email_recipient: Optional[str] = Field(default=None, alias="EMAIL_RECIPIENT")

    telegram_enabled: bool = Field(default=False, alias="TELEGRAM_ENABLED")
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_CHAT_ID")

    webhook_enabled: bool = Field(default=False, alias="WEBHOOK_ENABLED")
    webhook_url: Optional[HttpUrl] = Field(default=None, alias="WEBHOOK_URL")

    # MLflow
    mlflow_tracking_uri: str = Field(default="./mlruns", alias="MLFLOW_TRACKING_URI")

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data")
    snapshots_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "processed" / "snapshots")
    heatmaps_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "processed" / "heatmaps")
    faiss_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "processed" / "faiss")

    def ensure_dirs(self) -> None:
        """Create directories required for runtime artifacts.

        Example:
            >>> from config.settings import get_settings
            >>> get_settings().ensure_dirs()
        """

        for p in (self.snapshots_dir, self.heatmaps_dir, self.faiss_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance.

    Example:
        >>> from config.settings import get_settings
        >>> s1 = get_settings()
        >>> s2 = get_settings()
        >>> s1 is s2
        True
    """

    s = Settings()
    s.ensure_dirs()
    return s

