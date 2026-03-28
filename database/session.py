"""SQLAlchemy session factory for PyroSense AI.

Example:
    >>> from database.session import SessionLocal, init_engine
    >>> engine = init_engine()
    >>> db = SessionLocal()
    >>> db.close()
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from config.settings import get_settings
from utils.logger import logger

SessionLocal = sessionmaker(autocommit=False, autoflush=False)
_ENGINE: Engine | None = None


def init_engine() -> Engine:
    """Initialize and return a SQLAlchemy engine based on settings.

    Example:
        >>> from database.session import init_engine
        >>> _ = init_engine()
    """

    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    settings = get_settings()
    url = settings.database_url
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    try:
        _ENGINE = create_engine(url, echo=False, future=True, connect_args=connect_args)
        SessionLocal.configure(bind=_ENGINE)
        return _ENGINE
    except Exception as e:
        logger.error(f"Failed to initialize database engine: {e}")
        raise


# Bind the session factory early for convenience (tests/alerts can use DB without API startup).
try:
    init_engine()
except Exception:
    # DB might be misconfigured; callers will see errors when opening sessions.
    pass

