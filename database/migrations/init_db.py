"""Database schema creation script for PyroSense AI.

Run:
    python -m database.migrations.init_db

Example:
    >>> from database.migrations.init_db import init_db
    >>> init_db()
"""

from __future__ import annotations

from sqlalchemy.exc import SQLAlchemyError

from database.models import Base
from database.session import init_engine
from utils.logger import logger


def init_db() -> None:
    """Create database tables if they don't exist."""

    engine = init_engine()
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema ready.")
    except SQLAlchemyError as e:
        logger.error(f"Failed to initialize database schema: {e}")
        raise


if __name__ == "__main__":
    init_db()

