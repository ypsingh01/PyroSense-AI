"""FastAPI dependencies (DB session injection, auth hooks if needed).

Example:
    >>> from api.dependencies import get_db
    >>> gen = get_db()
    >>> next(gen) is not None
    True
"""

from __future__ import annotations

from typing import Generator

from database.session import SessionLocal


def get_db() -> Generator:
    """Yield a SQLAlchemy session for request lifetime."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

