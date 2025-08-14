"""
Shared database connection and session management for Benjamin AI System
"""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool


# Database configuration
# Use only service-specific variables (principle of least privilege)
DATABASE_URL = (
    os.environ.get("DATA_APP_DATABASE_URL")
    or os.environ.get("AGENTIC_APP_DATABASE_URL")
)
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not configured. Set a service-specific variable: DATA_APP_DATABASE_URL or AGENTIC_APP_DATABASE_URL."
    )

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session.
    Used with FastAPI Depends().
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Use for standalone database operations.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection() -> bool:
    """Test database connectivity"""
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def create_all_tables():
    """Create all tables defined in models"""
    from . import models  # Import models module
    Base.metadata.create_all(bind=engine)


def ensure_column_exists(table_name: str, column_name: str, column_sql_type: str) -> None:
    """Ensure a column exists on a table (PostgreSQL). Safe to call repeatedly.

    Example: ensure_column_exists('garmin_health', 'raw_training_status', 'JSON')
    """
    try:
        from sqlalchemy import text
        ddl = text(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {column_name} {column_sql_type}")
        with engine.connect() as conn:
            conn.execute(ddl)
            conn.commit()
    except Exception:
        # Intentionally swallow errors to avoid breaking runtime; caller can log
        pass


def drop_all_tables():
    """Drop all tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)
