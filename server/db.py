"""Async engine + session factory.

Postgres, not SQLite: the DB must NOT live on /workspace (MooseFS). Advisory locks
and WAL shared-memory don't behave over a network filesystem, and we already have
three incidents from that mount — see docs/LATENCY.md.

In production this runs on the always-on control plane, separate from the GPU pod,
so sign-in and account deletion keep working while the pod is stopped.
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+asyncpg://localhost/wemend_dev")

engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # the control plane may idle for hours between sessions
    echo=bool(os.environ.get("SQL_ECHO")),
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency. Commits on success, rolls back on any exception."""
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
