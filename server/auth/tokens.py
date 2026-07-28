"""App session tokens and the auth dependency.

Opaque bearer tokens, sha256-hashed at rest. Deliberately not JWTs: one indexed
lookup per request is nothing next to a ~2s voice turn, and it buys instant
server-side revocation — which matters in an app where someone may urgently need to
sign a device out of a shared household.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_db
from ..tables import AppSession, User

TOKEN_BYTES = 32
DEFAULT_TTL = timedelta(days=90)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def issue_token(db: AsyncSession, user: User, *, device_name: str | None = None,
                      ttl: timedelta = DEFAULT_TTL) -> str:
    """Mint a session token. The plaintext is returned once and never stored."""
    token = secrets.token_urlsafe(TOKEN_BYTES)
    db.add(AppSession(
        token_hash=_hash(token),
        user_id=user.id,
        device_name=device_name,
        expires_at=datetime.now(timezone.utc) + ttl,
    ))
    await db.flush()
    return token


async def revoke_token(db: AsyncSession, token: str) -> None:
    row = await db.get(AppSession, _hash(token))
    if row and row.revoked_at is None:
        row.revoked_at = datetime.now(timezone.utc)


async def revoke_all_for_user(db: AsyncSession, user_id) -> None:
    """Sign every device out. Used on account deletion and on unpair-for-safety."""
    rows = (await db.execute(
        select(AppSession).where(AppSession.user_id == user_id,
                                 AppSession.revoked_at.is_(None)))).scalars()
    now = datetime.now(timezone.utc)
    for r in rows:
        r.revoked_at = now


async def current_user(
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Resolve the caller, or 401.

    Attached at app level with an explicit allow-list, NOT per endpoint: the pre-auth
    service had four IDOR-able endpoints precisely because authentication was opt-in.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(401, "missing bearer token")

    row = await db.get(AppSession, _hash(token))
    now = datetime.now(timezone.utc)
    # One indistinguishable failure for unknown / revoked / expired: a caller should
    # not be able to tell a real-but-expired token from a fabricated one.
    if (row is None or row.revoked_at is not None
            or (row.expires_at is not None and row.expires_at < now)):
        raise HTTPException(401, "invalid or expired token")

    user = await db.get(User, row.user_id)
    if user is None or user.deleted_at is not None:
        raise HTTPException(401, "invalid or expired token")

    row.last_seen_at = now
    return user
