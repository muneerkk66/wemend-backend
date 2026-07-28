"""Data access for owned content.

Every accessor here takes a `user_id` and filters on it. **There is deliberately no
variant that omits the filter**, because that absence is the privacy model: one
partner cannot read the other's private transcript since no code path can express it.

If you ever need "all turns for a session", add the owner check to the new function
too — do not add an unfiltered helper "just for admin".
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .tables import MediationSession, Relay, Turn


async def get_owned_session(db: AsyncSession, session_id: uuid.UUID,
                            user_id: uuid.UUID) -> MediationSession | None:
    """A session, only if this user owns it. Returns None rather than raising so
    callers can decide between 403 and 404 (we use 403 everywhere: a 404 would
    confirm which session ids exist)."""
    return (await db.execute(
        select(MediationSession).where(
            MediationSession.id == session_id,
            MediationSession.owner_user_id == user_id,
        ))).scalar_one_or_none()


async def get_turns(db: AsyncSession, session_id: uuid.UUID,
                    user_id: uuid.UUID, limit: int = 8) -> list[Turn]:
    """Recent turns for a session THIS USER OWNS.

    The join to mediation_sessions is what enforces ownership. Querying `turns` by
    session_id alone would be the bug that leaks a partner's private words.
    """
    return list((await db.execute(
        select(Turn)
        .join(MediationSession, Turn.session_id == MediationSession.id)
        .where(Turn.session_id == session_id,
               MediationSession.owner_user_id == user_id)
        .order_by(Turn.created_at.desc())
        .limit(limit)
    )).scalars())[::-1]


async def get_inbox(db: AsyncSession, user_id: uuid.UUID) -> list[Relay]:
    """Relays addressed TO this user and actually delivered/approved.

    Note it filters on `to_user_id`, never `couple_id`: membership in a couple does
    not entitle you to a relay you are not the recipient of.
    """
    return list((await db.execute(
        select(Relay)
        .where(Relay.to_user_id == user_id,
               Relay.state.in_(("approved", "scheduled", "delivered")))
        .order_by(Relay.created_at.desc())
    )).scalars())
