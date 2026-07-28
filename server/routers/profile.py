"""
Profile and consent endpoints.

This is the tap surface, built before the voice intake on purpose: it is the fallback
for every extraction failure in Phase 3 and the permanent edit surface in Settings.
Voice fills this form; it does not replace it.

Like `auth.py`, this router must not import the ML holder — profile edits and consent
withdrawal have to work while the GPU pod is stopped.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth.tokens import current_user
from ..db import get_db
from ..tables import (
    Consent, ConsentKind, PacingPreference, Profile, Pronouns,
    RelationshipStatus, User,
)

router = APIRouter(prefix="/profile", tags=["profile"])

# Bumped when the wording changes. Stored per consent row so a later edit to the copy
# cannot retroactively rewrite what someone agreed to.
CONSENT_VERSION = "2026-07-28"

# The exact text shown to the user, hashed into each consent row. Keeping it
# server-side means the record is of what we *served*, not what a client claims.
CONSENT_TEXT: dict[ConsentKind, str] = {
    ConsentKind.ai_disclosure: (
        "WeMendAI is software, not a person, and not a therapist. "
        "I will never repeat anything to your partner unless you approve the exact "
        "words first."
    ),
    ConsentKind.relay_sharing: (
        "When you approve a message, the exact words you approved are delivered to "
        "your partner. Nothing else from your private sessions is ever shared."
    ),
    ConsentKind.tos: "I agree to the Terms of Service.",
    ConsentKind.privacy: "I agree to the Privacy Policy.",
}


def wording_hash(kind: ConsentKind) -> str:
    return hashlib.sha256(CONSENT_TEXT[kind].encode()).hexdigest()


# ────────────────────────────────── schemas ─────────────────────────────────
class ProfileOut(BaseModel):
    display_name: str | None
    pronouns: Pronouns
    relationship_status: RelationshipStatus
    together_months: int | None
    pacing_preference: PacingPreference
    goal_text: str | None
    onboarding_complete: bool
    # Per-field provenance from the voice intake, so the UI can show which values
    # came from speech (and may be wrong) versus a deliberate tap.
    extraction_meta: dict | None


class ProfilePatch(BaseModel):
    """All optional: the client sends only what changed.

    Every enum includes `not_stated`, which is what makes "prefer not to say" a real
    answer rather than a blank the model or the UI will try to fill in.
    """
    display_name: str | None = Field(default=None, max_length=120)
    pronouns: Pronouns | None = None
    relationship_status: RelationshipStatus | None = None
    together_months: int | None = Field(default=None, ge=0, le=1200)   # 100 years
    pacing_preference: PacingPreference | None = None
    goal_text: str | None = Field(default=None, max_length=2000)
    complete_onboarding: bool = False

    @field_validator("display_name", "goal_text")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None


class ConsentIn(BaseModel):
    kind: ConsentKind
    granted: bool = True


class ConsentOut(BaseModel):
    kind: ConsentKind
    version: str
    text: str
    granted_at: datetime | None
    withdrawn_at: datetime | None


# ─────────────────────────────────── routes ─────────────────────────────────
async def _get_or_create(db: AsyncSession, user: User) -> Profile:
    prof = await db.get(Profile, user.id)
    if prof is None:
        # A user can exist without a profile if sign-up was interrupted; don't 404 at
        # them for our own partial state.
        prof = Profile(user_id=user.id, display_name=user.display_name)
        db.add(prof)
        await db.flush()
    return prof


def _out(prof: Profile) -> ProfileOut:
    return ProfileOut(
        display_name=prof.display_name,
        pronouns=prof.pronouns,
        relationship_status=prof.relationship_status,
        together_months=prof.together_months,
        pacing_preference=prof.pacing_preference,
        goal_text=prof.goal_text,
        onboarding_complete=prof.onboarding_completed_at is not None,
        extraction_meta=prof.extraction_meta,
    )


@router.get("", response_model=ProfileOut)
async def get_profile(user: User = Depends(current_user),
                      db: AsyncSession = Depends(get_db)) -> ProfileOut:
    return _out(await _get_or_create(db, user))


@router.patch("", response_model=ProfileOut)
async def patch_profile(patch: ProfilePatch,
                        user: User = Depends(current_user),
                        db: AsyncSession = Depends(get_db)) -> ProfileOut:
    prof = await _get_or_create(db, user)

    for field in ("display_name", "pronouns", "relationship_status",
                  "together_months", "pacing_preference", "goal_text"):
        value = getattr(patch, field)
        if value is not None:
            setattr(prof, field, value)
            # A tap overrides whatever the voice intake guessed, and records that it
            # did — so a later screen doesn't present a hand-edited value as
            # model-extracted.
            if prof.extraction_meta and field in prof.extraction_meta:
                prof.extraction_meta = {
                    **prof.extraction_meta,
                    field: {"source": "user_edit", "confidence": 1.0},
                }

    if patch.complete_onboarding:
        # Require the disclosure consent before onboarding can be called complete:
        # "we told you this is software" is not optional.
        has_disclosure = (await db.execute(
            select(Consent).where(
                Consent.user_id == user.id,
                Consent.kind == ConsentKind.ai_disclosure,
                Consent.withdrawn_at.is_(None)))).scalar_one_or_none()
        if has_disclosure is None:
            raise HTTPException(409, "ai_disclosure consent required before completing onboarding")
        prof.onboarding_completed_at = datetime.now(timezone.utc)

    prof.updated_at = datetime.now(timezone.utc)
    await db.flush()
    return _out(prof)


@router.get("/consents", response_model=list[ConsentOut])
async def list_consents(user: User = Depends(current_user),
                        db: AsyncSession = Depends(get_db)) -> list[ConsentOut]:
    """Current state per kind, including the exact wording served.

    Returned so the app can show what was agreed to rather than a bare checkbox —
    and so withdrawal is a visible, available action.
    """
    rows = list((await db.execute(
        select(Consent).where(Consent.user_id == user.id)
        .order_by(Consent.granted_at.desc()))).scalars())
    latest: dict[ConsentKind, Consent] = {}
    for r in rows:
        latest.setdefault(r.kind, r)
    return [
        ConsentOut(kind=k, version=CONSENT_VERSION, text=CONSENT_TEXT[k],
                   granted_at=latest[k].granted_at if k in latest else None,
                   withdrawn_at=latest[k].withdrawn_at if k in latest else None)
        for k in ConsentKind
    ]


@router.post("/consents", response_model=ConsentOut)
async def set_consent(body: ConsentIn,
                      user: User = Depends(current_user),
                      db: AsyncSession = Depends(get_db)) -> ConsentOut:
    """Grant or withdraw one consent.

    Granular by design: one row per kind, so withdrawing relay_sharing does not
    disturb the disclosure record, and neither is bundled into a single "I accept".
    """
    existing = (await db.execute(
        select(Consent).where(Consent.user_id == user.id, Consent.kind == body.kind,
                              Consent.withdrawn_at.is_(None))
        .order_by(Consent.granted_at.desc()))).scalars().first()

    now = datetime.now(timezone.utc)
    if body.granted:
        if existing is not None and existing.version == CONSENT_VERSION:
            row = existing                       # already granted for this wording
        else:
            if existing is not None:
                # Wording changed: close the old record rather than mutate it, so the
                # audit trail shows what was agreed to and when.
                existing.withdrawn_at = now
            row = Consent(user_id=user.id, kind=body.kind, version=CONSENT_VERSION,
                          wording_hash=wording_hash(body.kind), method="tap")
            db.add(row)
            await db.flush()
    else:
        if existing is None:
            raise HTTPException(404, "no active consent of that kind to withdraw")
        existing.withdrawn_at = now
        row = existing

    return ConsentOut(kind=row.kind, version=row.version, text=CONSENT_TEXT[row.kind],
                      granted_at=row.granted_at, withdrawn_at=row.withdrawn_at)
