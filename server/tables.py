"""
Database schema.

Named `tables` rather than `models` because `app.py` already uses `Models` for the
ML model holder, and two different "models" in one service is a trap.

The access-control root is `MediationSession.owner_user_id`: exactly one owner per
session, always non-null. Everything private hangs off that.

**The privacy model is the turns/relays split, and it is structural.** A partner
cannot read your private session content because no query path exists that returns
`Turn` rows for a session you don't own — see `data.py`, where the accessor takes a
`user_id` and always filters on it, with no variant that doesn't. `Relay` is the only
table that deliberately crosses the couple boundary, and it carries only text the
speaker explicitly approved.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, Enum, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


# ──────────────────────────────── identity ─────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    # Apple's `sub`. THE identity key. Never key on email: it may be a
    # privaterelay.appleid.com address, may be hidden, and can change.
    apple_sub: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    email_relay: Mapped[str | None] = mapped_column(String(320))
    # Apple returns the full name only on the FIRST authorization, ever — not on
    # reinstall. Persist it there or it is gone permanently.
    display_name: Mapped[str | None] = mapped_column(String(120))
    locale: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = _now()
    # Soft-delete marker. Hard deletion of owned rows happens in the deletion flow;
    # this exists so an in-flight request can tell a deleted account apart from an
    # unknown one.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    apple_credential: Mapped[AppleCredential | None] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan")
    profile: Mapped[Profile | None] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan")


class AppleCredential(Base):
    """Refresh token from the authorization-code exchange.

    Required, not optional: Apple's /auth/revoke needs a refresh token, and the only
    way to obtain one is to exchange the `authorizationCode` at sign-up. Verifying
    just the identityToken leaves you unable to revoke — which fails App Review's
    account-deletion requirement (guideline 5.1.1(v)) at the worst moment.
    """
    __tablename__ = "apple_credentials"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    refresh_token_enc: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _now()
    rotated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="apple_credential")


class AppSession(Base):
    """Opaque bearer token, hashed at rest.

    Deliberately not a JWT pair. One indexed lookup per request is nothing next to a
    ~2s voice turn, and it buys instant server-side revocation — which matters a lot
    in an app where someone may urgently need to sign a device out.
    """
    __tablename__ = "app_sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)  # sha256 hex
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    device_name: Mapped[str | None] = mapped_column(String(120))
    created_at: Mapped[datetime] = _now()
    last_seen_at: Mapped[datetime] = _now()
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PushToken(Base):
    __tablename__ = "push_tokens"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    apns_token: Mapped[str] = mapped_column(String(200), nullable=False)
    environment: Mapped[str] = mapped_column(String(16), default="production")
    updated_at: Mapped[datetime] = _now()

    __table_args__ = (UniqueConstraint("user_id", "apns_token", name="uq_push_user_token"),)


# ───────────────────────────── profile & consent ────────────────────────────
class Pronouns(str, enum.Enum):
    he_him = "he_him"
    she_her = "she_her"
    they_them = "they_them"
    not_stated = "not_stated"


class RelationshipStatus(str, enum.Enum):
    married = "married"
    partners = "partners"
    dating = "dating"
    separated = "separated"
    other = "other"
    not_stated = "not_stated"


class PacingPreference(str, enum.Enum):
    """Replaces the cycle question for v1.

    Captures what cycle data would actually have been used for — how long to hold a
    cooling-off gap and how directly to push — without touching Article 9
    special-category health data, and without a gender gate.
    """
    needs_time = "needs_time"
    right_away = "right_away"
    depends = "depends"
    not_stated = "not_stated"


class Profile(Base):
    __tablename__ = "profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String(120))
    # Pronouns, not gender. Gender has no use in this product; pronouns do, because
    # the mediator speaks ABOUT you to your partner and rule 1 of
    # prompts/relay_distill.md depends on getting he/she right.
    pronouns: Mapped[Pronouns] = mapped_column(
        Enum(Pronouns, name="pronouns"), default=Pronouns.not_stated, nullable=False)
    relationship_status: Mapped[RelationshipStatus] = mapped_column(
        Enum(RelationshipStatus, name="relationship_status"),
        default=RelationshipStatus.not_stated, nullable=False)
    together_months: Mapped[int | None] = mapped_column(Integer)
    pacing_preference: Mapped[PacingPreference] = mapped_column(
        Enum(PacingPreference, name="pacing_preference"),
        default=PacingPreference.not_stated, nullable=False)
    goal_text: Mapped[str | None] = mapped_column(Text)
    onboarding_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Per-field provenance from the voice intake: {field: {source, confidence, quote}}.
    # Keeps the extraction audit trail without a table per field, and lets the UI show
    # which chips came from speech versus a tap.
    extraction_meta: Mapped[dict | None] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = _now()

    user: Mapped[User] = relationship(back_populates="profile")


class ConsentKind(str, enum.Enum):
    tos = "tos"
    privacy = "privacy"
    ai_disclosure = "ai_disclosure"
    relay_sharing = "relay_sharing"


class Consent(Base):
    """Granular rows, never one boolean.

    Each records the exact wording consented to (`wording_hash`), so a later change to
    the copy doesn't retroactively rewrite what someone agreed to. Separate rows are
    also what makes withdrawal of one consent possible without touching the others.
    """
    __tablename__ = "consents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    kind: Mapped[ConsentKind] = mapped_column(Enum(ConsentKind, name="consent_kind"), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    wording_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    granted_at: Mapped[datetime] = _now()
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    method: Mapped[str] = mapped_column(String(32), default="tap")


# ───────────────────────────── couple & pairing ─────────────────────────────
class CoupleStatus(str, enum.Enum):
    active = "active"
    dissolved = "dissolved"


class Couple(Base):
    __tablename__ = "couples"

    id: Mapped[uuid.UUID] = _uuid_pk()
    status: Mapped[CoupleStatus] = mapped_column(
        Enum(CoupleStatus, name="couple_status"), default=CoupleStatus.active, nullable=False)
    created_at: Mapped[datetime] = _now()
    dissolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    dissolved_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    cooldown_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CoupleMember(Base):
    __tablename__ = "couple_members"

    couple_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("couples.id", ondelete="CASCADE"), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    joined_at: Mapped[datetime] = _now()
    left_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # A user may have at most one ACTIVE couple. Partial index so historical
        # memberships (left_at set) don't block forming a new one.
        Index("uq_one_active_couple", "user_id", unique=True,
              postgresql_where=(left_at.is_(None))),
    )


class InviteState(str, enum.Enum):
    pending = "pending"
    accepted = "accepted"
    declined = "declined"
    expired = "expired"
    revoked = "revoked"


class Invite(Base):
    """Pairing invite. Stores a HASH of the code, never the code.

    A DB read shouldn't hand over live invites. Short TTL, single use, and an attempt
    counter because a 6-digit code is brute-forceable.
    """
    __tablename__ = "invites"

    id: Mapped[uuid.UUID] = _uuid_pk()
    code_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_by: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at: Mapped[datetime] = _now()
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    consumed_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PairingRequest(Base):
    """The invitee's explicit decision.

    Separate from `Invite` because entering a code must NOT itself constitute
    consent — the invitee sees what pairing means and then accepts or declines.
    """
    __tablename__ = "pairing_requests"

    id: Mapped[uuid.UUID] = _uuid_pk()
    invite_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("invites.id", ondelete="CASCADE"), nullable=False, index=True)
    invitee_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    state: Mapped[InviteState] = mapped_column(
        Enum(InviteState, name="invite_state"), default=InviteState.pending, nullable=False)
    created_at: Mapped[datetime] = _now()
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ──────────────────────────────── conversation ──────────────────────────────
class SessionKind(str, enum.Enum):
    intro = "intro"                    # the onboarding voice call
    private = "private"                # venting to the mediator
    relay_delivery = "relay_delivery"  # hearing the partner's approved message


class MediationSession(Base):
    __tablename__ = "mediation_sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    # The access-control root. Never null: every session has exactly one owner.
    owner_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    # Nullable ON PURPOSE: solo mode is a real mode, not a degraded state. Most
    # invites aren't accepted quickly, it's the only safe mode when abuse is flagged,
    # and App Review will test this app without a second human.
    couple_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("couples.id", ondelete="SET NULL"))
    kind: Mapped[SessionKind] = mapped_column(
        Enum(SessionKind, name="session_kind"), default=SessionKind.private, nullable=False)
    started_at: Mapped[datetime] = _now()
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TurnRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"


class Turn(Base):
    """A single utterance in a private session. **Owner-only, forever.**

    There is deliberately no `couple_id` here and no accessor that returns turns
    without a `user_id` filter. This is the table whose leakage would break the
    product's core promise, so the protection is structural rather than policy.
    """
    __tablename__ = "turns"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("mediation_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[TurnRole] = mapped_column(Enum(TurnRole, name="turn_role"), nullable=False)
    text_enc: Mapped[str] = mapped_column(Text, nullable=False)
    audio_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = _now()


class RelayState(str, enum.Enum):
    draft = "draft"
    approved = "approved"
    scheduled = "scheduled"
    delivered = "delivered"
    withdrawn = "withdrawn"


class Relay(Base):
    """The shuttle payload — **the only table that crosses the couple boundary.**

    `approved_text_enc` is populated only after the speaker approves the exact words;
    `draft_json_enc` never leaves the speaker. The listener's LLM context is built
    from `approved_text_enc` alone, never from the speaker's `Turn` rows.
    """
    __tablename__ = "relays"

    id: Mapped[uuid.UUID] = _uuid_pk()
    couple_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("couples.id", ondelete="CASCADE"), nullable=False, index=True)
    from_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    to_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    source_session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("mediation_sessions.id", ondelete="SET NULL"))
    draft_json_enc: Mapped[str | None] = mapped_column(Text)
    approved_text_enc: Mapped[str | None] = mapped_column(Text)
    state: Mapped[RelayState] = mapped_column(
        Enum(RelayState, name="relay_state"), default=RelayState.draft, nullable=False, index=True)
    # Computed by /distill. With real delivery this must BLOCK, not merely inform —
    # currently it is computed and ignored by the client.
    perspective_warning: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    abuse_flag: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deliver_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    audio_id: Mapped[str | None] = mapped_column(String(64))
    # Set when the author deletes their account: the words stay (the partner already
    # heard them) but the authorship is tombstoned.
    author_tombstoned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = _now()


class AudioAsset(Base):
    __tablename__ = "audio_assets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)   # uuid4().hex, as today
    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("mediation_sessions.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(32), default="reply")
    created_at: Mapped[datetime] = _now()
    # Retention: audio is deleted after transcription by default, per
    # docs/ARCHITECTURE.md. The reaper uses this.
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SafetyEvent(Base):
    """Abuse / self-harm indicators, for the human escalation path."""
    __tablename__ = "safety_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("mediation_sessions.id", ondelete="SET NULL"))
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    model_output: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _now()
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    action: Mapped[str | None] = mapped_column(String(64))
