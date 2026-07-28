"""
Phase 1 tests against a real Postgres.

    createdb wemend_test
    DATABASE_URL=postgresql+asyncpg://localhost/wemend_test .venv/bin/pytest tests -q

The important one is `test_partner_cannot_read_your_turns`. Everything else is
plumbing; that test is the product promise, and it should fail loudly if anyone ever
adds an unfiltered accessor.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from server import data
from server.auth import tokens
from server.tables import (
    AppSession, Base, Couple, CoupleMember, MediationSession, Profile, Relay,
    RelayState, SessionKind, Turn, TurnRole, User,
)

TEST_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://localhost/wemend_test")


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine(TEST_URL)
    async with engine.begin() as c:
        await c.run_sync(Base.metadata.drop_all)
        await c.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


async def _user(db, sub: str, name: str) -> User:
    u = User(apple_sub=sub, display_name=name)
    db.add(u)
    await db.flush()
    db.add(Profile(user_id=u.id, display_name=name))
    await db.flush()
    return u


# ─────────────────────────── the one that matters ───────────────────────────
@pytest.mark.asyncio
async def test_partner_cannot_read_your_turns(db):
    """A paired partner must not be able to read your private session content.

    This is the product's core promise. If someone adds an accessor that queries
    `turns` without an owner filter, this test is what catches it.
    """
    adam = await _user(db, "sub-adam", "Adam")
    sara = await _user(db, "sub-sara", "Sara")

    # Pair them — being in a couple must NOT grant read access.
    couple = Couple()
    db.add(couple)
    await db.flush()
    db.add_all([CoupleMember(couple_id=couple.id, user_id=adam.id),
                CoupleMember(couple_id=couple.id, user_id=sara.id)])

    s = MediationSession(owner_user_id=adam.id, couple_id=couple.id,
                         kind=SessionKind.private)
    db.add(s)
    await db.flush()
    db.add(Turn(session_id=s.id, role=TurnRole.user,
                text_enc="something Adam would never want repeated verbatim"))
    await db.flush()

    # Adam sees his own words.
    assert len(await data.get_turns(db, s.id, adam.id)) == 1
    # Sara, his paired partner, sees nothing — despite sharing the couple.
    assert await data.get_turns(db, s.id, sara.id) == []
    # And she cannot even resolve the session.
    assert await data.get_owned_session(db, s.id, sara.id) is None


@pytest.mark.asyncio
async def test_inbox_filters_on_recipient_not_couple(db):
    """Couple membership must not entitle you to a relay you aren't addressed in."""
    adam = await _user(db, "sub-a", "Adam")
    sara = await _user(db, "sub-b", "Sara")
    couple = Couple()
    db.add(couple)
    await db.flush()
    db.add(Relay(couple_id=couple.id, from_user_id=adam.id, to_user_id=sara.id,
                 approved_text_enc="Adam said he feels unheard.",
                 state=RelayState.delivered))
    await db.flush()

    assert len(await data.get_inbox(db, sara.id)) == 1   # recipient
    assert await data.get_inbox(db, adam.id) == []       # author is not the recipient


@pytest.mark.asyncio
async def test_draft_relays_are_not_in_the_inbox(db):
    """An unapproved draft must never reach the partner — the consent gate."""
    adam = await _user(db, "sub-c", "Adam")
    sara = await _user(db, "sub-d", "Sara")
    couple = Couple()
    db.add(couple)
    await db.flush()
    db.add(Relay(couple_id=couple.id, from_user_id=adam.id, to_user_id=sara.id,
                 draft_json_enc='{"relay": "unapproved wording"}',
                 state=RelayState.draft))
    await db.flush()
    assert await data.get_inbox(db, sara.id) == []


# ────────────────────────────── session tokens ──────────────────────────────
@pytest.mark.asyncio
async def test_token_is_hashed_at_rest(db):
    u = await _user(db, "sub-e", "Adam")
    token = await tokens.issue_token(db, u)
    row = (await db.execute(select(AppSession))).scalar_one()
    assert row.token_hash != token, "token stored in plaintext"
    assert row.token_hash == hashlib.sha256(token.encode()).hexdigest()


@pytest.mark.asyncio
async def test_revocation_takes_effect(db):
    u = await _user(db, "sub-f", "Adam")
    t = await tokens.issue_token(db, u)
    await tokens.revoke_token(db, t)
    row = await db.get(AppSession, hashlib.sha256(t.encode()).hexdigest())
    assert row.revoked_at is not None


@pytest.mark.asyncio
async def test_revoke_all_signs_out_every_device(db):
    u = await _user(db, "sub-g", "Adam")
    for _ in range(3):
        await tokens.issue_token(db, u)
    await tokens.revoke_all_for_user(db, u.id)
    rows = (await db.execute(select(AppSession))).scalars().all()
    assert rows and all(r.revoked_at is not None for r in rows)


# ─────────────────────────────── schema invariants ──────────────────────────
@pytest.mark.asyncio
async def test_one_active_couple_per_user(db):
    """The partial unique index must stop a second concurrent pairing."""
    from sqlalchemy.exc import IntegrityError

    adam = await _user(db, "sub-h", "Adam")
    c1, c2 = Couple(), Couple()
    db.add_all([c1, c2])
    await db.flush()
    db.add(CoupleMember(couple_id=c1.id, user_id=adam.id))
    await db.flush()
    db.add(CoupleMember(couple_id=c2.id, user_id=adam.id))
    with pytest.raises(IntegrityError):
        await db.flush()


@pytest.mark.asyncio
async def test_solo_session_allowed(db):
    """couple_id is nullable on purpose: solo mode is a real mode."""
    adam = await _user(db, "sub-i", "Adam")
    s = MediationSession(owner_user_id=adam.id, couple_id=None)
    db.add(s)
    await db.flush()
    assert s.couple_id is None
    assert await data.get_owned_session(db, s.id, adam.id) is not None


@pytest.mark.asyncio
async def test_apple_sub_is_unique(db):
    """Two accounts must never share an Apple sub."""
    from sqlalchemy.exc import IntegrityError

    await _user(db, "sub-dupe", "Adam")
    db.add(User(apple_sub="sub-dupe", display_name="Impostor"))
    with pytest.raises(IntegrityError):
        await db.flush()


# ──────────────────────────── Apple token handling ──────────────────────────
def test_apple_config_reports_what_is_missing(monkeypatch):
    from server.auth.apple import AppleAuthError, AppleConfig

    for k in ("APPLE_TEAM_ID", "APPLE_CLIENT_ID", "APPLE_KEY_ID",
              "APPLE_PRIVATE_KEY_PEM", "APPLE_PRIVATE_KEY_PATH"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(AppleAuthError) as e:
        AppleConfig.from_env()
    # A misconfigured deploy should say which variable, not just "unauthorized".
    assert "APPLE_TEAM_ID" in str(e.value)


def test_client_secret_is_es256_with_kid():
    """Apple rejects the client_secret unless it is ES256 and carries the kid."""
    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    from server.auth.apple import AppleConfig, _client_secret

    pem = ec.generate_private_key(ec.SECP256R1()).private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    cfg = AppleConfig(team_id="TEAM123456", client_id="com.wemendai.app",
                      key_id="KEY1234567", private_key=pem)
    tok = _client_secret(cfg)
    hdr = jwt.get_unverified_header(tok)
    assert hdr["alg"] == "ES256"
    assert hdr["kid"] == "KEY1234567"
    claims = jwt.decode(tok, options={"verify_signature": False},
                        audience="https://appleid.apple.com")
    assert claims["iss"] == "TEAM123456"
    assert claims["sub"] == "com.wemendai.app"
    assert claims["exp"] > claims["iat"]


def test_nonce_mismatch_is_rejected(monkeypatch):
    """A token captured from one sign-in must not replay into another session."""
    import jwt as pyjwt

    from server.auth import apple as mod

    cfg = mod.AppleConfig(team_id="T", client_id="com.wemendai.app",
                          key_id="K", private_key="unused")

    def fake_decode(*a, **k):
        return {"sub": "apple-sub-1", "aud": "com.wemendai.app",
                "iss": mod.APPLE_ISSUER, "exp": 9e9, "nonce": "hash-of-nonce-A"}

    monkeypatch.setattr(mod, "_jwks", lambda: type("K", (), {
        "get_signing_key_from_jwt": staticmethod(lambda t: type("S", (), {"key": "k"})())})())
    monkeypatch.setattr(pyjwt, "decode", fake_decode)

    with pytest.raises(mod.AppleAuthError, match="nonce mismatch"):
        mod.verify_identity_token("tok", expected_nonce_sha256="hash-of-nonce-B", cfg=cfg)

    ident = mod.verify_identity_token("tok", expected_nonce_sha256="hash-of-nonce-A", cfg=cfg)
    assert ident.sub == "apple-sub-1"
