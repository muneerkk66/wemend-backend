"""Auth endpoint integration tests.

Apple's network calls are stubbed — the credentials themselves are already verified
live against appleid.apple.com (see the invalid_grant check). What's tested here is
OUR logic: that a first sign-in requires the code, that the name is captured only
once, and that deletion aborts rather than half-completing.
"""
from __future__ import annotations

import hashlib
import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from server.auth import apple
from server.tables import AppleCredential, Base, Profile, Relay, User

TEST_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://localhost/wemend_test")
CFG = apple.AppleConfig(team_id="T", client_id="com.necsca.wemendai",
                        key_id="K", private_key="unused")


@pytest_asyncio.fixture
async def client(monkeypatch):
    from fastapi import FastAPI

    from server import db as dbmod
    from server.routers import auth as auth_router

    engine = create_async_engine(TEST_URL)
    async with engine.begin() as c:
        await c.run_sync(Base.metadata.drop_all)
        await c.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async def override_db():
        async with maker() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    app = FastAPI()
    app.include_router(auth_router.router)
    app.dependency_overrides[dbmod.get_db] = override_db

    # Stub Apple: config, token verification, code exchange, revoke.
    monkeypatch.setattr(apple.AppleConfig, "from_env", classmethod(lambda cls: CFG))
    monkeypatch.setattr(auth_router.apple.AppleConfig, "from_env", classmethod(lambda cls: CFG))
    monkeypatch.setattr(auth_router.apple, "verify_identity_token",
                        lambda tok, *, expected_nonce_sha256, cfg: apple.AppleIdentity(
                            sub=f"sub-for-{tok}", email="rly@privaterelay.appleid.com",
                            is_private_email=True))

    async def fake_exchange(code, *, cfg): return f"refresh-{code}"
    revoked: list[str] = []
    async def fake_revoke(rt, *, cfg): revoked.append(rt)
    monkeypatch.setattr(auth_router.apple, "exchange_code", fake_exchange)
    monkeypatch.setattr(auth_router.apple, "revoke", fake_revoke)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        c.revoked = revoked            # type: ignore[attr-defined]
        c.maker = maker                # type: ignore[attr-defined]
        yield c
    await engine.dispose()


TOK = "eyJhbGciOiJSUzI1NiJ9.stub-identity-token-payload.sig"


async def _signin(c, tok=TOK, code="code1", name="Adam"):
    r = await c.post("/auth/apple", json={"identity_token": tok,
                                          "authorization_code": code,
                                          "nonce": "n", "full_name": name})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.asyncio
async def test_first_signin_requires_authorization_code(client):
    """Without the code there is no refresh token, so revocation — and therefore
    App Store compliant deletion — would be impossible. Refuse up front."""
    r = await client.post("/auth/apple", json={"identity_token": TOK, "nonce": "n"})
    assert r.status_code == 400
    assert "authorization_code" in r.text


@pytest.mark.asyncio
async def test_signin_creates_user_profile_and_stores_refresh_token(client):
    body = await _signin(client)
    assert body["token"]
    assert body["user"]["display_name"] == "Adam"
    assert body["user"]["onboarding_complete"] is False
    assert body["user"]["has_partner"] is False

    async with client.maker() as s:
        user = (await s.execute(select(User))).scalar_one()
        assert user.apple_sub == f"sub-for-{TOK}"
        cred = await s.get(AppleCredential, user.id)
        assert cred.refresh_token_enc == "refresh-code1"   # exchange actually happened
        assert (await s.get(Profile, user.id)) is not None


@pytest.mark.asyncio
async def test_second_signin_reuses_account_and_keeps_edited_name(client):
    """Apple stops sending the name after the first authorization, and the user may
    have edited it since — a later sign-in must not clobber it.

    Note there are two name fields by design: User.display_name records what Apple
    supplied (a one-time gift we must not lose), while Profile.display_name is the
    editable one shown in the app. The profile wins when both are set.
    """
    first = await _signin(client, name="Adam")
    async with client.maker() as s:
        u = (await s.execute(select(User))).scalar_one()
        prof = await s.get(Profile, u.id)
        prof.display_name = "Adam Edited"        # the user renames themselves
        await s.commit()

    second = await _signin(client, name="Adam")     # Apple re-sends the original
    assert second["user"]["display_name"] == "Adam Edited"
    assert second["token"] != first["token"]        # a new device gets its own token
    async with client.maker() as s:
        assert len((await s.execute(select(User))).scalars().all()) == 1


@pytest.mark.asyncio
async def test_me_requires_a_valid_token(client):
    assert (await client.get("/auth/me")).status_code == 401
    assert (await client.get("/auth/me",
            headers={"Authorization": "Bearer nope"})).status_code == 401
    body = await _signin(client)
    r = await client.get("/auth/me", headers={"Authorization": f"Bearer {body['token']}"})
    assert r.status_code == 200 and r.json()["display_name"] == "Adam"


@pytest.mark.asyncio
async def test_signout_invalidates_the_token(client):
    body = await _signin(client)
    h = {"Authorization": f"Bearer {body['token']}"}
    assert (await client.post("/auth/signout", headers=h)).status_code == 200
    assert (await client.get("/auth/me", headers=h)).status_code == 401


@pytest.mark.asyncio
async def test_delete_account_revokes_with_apple_then_erases(client):
    body = await _signin(client)
    h = {"Authorization": f"Bearer {body['token']}"}
    r = await client.delete("/auth/account", headers=h)
    assert r.status_code == 200 and r.json()["deleted"] is True
    # Apple revocation used the stored refresh token.
    assert client.revoked == ["refresh-code1"]
    async with client.maker() as s:
        assert (await s.execute(select(User))).scalars().all() == []
    assert (await client.get("/auth/me", headers=h)).status_code == 401


@pytest.mark.asyncio
async def test_delete_aborts_and_keeps_data_if_apple_revoke_fails(client, monkeypatch):
    """A local wipe while the Apple grant is still live is worse than no deletion:
    the user believes they are gone when they are not. Abort with data intact."""
    from server.routers import auth as auth_router

    body = await _signin(client)
    h = {"Authorization": f"Bearer {body['token']}"}

    async def boom(rt, *, cfg): raise apple.AppleAuthError("apple is down")
    monkeypatch.setattr(auth_router.apple, "revoke", boom)

    r = await client.delete("/auth/account", headers=h)
    assert r.status_code == 502 and "nothing deleted" in r.text
    async with client.maker() as s:
        assert len((await s.execute(select(User))).scalars().all()) == 1
    assert (await client.get("/auth/me", headers=h)).status_code == 200


@pytest.mark.asyncio
async def test_delivered_relays_survive_deletion_tombstoned(client):
    """Words the partner already heard are part of the partner's record too.
    Erase the author, not the other data subject's history."""
    from datetime import datetime, timezone

    from server.tables import Couple, RelayState

    body = await _signin(client)
    h = {"Authorization": f"Bearer {body['token']}"}
    async with client.maker() as s:
        author = (await s.execute(select(User))).scalar_one()
        partner = User(apple_sub="sub-partner", display_name="Sara")
        couple = Couple()
        s.add_all([partner, couple])
        await s.flush()
        s.add(Relay(couple_id=couple.id, from_user_id=author.id, to_user_id=partner.id,
                    approved_text_enc="Adam said he feels unheard.",
                    state=RelayState.delivered,
                    delivered_at=datetime.now(timezone.utc)))
        s.add(Relay(couple_id=couple.id, from_user_id=author.id, to_user_id=partner.id,
                    draft_json_enc="never approved", state=RelayState.draft))
        await s.commit()

    assert (await client.delete("/auth/account", headers=h)).status_code == 200
    async with client.maker() as s:
        rows = (await s.execute(select(Relay))).scalars().all()
        assert len(rows) == 1, "undelivered draft should be erased"
        assert rows[0].author_tombstoned is True
        assert rows[0].from_user_id is None
        assert rows[0].approved_text_enc == "Adam said he feels unheard."
