"""Profile and consent endpoints."""
from __future__ import annotations

import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from server.auth import tokens
from server.tables import Base, Profile, User

TEST_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://localhost/wemend_test")


@pytest_asyncio.fixture
async def client(monkeypatch):
    from fastapi import FastAPI

    from server import db as dbmod
    from server.routers import profile as profile_router

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
    app.include_router(profile_router.router)
    app.dependency_overrides[dbmod.get_db] = override_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        c.maker = maker          # type: ignore[attr-defined]
        yield c
    await engine.dispose()


async def _auth(maker, name="Adam") -> dict:
    async with maker() as s:
        u = User(apple_sub=f"sub-{name}", display_name=name)
        s.add(u)
        await s.flush()
        s.add(Profile(user_id=u.id, display_name=name))
        tok = await tokens.issue_token(s, u)
        await s.commit()
    return {"Authorization": f"Bearer {tok}"}


@pytest.mark.asyncio
async def test_profile_requires_auth(client):
    assert (await client.get("/profile")).status_code == 401
    assert (await client.patch("/profile", json={})).status_code == 401
    assert (await client.post("/profile/consents", json={"kind": "tos"})).status_code == 401


@pytest.mark.asyncio
async def test_defaults_are_not_stated_not_null(client):
    """Every enum defaults to not_stated so "prefer not to say" is a real answer and
    nothing downstream has to guess what a null means."""
    h = await _auth(client.maker)
    body = (await client.get("/profile", headers=h)).json()
    assert body["pronouns"] == "not_stated"
    assert body["relationship_status"] == "not_stated"
    assert body["pacing_preference"] == "not_stated"
    assert body["onboarding_complete"] is False


@pytest.mark.asyncio
async def test_patch_updates_only_what_is_sent(client):
    h = await _auth(client.maker)
    r = await client.patch("/profile", json={"pronouns": "he_him"}, headers=h)
    assert r.status_code == 200 and r.json()["pronouns"] == "he_him"
    # display_name untouched by a patch that didn't mention it
    assert r.json()["display_name"] == "Adam"

    r = await client.patch("/profile", json={"relationship_status": "married",
                                            "together_months": 36}, headers=h)
    b = r.json()
    assert b["pronouns"] == "he_him"          # still there
    assert b["relationship_status"] == "married" and b["together_months"] == 36


@pytest.mark.asyncio
async def test_invalid_enum_is_rejected_not_coerced(client):
    h = await _auth(client.maker)
    r = await client.patch("/profile", json={"pronouns": "whatever"}, headers=h)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_together_months_bounds(client):
    h = await _auth(client.maker)
    assert (await client.patch("/profile", json={"together_months": -1}, headers=h)).status_code == 422
    assert (await client.patch("/profile", json={"together_months": 99999}, headers=h)).status_code == 422


@pytest.mark.asyncio
async def test_onboarding_cannot_complete_without_disclosure_consent(client):
    """"We told you this is software" is not an optional step."""
    h = await _auth(client.maker)
    r = await client.patch("/profile", json={"complete_onboarding": True}, headers=h)
    assert r.status_code == 409 and "ai_disclosure" in r.text

    await client.post("/profile/consents", json={"kind": "ai_disclosure"}, headers=h)
    r = await client.patch("/profile", json={"complete_onboarding": True}, headers=h)
    assert r.status_code == 200 and r.json()["onboarding_complete"] is True


@pytest.mark.asyncio
async def test_consent_records_the_exact_wording_served(client):
    h = await _auth(client.maker)
    r = await client.post("/profile/consents", json={"kind": "ai_disclosure"}, headers=h)
    assert r.status_code == 200
    body = r.json()
    # The record is of what WE served, not what a client claims it agreed to.
    assert "not a person" in body["text"]
    assert body["granted_at"] and body["withdrawn_at"] is None


@pytest.mark.asyncio
async def test_consents_are_granular_and_withdrawable_independently(client):
    h = await _auth(client.maker)
    for k in ("ai_disclosure", "relay_sharing"):
        await client.post("/profile/consents", json={"kind": k}, headers=h)

    r = await client.post("/profile/consents",
                          json={"kind": "relay_sharing", "granted": False}, headers=h)
    assert r.status_code == 200 and r.json()["withdrawn_at"] is not None

    listing = {c["kind"]: c for c in (await client.get("/profile/consents", headers=h)).json()}
    assert listing["relay_sharing"]["withdrawn_at"] is not None
    # Withdrawing one must not disturb the other.
    assert listing["ai_disclosure"]["withdrawn_at"] is None


@pytest.mark.asyncio
async def test_withdrawing_something_never_granted_is_404(client):
    h = await _auth(client.maker)
    r = await client.post("/profile/consents", json={"kind": "tos", "granted": False}, headers=h)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_regrant_is_idempotent_for_the_same_wording(client):
    h = await _auth(client.maker)
    a = await client.post("/profile/consents", json={"kind": "tos"}, headers=h)
    b = await client.post("/profile/consents", json={"kind": "tos"}, headers=h)
    assert a.json()["granted_at"] == b.json()["granted_at"], "duplicate consent row created"


@pytest.mark.asyncio
async def test_a_tap_marks_the_field_as_user_edited(client):
    """A hand-corrected value must stop being presented as model-extracted, or the
    confirm card in Phase 3 will keep flagging it as uncertain."""
    h = await _auth(client.maker)
    async with client.maker() as s:
        from sqlalchemy import select
        u = (await s.execute(select(User))).scalar_one()
        prof = await s.get(Profile, u.id)
        prof.extraction_meta = {"pronouns": {"source": "voice", "confidence": 0.4,
                                             "quote": "I'm a guy"}}
        await s.commit()

    r = await client.patch("/profile", json={"pronouns": "he_him"}, headers=h)
    meta = r.json()["extraction_meta"]
    assert meta["pronouns"]["source"] == "user_edit"
    assert meta["pronouns"]["confidence"] == 1.0


@pytest.mark.asyncio
async def test_profile_is_created_if_signup_was_interrupted(client):
    """A user can exist without a profile row; don't 404 at them for our own state."""
    async with client.maker() as s:
        u = User(apple_sub="sub-noprofile", display_name="Orphan")
        s.add(u)
        await s.flush()
        tok = await tokens.issue_token(s, u)
        await s.commit()
    h = {"Authorization": f"Bearer {tok}"}
    r = await client.get("/profile", headers=h)
    assert r.status_code == 200 and r.json()["display_name"] == "Orphan"
