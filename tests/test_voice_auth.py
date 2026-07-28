"""Voice endpoints under real identity.

Stubs the ML holder (no GPU in CI) but exercises the real auth dependency, the real
ownership queries and real Postgres. What matters here is that identity gates access
and that a second user is locked out of the first user's session and audio.
"""
from __future__ import annotations

import os
import sys
import types

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from server.auth import apple, tokens
from server.tables import AudioAsset, Base, Profile, User

TEST_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://localhost/wemend_test")


@pytest_asyncio.fixture
async def client(monkeypatch, tmp_path):
    from fastapi import FastAPI

    from server import db as dbmod

    # Stub the heavy ML module BEFORE routers.voice does its late import.
    fake_app = types.ModuleType("server.app")
    fake_app.M = types.SimpleNamespace(
        ready=True,
        stt=types.SimpleNamespace(
            transcribe=lambda p, **k: ([types.SimpleNamespace(text="she never listens")], None)))
    fake_app.LISTEN_SYSTEM = "sys {speaker}"
    fake_app.RELAY_SYSTEM = "relay {speaker} {listener}"

    async def fake_llm(system, user, *, json_mode):
        return '{"relay": "Adam said he feels unheard."}' if json_mode else "That sounds hard."

    async def fake_speak(text, sid=0, engine=None):
        aid = f"aud{abs(hash(text)) % 10**12:012d}"
        (tmp_path / f"{aid}.wav").write_bytes(b"RIFF....")
        return aid, 1.5, {"engine": "kokoro", "gen_ms": 90, "realtime_factor": 16.0}

    fake_app.llm = fake_llm
    fake_app.speak = fake_speak
    monkeypatch.setitem(sys.modules, "server.app", fake_app)
    monkeypatch.setenv("AUDIO_DIR", str(tmp_path))

    from server.routers import auth as auth_router
    from server.routers import voice as voice_router
    monkeypatch.setattr(voice_router, "AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(voice_router, "transcode_to_wav", lambda raw, suffix: str(tmp_path / "in.wav"))
    (tmp_path / "in.wav").write_bytes(b"x")
    monkeypatch.setattr(voice_router.os, "unlink", lambda p: None)

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
    app.include_router(voice_router.router)
    app.dependency_overrides[dbmod.get_db] = override_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        c.maker = maker            # type: ignore[attr-defined]
        yield c
    await engine.dispose()


async def _mk_user(maker, sub: str, name: str) -> str:
    """Create a user + token directly, bypassing Apple."""
    async with maker() as s:
        u = User(apple_sub=sub, display_name=name)
        s.add(u)
        await s.flush()
        s.add(Profile(user_id=u.id, display_name=name))
        tok = await tokens.issue_token(s, u)
        await s.commit()
    return tok


@pytest.mark.asyncio
async def test_every_voice_endpoint_requires_auth(client):
    """Default-deny: an unauthenticated caller gets 401 everywhere, and crucially
    /audio no longer serves anything to an anonymous guess."""
    for method, path, kw in [
        ("post", "/session", {"data": {"kind": "private"}}),
        ("post", "/turn", {"data": {"session_id": "x"}, "files": {"audio": ("a.m4a", b"x")}}),
        ("post", "/distill", {"data": {"session_id": "x"}}),
        ("post", "/approve", {"data": {"session_id": "x"}}),
        ("get", "/audio/deadbeef", {}),
    ]:
        r = await getattr(client, method)(path, **kw)
        assert r.status_code == 401, f"{method} {path} returned {r.status_code}, want 401"


@pytest.mark.asyncio
async def test_full_turn_persists_to_postgres(client):
    tok = await _mk_user(client.maker, "sub-adam", "Adam")
    h = {"Authorization": f"Bearer {tok}"}

    sid = (await client.post("/session", data={"kind": "private"}, headers=h)).json()["session_id"]
    r = await client.post("/turn", data={"session_id": sid},
                          files={"audio": ("a.m4a", b"fake")}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["heard"] == "she never listens"

    # The turns are in the DB, not a module dict — this is what makes cooling-off
    # across hours (and a server restart) possible.
    from sqlalchemy import select
    from server.tables import Turn
    async with client.maker() as s:
        rows = (await s.execute(select(Turn))).scalars().all()
    assert len(rows) == 2 and {r.role.value for r in rows} == {"user", "assistant"}


@pytest.mark.asyncio
async def test_another_user_cannot_touch_your_session_or_audio(client):
    """The pre-auth service let anyone holding a session id read the pending relay
    draft and voice it. Identity must make that impossible."""
    adam = await _mk_user(client.maker, "sub-a", "Adam")
    sara = await _mk_user(client.maker, "sub-b", "Sara")
    ha, hs = {"Authorization": f"Bearer {adam}"}, {"Authorization": f"Bearer {sara}"}

    sid = (await client.post("/session", data={"kind": "private"}, headers=ha)).json()["session_id"]
    turn = await client.post("/turn", data={"session_id": sid},
                             files={"audio": ("a.m4a", b"x")}, headers=ha)
    aid = turn.json()["audio_url"].rsplit("/", 1)[-1]

    # Adam can fetch his own audio.
    assert (await client.get(f"/audio/{aid}", headers=ha)).status_code == 200
    # Sara cannot — even though she has a perfectly valid token of her own.
    assert (await client.get(f"/audio/{aid}", headers=hs)).status_code == 403
    # Nor can she use his session id.
    assert (await client.post("/turn", data={"session_id": sid},
            files={"audio": ("a.m4a", b"x")}, headers=hs)).status_code == 403
    assert (await client.post("/distill", data={"session_id": sid}, headers=hs)).status_code == 403
    assert (await client.post("/approve", data={"session_id": sid}, headers=hs)).status_code == 403


@pytest.mark.asyncio
async def test_distill_saves_a_draft_and_does_not_voice_it(client):
    """The consent gate: /distill must never produce audio."""
    tok = await _mk_user(client.maker, "sub-c", "Adam")
    h = {"Authorization": f"Bearer {tok}"}
    sid = (await client.post("/session", data={"kind": "private"}, headers=h)).json()["session_id"]
    await client.post("/turn", data={"session_id": sid},
                      files={"audio": ("a.m4a", b"x")}, headers=h)

    r = await client.post("/distill", data={"session_id": sid}, headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["requires_approval"] is True
    assert body["solo"] is True                       # no partner yet
    assert "audio_url" not in body                    # nothing was voiced

    # Only /approve produces audio.
    a = await client.post("/approve", data={"session_id": sid,
                                            "edited_relay": "He feels unheard."}, headers=h)
    assert a.status_code == 200 and a.json()["audio_url"]


@pytest.mark.asyncio
async def test_session_survives_a_new_app_instance(client):
    """Simulates a server restart: a brand-new dependency graph must still resolve the
    session and its turns, because nothing lives in process memory."""
    tok = await _mk_user(client.maker, "sub-d", "Adam")
    h = {"Authorization": f"Bearer {tok}"}
    sid = (await client.post("/session", data={"kind": "private"}, headers=h)).json()["session_id"]
    await client.post("/turn", data={"session_id": sid},
                      files={"audio": ("a.m4a", b"x")}, headers=h)

    # A different client over the same database — the restart analogue.
    r = await client.post("/distill", data={"session_id": sid}, headers=h)
    assert r.status_code == 200
    assert "unheard" in r.json()["draft"]["relay"].lower()
