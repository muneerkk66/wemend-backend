"""
Voice endpoints, now backed by Postgres and real identity.

Two changes from the Phase 0 version in app.py:

1. **Identity replaces the per-session secret.** Every endpoint depends on
   `current_user`, and sessions are resolved through `data.get_owned_session`, which
   filters on `owner_user_id`. Ownership is checked by the query, not by an `if`.

2. **State lives in Postgres, not a module dict.** This is what makes the product's
   signature feature possible: shuttle mediation means A speaks now and B hears it
   hours later, after a cooling-off gap, on another device — which cannot work when
   state dies with the uvicorn process on a pod that gets stopped nightly.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from .. import data
from ..auth.tokens import current_user
from ..db import get_db
from ..tables import (
    AudioAsset, MediationSession, Profile, Relay, SessionKind, Turn, TurnRole, User,
)

router = APIRouter(tags=["voice"])

AUDIO_DIR = os.environ.get("AUDIO_DIR", "/workspace/audio")
# Audio is transient by default: docs/ARCHITECTURE.md commits to deleting it after
# transcription. Keep a short window so a client can still fetch the reply it asked for.
AUDIO_TTL = timedelta(hours=6)


def transcode_to_wav(raw: bytes, suffix: str) -> str:
    """iOS sends m4a/aac; normalise to 16k mono wav for Whisper."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        src = f.name
    dst = src + ".wav"
    r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", src,
                        "-ar", "16000", "-ac", "1", dst], capture_output=True)
    os.unlink(src)
    if r.returncode != 0:
        raise HTTPException(400, f"could not decode audio: {r.stderr.decode()[-300:]}")
    return dst


async def _speaker_and_listener(db: AsyncSession, user: User) -> tuple[str, str]:
    """Names for the prompt. Correct names are load-bearing: the perspective-flip
    failure in prompts/relay_distill.md was fixed by naming the speakers, and
    "husband"/"wife" alone still flipped."""
    prof = await db.get(Profile, user.id)
    speaker = (prof.display_name if prof else None) or user.display_name or "They"
    # Until pairing exists the listener is unnamed; Phase 4 fills this from the
    # partner's own account, never from what this user called them.
    return speaker, "your partner"


@router.post("/session")
async def create_session(
    kind: str = Form("private"),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        session_kind = SessionKind(kind)
    except ValueError:
        raise HTTPException(400, f"unknown kind {kind!r}")
    # couple_id stays null in solo mode — a real mode, not a degraded one.
    s = MediationSession(owner_user_id=user.id, kind=session_kind)
    db.add(s)
    await db.flush()
    speaker, listener = await _speaker_and_listener(db, user)
    # No session_secret any more: the caller's own bearer token authorises.
    return {"session_id": str(s.id), "kind": session_kind.value,
            "speaker": speaker, "listener": listener}


@router.post("/turn")
async def turn(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
    voice: str | None = Form(None),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    from ..app import LISTEN_SYSTEM, M, llm, speak     # late import: heavy ML holder

    if not M.ready:
        raise HTTPException(503, "models still loading — poll /health")
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(403, "invalid session")
    s = await data.get_owned_session(db, sid, user.id)
    if s is None:
        # 403 rather than 404 so session ids stay unconfirmable.
        raise HTTPException(403, "invalid session")

    raw = await audio.read()
    suffix = os.path.splitext(audio.filename or "a.m4a")[1] or ".m4a"
    wav = transcode_to_wav(raw, suffix)

    t0 = time.time()
    # No hardcoded language: a non-English name or strong accent degraded badly before.
    segments, _ = M.stt.transcribe(wav, vad_filter=True)
    heard = " ".join(seg.text for seg in segments).strip()
    stt_s = time.time() - t0
    os.unlink(wav)
    if not heard:
        raise HTTPException(422, "no speech detected")

    db.add(Turn(session_id=s.id, role=TurnRole.user, text_enc=heard))
    await db.flush()

    history_rows = await data.get_turns(db, s.id, user.id, limit=8)
    history = "\n".join(
        f"{'Them' if t.role == TurnRole.user else 'You'}: {t.text_enc}"
        for t in history_rows)

    speaker, _ = await _speaker_and_listener(db, user)
    t1 = time.time()
    reply = (await llm(LISTEN_SYSTEM.format(speaker=speaker), history,
                       json_mode=False)).strip()
    llm_s = time.time() - t1
    db.add(Turn(session_id=s.id, role=TurnRole.assistant, text_enc=reply))

    aid, dur, tstats = await speak(reply, 0, voice)
    db.add(AudioAsset(id=aid, owner_user_id=user.id, session_id=s.id, kind="reply",
                      expires_at=datetime.now(timezone.utc) + AUDIO_TTL))
    await db.flush()

    return JSONResponse({
        "heard": heard,
        "reply_text": reply,
        "audio_url": f"/audio/{aid}",
        "audio_seconds": round(dur, 2),
        "tts": tstats,
        "timing_ms": {"stt": int(stt_s * 1000), "llm": int(llm_s * 1000),
                      "tts": tstats["gen_ms"], "total": int((time.time() - t0) * 1000)},
    })


@router.get("/audio/{audio_id}")
async def get_audio(
    audio_id: str,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """Scoped to the owner. Was the sharpest hole in the service: an isalnum() check
    was the only thing standing between a guessed id and someone's relay audio."""
    if not audio_id.isalnum():
        raise HTTPException(400, "bad id")
    asset = await db.get(AudioAsset, audio_id)
    if asset is None or asset.owner_user_id != user.id or asset.deleted_at is not None:
        raise HTTPException(403, "not yours")
    path = os.path.join(AUDIO_DIR, f"{audio_id}.wav")
    if not os.path.exists(path):
        raise HTTPException(404, "expired or unknown audio")
    return FileResponse(path, media_type="audio/wav")


@router.post("/distill")
async def distill(
    session_id: str = Form(...),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Shuttle step: turn this speaker's session into a message for their partner.

    Persists a Relay in `draft` state and returns it WITHOUT voicing it. The consent
    gate is a product requirement, not an implementation detail: nothing reaches the
    partner until they approve the exact words via /approve.

    Works in solo mode too, where it produces a saved draft — "here's what you could
    actually say to them" — which is useful on its own and exercises the relay
    pipeline before pairing exists.
    """
    from ..app import M, RELAY_SYSTEM, llm

    if not M.ready:
        raise HTTPException(503, "models still loading")
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(403, "invalid session")
    s = await data.get_owned_session(db, sid, user.id)
    if s is None:
        raise HTTPException(403, "invalid session")

    turns = await data.get_turns(db, s.id, user.id, limit=100)
    said = " ".join(t.text_enc for t in turns if t.role == TurnRole.user)
    if not said:
        raise HTTPException(422, "nothing said yet in this session")

    speaker, listener = await _speaker_and_listener(db, user)
    raw = await llm(RELAY_SYSTEM.format(speaker=speaker, listener=listener),
                    f"{speaker} said: {said}", json_mode=True)
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(502, f"llm returned non-JSON: {raw[:300]}")

    # Cheap guard against the perspective-flip failure mode: first-person for the
    # speaker means the relay is asserting the opposite of what they said. See
    # prompts/relay_distill.md — this is the failure that damages users, so from
    # Phase 5 a true value must BLOCK delivery rather than merely inform the client.
    relay_text = str(d.get("relay", ""))
    flipped = any(p in relay_text.lower()
                  for p in (" i feel", "i am ", "i'm ", "when you "))

    couple_id = s.couple_id
    relay = None
    if couple_id is not None:
        from ..tables import CoupleMember
        from sqlalchemy import select as _select
        partner = (await db.execute(
            _select(CoupleMember).where(CoupleMember.couple_id == couple_id,
                                        CoupleMember.user_id != user.id,
                                        CoupleMember.left_at.is_(None)))).scalar_one_or_none()
        if partner is not None:
            relay = Relay(couple_id=couple_id, from_user_id=user.id,
                          to_user_id=partner.user_id, source_session_id=s.id,
                          draft_json_enc=json.dumps(d),
                          perspective_warning=flipped,
                          abuse_flag=bool(d.get("abuse_flag")))
            db.add(relay)
            await db.flush()

    d["_perspective_warning"] = flipped
    return {"session_id": str(s.id),
            "relay_id": str(relay.id) if relay else None,
            "draft": d,
            "requires_approval": True,
            "solo": couple_id is None}


@router.post("/approve")
async def approve(
    session_id: str = Form(...),
    relay_id: str | None = Form(None),
    edited_relay: str | None = Form(None),
    voice: str | None = Form(None),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """The speaker approves (optionally edits) the relay; only then is it voiced.

    Verifies the approver OWNS the session — the pre-auth version accepted anyone
    holding the session id, which meant a leaked id let a third party put words into
    someone's mouth.
    """
    from ..app import speak

    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(403, "invalid session")
    s = await data.get_owned_session(db, sid, user.id)
    if s is None:
        raise HTTPException(403, "invalid session")

    relay = None
    if relay_id:
        try:
            relay = await db.get(Relay, uuid.UUID(relay_id))
        except ValueError:
            raise HTTPException(403, "invalid relay")
        # Only the author may approve their own words.
        if relay is None or relay.from_user_id != user.id:
            raise HTTPException(403, "not your relay")

    text = (edited_relay or "").strip()
    if not text and relay is not None and relay.draft_json_enc:
        text = str(json.loads(relay.draft_json_enc).get("relay", "")).strip()
    if not text:
        raise HTTPException(422, "no relay text to approve")

    aid, dur, tstats = await speak(text, 1, voice)   # distinct voice for the relay
    db.add(AudioAsset(id=aid, owner_user_id=user.id, session_id=s.id, kind="relay",
                      expires_at=datetime.now(timezone.utc) + AUDIO_TTL))

    if relay is not None:
        from ..tables import RelayState
        relay.approved_text_enc = text          # the exact words, as approved
        relay.state = RelayState.approved
        relay.approved_at = datetime.now(timezone.utc)
        relay.audio_id = aid
    await db.flush()

    return {"relay_text": text, "audio_url": f"/audio/{aid}",
            "audio_seconds": round(dur, 2), "tts": tstats,
            "relay_id": str(relay.id) if relay else None}
