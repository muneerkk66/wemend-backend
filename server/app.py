"""
WeMendAI voice API — FastAPI, turn-based.

Pipeline:  audio in → faster-whisper (STT) → Gemma 4 (Ollama) → CSM-1B (TTS) → audio out

Turn-based, not streaming, deliberately: CSM-1B measured 0.43x realtime on the
RTX 4090, so audio cannot be produced faster than it is consumed. See
docs/LATENCY.md. The client records a full utterance, uploads it, polls, then
plays the reply.

Models are loaded ONCE at startup and held in VRAM (whisper ~2GB + CSM ~4.2GB;
Gemma lives in the Ollama process, ~4.8GB). Loading CSM costs ~52s, so never
load per request.

Run:
    HF_HOME=/workspace/.hf PYTHONPATH=/workspace/csm \
    /opt/venv-voice/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import io
import os
import secrets
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

import httpx
import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

import tts as tts_mod

sys.path.insert(0, "/workspace/csm")

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4:e4b")
AUDIO_DIR = os.environ.get("AUDIO_DIR", "/workspace/audio")
# CSM is the default voice by product choice; kokoro is loaded alongside so the
# app can switch per request and you can hear the tradeoff directly.
DEFAULT_ENGINE = os.environ.get("TTS_ENGINE", "csm")
ENGINES = [e for e in os.environ.get("TTS_ENGINES", "csm,kokoro").split(",") if e]
os.makedirs(AUDIO_DIR, exist_ok=True)

# ── the relay prompt. see prompts/relay_distill.md for why each rule exists ──
RELAY_SYSTEM = """You relay messages between two partners in conflict.

SPEAKER = {speaker}. LISTENER = {listener}.
You will hear what {speaker} said privately. You will then SPEAK ALOUD TO {listener}.

Rules for the relay text:
1. Write it as YOU (the mediator) speaking TO {listener} ABOUT {speaker}. Use
   "{speaker}" and he/she, never "I" or "you" for {speaker}. Never swap who did what.
2. KEEP the specific concrete behaviour {speaker} named. Do not generalise it away.
3. KEEP any statement about withdrawing, giving up, or leaving. Never drop it.
4. Remove insults, contempt, and absolutes (never/always).

Return JSON keys: relay, underlying_need, concrete_behaviour,
withdrawal_signal (boolean), abuse_flag (boolean), reason."""

# Spoken back to the person currently on the line, while they talk to the AI.
LISTEN_SYSTEM = """You are a warm, calm relationship mediator on a voice call with {speaker}.
You are NOT their partner and you are NOT a therapist — say so if asked.
Keep replies under 40 words: this is spoken aloud, and long replies feel like a lecture.
Reflect what you heard, then ask one open question. Never take sides. Never give advice
about whether to stay or leave."""


# ─────────────────────────────── model holder ───────────────────────────────
class Models:
    """Loaded once at startup, shared across requests."""

    def __init__(self) -> None:
        self.stt = None
        self.tts_lock = asyncio.Lock()   # CSM is not safe to call concurrently
        self.ready = False

    def load(self) -> None:
        from faster_whisper import WhisperModel

        t0 = time.time()
        # int8_float16 keeps VRAM ~1GB instead of ~2GB with no meaningful WER cost
        self.stt = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
        print(f"[models] whisper loaded in {time.time()-t0:.1f}s", flush=True)

        tts_mod.load(ENGINES)
        print(f"[models] warm. total {time.time()-t0:.1f}s "
              f"VRAM {torch.cuda.memory_allocated()/2**30:.2f}GB", flush=True)
        self.ready = True


M = Models()


# ─────────────────────────────── session state ──────────────────────────────
@dataclass
class Turn:
    role: Literal["user", "assistant"]
    text: str


@dataclass
class Session:
    id: str
    speaker: str
    listener: str
    # Bearer secret returned once at creation and required by every other endpoint.
    # A stopgap until Phase 1 replaces it with real Apple-backed auth: it stops the
    # session id alone (which appears in URLs and logs) from granting access.
    secret: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    turns: list[Turn] = field(default_factory=list)
    # Set when the speaker's side is distilled and awaiting their approval.
    pending_relay: dict | None = None
    # Audio ids this session produced. /audio is scoped to this set, so a guessed
    # id no longer yields another person's relay audio.
    audio_ids: set[str] = field(default_factory=set)
    created: float = field(default_factory=time.time)


SESSIONS: dict[str, Session] = {}


# ────────────────────────────────── auth ────────────────────────────────────
def authed_session(
    session_id: str = Form(...),
    authorization: str | None = Header(None),
) -> Session:
    """Resolve a session from its id AND its bearer secret.

    Every mutating endpoint depends on this. Previously they took a bare
    session_id, so anyone holding an id could read another person's pending relay
    draft and voice it. Compared with compare_digest to avoid leaking the secret
    through timing.
    """
    s = SESSIONS.get(session_id)
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    # Same error whether the session is unknown or the secret is wrong — a
    # distinguishable 404 would confirm which session ids exist.
    if s is None or not token or not secrets.compare_digest(token, s.secret):
        raise HTTPException(403, "invalid session or token")
    return s


def register_audio(s: Session, audio_id: str) -> None:
    s.audio_ids.add(audio_id)


# ────────────────────────────────── helpers ─────────────────────────────────
def transcode_to_wav(raw: bytes, suffix: str) -> str:
    """iOS sends m4a/aac. Whisper wants a file ffmpeg can read; normalise to 16k mono wav."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        src = f.name
    dst = src + ".wav"
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", src, "-ar", "16000", "-ac", "1", dst],
        capture_output=True,
    )
    os.unlink(src)
    if r.returncode != 0:
        raise HTTPException(400, f"could not decode audio: {r.stderr.decode()[-300:]}")
    return dst


async def llm(system: str, user: str, *, json_mode: bool) -> str:
    body = {
        "model": LLM_MODEL,
        "stream": False,
        # Gemma 4 has thinking ON by default: measured 262 eval tokens / 4.53s vs
        # 28 tokens / 2.14s with it off, for an indistinguishable 30-word reply.
        # A spoken mediator reply does not need chain-of-thought — keep it off.
        "think": False,
        "options": {"temperature": 0.2 if json_mode else 0.6, "num_ctx": 8192},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    if json_mode:
        body["format"] = "json"
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(f"{OLLAMA}/api/chat", json=body)
    if r.status_code != 200:
        raise HTTPException(502, f"llm error: {r.text[:300]}")
    d = r.json()
    if "message" not in d:
        raise HTTPException(502, f"llm returned no message: {str(d)[:300]}")
    return d["message"]["content"]


async def speak(text: str, speaker_id: int = 0, engine: str = None) -> tuple[str, float, dict]:
    """Synthesize text -> wav on disk. Returns (audio_id, seconds, stats)."""
    eng = tts_mod.get(engine or DEFAULT_ENGINE)
    async with M.tts_lock:                      # serialize: CSM is single-stream
        loop = asyncio.get_running_loop()
        t0 = time.time()
        audio = await loop.run_in_executor(None, lambda: eng.synth(text, speaker_id))
        gen_s = time.time() - t0

    aid = uuid.uuid4().hex
    path = os.path.join(AUDIO_DIR, f"{aid}.wav")
    sf.write(path, audio, eng.sample_rate)
    dur = len(audio) / eng.sample_rate
    rt = dur / gen_s if gen_s else 0.0
    print(f"[tts:{eng.name}] {dur:.2f}s audio in {gen_s:.2f}s = {rt:.2f}x realtime", flush=True)
    return aid, dur, {"engine": eng.name, "gen_ms": int(gen_s * 1000),
                      "realtime_factor": round(rt, 2)}


# ──────────────────────────────────── app ──────────────────────────────────
app = FastAPI(title="WeMendAI Voice API")

# No CORS middleware on purpose. A native iOS client sends no Origin header and
# does not need CORS; the previous allow_origins=["*"] only made the API callable
# from any web page. Add a narrow allowlist here if a browser client ever exists.


@app.on_event("startup")
async def _startup() -> None:
    # Load in a thread so uvicorn can bind the port and answer /health immediately;
    # otherwise a 60s load looks like a dead server to the client.
    asyncio.get_running_loop().run_in_executor(None, M.load)


@app.get("/health")
async def health() -> dict:
    return {
        "ready": M.ready,
        "vram_gb": round(torch.cuda.memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
        "llm": LLM_MODEL,
        "tts_engines": tts_mod.loaded(),
        "tts_default": DEFAULT_ENGINE,
    }


@app.post("/session")
async def create_session(speaker: str = Form(...), listener: str = Form(...)) -> dict:
    s = Session(id=uuid.uuid4().hex, speaker=speaker, listener=listener)
    SESSIONS[s.id] = s
    # The secret is returned exactly once. Clients must send it as
    # `Authorization: Bearer <secret>` on every other call.
    return {"session_id": s.id, "session_secret": s.secret,
            "speaker": s.speaker, "listener": s.listener}


@app.post("/turn")
async def turn(
    audio: UploadFile = File(...),
    voice: str = Form(None),
    s: Session = Depends(authed_session),
) -> JSONResponse:
    """One conversational turn: the speaker talks to the AI, the AI answers aloud."""
    if not M.ready:
        raise HTTPException(503, "models still loading — poll /health")

    raw = await audio.read()
    suffix = os.path.splitext(audio.filename or "a.m4a")[1] or ".m4a"
    wav = transcode_to_wav(raw, suffix)

    t0 = time.time()
    segments, info = M.stt.transcribe(wav, language="en", vad_filter=True)
    heard = " ".join(seg.text for seg in segments).strip()
    stt_s = time.time() - t0
    os.unlink(wav)

    if not heard:
        raise HTTPException(422, "no speech detected")

    s.turns.append(Turn("user", heard))

    history = "\n".join(f"{'Them' if t.role=='user' else 'You'}: {t.text}" for t in s.turns[-8:])
    t1 = time.time()
    reply = (await llm(LISTEN_SYSTEM.format(speaker=s.speaker), history, json_mode=False)).strip()
    llm_s = time.time() - t1
    s.turns.append(Turn("assistant", reply))

    aid, dur, tstats = await speak(reply, 0, voice)
    register_audio(s, aid)

    return JSONResponse({
        "heard": heard,
        "reply_text": reply,
        "audio_url": f"/audio/{aid}",
        "audio_seconds": round(dur, 2),
        "tts": tstats,
        "timing_ms": {
            "stt": int(stt_s * 1000),
            "llm": int(llm_s * 1000),
            "tts": tstats["gen_ms"],
            "total": int((time.time() - t0) * 1000),
        },
    })


@app.post("/distill")
async def distill(s: Session = Depends(authed_session)) -> dict:
    """Shuttle step: turn this speaker's session into a message for the partner.

    Returns the draft WITHOUT voicing it. The speaker must approve via /approve —
    that consent gate is a product requirement, not an implementation detail.
    """
    if not M.ready:
        raise HTTPException(503, "models still loading")

    said = " ".join(t.text for t in s.turns if t.role == "user")
    if not said:
        raise HTTPException(422, "nothing said yet in this session")

    import json
    raw = await llm(
        RELAY_SYSTEM.format(speaker=s.speaker, listener=s.listener),
        f"{s.speaker} said: {said}",
        json_mode=True,
    )
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(502, f"llm returned non-JSON: {raw[:300]}")

    # Cheap guard against the perspective-flip failure mode (see prompts/relay_distill.md).
    relay = str(d.get("relay", ""))
    flipped = any(p in relay.lower() for p in (" i feel", "i am ", "i'm ", "when you "))
    d["_perspective_warning"] = flipped

    s.pending_relay = d
    return {"session_id": s.id, "draft": d, "requires_approval": True}


@app.post("/approve")
async def approve(edited_relay: str = Form(None), voice: str = Form(None),
                  s: Session = Depends(authed_session)) -> dict:
    """Speaker approves (optionally edits) the relay; only then is it voiced."""
    if s.pending_relay is None:
        raise HTTPException(404, "no pending relay for this session")

    text = (edited_relay or s.pending_relay["relay"]).strip()
    aid, dur, tstats = await speak(text, 1, voice)   # distinct voice for the relay
    register_audio(s, aid)
    s.pending_relay = None
    return {"relay_text": text, "audio_url": f"/audio/{aid}",
            "audio_seconds": round(dur, 2), "tts": tstats}


@app.get("/audio/{audio_id}")
async def get_audio(
    audio_id: str,
    session_id: str = "",
    authorization: str | None = Header(None),
) -> FileResponse:
    """Fetch generated audio. Scoped to the session that produced it.

    This was the sharpest hole in the service: the only check was isalnum(), so any
    wav ever generated — including approved relay audio — was readable by guessing a
    32-hex id on a public URL. Now the caller must prove they own the session that
    created it. session_id is a query param because <audio> tags cannot set headers,
    so the client appends ?session_id=...; the bearer secret still travels in the
    header and is what actually authorises.
    """
    if not audio_id.isalnum():                  # no path traversal
        raise HTTPException(400, "bad id")

    s = SESSIONS.get(session_id)
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if s is None or not token or not secrets.compare_digest(token, s.secret):
        raise HTTPException(403, "invalid session or token")
    if audio_id not in s.audio_ids:
        raise HTTPException(403, "audio does not belong to this session")

    path = os.path.join(AUDIO_DIR, f"{audio_id}.wav")
    if not os.path.exists(path):
        raise HTTPException(404, "expired or unknown audio")
    return FileResponse(path, media_type="audio/wav")
