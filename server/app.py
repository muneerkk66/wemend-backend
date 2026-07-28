"""
WeMendAI voice API — FastAPI, turn-based.

Pipeline:  audio in → faster-whisper (STT) → Gemma 4 (Ollama) → CSM-1B (TTS) → audio out

Turn-based, not streaming, deliberately: CSM measured 0.43x realtime on the RTX 4090
and 0.47-0.53x on Blackwell, so audio cannot be produced faster than it is consumed.
See docs/LATENCY.md.

Routers are split so the identity half never imports this module's ML holder:
`routers/auth.py` must keep answering while the GPU pod is stopped, otherwise sign-in
and account deletion break every night. `routers/voice.py` owns the endpoints that do
need the models, and reaches back into this module late.

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
import sys
import time
import uuid

import httpx
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException

from . import tts as tts_mod

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


# ────────────────────────────────── helpers ─────────────────────────────────
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


# Routers. Auth first so it is obvious it does not depend on the models.
from .routers import auth as auth_router        # noqa: E402
from .routers import profile as profile_router  # noqa: E402
from .routers import voice as voice_router      # noqa: E402

app.include_router(auth_router.router)
app.include_router(profile_router.router)
app.include_router(voice_router.router)

# Authentication is default-deny by construction: every route in the routers above
# declares Depends(current_user) except the explicit exceptions here (/health and
# /auth/apple). The pre-auth service had four IDOR-able endpoints precisely because
# authentication was opt-in per endpoint.
PUBLIC_PATHS = {"/health", "/auth/apple", "/docs", "/openapi.json"}


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


