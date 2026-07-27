"""
Pluggable TTS backends.

Two engines, same interface, chosen per-request so you can A/B the actual feel
in the app instead of trusting a benchmark table.

Measured warm on the RTX 4090 (see docs/LATENCY.md):

    csm      0.43x realtime   4.2 GB VRAM   Sesame prosody, the good voice
    kokoro   52x   realtime   0.54 GB       near-instant, plainer voice

The 0.43x figure has a consequence worth understanding: to play D seconds of
audio without underrunning, generation must be ≥1x realtime. At 0.43x you have
to buffer ~1.33*D before starting playback, so a 5s reply implies ~6.6s of dead
air. Chunked streaming does not fix that — it only reduces time-to-first-audio
for engines already faster than realtime.

Recommended split (not enforced here — your call per request):
  • live conversational turns  -> whichever you prefer the feel of
  • the relay message          -> csm, pre-generated during the between-session
                                  gap where its slowness is invisible and its
                                  warmth matters most
"""
from __future__ import annotations

import os
import time
from typing import Protocol

import numpy as np
import torch

CSM_SR = 24000
KOKORO_SR = 24000


class Engine(Protocol):
    name: str
    sample_rate: int

    def synth(self, text: str, speaker: int = 0) -> np.ndarray: ...


class CSMEngine:
    """Sesame CSM-1B. Best voice, 0.43x realtime — expect a wait."""

    name = "csm"

    def __init__(self) -> None:
        import sys
        sys.path.insert(0, "/workspace/csm")
        from generator import load_csm_1b

        t0 = time.time()
        self._g = load_csm_1b(device="cuda")
        self.sample_rate = self._g.sample_rate
        # First generation is ~3x slower than steady state; burn it at load so
        # the first real user request isn't the cold one.
        self._g.generate(text="Ready.", speaker=0, context=[], max_audio_length_ms=2000)
        print(f"[tts:csm] ready in {time.time()-t0:.1f}s "
              f"VRAM {torch.cuda.memory_allocated()/2**30:.2f}GB", flush=True)

    def synth(self, text: str, speaker: int = 0) -> np.ndarray:
        a = self._g.generate(text=text, speaker=speaker, context=[],
                             max_audio_length_ms=30000)
        return a.detach().cpu().numpy()


class KokoroEngine:
    """Kokoro-82M. 52x realtime, 0.54GB. Plainer voice, no perceptible wait."""

    name = "kokoro"
    sample_rate = KOKORO_SR
    # speaker id -> voice. 0 = the mediator, 1 = the relay voice.
    VOICES = {0: "af_heart", 1: "am_michael"}

    def __init__(self) -> None:
        from kokoro import KPipeline

        t0 = time.time()
        self._p = KPipeline(lang_code="a")
        self.synth("Ready.")                      # warm it (cold call is ~1x)
        print(f"[tts:kokoro] ready in {time.time()-t0:.1f}s", flush=True)

    def synth(self, text: str, speaker: int = 0) -> np.ndarray:
        voice = self.VOICES.get(speaker, "af_heart")
        chunks = [a for _, _, a in self._p(text, voice=voice, speed=1.0)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(
            [c.numpy() if hasattr(c, "numpy") else np.asarray(c) for c in chunks]
        )


_ENGINES: dict[str, Engine] = {}


def load(names: list[str]) -> None:
    for n in names:
        if n in _ENGINES:
            continue
        _ENGINES[n] = CSMEngine() if n == "csm" else KokoroEngine()


def get(name: str) -> Engine:
    if name not in _ENGINES:
        raise KeyError(f"tts engine '{name}' not loaded (loaded: {sorted(_ENGINES)})")
    return _ENGINES[name]


def loaded() -> list[str]:
    return sorted(_ENGINES)
