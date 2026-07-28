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
    """Sesame CSM-1B. Best voice, 0.43x realtime — expect a wait.

    Pinned to a fixed speaker via a reference context. CSM is a *conversational*
    model: it infers speaker identity from context, so calling it with
    `context=[]` invents a new voice on every request. Measured timbre similarity
    across three generations of the same line:

        context=[]        0.893  (range 0.837-0.979)  -> audibly drifts
        fixed reference   0.997  (range 0.996-0.997)  -> stable

    The reference costs nothing measurable: 0.40x realtime either way, because
    generation dominates and 8s of prefill is cheap.
    """

    name = "csm"

    # Sesame ships speaker prompts in the model repo. conversational_a is warm and
    # unhurried, which suits a mediator.
    REF_FILE = "prompts/conversational_a.wav"
    REF_SECONDS = 8            # enough to fix timbre, cheap in context
    # Transcript of the first 8s, produced by transcribing the clip with
    # faster-whisper. It must match the audio or conditioning degrades.
    REF_TEXT = ("like revising for an exam I'd have to try and like keep up the "
                "momentum because I'd start really early I'd be like okay I'm "
                "gonna start revising now and then like")

    def __init__(self) -> None:
        import glob
        import sys

        sys.path.insert(0, "/workspace/csm")
        import torchaudio
        from generator import load_csm_1b, Segment

        t0 = time.time()
        self._g = load_csm_1b(device="cuda")
        self.sample_rate = self._g.sample_rate

        snaps = glob.glob(os.path.expanduser(
            f"{os.environ.get('HF_HOME', '~/.cache/huggingface')}"
            "/hub/models--sesame--csm-1b/snapshots/*"))
        if not snaps:
            raise RuntimeError("csm-1b snapshot not found; run bootstrap phase2")
        wav, sr = torchaudio.load(os.path.join(snaps[0], self.REF_FILE))
        ref = torchaudio.functional.resample(
            wav.mean(0), sr, self.sample_rate)[: self.REF_SECONDS * self.sample_rate]
        self._context = [Segment(text=self.REF_TEXT, speaker=0, audio=ref)]

        # First generation is ~3x slower than steady state; burn it at load.
        self._g.generate(text="Ready.", speaker=0, context=self._context,
                         max_audio_length_ms=2000)
        print(f"[tts:csm] ready in {time.time()-t0:.1f}s "
              f"(voice pinned to {self.REF_FILE}) "
              f"VRAM {torch.cuda.memory_allocated()/2**30:.2f}GB", flush=True)

    def synth(self, text: str, speaker: int = 0) -> np.ndarray:
        # speaker is ignored: the reference context defines the voice, and passing
        # a different id with this context would fight the conditioning.
        a = self._g.generate(text=text, speaker=0, context=self._context,
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
