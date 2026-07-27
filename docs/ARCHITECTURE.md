# WeMendAI — Architecture

Voice-first AI mediator for couples in conflict. iOS app.

## Decisions locked

| | Choice |
|---|---|
| Mediation model | **Shuttle / turn-based** — one partner on the line at a time |
| Devices | **Two devices, paired accounts** |
| TTS | Sesame CSM-1B (Apache 2.0) |
| LLM | Gemma 4 on RunPod now → Claude API later |
| STT | faster-whisper `large-v3-turbo` |

## The key distinction

Shuttle mediation is turn-based **between partners**, but each individual session is a
**live, interruptible voice conversation** with the AI. So we still need real-time voice
agent infrastructure — we just don't need to synchronise two humans.

```
┌─ Session 1: Husband ─────────────┐
│  live voice conv w/ AI           │   <- real-time: VAD, barge-in, ~700ms turns
│  AI listens, reflects, clarifies │
└──────────────┬───────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ DISTILL (no time pressure)      │   <- the actual product value
        │ • strip contempt/blame          │
        │ • extract underlying need       │
        │ • draft what to convey          │
        │ • husband APPROVES before send  │   <- consent gate, see below
        └──────┬──────────────────────────┘
               │  push notification
┌──────────────▼───────────────────┐
│  Session 2: Wife                 │
│  AI conveys, then hears her side │
└──────────────────────────────────┘
```

## Why "Sesame" alone isn't enough

CSM-1B is **text-to-speech with conversational context**. It is *not* speech-to-speech and
does *not* do recognition — the viral Maya/Miles demo was Sesame's closed product, not the
open release. So the pipeline is three models, not one:

```
mic ─> faster-whisper ─> Gemma 4 ─> CSM-1B ─> speaker
        30-80ms         150-300ms    ~150ms TTFA
                                    total ≈ 400-700ms + network
```

700ms is fine for a reflective, therapeutic pace. Don't over-optimise this.

CSM's real advantage over a generic TTS is **prosody carried across conversational
context** — it can sound gentle rather than clinical, which matters a lot here.

### Gotchas

- Both `sesame/csm-1b` **and** `meta-llama/Llama-3.2-1B` (tokenizer) are **gated** on HF.
  Accept both licences or the download 401s.
- CSM pins its own `torch`/`torchaudio`; installing them clobbers the RunPod image's
  working CUDA build. `infra/runpod_bootstrap.sh` strips those pins.
- Set `HF_HOME=/workspace/.hf` — the container disk is small, the volume is not.

## Orchestration

Use **Pipecat** or **LiveKit Agents** rather than hand-rolling a WebSocket server. The hard
part of voice agents isn't the models — it's VAD, endpointing, barge-in, and turn-taking.
Both frameworks accept custom STT/LLM/TTS services, so CSM-1B and Gemma 4 drop in.

## Design constraints that are not optional

These are product-shaping, not legal boilerplate:

1. **Disclose the AI.** When the app speaks to the wife, she must know she's talking to
   software, not a person. "Act like a real person" is fine as a *quality* bar (warm,
   natural, understands nuance) — not as concealment.

2. **Consent gate before every relay.** The speaker approves the distilled message before
   it's conveyed. Without this, users can't trust the app with anything real, and the AI
   becomes a channel for things people didn't intend to send.

3. **Abuse screening + human escalation.** Relationship conflict overlaps with coercive
   control. A mediator that faithfully relays threats causes harm. Need detection for
   abuse/self-harm indicators and a hard handoff to human professionals and hotlines.
   Also the reason the AI must never relay a disclosure of abuse back to an abuser.

4. **These recordings are the most sensitive data a couple owns.** Encrypt at rest and in
   transit, define retention up front (default: delete audio after transcription), and
   never train on it. Self-hosting on RunPod actually helps here vs. third-party APIs.

5. **Not therapy.** Position as communication support. Apple reviews mental-health apps
   closely, and clinical claims invite regulatory exposure.

## Repo layout

```
infra/runpod_bootstrap.sh   GPU pod: CSM-1B + faster-whisper + vLLM/Gemma 4
```
