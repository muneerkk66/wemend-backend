# wemend-backend

Voice backend for **WeMendAI** — an AI mediator for couples in conflict. Each partner
speaks with the app privately; the app distils what they said and relays it to the
other. Voice, not chat.

Runs self-hosted on a single RTX 4090 (RunPod). No third-party inference API, which
matters because these are the most sensitive recordings a couple owns.

## Pipeline

```
mic ──> faster-whisper ──> Gemma 4 E4B ──> CSM-1B / Kokoro ──> speaker
        ~0.5s              ~2.1s           25s / 0.1s
```

Both TTS engines load side by side and are selectable per request
(`voice=csm|kokoro`) so you can A/B the tradeoff by ear. **Read
[docs/LATENCY.md](docs/LATENCY.md) before choosing** — CSM sounds better but runs at
0.43× realtime, which rules out live streaming.

## Layout

| Path | What |
|---|---|
| `infra/runpod_bootstrap.sh` | Provisions the pod: CSM + Whisper + Ollama/Gemma 4 |
| `server/app.py` | FastAPI: `/session` `/turn` `/distill` `/approve` `/audio` |
| `server/tts.py` | Pluggable TTS engines (CSM, Kokoro) |
| `prompts/relay_distill.md` | **The core prompt** + the three failure modes it fixes |
| `docs/ARCHITECTURE.md` | Product architecture and non-negotiable constraints |
| `docs/LATENCY.md` | Measured numbers on real hardware |

## Setup

```bash
# On the pod. Both HF repos are gated — accept the licences first:
#   https://huggingface.co/sesame/csm-1b
#   https://huggingface.co/meta-llama/Llama-3.2-1B
bash infra/runpod_bootstrap.sh phase1              # deps, CSM source, Ollama, Gemma 4
HF_TOKEN=hf_xxx bash infra/runpod_bootstrap.sh phase2   # gated weights + smoke test

# Keep Gemma resident — the default 5-min unload costs ~30s on the next request
OLLAMA_MODELS=/workspace/.ollama OLLAMA_KEEP_ALIVE=-1 ollama serve &

cd server && HF_HOME=/workspace/.hf PYTHONPATH=/workspace/csm:. \
  /opt/venv-voice/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

`/health` answers immediately while models load in a background thread (CSM ~45 s,
Whisper ~2 min). Poll it for `"ready": true`.

## Stopping and resuming the pod

**Stop the pod, don't terminate it.** Stopping halts GPU billing (~$248/mo → ~$5–18/mo
storage). Terminating destroys the volume, and that means re-downloading 26 GB of CSM
weights and 11 GB of Ollama blobs.

Note RunPod bills **stopped** storage at double: $0.20/GB/month vs $0.10 running.

What survives a stop (the volume), and what doesn't (the container disk):

| Path | Survives? | Rebuild cost |
|---|---|---|
| `/workspace/.hf` (26 GB weights) | ✅ | — |
| `/workspace/.ollama` (11 GB blobs) | ✅ | — |
| `/workspace/{csm,server,infra}` | ✅ | — |
| `/opt/venv-voice` | ❌ | ~5 min (pip) |
| `/opt/ollama` (local blob copy) | ❌ | ~5 s (`cp` from volume) |
| `ollama` binary, apt packages | ❌ | ~1 min |

After restarting, one command restores all of it and starts both services:

```bash
bash /workspace/infra/resume.sh
```

It's idempotent, so it doubles as a health check. If the **pod id** changed, update
`Config.swift` in `wemend-ios` — the app's URL is `https://<pod-id>-8888.proxy.runpod.net`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | readiness, VRAM, loaded engines |
| `POST` | `/session` | start a session (`speaker`, `listener`) |
| `POST` | `/turn` | audio in → mediator replies aloud (`voice=csm\|kokoro`) |
| `POST` | `/distill` | shuttle step: draft the relay. **Does not voice it.** |
| `POST` | `/approve` | speaker approves/edits → then it's voiced |
| `GET` | `/audio/{id}` | fetch generated wav |

`/distill` never speaks. The consent gate is a product requirement, not an
implementation detail — see `docs/ARCHITECTURE.md`.

## Hard-won gotchas

- **Venv must live on the container disk, not `/workspace`.** `/workspace` is MooseFS;
  a venv is ~30k tiny files. Measured 62 packages in 34 min there vs. the whole
  install in <9 min locally.
- **Ollama's log must not live on `/workspace` either** — a stale network-storage fd
  killed `llama-server` with SIGPIPE (`broken pipe`) on every model load.
- **Ollama's installer needs `zstd`**, which isn't in the RunPod image.
- **Never interrupt an HF download.** A killed transfer left `model.safetensors` 3 MiB
  *larger* than its header declared; `safetensors` reports that as
  `MetadataIncompleteBuffer`, which reads like corruption but is a bad resume.
- **`think: false` on Gemma 4.** On by default; 262 eval tokens → 28 with it off.

## Not production ready

CORS is `*`, sessions are in-memory, `/audio` files are never pruned, and there is no
auth. See the constraints section of `docs/ARCHITECTURE.md` — particularly abuse
screening and the consent gate — before this touches a real couple.
