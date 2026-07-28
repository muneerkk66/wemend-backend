# Measured latency — RTX 4090 (24 GB), RunPod

Everything here was measured on the actual pod, warm, not taken from a vendor
benchmark. Re-measure before trusting any of it on different hardware.

Hardware: RTX 4090 24.5 GB · 48 vCPU · 251 GB RAM · torch 2.4.1+cu124 · Ubuntu 22.04

## Text-to-speech — this is the bottleneck

| Engine | Warm speed | RTF | VRAM | Time for a 10 s reply |
|---|---|---|---|---|
| Sesame CSM-1B | **0.43× realtime** | ~2.3 | 4.2 GB | **~25 s** |
| Kokoro-82M | **52× realtime** | ~0.019 | 0.54 GB | **~0.2 s** |

CSM measured across four consecutive warm generations: 0.41, 0.43, 0.42, 0.40 —
stable, not a warmup artifact. `torch.compile` made no difference (0.44× vs 0.43×)
and cost 13.5 s on the first call, so it is left disabled.

### Why CSM cannot stream, at any chunk size

To play audio without underrunning, generation must be **faster than realtime**.
CSM is 2.3× slower. If you chunk the output and start playback early, the buffer
drains and the voice stutters.

The floor: to play `D` seconds gapless at 0.43×, you must buffer `~1.33 × D`
first. A 5 s reply means **~6.6 s of dead air before it speaks**, and chunking
cannot reduce that — it only helps engines already above 1× realtime.

Kokoro at 52× has enormous headroom, so sentence-level streaming works there: emit
each sentence as the LLM produces it and time-to-first-audio drops to ~1 s.

### Where CSM is still the right choice

Its slowness is free wherever the user isn't waiting. In the shuttle architecture
the relay message is delivered in a *separate later session*, so a 25 s synthesis
during the cooling-off gap is invisible — and that message is exactly where warmth
and prosody matter most. Use CSM there; use something fast for live turns.

## CSM voice consistency — pin it with a reference context

CSM is a *conversational* model: it infers speaker identity from context, so
`generate(context=[])` invents a new voice on every call. Measured timbre
similarity (cosine of mean-MFCC) across three generations of the same line:

| Context | Pairwise similarity | Result |
|---|---|---|
| `[]` | **0.893** (0.837–0.979) | audibly drifts between turns |
| Fixed 8s reference | **0.997** (0.996–0.997) | stable |

Costs nothing: 0.40× realtime either way, because generation dominates and 8 s of
prefill is negligible. The reference is `prompts/conversational_a.wav` from the
model repo, trimmed to 8 s; its transcript was produced with faster-whisper (the
`Segment` text must match the audio or conditioning degrades).

## Ollama: keep model blobs on local disk

`OLLAMA_MODELS` on the MooseFS volume vs the container disk:

| Blob location | Warm TTFT |
|---|---|
| `/workspace` (MooseFS) | 1785 ms |
| `/opt` (container disk) | **1090 ms** |

~40% off, for a `cp`. Same root cause as the venv and the SIGPIPE — llama.cpp mmaps
the GGUF, and page faults over the network are expensive.

### A lighter model does NOT help

Measured on the identical stack: `llama3.2:3b` gives **1707 ms** TTFT vs Gemma 4
E4B's **1785 ms**. Gemma's actual generation is only ~200–370 ms; the rest is
serving overhead every model pays. Shrinking the model trades real quality for a
few percent. A bigger GPU doesn't help either — the overhead isn't GPU compute.

## Speech-to-text

faster-whisper `large-v3-turbo`, `int8_float16`: **~520–970 ms** for an 8 s clip.
Cold load ~116 s, so load once at startup, never per request.

Worth noting: STT is also **lossy in a way that matters here**. Converting speech to
text discards tone, pacing, and whether someone sounds frustrated or calm — which is
central to mediation. Gemma 4 E4B reports a native `audio` capability, so feeding it
speech directly (skipping Whisper) is the more promising path. Not yet benchmarked.

## LLM — Gemma 4 E4B via Ollama

Q4_K_M, 9.6 GB on disk, ~3.4 GB VRAM resident, 8.0 B params, 131 k context.

| Config | eval tokens | Time |
|---|---|---|
| `think: true` (**Ollama default**) | 262 | 4.53 s |
| `think: false` | **28** | **2.14 s** |

**Set `think: false`.** Thinking is on by default and burned 90% of the tokens for
an indistinguishable 30-word reply. Also set `OLLAMA_KEEP_ALIVE=-1` — the default
5-minute unload cost 30 s on the first request after idle.

Raw generation is ~114 tok/s; most of the remaining 2 s is per-request overhead.

## Full turn, measured end to end

| | STT | LLM | TTS | **Total** |
|---|---|---|---|---|
| CSM | 970 ms | 35.6 s (cold) | 25.4 s | **62.0 s** |
| Kokoro | 528 ms | 5.6 s | 117 ms | **6.3 s** |
| Kokoro, LLM pinned | 521 ms | 4.8 s | 111 ms | **5.5 s** |

With `think: false` applied, expect roughly **STT 0.5 s + LLM 2.1 s + TTS 0.1 s ≈ 2.7 s**
for a Kokoro turn. Streaming the LLM into per-sentence TTS should bring
time-to-first-audio near ~1 s.

## VRAM budget (24.5 GB total)

```
Gemma 4 E4B (Ollama, resident)   3.4 GB
Sesame CSM-1B                    4.2 GB
faster-whisper turbo int8        ~1.0 GB
Kokoro-82M                       0.54 GB
────────────────────────────────────────
                                ~9.2 GB   → ~15 GB headroom
```

All four fit comfortably. VRAM is not the constraint; TTS speed is.

## Would a bigger GPU fix CSM?

Probably not to a comfortable place. Batch-1 autoregressive decoding is bound by
memory bandwidth: 4090 ≈ 1008 GB/s, H100 SXM ≈ 3350 GB/s (3.3×). That maps to
roughly **1.0–1.4× realtime** — technically past the streaming threshold, with no
headroom for a hiccup.

Two caveats make even that optimistic: CSM runs ~1 backbone pass plus ~32 sequential
codebook passes per audio frame (~12.5 frames/s), so it is latency- and
launch-overhead-bound as much as bandwidth-bound, and won't scale linearly. And an
H100 pod is ~$2.4–3/hr vs ~$0.34–0.69 for the 4090 — 4–9× the cost to go from
"impossible" to "marginal".

Cheaper paths to Sesame-like warmth at realtime speed: voice-clone the CSM voice
character into a fast zero-shot model (XTTS-v2, F5-TTS, Chatterbox), or use CSM only
for the pre-generated relay. Neither is benchmarked here yet.

## Reproducing

```bash
# TTS comparison
python - <<'PY'
from kokoro import KPipeline; import time, numpy as np
p = KPipeline(lang_code="a")
for _ in range(3):
    t=time.time(); a=np.concatenate([c.numpy() for _,_,c in p("...", voice="af_heart")])
    print(len(a)/24000/(time.time()-t), "x realtime")
PY

# LLM thinking on/off
curl -s localhost:11434/api/chat -d '{"model":"gemma4:e4b","stream":false,"think":false,
  "messages":[{"role":"user","content":"hi"}]}' | jq '.eval_count, .total_duration'
```
