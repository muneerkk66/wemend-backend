#!/usr/bin/env bash
# WeMendAI — RunPod bootstrap.
# Target: RTX 4090 (24GB VRAM), Ubuntu 22.04, torch 2.4.1+cu124, Python 3.11.
#
# VRAM budget (24.5GB total):
#   Gemma 4 E4B via Ollama, Q4    ~5GB
#   Sesame CSM-1B (bf16)          ~3GB
#   faster-whisper large-v3-turbo ~2GB
#   KV cache + activations        ~6GB
#   ------------------------------------
#   ~16GB used, ~8GB headroom
#
# Two venvs on purpose: vLLM/Ollama tooling must never upgrade the torch that CSM
# runs on. Container disk is only 20GB, so everything heavy lives on /workspace.
#
# Phase 1 (no credentials):  bash runpod_bootstrap.sh phase1
# Phase 2 (needs HF_TOKEN):  HF_TOKEN=hf_xxx bash runpod_bootstrap.sh phase2
set -euo pipefail

WORK=/workspace
# Venv on the CONTAINER disk, not /workspace. /workspace is MooseFS network
# storage: fine for a few large files, terrible for the ~30k tiny files in a
# venv (measured: 62 packages in 34 min, 3% CPU — pure I/O stall). It would also
# tax every model load at runtime. Venv is cheap to rebuild from this script;
# the weights are the expensive artifact, so those live on the volume.
VOICE_VENV=/opt/venv-voice           # CSM + Whisper, inherits system torch 2.4.1
export HF_HOME=$WORK/.hf             # ~13GB of weights → volume
export OLLAMA_MODELS=$WORK/.ollama

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

phase1() {
  log "Hardware"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  echo "Disk / : $(df -h / | awk 'NR==2{print $4}') free   (container, small)"
  echo "Disk $WORK : $(df -h $WORK | awk 'NR==2{print $4}') free   (volume, use this)"

  log "System packages"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  # zstd is required by the Ollama installer's extraction step (fails without it)
  apt-get install -y -qq git git-lfs ffmpeg build-essential curl zstd >/dev/null
  git lfs install --skip-repo >/dev/null

  log "Voice venv (inherits system torch 2.4.1+cu124)"
  mkdir -p "$HF_HOME" "$OLLAMA_MODELS"
  [ -d "$VOICE_VENV" ] || python3 -m venv "$VOICE_VENV" --system-site-packages
  # shellcheck disable=SC1091
  source "$VOICE_VENV/bin/activate"
  pip install -q --upgrade pip wheel
  python -c 'import torch;assert torch.cuda.is_available();print("torch",torch.__version__,"CUDA ok")'

  log "STT — faster-whisper"
  pip install -q faster-whisper "huggingface_hub[cli]"

  log "Sesame CSM-1B source + deps"
  cd "$WORK"
  [ -d csm ] || git clone -q https://github.com/SesameAILabs/csm.git
  cd csm
  # Strip the repo's torch/torchaudio pins — installing them would replace the
  # image's working CUDA build and cost ~3GB on a 20GB disk.
  grep -viE '^(torch|torchaudio)([=<>~!]|$)' requirements.txt > /tmp/csm-reqs.txt
  echo "  (skipped pins: $(grep -iE '^(torch|torchaudio)([=<>~!]|$)' requirements.txt | tr '\n' ' '))"
  pip install -q -r /tmp/csm-reqs.txt
  pip install -q moshi torchtune torchao

  log "Ollama + Gemma 4 E4B"
  command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
  pgrep -x ollama >/dev/null || { nohup ollama serve >$WORK/ollama.log 2>&1 & sleep 5; }
  ollama pull gemma4:e4b || echo "  !! tag 'gemma4:e4b' not found — check 'ollama search gemma'"

  log "Phase 1 done. Next:  HF_TOKEN=hf_xxx bash $0 phase2"
}

phase2() {
  [ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN not set.
  1. Token:   https://huggingface.co/settings/tokens   (read access is enough)
  2. Accept BOTH gated licences, or the download 401s:
       https://huggingface.co/sesame/csm-1b
       https://huggingface.co/meta-llama/Llama-3.2-1B   (CSM's tokenizer)"

  # shellcheck disable=SC1091
  source "$VOICE_VENV/bin/activate"

  log "Downloading gated weights (~2.6GB + ~2.5GB) to $HF_HOME"
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
for repo in ("sesame/csm-1b", "meta-llama/Llama-3.2-1B"):
    try:
        snapshot_download(repo, token=tok)
        print(f"  OK   {repo}")
    except Exception as e:
        raise SystemExit(
            f"  FAIL {repo}\n  {type(e).__name__}: {e}\n\n"
            f"  -> Accept the licence at https://huggingface.co/{repo} then re-run phase2."
        )
PY

  log "Smoke test"
  python "$WORK/test_csm.py"
}

write_helpers() {
  cat > "$WORK/test_csm.py" <<'PY'
"""Verify CSM-1B generates audio; report realtime factor and VRAM cost."""
import os, sys, time
os.environ.setdefault("NO_TORCH_COMPILE", "1")
sys.path.insert(0, "/workspace/csm")
import torch, torchaudio
from generator import load_csm_1b

t0 = time.time(); gen = load_csm_1b(device="cuda")
print(f"loaded in {time.time()-t0:.1f}s | VRAM {torch.cuda.memory_allocated()/2**30:.2f}GB")

t1 = time.time()
audio = gen.generate(
    text="I hear that you felt dismissed. Let me carry that across for you.",
    speaker=0, context=[], max_audio_length_ms=10_000,
)
el = time.time() - t1; dur = audio.shape[-1] / gen.sample_rate
print(f"{dur:.2f}s audio in {el:.2f}s = {dur/el:.1f}x realtime")
torchaudio.save("/workspace/out.wav", audio.unsqueeze(0).cpu(), gen.sample_rate)
print("wrote /workspace/out.wav")
PY

  cat > "$WORK/env.sh" <<EOF
# source this before running anything
source $VOICE_VENV/bin/activate
export HF_HOME=$HF_HOME
export OLLAMA_MODELS=$OLLAMA_MODELS
export NO_TORCH_COMPILE=1
export PYTHONPATH=$WORK/csm:\${PYTHONPATH:-}
EOF
  chmod +x "$WORK/env.sh"
}

write_helpers
case "${1:-phase1}" in
  phase1) phase1 ;;
  phase2) phase2 ;;
  *) die "usage: $0 [phase1|phase2]" ;;
esac
