#!/usr/bin/env bash
# Restore a stopped-and-restarted RunPod pod, then start everything.
#
#   bash /workspace/infra/resume.sh
#
# Why this is needed: stopping a pod preserves the /workspace VOLUME but wipes the
# CONTAINER DISK. Everything expensive lives on the volume and survives:
#
#   /workspace/.hf       26GB  CSM-1B + Llama tokenizer weights
#   /workspace/.ollama   11GB  Gemma 4 blobs (canonical copy)
#   /workspace/csm             CSM source
#   /workspace/server          the API
#
# What the wipe costs us, and this script rebuilds:
#
#   /opt/venv-voice      python env      ~5 min  (pip, from PyPI)
#   /opt/ollama          local model copy  ~5 s  (cp from the volume — this is the
#                                                 40% TTFT win, see docs/LATENCY.md)
#   /usr/local/bin/ollama                 ~30 s
#   apt: ffmpeg zstd git-lfs pciutils     ~1 min
#
# Idempotent: safe to run when nothing is missing (it becomes a health check).
set -euo pipefail

WORK=/workspace
VENV=/opt/venv-voice
OLLAMA_LOCAL=/opt/ollama
PORT="${PORT:-8888}"          # 8888 is proxied by default RunPod pods; see README

# Wheels cached on the VOLUME so a container-disk wipe doesn't re-download ~500MB.
# The venv itself stays on the container disk deliberately — on /workspace it took
# 34 min for 62 packages, and it costs a permanent ~40% on Gemma TTFT because
# llama.cpp mmaps the GGUF and page-faults over the network. Cache the downloads,
# not the installed tree.
export PIP_CACHE_DIR="$WORK/.pipcache"
# authorized_keys lives on the container disk, so a stop wipes SSH access and locks
# you out of the very box you need to fix. Keep a copy on the volume.
SSH_BACKUP="$WORK/.ssh/authorized_keys"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

log "Sanity: is the volume actually mounted?"
[ -d "$WORK/.hf/hub" ] || die "$WORK/.hf missing — the volume did not mount, or the
pod was TERMINATED rather than stopped. If terminated, re-run:
    bash $WORK/infra/runpod_bootstrap.sh phase1
    HF_TOKEN=hf_xxx bash $WORK/infra/runpod_bootstrap.sh phase2"
[ -d "$WORK/.ollama/blobs" ] || die "$WORK/.ollama missing — re-run bootstrap phase1"

log "Restoring SSH access if the wipe took it"
# Do this BEFORE the GPU check: if the GPU is missing you still want to be able to
# get back in over SSH to investigate, rather than being stuck in a web terminal.
mkdir -p "$WORK/.ssh" "$HOME/.ssh"; chmod 700 "$WORK/.ssh" "$HOME/.ssh"
if [ -s "$SSH_BACKUP" ]; then
  # Merge rather than overwrite, and de-dupe, so a key added by either side survives.
  cat "$SSH_BACKUP" "$HOME/.ssh/authorized_keys" 2>/dev/null \
    | sort -u | grep -v '^[[:space:]]*$' > /tmp/ak.merged
  install -m 600 /tmp/ak.merged "$HOME/.ssh/authorized_keys"
  echo "  restored $(wc -l < "$HOME/.ssh/authorized_keys") key(s) from the volume"
elif [ -s "$HOME/.ssh/authorized_keys" ]; then
  install -m 600 "$HOME/.ssh/authorized_keys" "$SSH_BACKUP"
  echo "  backed up $(wc -l < "$SSH_BACKUP") key(s) to the volume for next restart"
else
  echo "  !! no authorized_keys anywhere. Add your key now or the next stop locks you out:"
  echo "     echo '<your-public-key>' >> $SSH_BACKUP"
fi

log "GPU"
# Diagnose properly. A restarted pod can come back with the host driver visible but
# no GPU passed through to the container, which is what happened on 2026-07-28: the
# old check just printed "no GPU" and gave no clue where to look.
if [ ! -e /dev/nvidiactl ] && ! ls /dev/nvidia[0-9]* >/dev/null 2>&1; then
  printf '\n\033[1;31m'
  echo "No GPU in this container."
  if [ -r /proc/driver/nvidia/version ]; then
    echo "  The HOST driver is present ($(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version))"
    echo "  but no /dev/nvidia* devices are mapped in — the GPU was not passed"
    echo "  through when the pod started."
  else
    echo "  No host driver either."
  fi
  printf '\033[0m'
  cat <<'EOG'

  Fix in the RunPod dashboard (nothing here can attach a GPU):
    1. Stop the pod, then Start it again — RunPod may reallocate.
    2. If it comes back without a GPU, terminate and create a NEW pod attached to
       the SAME network volume. /workspace is mfs#...runpod.net (a network volume),
       so it survives termination and you lose none of the 51GB of weights.

EOG
  exit 1
fi
command -v nvidia-smi >/dev/null \
  && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || echo "  devices present but nvidia-smi missing (unusual — torch may still work)"

log "System packages"
if ! command -v ffmpeg >/dev/null || ! command -v zstd >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  # zstd is required by the Ollama installer; pciutils lets it detect the GPU
  apt-get install -y -qq git git-lfs ffmpeg build-essential curl zstd pciutils >/dev/null
  echo "  installed"
else
  echo "  already present"
fi

log "Python venv ($VENV — container disk, NOT the volume: see docs/LATENCY.md)"
if [ -x "$VENV/bin/python" ]; then
  echo "  already present"
else
  echo "  (pip cache on the volume: $PIP_CACHE_DIR — no re-download after a wipe)"
  python3 -m venv "$VENV" --system-site-packages
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install -q --upgrade pip wheel
  python -c 'import torch;assert torch.cuda.is_available()' || die "torch cannot see CUDA"
  pip install -q faster-whisper "huggingface_hub[cli]" moshi torchtune torchao \
                 fastapi "uvicorn[standard]" python-multipart httpx kokoro soundfile
  # CSM's own requirements, minus the torch pins that would replace the image's
  # working CUDA build.
  grep -viE '^(torch|torchaudio)([=<>~!]|$)' "$WORK/csm/requirements.txt" > /tmp/csm-reqs.txt
  pip install -q -r /tmp/csm-reqs.txt
  echo "  built"
fi

log "Ollama binary"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
ollama --version 2>/dev/null | head -1 || true

log "Model blobs -> container disk (this is the ~40% TTFT win)"
if [ -d "$OLLAMA_LOCAL/blobs" ]; then
  echo "  already present"
else
  need=$(du -sm "$WORK/.ollama" | cut -f1)
  free=$(df -m / | awk 'NR==2{print $4}')
  if [ "$free" -lt $((need + 2048)) ]; then
    echo "  !! only ${free}MB free on / but need ~${need}MB — staying on the volume."
    echo "     Expect TTFT ~1.8s instead of ~1.1s. Raise the pod's container disk to fix."
    OLLAMA_LOCAL="$WORK/.ollama"
  else
    mkdir -p "$OLLAMA_LOCAL"
    cp -r "$WORK/.ollama/." "$OLLAMA_LOCAL/"
    echo "  copied ${need}MB"
  fi
fi

log "Starting Ollama (resident, logging to LOCAL disk)"
# KEEP_ALIVE=-1: the default 5-min unload costs ~30s on the next request.
# Log to /var/log, NOT the volume: a stale network-storage fd killed llama-server
# with SIGPIPE on every model load.
if ! curl -s --max-time 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  setsid nohup env OLLAMA_MODELS="$OLLAMA_LOCAL" OLLAMA_KEEP_ALIVE=-1 \
    ollama serve > /var/log/ollama.log 2>&1 < /dev/null &
  for _ in $(seq 1 20); do
    curl -s --max-time 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
    sleep 2
  done
fi
curl -s http://127.0.0.1:11434/api/version || die "ollama did not come up"
echo ""

log "Starting the API on :$PORT"
PIDS=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | sort -u || true)
[ -n "$PIDS" ] && { kill $PIDS 2>/dev/null || true; sleep 3; }
cd "$WORK/server"
setsid nohup env HF_HOME="$WORK/.hf" PYTHONPATH="$WORK/csm:$WORK/server" \
  "$VENV/bin/python" -m uvicorn app:app --host 0.0.0.0 --port "$PORT" \
  > /var/log/wemendai.log 2>&1 < /dev/null &

log "Waiting for models (CSM ~40s, Whisper ~2min on a cold volume)"
for _ in $(seq 1 60); do
  curl -s --max-time 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ready":true' && break
  sleep 8
done

echo ""
curl -s "http://127.0.0.1:$PORT/health"; echo ""
grep -a 'tts:csm\|\[models\]' /var/log/wemendai.log | tail -3

cat <<EOF

  pip cache on volume: $(du -sh "$PIP_CACHE_DIR" 2>/dev/null | cut -f1 || echo 0)
  SSH keys backed up:  $SSH_BACKUP

  READY. Public URL (RunPod proxies :$PORT):
    https://<pod-id>-$PORT.proxy.runpod.net/health

  The pod id changes if the pod is recreated — update Config.swift in wemend-ios
  if the app can't reach it. Logs: /var/log/wemendai.log, /var/log/ollama.log
EOF
