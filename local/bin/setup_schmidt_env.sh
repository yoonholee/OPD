#!/usr/bin/env bash
# One-time setup on Schmidt login node: install verl venv + prefetch HF models.
# Run from /scratch/schmidt/ssci-cbfinn/$USER/repos/opd/ on login.
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
VENV=${VENV:-$ROOT/.venv-qwen35-2b}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
SMOKE_PROJECT=${SMOKE_PROJECT:-$ROOT/local/qwen35_2b_smoke}

# Schmidt scratch + cache layout (mirror of starters/schmidt-vllm/env.sh).
export PROJECT="${PROJECT:-/scratch/schmidt/ssci-cbfinn}"
export USER_SCRATCH="${USER_SCRATCH:-$PROJECT/$USER}"
mkdir -p "$USER_SCRATCH"/cache/{hf,uv,vllm,tmp} "$USER_SCRATCH/slurm-logs"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$USER_SCRATCH/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$USER_SCRATCH/cache/uv}"
export TMPDIR="${TMPDIR:-$USER_SCRATCH/cache/tmp}"
export UV_LINK_MODE=symlink
export HF_XET_HIGH_PERFORMANCE=0
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_DOWNLOAD_MAX_WORKERS=1

cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck source=/dev/null
  source "$HOME/.local/bin/env"
fi

uv venv --python "$PYTHON_VERSION" "$VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"

UV_PROJECT_ENVIRONMENT="$VENV" uv sync --project "$SMOKE_PROJECT" --frozen --no-dev

# Editable verl install on top, mirroring the GCP setup.
uv pip install -e ./verl --no-deps

# Prefetch the student + teacher models so GPU jobs run offline.
echo "=== prefetching models ==="
for MODEL_ID in Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base; do
  echo "--- $MODEL_ID ---"
  lock="$HF_HOME/prefetch.lock"
  flock "$lock" python - "$MODEL_ID" <<'PY'
import os, sys, time
from huggingface_hub import snapshot_download
mid = sys.argv[1]
t0 = time.monotonic()
path = snapshot_download(
    repo_id=mid,
    max_workers=int(os.environ.get("HF_DOWNLOAD_MAX_WORKERS", "1")),
    allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.tiktoken","*.py","*.jinja","tokenizer*","vocab*","merges.txt"],
)
print(f"prefetched {mid} -> {path} in {time.monotonic()-t0:.1f}s", flush=True)
PY
done

# Smoke env probe.
python - <<'PY'
import torch, transformers, vllm
print({"torch": torch.__version__, "transformers": transformers.__version__, "vllm": vllm.__version__})
PY

echo "=== setup_schmidt_env.sh done ==="
