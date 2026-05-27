#!/usr/bin/env bash
# Create a separate venv for LlamaFactory SFT.
# LF pins transformers<=5.2, trl<=0.24; incompatible with the verl smoke env's transformers==5.9.
# This env is dedicated to the SFT leg only.
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
VENV=${VENV:-$ROOT/.venv-lf-sft}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}

cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck source=/dev/null
  source "$HOME/.local/bin/env"
fi

uv venv --python "$PYTHON_VERSION" "$VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# LF wants its own deps; install editable from the local clone.
uv pip install -e "$ROOT/LlamaFactory"
uv pip install deepspeed swanlab "datasets>=2.16,<=4.0"

python - <<'PY'
import torch, transformers, trl, deepspeed
import llamafactory
print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "trl": trl.__version__,
    "deepspeed": deepspeed.__version__,
    "llamafactory": llamafactory.__version__,
})
PY
