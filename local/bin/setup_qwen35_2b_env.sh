#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
VENV=${VENV:-$ROOT/.venv-qwen35-2b}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
SMOKE_PROJECT=${SMOKE_PROJECT:-$ROOT/local/qwen35_2b_smoke}
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-0}

cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck source=/dev/null
  source "$HOME/.local/bin/env"
fi

uv venv --python "$PYTHON_VERSION" "$VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"

SYNC_ARGS=(--project "$SMOKE_PROJECT" --frozen --no-dev)
if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  SYNC_ARGS+=(--extra flash-attn)
fi
UV_PROJECT_ENVIRONMENT="$VENV" uv sync "${SYNC_ARGS[@]}"

# Avoid verl[vllm], whose vendored constraints can lag current pinned vLLM.
uv pip install -e ./verl --no-deps

python - <<'PY'
import torch, transformers, vllm
from transformers import AutoModelForImageTextToText

print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "image_text_cls": AutoModelForImageTextToText.__name__,
})
PY
