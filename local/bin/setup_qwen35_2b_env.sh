#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
VENV=${VENV:-$ROOT/.venv-qwen35-2b}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
TRANSFORMERS_SPEC=${TRANSFORMERS_SPEC:-"transformers @ git+https://github.com/huggingface/transformers.git@main"}
VLLM_INDEX=${VLLM_INDEX:-https://wheels.vllm.ai/nightly}

cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck source=/dev/null
  source "$HOME/.local/bin/env"
fi

uv venv --python "$PYTHON_VERSION" "$VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"

uv pip install -U pip setuptools wheel packaging ninja
uv pip install \
  accelerate \
  codetiming \
  datasets \
  dill \
  hf-transfer \
  hydra-core \
  "math-verify" \
  "numpy<2" \
  pandas \
  peft \
  "pyarrow>=19" \
  pybind11 \
  pylatexenc \
  "ray[default]>=2.41.0" \
  tensorboard \
  "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
  torchdata \
  torchvision \
  wandb \
  qwen-vl-utils

uv pip install "$TRANSFORMERS_SPEC"
uv pip install --torch-backend=auto --extra-index-url "$VLLM_INDEX" vllm
uv pip install "numpy<2"

# Avoid verl[vllm], whose vendored constraint excludes vLLM nightly.
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
