#!/usr/bin/env bash
# Sync the local opd workdir to Schmidt scratch.
# Anchored excludes (leading '/'): only exclude root-level dirs, not nested ones.
# Unanchored `--exclude='checkpoint/'` would have silently nuked
# verl/verl/utils/checkpoint/ and verl/verl/third_party/torch/distributed/checkpoint/,
# breaking `from verl.utils.checkpoint.checkpoint_manager import ...` on Schmidt.
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
REMOTE=${REMOTE:-ssci:/scratch/schmidt/ssci-cbfinn/ssci-yoonho/repos/opd/}

cd "$ROOT"

# Leading '/' anchors the pattern to the source root, so nested dirs of the same
# name (e.g. verl/verl/utils/checkpoint/) are NOT excluded.
rsync -az --stats \
  --exclude='/.git' \
  --exclude='/.venv*' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='/agent_notes/gcp_runs/' \
  --exclude='/checkpoint/' \
  --exclude='/model/' \
  ./ "$REMOTE"

echo "synced to $REMOTE"
