#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PROJECT=${PROJECT:-$(gcloud config get-value project)}
ZONE=${ZONE:-us-west1-a}
NAME=${NAME:-opd-qwen35-2b-l4-$(date +%m%d-%H%M)}
MACHINE=${MACHINE:-g2-standard-48}
ACCELERATOR=${ACCELERATOR-type=nvidia-l4,count=4}
PROVISIONING_MODEL=${PROVISIONING_MODEL:-STANDARD}
IMAGE_FAMILY=${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}
IMAGE_PROJECT=${IMAGE_PROJECT:-deeplearning-platform-release}
BOOT_DISK_SIZE=${BOOT_DISK_SIZE:-300GB}
BOOT_DISK_TYPE=${BOOT_DISK_TYPE:-pd-balanced}
LOCAL_RUN_DIR=${LOCAL_RUN_DIR:-$ROOT/agent_notes/gcp_runs/${NAME}}
REMOTE_ROOT=${REMOTE_ROOT:-/home/yoonholee/opd}
RUN_COMMANDS=${RUN_COMMANDS:-"
TIMEOUT=30m bash local/bin/run_qwen35_2b_smoke.sh probe
TIMEOUT=45m bash local/bin/run_qwen35_2b_smoke.sh grpo
TIMEOUT=45m bash local/bin/run_qwen35_2b_smoke.sh opd
"}

mkdir -p "$LOCAL_RUN_DIR"

cleanup() {
  set +e
  gcloud compute scp --project "$PROJECT" --zone "$ZONE" --recurse \
    "$NAME:$REMOTE_ROOT/agent_notes/gcp_runs" "$LOCAL_RUN_DIR/" >/dev/null 2>&1
  gcloud compute instances delete "$NAME" --project "$PROJECT" --zone "$ZONE" --quiet >/dev/null 2>&1
}
trap cleanup EXIT

cd "$ROOT"

CREATE_ARGS=(
  "$NAME"
  --project "$PROJECT" \
  --zone "$ZONE" \
  --machine-type "$MACHINE" \
  --maintenance-policy TERMINATE \
  --provisioning-model "$PROVISIONING_MODEL" \
  --image-family "$IMAGE_FAMILY" \
  --image-project "$IMAGE_PROJECT" \
  --boot-disk-size "$BOOT_DISK_SIZE" \
  --boot-disk-type "$BOOT_DISK_TYPE" \
  --metadata install-nvidia-driver=True \
  --no-service-account \
  --no-scopes \
  --quiet
)
if [[ -n "$ACCELERATOR" ]]; then
  CREATE_ARGS+=(--accelerator "$ACCELERATOR")
fi
if [[ "${SPOT_TERMINATION_ACTION:-}" ]]; then
  CREATE_ARGS+=(--instance-termination-action "$SPOT_TERMINATION_ACTION")
fi

gcloud compute instances create "${CREATE_ARGS[@]}"

for _ in {1..60}; do
  if gcloud compute ssh "$NAME" --project "$PROJECT" --zone "$ZONE" --command "true" --quiet >/dev/null 2>&1; then
    break
  fi
  sleep 10
done

TARBALL=/tmp/${NAME}.tar.gz
COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  --exclude .git \
  --exclude '.venv*' \
  --exclude __pycache__ \
  --exclude 'agent_notes/gcp_runs/*' \
  -czf "$TARBALL" .

gcloud compute scp "$TARBALL" "$NAME:/home/yoonholee/opd.tar.gz" --project "$PROJECT" --zone "$ZONE" --quiet

gcloud compute ssh "$NAME" --project "$PROJECT" --zone "$ZONE" --quiet --command "
set -euo pipefail
rm -rf '$REMOTE_ROOT'
mkdir -p '$REMOTE_ROOT'
tar -xzf /home/yoonholee/opd.tar.gz -C '$REMOTE_ROOT'
cd '$REMOTE_ROOT'
sudo apt-get update
sudo apt-get install -y build-essential curl git
bash local/bin/setup_qwen35_2b_env.sh
$RUN_COMMANDS
"

echo "$LOCAL_RUN_DIR"
