#!/usr/bin/env bash
# Launch the GRPO and OPD paper probes on GCP H100 8x spot.
# SFT moved to native verl and is not part of this legacy launcher.
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
NAME=${NAME:-opd-paperpair-h100-$(date +%m%d-%H%M)}

export MACHINE=${MACHINE:-a3-highgpu-8g}
# a3-highgpu-8g bundles 8x H100 80GB; do not pass --accelerator.
export ACCELERATOR=${ACCELERATOR:-}
export PROVISIONING_MODEL=${PROVISIONING_MODEL:-SPOT}
export SPOT_TERMINATION_ACTION=${SPOT_TERMINATION_ACTION:-DELETE}
export BOOT_DISK_TYPE=${BOOT_DISK_TYPE:-pd-balanced}
export BOOT_DISK_SIZE=${BOOT_DISK_SIZE:-1000GB}
export ZONE=${ZONE:-us-central1-a}
export NAME

STUDENT=Qwen/Qwen3-1.7B-Base
TEACHER=Qwen/Qwen3-4B-Base

# Shared knobs for verl GRPO/OPD legs.
# Tuning round 2 (2026-05-27): GRPO mode-collapsed at step 60 of prior run.
# Fix: enable KL anchor (use_kl_loss=True), lower LR by 2x, bump SMOKE_N for 100+ steps.
COMMON_ENV="\
export MODEL='$STUDENT'
export NGPUS=8
export TRAIN_STEPS=100
export SMOKE_N=4096
export MAX_PROMPT_LENGTH=1280
export MAX_RESPONSE_LENGTH=1024
export ROLLOUT_N=4
export PPO_MAX_TOKEN_LEN_PER_GPU=8192
export VLLM_MAX_MODEL_LEN=2560
export VLLM_MAX_NUM_BATCHED_TOKENS=4096
export VLLM_MAX_NUM_SEQS=128
export VLLM_GPU_MEMORY_UTILIZATION=0.55
export TRAIN_BATCH_SIZE=32
export PPO_MINI_BATCH_SIZE=32
export ACTOR_LR=5e-7
export USE_KL_LOSS=True
export KL_LOSS_COEF=0.001
export KL_LOSS_TYPE=low_var_kl
export ENTROPY_COEFF=0"

export RUN_COMMANDS="
ulimit -n 1048576 || true
$COMMON_ENV
TIMEOUT=90m bash local/bin/run_qwen35_2b_smoke.sh grpo
$COMMON_ENV
export TEACHER_MODEL='$TEACHER'
TIMEOUT=120m bash local/bin/run_qwen35_2b_smoke.sh opd
"

exec bash "$ROOT/local/bin/gcp_qwen35_2b_l4_smoke.sh"
