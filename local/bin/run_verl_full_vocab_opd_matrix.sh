#!/usr/bin/env bash
set -u -o pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT" || exit 1

BASE_LOGDIR=${BASE_LOGDIR:-$ROOT/agent_notes/gcp_runs/$(date +%Y%m%d_%H%M%S)_verl_full_vocab_opd}
mkdir -p "$BASE_LOGDIR"

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-1.7B}
NGPUS=${NGPUS:-1}
TRAIN_STEPS=${TRAIN_STEPS:-2}
SMOKE_N=${SMOKE_N:-4}
TIMEOUT=${TIMEOUT:-50m}
OBJECTIVES=${OBJECTIVES:-reverse_kl forward_kl jsd sym_kl topk_rkl entropy_aware}
INCLUDE_BASELINES=${INCLUDE_BASELINES:-True}

COMMON_ENV=(
  ROOT="$ROOT"
  MODEL="$MODEL"
  TEACHER_MODEL="$TEACHER_MODEL"
  NGPUS="$NGPUS"
  TRAIN_STEPS="$TRAIN_STEPS"
  SMOKE_N="$SMOKE_N"
  VAL_FILES="[datasets/local_smoke/dapo_math_train_${SMOKE_N}.parquet]"
  TRAIN_BATCH_SIZE=1
  PPO_MINI_BATCH_SIZE=1
  PPO_MICRO_BATCH_SIZE_PER_GPU=1
  LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
  REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU=1
  MAX_PROMPT_LENGTH=256
  MAX_RESPONSE_LENGTH=32
  PPO_MAX_TOKEN_LEN_PER_GPU=512
  VLLM_MAX_MODEL_LEN=512
  VLLM_MAX_NUM_BATCHED_TOKENS=1024
  VLLM_MAX_NUM_SEQS=16
  VLLM_GPU_MEMORY_UTILIZATION=0.45
  ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
  ROLLOUT_N=1
  USE_REMOVE_PADDING=False
  USE_ACTIVATION_OFFLOAD=False
  USE_TORCH_COMPILE=False
  ATTN_IMPLEMENTATION=sdpa
  VAL_BEFORE_TRAIN=False
  TEST_FREQ=9999
  VAL_N=1
  TIMEOUT="$TIMEOUT"
)

run_case() {
  local name=$1
  shift
  local dir="$BASE_LOGDIR/$name"
  mkdir -p "$dir"
  printf '%s START %s\n' "$name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$dir/matrix.status"
  (
    set -x
    env "${COMMON_ENV[@]}" LOGDIR="$dir" EXPERIMENT_NAME="$name" "$@"
  ) >"$dir/matrix.log" 2>&1
  local rc=$?
  printf 'rc=%s\n%s END %s\n' "$rc" "$name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$dir/matrix.status"
  return "$rc"
}

FAILED=0
if [[ "$INCLUDE_BASELINES" == "True" || "$INCLUDE_BASELINES" == "true" || "$INCLUDE_BASELINES" == "1" ]]; then
  run_case grpo bash local/bin/run_qwen35_2b_smoke.sh grpo || FAILED=1
  run_case topk_opd bash local/bin/run_qwen35_2b_smoke.sh opd || FAILED=1
fi

for objective in $OBJECTIVES; do
  run_case "full_${objective}" \
    env LOG_PROB_TOP_K=0 FULL_VOCAB_OBJECTIVE="$objective" \
    bash local/bin/run_qwen35_2b_smoke.sh opd || FAILED=1
done

python3 local/bin/summarize_verl_opd_runs.py "$BASE_LOGDIR" || true
printf '%s\n' "$BASE_LOGDIR" | tee "$ROOT/agent_notes/latest_full_vocab_opd_run_dir.txt"
exit "$FAILED"
