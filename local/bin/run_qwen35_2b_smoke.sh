#!/usr/bin/env bash
set -u -o pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
VENV=${VENV:-$ROOT/.venv-qwen35-2b}
MODEL=${MODEL:-Qwen/Qwen3.5-2B}
TEACHER_MODEL=${TEACHER_MODEL:-$MODEL}
MODE=${1:-all}
NGPUS=${NGPUS:-4}
TRAIN_STEPS=${TRAIN_STEPS:-1}
SMOKE_N=${SMOKE_N:-8}
SLURM_TAG=${SLURM_JOB_ID:+_slurm${SLURM_JOB_ID}}
LOGDIR=${LOGDIR:-$ROOT/agent_notes/gcp_runs/$(date +%Y%m%d_%H%M%S)_qwen35_2b${SLURM_TAG}}
GPU_LIST=${GPU_LIST:-$(seq -s, 0 $((NGPUS - 1)))}
NCCL_SOCKET_IFNAME_DEFAULT=${NCCL_SOCKET_IFNAME:-$(ip route get 8.8.8.8 2>/dev/null | awk '/dev/ {for (i=1; i<=NF; i++) if ($i == "dev") {print $(i+1); exit}}')}
NCCL_SOCKET_IFNAME_DEFAULT=${NCCL_SOCKET_IFNAME_DEFAULT:-ens7}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$NGPUS}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-512}
ROLLOUT_N=${ROLLOUT_N:-1}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.45}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-512}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-${VLLM_MAX_BATCHED_TOKENS:-1024}}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-32}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}
USE_ACTIVATION_OFFLOAD=${USE_ACTIVATION_OFFLOAD:-False}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-False}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-4}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-$VLLM_GPU_MEMORY_UTILIZATION}
MAX_TOKENS_VAL=${MAX_TOKENS_VAL:-$MAX_RESPONSE_LENGTH}
REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU:-1}
RETURN_MULTI_MODAL_INPUTS=${RETURN_MULTI_MODAL_INPUTS:-False}
ACTOR_LR=${ACTOR_LR:-1e-6}
USE_KL_LOSS=${USE_KL_LOSS:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
PEDAGOGICAL_ENABLED=${PEDAGOGICAL_ENABLED:-False}
PEDAGOGICAL_SPIKE_LAMBDA=${PEDAGOGICAL_SPIKE_LAMBDA:-0.2}
PEDAGOGICAL_SPIKE_BETA=${PEDAGOGICAL_SPIKE_BETA:-5.0}
PEDAGOGICAL_REWARD_GATE=${PEDAGOGICAL_REWARD_GATE:-True}
PEDAGOGICAL_REWARD_GATE_MODE=${PEDAGOGICAL_REWARD_GATE_MODE:-raw}
PEDAGOGICAL_TOKEN_GATE=${PEDAGOGICAL_TOKEN_GATE:-False}
PEDAGOGICAL_GATE_GAMMA=${PEDAGOGICAL_GATE_GAMMA:--8.0}
PEDAGOGICAL_GATE_KAPPA=${PEDAGOGICAL_GATE_KAPPA:-1.0}
VAL_FILES=${VAL_FILES:-"[datasets/test_data/MATH-500/test.parquet]"}
VAL_N=${VAL_N:-8}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TEST_FREQ=${TEST_FREQ:-$TRAIN_STEPS}
EXIT_CODE=0

mkdir -p "$LOGDIR"
cd "$ROOT" || exit 1

# File-descriptor headroom for Ray on 8+ GPU shapes (was hitting fd-limit SIGABRTs on a3-highgpu-8g).
ulimit -n 1048576 2>/dev/null || true
ulimit -u 65536 2>/dev/null || true

# Bare `ray.init()` consults RAY_ADDRESS then /tmp/ray/ray_current_cluster; on shared SLURM nodes
# this latches onto a neighbor job's cluster. Unset the env var and rely on per-job RAY_TMPDIR below
# (driver code that uses ray.init(address=...) is unaffected).
unset RAY_ADDRESS

# OpenMP thread explosion from default cpu-count detection has triggered worker fd/thread exhaustion
# on multi-tenant shared nodes (Ray issue #54225, #36936). Cap at 1; verl/vllm don't need OMP threads.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# NCCL env exported globally so subprocesses (incl. LF's torchrun) inherit them.
# On H100 a3-highgpu-8g, NCCL must fall back to Socket because IB/EFA plugins aren't auto-detected via the DLVM image.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_NET=${NCCL_NET:-Socket}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME_DEFAULT}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}


now_iso() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

run() {
  local name=$1
  shift
  echo "===== $name START $(now_iso) =====" | tee "$LOGDIR/$name.status"
  (
    set -e
    timeout "${TIMEOUT:-35m}" bash -lc "set -euo pipefail; $*"
  ) >"$LOGDIR/$name.log" 2>&1
  local rc=$?
  echo "rc=$rc" | tee -a "$LOGDIR/$name.status"
  echo "===== $name END $(now_iso) =====" | tee -a "$LOGDIR/$name.status"
  if [[ "$rc" -ne 0 ]]; then
    EXIT_CODE=$rc
  fi
  return 0
}

BASE_ENV="source '$VENV/bin/activate';
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES;
export CUDA_VISIBLE_DEVICES=$GPU_LIST;
export HF_XET_HIGH_PERFORMANCE=1;
export PYTHONPATH=\$PWD/verl:\${PYTHONPATH:-};
export WANDB_MODE=offline;
export TOKENIZERS_PARALLELISM=true;
export HYDRA_FULL_ERROR=1;
export CUDA_DEVICE_MAX_CONNECTIONS=1;
export NCCL_DEBUG=WARN;
export NCCL_CUMEM_ENABLE=0;
export NCCL_CUMEM_HOST_ENABLE=0;
export NCCL_IB_DISABLE=1;
export NCCL_NET=Socket;
export NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME_DEFAULT};
export RAY_memory_usage_threshold=0.99;
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0};
unset VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_BATCHED_TOKENS VLLM_MAX_BATCHED_TOKENS VLLM_MAX_NUM_SEQS VLLM_GDN_PREFILL_BACKEND;
RAY_FLAGS=\"--num-gpus=${NGPUS}\";
RAY_STOP_CMD=\"ray stop --force || true\";
if [ -n \"\${SLURM_JOB_ID:-}\" ]; then
  # Assign a job-derived port block. 40 slices keeps near-concurrent Slurm job IDs apart
  # while leaving a 1k worker-port range per job.
  # All eight Ray ports must be pinned: --dashboard-agent-listen-port defaults to 52365 (FIXED!),
  # which silently collides if two jobs land on the same node. Same for --metrics-export-port,
  # --runtime-env-agent-port, --node-manager-port, --object-manager-port.
  JOB_SLOT=\$((SLURM_JOB_ID % 40));
  BASE_PORT=\$((6000 + JOB_SLOT * 10));
  RAY_PORT=\$((BASE_PORT + 0));
  RAY_CLIENT=\$((BASE_PORT + 1));
  RAY_DASH=\$((BASE_PORT + 2));
  RAY_DASH_AGENT=\$((BASE_PORT + 3));
  RAY_DASH_AGENT_GRPC=\$((BASE_PORT + 4));
  RAY_RUNTIME_AGENT=\$((BASE_PORT + 5));
  RAY_METRICS=\$((BASE_PORT + 6));
  RAY_NODE_MGR=\$((BASE_PORT + 7));
  RAY_OBJ_MGR=\$((BASE_PORT + 8));
  MIN_WORKER=\$((20000 + JOB_SLOT*1000));
  MAX_WORKER=\$((MIN_WORKER + 999));
  export RAY_TMPDIR=\${TMPDIR:-/tmp}/ray-\${SLURM_JOB_ID};
  mkdir -p \$RAY_TMPDIR/plasma;
  RAY_FLAGS=\"--num-gpus=${NGPUS} --temp-dir=\$RAY_TMPDIR --plasma-directory=\$RAY_TMPDIR/plasma --port=\$RAY_PORT --ray-client-server-port=\$RAY_CLIENT --dashboard-port=\$RAY_DASH --dashboard-agent-listen-port=\$RAY_DASH_AGENT --dashboard-agent-grpc-port=\$RAY_DASH_AGENT_GRPC --runtime-env-agent-port=\$RAY_RUNTIME_AGENT --metrics-export-port=\$RAY_METRICS --node-manager-port=\$RAY_NODE_MGR --object-manager-port=\$RAY_OBJ_MGR --min-worker-port=\$MIN_WORKER --max-worker-port=\$MAX_WORKER\";
  RAY_STOP_CMD=\"true\";
fi;
eval \"\$RAY_STOP_CMD\";
ray start --head \$RAY_FLAGS;"

DATA_ARGS="data.shuffle=False \
data.train_files=datasets/local_smoke/dapo_math_train_${SMOKE_N}.parquet \
data.val_files='${VAL_FILES}' \
data.train_batch_size=${TRAIN_BATCH_SIZE} \
data.max_prompt_length=${MAX_PROMPT_LENGTH} \
data.max_response_length=${MAX_RESPONSE_LENGTH} \
data.filter_overlong_prompts=True \
data.truncation=error \
data.return_raw_chat=True \
data.return_multi_modal_inputs=${RETURN_MULTI_MODAL_INPUTS} \
+data.apply_chat_template_kwargs.enable_thinking=False"

MODEL_ARGS="actor_rollout_ref.model.path=${MODEL} \
actor_rollout_ref.model.trust_remote_code=True \
actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING} \
actor_rollout_ref.model.enable_activation_offload=${USE_ACTIVATION_OFFLOAD} \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}"

ACTOR_ARGS="actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE} \
actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
actor_rollout_ref.actor.use_torch_compile=${USE_TORCH_COMPILE} \
actor_rollout_ref.actor.loss_agg_mode=token-mean \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
actor_rollout_ref.ref.use_torch_compile=${USE_TORCH_COMPILE} \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"

ROLLOUT_ARGS="actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
actor_rollout_ref.rollout.max_model_len=${VLLM_MAX_MODEL_LEN} \
actor_rollout_ref.rollout.max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS} \
actor_rollout_ref.rollout.max_num_seqs=${VLLM_MAX_NUM_SEQS} \
actor_rollout_ref.rollout.n=${ROLLOUT_N} \
actor_rollout_ref.rollout.temperature=1.0 \
actor_rollout_ref.rollout.val_kwargs.do_sample=True \
+actor_rollout_ref.rollout.val_kwargs.max_tokens=${MAX_TOKENS_VAL} \
actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
actor_rollout_ref.rollout.calculate_log_probs=True \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
actor_rollout_ref.rollout.enable_prefix_caching=False \
+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton \
+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=0 \
+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.video=0 \
+actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode=processed_logprobs \
+actor_rollout_ref.rollout.engine_kwargs.vllm.generation_config=vllm"

PEDAGOGICAL_ARGS="++algorithm.pedagogical.enabled=${PEDAGOGICAL_ENABLED} \
++algorithm.pedagogical.spike_lambda=${PEDAGOGICAL_SPIKE_LAMBDA} \
++algorithm.pedagogical.spike_beta=${PEDAGOGICAL_SPIKE_BETA} \
++algorithm.pedagogical.reward_gate=${PEDAGOGICAL_REWARD_GATE} \
++algorithm.pedagogical.reward_gate_mode=${PEDAGOGICAL_REWARD_GATE_MODE} \
++algorithm.pedagogical.token_gate=${PEDAGOGICAL_TOKEN_GATE} \
++algorithm.pedagogical.gate_gamma=${PEDAGOGICAL_GATE_GAMMA} \
++algorithm.pedagogical.gate_kappa=${PEDAGOGICAL_GATE_KAPPA}"

TRAINER_ARGS="trainer.val_before_train=${VAL_BEFORE_TRAIN} \
trainer.logger=[console] \
trainer.project_name=opd_smoke \
trainer.n_gpus_per_node=${NGPUS} \
trainer.nnodes=1 \
trainer.save_freq=-1 \
trainer.test_freq=${TEST_FREQ} \
trainer.total_epochs=1 \
trainer.total_training_steps=${TRAIN_STEPS} \
trainer.log_val_generations=2"

# shellcheck disable=SC2016
RAY_CLEANUP='eval "$RAY_STOP_CMD" || true'

CUSTOM_REWARD="custom_reward_function.path=verl/verl/utils/reward_score/ttrl_math/__init__.py custom_reward_function.name=reward_func"

run prepare_data "source '$VENV/bin/activate'; python local/bin/prepare_smoke_data.py --n '$SMOKE_N' --root '$ROOT'"

if [[ "$MODE" == "all" || "$MODE" == "probe" ]]; then
  run probe_transformers "source '$VENV/bin/activate'; CUDA_VISIBLE_DEVICES=0 python local/bin/probe_qwen35_compat.py --model '$MODEL' --backend transformers"
  run probe_vllm "source '$VENV/bin/activate'; CUDA_VISIBLE_DEVICES=0 python local/bin/probe_qwen35_compat.py --model '$MODEL' --backend vllm"
fi

if [[ "$MODE" == "sft" ]]; then
  echo "LlamaFactory SFT path removed. Use verl native SFT."
  exit 2
fi

if [[ "$MODE" == "all" || "$MODE" == "grpo" ]]; then
  run grpo_qwen35_2b "${BASE_ENV} python -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo algorithm.grpo_outcome_weight=1.0 \
${DATA_ARGS} ${MODEL_ARGS} ${ACTOR_ARGS} ${ROLLOUT_ARGS} \
reward_model.enable=False ${CUSTOM_REWARD} ${TRAINER_ARGS} \
trainer.experiment_name=grpo_qwen35_2b trainer.default_local_dir=checkpoint/qwen35_2b_grpo; ${RAY_CLEANUP}"
fi

if [[ "$MODE" == "all" || "$MODE" == "opd" ]]; then
  run opd_qwen35_2b "${BASE_ENV} python -m verl.trainer.main_ppo \
algorithm.adv_estimator=token_reward_direct \
${DATA_ARGS} ${MODEL_ARGS} ${ACTOR_ARGS} ${ROLLOUT_ARGS} \
+actor_rollout_ref.rollout.log_prob_top_k=${LOG_PROB_TOP_K} \
+actor_rollout_ref.rollout.top_k_strategy=only_stu \
+actor_rollout_ref.rollout.reward_weight_mode=student_p \
+actor_rollout_ref.rollout.teacher_temperature=1.0 \
reward_model.enable=True \
+reward_model.reward_kwargs.enable_format_reward=False \
reward_model.model.path=${TEACHER_MODEL} \
reward_model.model.trust_remote_code=True \
reward_model.model.input_tokenizer=null \
reward_model.model.use_remove_padding=${USE_REMOVE_PADDING} \
reward_model.model.fsdp_config.param_offload=True \
+reward_model.model.dtype=bfloat16 \
+reward_model.model.attn_implementation=${ATTN_IMPLEMENTATION} \
reward_model.micro_batch_size_per_gpu=${REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU} \
${CUSTOM_REWARD} ${PEDAGOGICAL_ARGS} ${TRAINER_ARGS} \
trainer.experiment_name=opd_qwen35_2b trainer.default_local_dir=checkpoint/qwen35_2b_opd trainer.is_plot=False; ${RAY_CLEANUP}"
fi

echo "$LOGDIR" | tee "$ROOT/agent_notes/latest_qwen35_2b_run_dir.txt"
exit "$EXIT_CODE"
