#!/usr/bin/env bash
set -u -o pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
VENV=${VENV:-$ROOT/.venv-qwen35-2b}
MODEL=${MODEL:-Qwen/Qwen3.5-2B}
MODE=${1:-all}
NGPUS=${NGPUS:-4}
TRAIN_STEPS=${TRAIN_STEPS:-1}
SMOKE_N=${SMOKE_N:-8}
LOGDIR=${LOGDIR:-$ROOT/agent_notes/gcp_runs/$(date +%Y%m%d_%H%M%S)_qwen35_2b}
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

mkdir -p "$LOGDIR"
cd "$ROOT" || exit 1

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
  return 0
}

BASE_ENV="source '$VENV/bin/activate';
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
ray stop --force || true;
ray start --head;"

DATA_ARGS="data.shuffle=False \
data.train_files=datasets/local_smoke/dapo_math_train_${SMOKE_N}.parquet \
data.val_files='[datasets/test_data/AIME24/test.parquet]' \
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

ACTOR_ARGS="actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
actor_rollout_ref.actor.use_kl_loss=False \
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
actor_rollout_ref.rollout.val_kwargs.n=1 \
actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
actor_rollout_ref.rollout.calculate_log_probs=True \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton \
+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=0 \
+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.video=0"

TRAINER_ARGS="trainer.val_before_train=False \
trainer.logger=[console] \
trainer.project_name=opd_smoke \
trainer.n_gpus_per_node=${NGPUS} \
trainer.nnodes=1 \
trainer.save_freq=-1 \
trainer.test_freq=-1 \
trainer.total_epochs=1 \
trainer.total_training_steps=${TRAIN_STEPS} \
trainer.log_val_generations=1"

CUSTOM_REWARD="custom_reward_function.path=verl/verl/utils/reward_score/ttrl_math/__init__.py custom_reward_function.name=reward_func"

run prepare_data "source '$VENV/bin/activate'; python local/bin/prepare_smoke_data.py --n '$SMOKE_N' --root '$ROOT'"

if [[ "$MODE" == "all" || "$MODE" == "probe" ]]; then
  run probe_transformers "source '$VENV/bin/activate'; CUDA_VISIBLE_DEVICES=0 python local/bin/probe_qwen35_compat.py --model '$MODEL' --backend transformers"
  run probe_vllm "source '$VENV/bin/activate'; CUDA_VISIBLE_DEVICES=0 python local/bin/probe_qwen35_compat.py --model '$MODEL' --backend vllm"
fi

if [[ "$MODE" == "all" || "$MODE" == "grpo" ]]; then
  run grpo_qwen35_2b "${BASE_ENV} python -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo algorithm.grpo_outcome_weight=1.0 \
${DATA_ARGS} ${MODEL_ARGS} ${ACTOR_ARGS} ${ROLLOUT_ARGS} \
reward_model.enable=False ${CUSTOM_REWARD} ${TRAINER_ARGS} \
trainer.experiment_name=grpo_qwen35_2b trainer.default_local_dir=checkpoint/qwen35_2b_grpo; ray stop --force || true"
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
reward_model.model.path=${MODEL} \
reward_model.model.trust_remote_code=True \
reward_model.model.input_tokenizer=null \
reward_model.model.use_remove_padding=${USE_REMOVE_PADDING} \
reward_model.model.fsdp_config.param_offload=True \
+reward_model.model.dtype=bfloat16 \
+reward_model.model.attn_implementation=${ATTN_IMPLEMENTATION} \
reward_model.micro_batch_size_per_gpu=${REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU} \
${CUSTOM_REWARD} ${TRAINER_ARGS} \
trainer.experiment_name=opd_qwen35_2b trainer.default_local_dir=checkpoint/qwen35_2b_opd trainer.is_plot=False; ray stop --force || true"
fi

echo "$LOGDIR" | tee "$ROOT/agent_notes/latest_qwen35_2b_run_dir.txt"
