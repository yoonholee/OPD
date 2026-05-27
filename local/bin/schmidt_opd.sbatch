#!/bin/bash
# Single-GPU OPD/GRPO smoke on Schmidt B200/B300.
# Submit per mode: sbatch --export=ALL,MODE=grpo schmidt_opd.sbatch  (or MODE=opd)
#
#SBATCH --partition=b300
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --job-name=opd-smoke
#SBATCH --output=/scratch/schmidt/ssci-cbfinn/%u/slurm-logs/%x-%j.out
#SBATCH --error=/scratch/schmidt/ssci-cbfinn/%u/slurm-logs/%x-%j.err

set -euo pipefail

module load cuda/13.0.2

# Schmidt scratch + cache layout (mirror starters/schmidt-vllm/env.sh).
export PROJECT=${PROJECT:-/scratch/schmidt/ssci-cbfinn}
export USER_SCRATCH=${USER_SCRATCH:-$PROJECT/$USER}
mkdir -p "$USER_SCRATCH"/cache/{hf,uv,vllm,tmp} "$USER_SCRATCH/slurm-logs"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=${HF_HOME:-$USER_SCRATCH/cache/hf}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$USER_SCRATCH/cache/uv}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$USER_SCRATCH/cache/vllm}
export TMPDIR=${TMPDIR:-$USER_SCRATCH/cache/tmp}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

# Per-mode knobs for single-GPU verl GRPO/OPD on B200/B300 (192GB HBM).
# Tuned hparams: KL anchor + low LR + entropy bonus (matches GCP G4 path).
export MODEL=Qwen/Qwen3-1.7B-Base
export NGPUS=1
export TRAIN_STEPS=${TRAIN_STEPS:-100}
export SMOKE_N=${SMOKE_N:-4096}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1280}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
export ROLLOUT_N=${ROLLOUT_N:-4}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-2560}
export VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.55}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
export ACTOR_LR=${ACTOR_LR:-5e-7}
export USE_KL_LOSS=${USE_KL_LOSS:-True}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
export KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}
export ENTROPY_COEFF=${ENTROPY_COEFF:-0}
export VAL_N=${VAL_N:-8}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export VAL_FILES='[datasets/test_data/MATH-500/test.parquet]'

if [ "${MODE:-grpo}" = "opd" ]; then
  export TEACHER_MODEL=Qwen/Qwen3-4B-Base
fi

cd "$SLURM_SUBMIT_DIR"
if [ ! -x .venv-qwen35-2b/bin/python ]; then
  echo "missing .venv-qwen35-2b; run local/bin/setup_schmidt_env.sh on login first" >&2
  exit 2
fi

ulimit -n 1048576 || true

hostname | tee -a "$USER_SCRATCH/slurm-logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.host"
nvidia-smi -L | tee -a "$USER_SCRATCH/slurm-logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.host"

bash local/bin/run_qwen35_2b_smoke.sh "${MODE:-grpo}"
