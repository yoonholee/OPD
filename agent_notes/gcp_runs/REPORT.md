# Qwen3.5-2B OPD smoke report, 2026-05-24

## Bottom line

THUNLP/OPD can run `Qwen/Qwen3.5-2B` on a 4-GPU single node with small local compatibility patches plus Transformers main and vLLM nightly.
This is smoke-scale only, not paper-scale reproduction.

Recommendation: keep THUNLP/OPD as the paper baseline, use these patches for Qwen3.5-2B, and keep `use_remove_padding=False` until the FA2/remove-padding crash is isolated.

## Current verified runs

Hardware: GCP `g2-standard-48`, 4x L4, `us-west1-a`.
Workload: DAPO-Math tiny local smoke, `SMOKE_N=8`, `TRAIN_STEPS=1`, response length 32.

| Run | Status | Step time | Throughput | Max alloc | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| GRPO, Qwen3.5-2B | rc 0 | 11.76s | 14.77 tok/s | 15.71 GB | Rule rewards all zero, so pg loss and grad norm were zero. |
| OPD, same-weight teacher/student | rc 0 | 12.34s | 14.08 tok/s | 15.70 GB | top-k overlap 1.0, reward mean 5.3e-8, grad norm 2.19e-6. |

Evidence:

- GRPO: `opd-qwen35-2b-l4-0524-1159-final/gcp_runs/20260524_122502_qwen35_2b_grpo_retry5/grpo_qwen35_2b.log`
- OPD: `opd-qwen35-2b-l4-0524-1159-final/gcp_runs/20260524_122711_qwen35_2b_opd_retry/opd_qwen35_2b.log`
- Teardown: `opd-qwen35-2b-l4-0524-1159-final/delete-proof.txt`

## What had to change

### Code patches

- Transformers auto-class shim: `verl/verl/utils/transformers_compat.py`
  - Added compatibility for `AutoModelForImageTextToText`.
  - Added `conditional_generation_auto_class()`, `auto_class_from_remote_name()`, and `mapping_keys()` helpers.
  - Reason: Qwen3.5 is a VLM-style repo using `qwen3_5` and image/text auto classes, not the old plain CausalLM path.

- Model loading: `verl/verl/utils/model.py`, `verl/verl/model_merger/base_model_merger.py`
  - Mapped `ForConditionalGeneration` and `ForImageTextToText` to the right HF auto class.
  - Reason: THUNLP's vendored veRL assumed older architecture names.

- Actor/ref and reward worker loading: `verl/verl/workers/fsdp_workers.py`
  - Actor/ref selection now uses the compatibility helpers.
  - RewardModelWorker now uses `get_hf_auto_model_class(model_config)` instead of hardcoded `AutoModelForCausalLM`.
  - RewardModelWorker accepts `attn_implementation` from config.
  - Reason: OPD teacher loading fails for Qwen3.5 if treated as CausalLM.

- vLLM LoRA import shim: `verl/verl/utils/vllm/utils.py`
  - Fallback import for `LoRAModel` from `vllm.lora.worker_manager` or `vllm.lora.lora_model`.
  - Reason: vLLM nightly no longer exposes `vllm.lora.models`.

- Local scripts: `local/bin/setup_qwen35_2b_env.sh`, `local/bin/run_qwen35_2b_smoke.sh`, `local/bin/gcp_qwen35_2b_l4_smoke.sh`
  - Added reproducible env setup, local data prep, Qwen3.5 probes, GRPO smoke, OPD smoke, and GCP lifecycle wrapper.

### Runtime knobs

Required for the passing smoke:

- Transformers main plus vLLM nightly.
  - Why: the frozen stack does not know `qwen3_5`; vLLM stable did not have the needed Qwen3.5 path.

- `actor_rollout_ref.model.use_remove_padding=False`.
  - Why: prior 4xL4 THUNLP runs killed Ray workers with remove-padding enabled, even for Qwen3 text models.

- `actor_rollout_ref.model.enable_activation_offload=False`.
  - Why: activation offload hit `activation_offload.py` tuple assertion during backward.

- `actor_rollout_ref.actor.use_torch_compile=False`, `actor_rollout_ref.ref.use_torch_compile=False`.
  - Why: avoids compile overhead and TorchInductor fragility on small L4 smokes.

- `actor_rollout_ref.rollout.gpu_memory_utilization=0.45`.
  - Why: 0.22 left no available KV/Mamba cache blocks.

- `actor_rollout_ref.rollout.max_num_seqs=32`.
  - Why: default 1024 exceeded available Mamba cache blocks.

- `limit_mm_per_prompt.image=0`, `limit_mm_per_prompt.video=0`.
  - Why: text-only Qwen3.5 vLLM probes still tried multimodal dummy inputs unless both limits were zero.

- `gdn_prefill_backend=triton`.
  - Why: avoids slow/fragile GDN prefill kernel path during vLLM init.

- GCP NCCL socket settings: `NCCL_NET=Socket`, `NCCL_SOCKET_IFNAME=ens7`, `NCCL_IB_DISABLE=1`.
  - Why: G2/L4 uses `ens7`; forcing `eth0` failed.

## Failures fixed on the way

| Failure | Root cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: vllm.lora.models` | vLLM nightly API drift | LoRAModel import fallback |
| No KV/Mamba cache blocks | vLLM memory cap too low | GPU memory utilization 0.45 |
| `max_num_seqs (1024) exceeds available Mamba cache blocks` | Qwen3.5 Mamba cache needs one block per decode seq | `max_num_seqs=32` |
| Activation offload tuple assertion | THUNLP activation offload incompatible with this Qwen3.5/Transformers-main checkpointing path | Disable activation offload |
| NCCL invalid usage / no socket interface | Wrong interface assumption on GCP G2 | `NCCL_SOCKET_IFNAME=ens7` |
| Tarball missed `verl.utils.checkpoint` | `--exclude checkpoint` matched vendored code path | Do not globally exclude `checkpoint` |

## Earlier baseline smokes

- THUNLP GRPO, Qwen3-0.6B, 4xL4, 1 step, `use_remove_padding=False`.
  - Evidence: `20260524_182510_l4_final/thunlp_grpo_qwen3_0p6b_no_remove_padding_1024.log`
  - rc 0, throughput 24.97 tok/s, max alloc 5.81 GB, step 6.96s.

- THUNLP OPD, same-weight teacher/student, Qwen3-0.6B, 4xL4, 1 step, `use_remove_padding=False`.
  - Evidence: `20260524_182510_l4_final/thunlp_opd_qwen3_0p6b_no_remove_padding_1024.log`
  - rc 0, throughput 27.33 tok/s, top-k overlap 1.0, grad norm 7.7e-7.

- Clean 4-GPU NCCL all-reduce on L4 passed after stopping Ray.
  - Evidence: `20260524_182510_l4_final/nccl_clean_after_ray_stop.log`

## Fork/import notes

Import now from THUNLP/OPD:

- OPD token reward path and top-k diagnostics.
- Paper-matching configs and DAPO-Math processing.

Import selectively from upstream veRL:

- `vexact` zero-mismatch design as a north star.
- Token rollout importance-sampling hooks.
- `fully_async_policy`, `one_step_off_policy`, and transfer queue only after the baseline is stable.

Import selectively from EasyR1:

- LoRA adapter tensor sync into vLLM.
- DAPO online filtering and clip low/high/dual config.
- Better small-model examples and config ergonomics.
- Use Docker/Apptainer if evaluating seriously. Bare source install was fragile.

Consider Prime-RL later for teacher-loop architecture:

- Explicit `rl`, `opd`, and `sft` modes.
- Separate orchestrator/trainer/inference services.
- Local vLLM teacher requirement for OPD token logprobs.
- Heavier install and submodules made quick source install fail.

Do not prioritize RAGEN/SkyRL for this baseline:

- RAGEN is agent/env oriented.
- SkyRL is interesting for Tinker-compatible APIs and async agent RL, not THUNLP OPD reproduction.

## Teacher-student probability diff mitigations

- Use identical tokenizer, chat template, dtype, attention implementation, TP size, backend, and logit processors.
- Use the same backend for teacher and student logprobs when measuring OPD reward.
- For vLLM, use `processed_logprobs` if temperature is not 1.0.
- Keep temperature 1.0, top_p 1.0, top_k disabled unless the rollout mask is preserved and reused.
- Add same-weight calibration or an epsilon deadband before using raw probability deltas as reward.
- Prefer `use_remove_padding=False` on this stack until rmpad is fixed.

## GCP resources

Attempted:

- 4xH100 spot: unavailable across tried zones.
- 1xH100 spot: provisioned, Qwen3.5 Transformers probe passed, then preempted.
- 4xL4 on-demand: completed final GRPO and OPD smokes.

Deleted:

- `opd-4xl4-0524-1054`, `us-west1-a`
- `opd-1xh100-0524-1032-uswest4a`, `us-west4-a`
- `opd-qwen35-2b-l4-0524-1159`, `us-west1-a`
