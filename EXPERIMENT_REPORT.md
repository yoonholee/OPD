# Qwen3.5-2B OPD experiment status

Date: 2026-05-25
Branch: `qwen35-2b-opd-smoke`
Latest pushed commit: `811bf42 feat: pin qwen35 opd smoke env`

## Summary

The THUNLP/OPD codebase now has a reproducible Qwen3.5-2B smoke stack.
The main successful setup is GCP G4, 4x RTX PRO 6000 Blackwell, stable Python package pins, and text-only Qwen3.5 handling.

This proves the infrastructure path for GRPO and OPD on one 4-GPU node.
It does not prove learning quality yet, because the GRPO reward smoke produced all-zero rewards and OPD used a same-weight teacher.

## Current working recipe

Environment:

- `local/qwen35_2b_smoke/pyproject.toml`
- `local/qwen35_2b_smoke/uv.lock`
- `transformers==5.9.0`
- `vllm==0.21.0`
- Torch `2.11.0+cu130`
- NumPy `2.3.5`
- Ray `2.55.1`

Runtime knobs:

- `data.return_multi_modal_inputs=False`
- `VLLM_USE_FLASHINFER_SAMPLER=0`
- `gdn_prefill_backend=triton`
- route-detected `NCCL_SOCKET_IFNAME`
- `use_remove_padding=False` by default, despite a small passing probe

## Main 10-step results

Hardware: GCP `g4-standard-192`, 4 GPUs, Spot, `us-west1-a`.
Dataset: local DAPO-Math smoke, `SMOKE_N=96`.
Batch: 8.
Response length: 64.
Metrics below exclude step 1 warmup.

| Run | Status | Step time | Throughput | Max GPU alloc | Signal |
| --- | --- | ---: | ---: | ---: | --- |
| GRPO | pass | 2.26s | 148.7 tok/s | 58.9 GB | rewards zero, no learning signal |
| OPD same-weight teacher | pass | 2.51s | 138.8 tok/s | 58.9 GB | nonzero reward/grad from train-teacher drift |
| GRPO remove-padding probe | pass, 3 steps | 2.00s | 63.4 tok/s | 49.4 GB | batch 4, not apples-to-apples |

Evidence:

- GRPO: `agent_notes/gcp_runs/opd-g4-qwen35-scale5-0524-1557/gcp_runs/20260524_225852_g4_grpo10_b8_r64_textonly_no_flashinfer_sampler/grpo_qwen35_2b.log`
- OPD: `agent_notes/gcp_runs/opd-g4-qwen35-opdfix-0524-1609/gcp_runs/20260524_231105_g4_opd10_b8_r64_reshape_textonly_no_flashinfer_sampler/opd_qwen35_2b.log`
- remove-padding: `agent_notes/gcp_runs/opd-g4-qwen35-scale5-0524-1557/gcp_runs/20260524_230425_g4_rmpad_probe_textonly_no_flashinfer_sampler/grpo_qwen35_2b.log`

## What changed

Code and config:

- Added locked smoke project under `local/qwen35_2b_smoke/`.
- Updated `local/bin/setup_qwen35_2b_env.sh` to use `uv sync --frozen`.
- Added tunable batch/length/vLLM knobs to `local/bin/run_qwen35_2b_smoke.sh`.
- Generalized `local/bin/gcp_qwen35_2b_l4_smoke.sh` for G4/A2/A3 shapes, Spot, boot disk type, and custom commands.
- Fixed OPD reward entropy by replacing `.view` with `.reshape` for non-contiguous logits.
- Added fallback padding helpers when `flash_attn.bert_padding` is unavailable.

Tests added:

- exact train/rollout logprob match gives identity importance weights
- token and sequence importance sampling ratios
- policy loss applies rollout IS weights
- OPD reward entropy accepts non-contiguous logits

## Root causes

- Qwen3.5-2B is VLM-style, not plain text CausalLM. Text-only batches still carried variable `multi_modal_inputs`, which broke batch collation. Fix: `data.return_multi_modal_inputs=False`.
- Stable packages are now sufficient. Nightly Transformers/vLLM is no longer needed for this smoke.
- OPD reward entropy failed because Qwen3.5 logits were non-contiguous and THUNLP used `.view`. Fix: `.reshape`.
- G4 uses `ens3`, G2/L4 used `ens7`, A2/A100 used `ens8`. Hardcoded NICs are brittle.
- G4 requires `hyperdisk-balanced`, not `pd-balanced`.
- vLLM FlashInfer sampler JIT can stall with colocated Ray workers. Disable it for now.
- Qwen3.5 GDN/FLA Triton warmup makes step 1 slow, around 25 to 26 seconds.

## GPU status

- G4 Spot: works and is the current best path.
- A100 Standard: created, reached `nvidia-smi`, then stalled in vLLM/FlashInfer init during the source run. Stopped to save cost.
- H100 Spot: stocked out in the probed zone.
- All GCP VMs were deleted. Final instance list was empty.

## Interpretation

GRPO:

- Infrastructure path works.
- The reward function/data smoke is too weak. All rewards were zero, so `pg_loss` and grad norm were zero.
- Next GRPO experiment needs a task/reward with nonzero variance.

OPD:

- The end-to-end OPD path works after the reshape fix.
- Same-weight teacher/student still yields nonzero probability deltas because train and inference paths are not exactly identical.
- Treat same-weight OPD as a systems calibration test, not a quality result.

Probability drift:

- GRPO train-vs-rollout max prob diff mean: 0.059.
- OPD train-vs-rollout max prob diff mean: 0.058.
- OPD top-k overlap mean: 0.989.
- Same-weight OPD produced nonzero gradients, so real OPD needs drift calibration or a deadband.

## Recommended next experiment

1. Keep this G4 locked setup as the baseline.
2. Run a real-reward GRPO task where reward variance is nonzero.
3. Run OPD with a stronger teacher or later-checkpoint teacher.
4. Add same-weight teacher calibration before using raw probability deltas as reward.
5. Only then test async or self-improving teacher loops.
6. Re-run remove-padding apples-to-apples at batch 8 before flipping the default.

## Verification run locally

- `bash -n local/bin/setup_qwen35_2b_env.sh local/bin/run_qwen35_2b_smoke.sh local/bin/gcp_qwen35_2b_l4_smoke.sh`
- `uv lock --project local/qwen35_2b_smoke --check`
- `pytest -q verl/tests/trainer/ppo/test_rollout_corr_matching.py`
- `pytest -q verl/tests/workers/test_fsdp_workers.py::test_reward_entropy_accepts_non_contiguous_logits`
- `ruff check --select E9,F63,F7,F82 ...`
- `gcloud compute instances list`, no rows
