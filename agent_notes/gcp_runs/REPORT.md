# Qwen3.5-2B OPD smoke report, 2026-05-24

## Bottom line

THUNLP/OPD now runs `Qwen/Qwen3.5-2B` on one 4-GPU GCP node with stable pins, not nightly wheels.
Use the committed `local/qwen35_2b_smoke/pyproject.toml` and `uv.lock`.

Best current recipe: G4 Spot, 4x RTX PRO 6000 Blackwell, `transformers==5.9.0`, `vllm==0.21.0`, Torch `2.11.0+cu130`, `data.return_multi_modal_inputs=False`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `gdn_prefill_backend=triton`.

## 10-step scale runs

Hardware: GCP `g4-standard-192`, 4x RTX PRO 6000 Blackwell, Spot, `us-west1-a`.
Data: local DAPO-Math smoke, `SMOKE_N=96`, batch 8 for 10-step GRPO/OPD, response 64.
Metrics below use steps 2-10 to remove first-step vLLM/GDN warmup.

| Run | Status | Step time | Throughput | Max alloc | Signal |
| --- | --- | ---: | ---: | ---: | --- |
| GRPO, batch 8 | rc 0 | 2.26s | 148.7 tok/s | 58.9 GB | Rewards all zero, so pg loss/grad norm zero. Infrastructure smoke only. |
| OPD same-weight teacher, batch 8 | rc 0 | 2.51s | 138.8 tok/s | 58.9 GB | Nonzero OPD rewards from train/teacher drift. Grad norm mean 0.79, top-k overlap mean 0.989. |
| GRPO remove-padding probe, batch 4 | rc 0 | 2.00s | 63.4 tok/s | 49.4 GB | `Actor use_remove_padding=True` ran 3 steps. Not yet apples-to-apples. |

Evidence:

- GRPO 10: `agent_notes/gcp_runs/opd-g4-qwen35-scale5-0524-1557/gcp_runs/20260524_225852_g4_grpo10_b8_r64_textonly_no_flashinfer_sampler/grpo_qwen35_2b.log`
- OPD 10 fixed: `agent_notes/gcp_runs/opd-g4-qwen35-opdfix-0524-1609/gcp_runs/20260524_231105_g4_opd10_b8_r64_reshape_textonly_no_flashinfer_sampler/opd_qwen35_2b.log`
- Remove-padding probe: `agent_notes/gcp_runs/opd-g4-qwen35-scale5-0524-1557/gcp_runs/20260524_230425_g4_rmpad_probe_textonly_no_flashinfer_sampler/grpo_qwen35_2b.log`

## What changed in this pass

- Replaced shell-floating/nightly installs with `local/qwen35_2b_smoke/pyproject.toml` plus `uv.lock`.
  - Direct pins: `transformers==5.9.0`, `vllm==0.21.0`, optional `flash-attn==2.8.3` extra.
  - Lock resolved Linux GPU env to Torch `2.11.0+cu130`, NumPy `2.3.5`, Ray `2.55.1`.
- `setup_qwen35_2b_env.sh` now runs `uv sync --frozen` into `.venv-qwen35-2b`, then installs local `verl` editable with `--no-deps`.
- `run_qwen35_2b_smoke.sh` gained tunable batch/length/vLLM knobs, GCP NIC autodetect, text-only VLM collation guard, and vLLM FlashInfer sampler disable.
- `gcp_qwen35_2b_l4_smoke.sh` now handles G4/A2/A3 shapes by allowing empty `ACCELERATOR`, `BOOT_DISK_TYPE`, Spot mode, and custom `RUN_COMMANDS`.
- `RewardModelWorker._compute_entropy_safe` now uses `reshape`, not `view`, so OPD reward entropy accepts non-contiguous logits from the Qwen3.5/Transformers 5 path.
- `attention_utils.py` falls back to PyTorch padding helpers if `flash_attn.bert_padding` is unavailable.
- Added probability correction tests for exact match and token/sequence importance sampling.
- Added a non-contiguous logits entropy test for the OPD reward path.

## Root causes found

- Nightly dependency root cause: Qwen3.5 initially needed newer Transformers/vLLM than THUNLP's frozen stack. Stable `transformers==5.9.0` plus `vllm==0.21.0` now cover the path.
- Batch collation root cause: `Qwen/Qwen3.5-2B` is VLM-style. Text-only batches still produced variable `multi_modal_inputs`; `data.return_multi_modal_inputs=False` fixes batch >1.
- OPD failure root cause: reward logits were non-contiguous. THUNLP code used `.view`; `.reshape` fixes it.
- GCP NIC root cause: interface names differ by machine. Observed G2/L4 `ens7`, G4 `ens3`, A2/A100 `ens8`; route-based autodetect is safer.
- G4 boot disk root cause: G4 requires `hyperdisk-balanced`; `pd-balanced` fails at create.
- vLLM init root cause: FlashInfer sampler JIT can stall/race under colocated Ray workers; `VLLM_USE_FLASHINFER_SAMPLER=0` avoids that path. GDN/FLA Triton warmup still costs the first step.

## GPU probes

- A100: `a2-highgpu-4g`, `us-west1-b`, Standard created and reached `nvidia-smi`, but source run was stopped to save cost after vLLM/FlashInfer JIT stall.
- H100: `a3-highgpu-4g`, `us-west1-a`, Spot stocked out.
- G4: `g4-standard-192`, `us-west1-a`, Spot created. Needs `hyperdisk-balanced`. This was the successful scale path.

Evidence:

- GPU probe logs: `agent_notes/gcp_runs/gpu_type_probe_20260524_150032/`, `agent_notes/gcp_runs/gpu_type_probe_g4_spot_20260524_150412/`
- Failed OPD before reshape: `agent_notes/gcp_runs/opd-g4-qwen35-scale5-0524-1557/gcp_runs/20260524_230239_g4_opd10_b8_r64_textonly_no_flashinfer_sampler/opd_qwen35_2b.log`
- A100 stall evidence: `agent_notes/gcp_runs/opd-a100-qwen35-scale4-std-0524-1540/`

## Throughput notes

- Warm steady-state is fast: GRPO step 2-10 mean 2.26s, OPD step 2-10 mean 2.51s.
- First step is not representative: ~25-26s because vLLM captures CUDA graphs and compiles Qwen3.5 GDN/FLA kernels.
- OPD adds ~0.25s/step over GRPO here, mostly teacher reward/logprob work. `compute_rm_score` after warmup is ~0.215s/step.
- Remove-padding no longer hard-crashes in the 3-step probe, but it needs an apples-to-apples batch-8 run before claiming throughput win.

## Probability drift / OPD caution

Same-weight teacher/student is not mathematically exact under this stack.
Observed train-vs-rollout probability diff in successful 10-step runs:

- GRPO max diff mean: 0.059, mean diff mean: 0.00346.
- OPD max diff mean: 0.058, mean diff mean: 0.00351.
- OPD same-weight teacher still produced rewards with max around 0.05 by step 3 and nonzero gradients.

Mitigations to keep:

- Same tokenizer/chat template/logit processors/dtype/attention backend where possible.
- Temperature 1.0, top-p 1.0, no top-k unless the mask is explicit.
- Use vLLM processed logprobs if sampling processors are enabled.
- Keep exact-match and importance-sampling tests around the train/inference server boundary.
- For real OPD, add same-weight calibration/deadband before treating tiny probability deltas as learning signal.

## Current resource state

All GCP VMs from this pass were deleted.
Final `gcloud compute instances list` returned no rows.
