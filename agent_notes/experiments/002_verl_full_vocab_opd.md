# 002: verl full-vocab OPD objective matrix

Status: complete for tiny GCP smoke.

## Question

Can verl run full-vocab distillation losses end-to-end for reverse KL, forward KL, JSD, symmetric KL, top-k reverse KL, and entropy-aware switching?

## Scope

Compared:
- GRPO baseline
- existing top-k OPD default: `only_stu + student_p`
- full-vocab `reverse_kl`
- full-vocab `forward_kl`
- full-vocab `jsd`
- full-vocab `sym_kl`
- full-vocab `topk_rkl`
- full-vocab `entropy_aware`

Not measured:
- accuracy, hparam quality, long-run stability, remove-padding path, or prompt/context selection.

## Setup

- Code commit: `c502338`
- GCP project: `soe-iris-gcp`
- Zone: `us-central1-b`
- Machine: `a2-highgpu-1g`, A100 spot
- Train steps: 2
- Smoke rows: 4
- Response length: 32
- Full-vocab path: `use_remove_padding=False`
- Unit check before each matrix: `PYTHONPATH=verl python verl/tests/trainer/ppo/test_full_vocab_distill.py`

Runs:
- Qwen3: `Qwen/Qwen3-0.6B` student, `Qwen/Qwen3-1.7B` teacher, VM `opd-fullvocab-qwen3-a100-c1b-0528-0836`
- Qwen3.5: `Qwen/Qwen3.5-0.8B` student, `Qwen/Qwen3.5-2B` teacher, VM `opd-fullvocab-qwen35-a100-c1b-0528-0858`

Ignored raw logs:
- `agent_notes/gcp_runs/opd-fullvocab-qwen3-a100-c1b-0528-0836/gcp_runs/20260528_154210_verl_full_vocab_opd/`
- `agent_notes/gcp_runs/opd-fullvocab-qwen35-a100-c1b-0528-0858/gcp_runs/20260528_160205_verl_full_vocab_opd/`

Repro shape:

```bash
MODEL=Qwen/Qwen3-0.6B TEACHER_MODEL=Qwen/Qwen3-1.7B \
NGPUS=1 TRAIN_STEPS=2 SMOKE_N=4 LOG_PROB_TOP_K=0 TIMEOUT=45m \
bash local/bin/run_verl_full_vocab_opd_matrix.sh

MODEL=Qwen/Qwen3.5-0.8B TEACHER_MODEL=Qwen/Qwen3.5-2B \
NGPUS=1 TRAIN_STEPS=2 SMOKE_N=4 LOG_PROB_TOP_K=0 TIMEOUT=60m \
bash local/bin/run_verl_full_vocab_opd_matrix.sh
```

## Claim

Full-vocab OPD objectives now run end-to-end in verl on Qwen3 and Qwen3.5 tiny A100 smokes.

Axis: runtime correctness, finite losses, gradient flow, and rough throughput.

This is not an accuracy or hparam claim.

## Results

Both matrices passed:
- Qwen3: 8/8 rc0
- Qwen3.5: 8/8 rc0

Qwen3, last logged step:

| case | loss | grad norm | tok/s |
|---|---:|---:|---:|
| grpo | 0.000 | 0.0 | 110.3 |
| topk_opd | 0.229 | 36.2 | 70.4 |
| full_reverse_kl | 0.159 | 27.8 | 67.4 |
| full_forward_kl | 0.076 | 15.9 | 66.9 |
| full_jsd | 0.028 | 5.4 | 67.5 |
| full_sym_kl | 0.128 | 25.2 | 68.0 |
| full_topk_rkl | 0.157 | 27.5 | 67.4 |
| full_entropy_aware | 0.116 | 25.5 | 67.5 |

Qwen3.5, last logged step:

| case | loss | grad norm | tok/s |
|---|---:|---:|---:|
| grpo | 0.000 | 0.0 | 42.6 |
| topk_opd | -0.053 | 596.0 | 35.8 |
| full_reverse_kl | 0.200 | 157.0 | 33.2 |
| full_forward_kl | 0.163 | 320.0 | 33.0 |
| full_jsd | 0.033 | 27.6 | 33.1 |
| full_sym_kl | 0.179 | 764.0 | 33.8 |
| full_topk_rkl | 0.200 | 280.0 | 33.3 |
| full_entropy_aware | 0.160 | 148.0 | 33.8 |

## Interpretation

The integration works: teacher full-vocab logprobs reach actor update, sampled rewards are zeroed, actor loss uses full-vocab divergence, and gradients are nonzero.

JSD has the smallest loss and gradient scale in both tiny smokes.

Qwen3.5 has much larger grad norms than Qwen3 for top-k OPD and several full-vocab losses, especially `sym_kl` and `forward_kl`.

Do not read this as quality. It is a two-step wiring and scale check.

## Regressions and limits

- Full-vocab path currently requires `use_remove_padding=False`.
- Full-vocab tensors were only tested at batch 1 and response length 32.
- Validation accuracy is intentionally not meaningful in this smoke.
- Qwen3.5 teacher/student pair is `0.8B -> 2B`; no `1.7B` Qwen3.5 public text pair was used.

## Decision

Keep full-vocab losses in the main stack.

Default next comparison set: `topk_opd`, `reverse_kl`, `forward_kl`, `jsd`, and `entropy_aware`.

Before longer runs, tune LR or grad clipping for Qwen3.5 full-vocab losses, with JSD as the likely safest scale baseline.
