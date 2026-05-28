# OPD full-vocab experiment report

Date: 2026-05-28
Branch: `qwen35-2b-opd-smoke`
Core code commit: `d02c857`
Latest run note commit before this report: `f25d994`

## Summary

We now have full-vocab distillation objectives wired into the local verl stack and verified on GCP A100.
The implemented objectives are reverse KL, forward KL, Jensen-Shannon divergence, symmetric KL, teacher-top-k reverse KL, and an entropy-aware reverse/forward KL switch.
The old top-k OPD reward path is still intact.

The latest useful run is a 50-step Qwen3.5 method matrix.
It used `Qwen/Qwen3.5-0.8B` as student and `Qwen/Qwen3.5-2B` as teacher on one A100 spot VM.
All eight methods completed rc0.

The strongest practical result is simple: `full_jsd` is the cleanest next default.
It decreased over 50 updates, had the smallest loss scale, and had the lowest max gradient norm among full-vocab objectives.
`full_reverse_kl`, `full_forward_kl`, and `full_entropy_aware` also decreased and are worth carrying forward.
Existing top-k OPD and `full_topk_rkl` moved in the wrong direction on this short run and had large gradient spikes.

This is still not a quality result.
Response length is only 32, validation is intentionally meaningless, and the data slice is smoke-scale.
Treat this as a wiring, stability, and loss-scale report.

## What changed in code

Added:

- `verl/verl/trainer/ppo/full_vocab_distill.py`
- `verl/tests/trainer/ppo/test_full_vocab_distill.py`
- `local/bin/run_verl_full_vocab_opd_matrix.sh`
- `local/bin/summarize_verl_opd_runs.py`

Wired:

- actor computes current student full-vocab logprobs on response tokens
- reward worker computes teacher full-vocab logprobs on the same tokens
- trainer passes full-vocab objective metadata through the reward and actor update path
- full-vocab mode disables top-k reward shaping and uses a direct actor divergence loss

Current limitation:

- full-vocab mode requires `use_remove_padding=False`

## Verification

Local static checks:

```bash
python3 -m py_compile \
  local/bin/summarize_verl_opd_runs.py \
  verl/verl/trainer/ppo/full_vocab_distill.py \
  verl/verl/workers/actor/dp_actor.py \
  verl/verl/workers/fsdp_workers.py \
  verl/verl/trainer/ppo/ray_trainer.py \
  verl/verl/workers/config/rollout.py \
  verl/tests/trainer/ppo/test_full_vocab_distill.py

shellcheck local/bin/run_qwen35_2b_smoke.sh local/bin/run_verl_full_vocab_opd_matrix.sh
```

GCP unit check before method matrices:

```bash
source .venv-qwen35-2b/bin/activate
PYTHONPATH=verl python verl/tests/trainer/ppo/test_full_vocab_distill.py
```

GCP GPU status after the latest run:

- no RUNNING GPU instances in `soe-iris-gcp`

## Experiment 001: top-k OPD surface

Purpose: check the existing verl top-k OPD support and weighting paths before adding full-vocab objectives.

Setup:

- commit: `aec0862`
- student: `Qwen/Qwen3-0.6B`
- teacher: `Qwen/Qwen3-1.7B`
- hardware: GCP A100 spot, `a2-highgpu-1g`
- train steps: 2
- top-k: 4

Result:

- 18/18 cases rc0
- `only_stu + student_p` was the best default smoke path
- `reward_weight_mode=none` produced unstable-scale gradients
- `union-intersection + teacher_p` nearly canceled the signal

Tracked note:

- `agent_notes/experiments/001_verl_opd_topk_matrix.md`

## Experiment 002: full-vocab wiring smoke

Purpose: verify full-vocab objectives run end-to-end on Qwen3 and Qwen3.5.

Setup:

- code commit: `c502338`
- hardware: GCP A100 spot, `a2-highgpu-1g`
- train steps: 2 per method
- response length: 32
- methods: GRPO, top-k OPD, reverse KL, forward KL, JSD, symmetric KL, top-k reverse KL, entropy-aware

Results:

- Qwen3 `0.6B -> 1.7B`: 8/8 rc0
- Qwen3.5 `0.8B -> 2B`: 8/8 rc0

Qwen3.5 last-step wiring smoke:

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

Tracked note:

- `agent_notes/experiments/002_verl_full_vocab_opd.md`

## Experiment 003: Qwen3.5 50-step method matrix

Purpose: run each method long enough to see basic loss direction and gradient scale.

Setup:

- code commit: `d02c857`
- run note commit: `f25d994`
- VM: `opd-fullvocab-qwen35-50step-a100-c1b-0528-0941`
- student: `Qwen/Qwen3.5-0.8B`
- teacher: `Qwen/Qwen3.5-2B`
- hardware: GCP A100 spot, `a2-highgpu-1g`, `us-central1-b`
- train steps: 50 per method
- smoke rows: 64
- response length: 32
- actor LR: `1e-6`
- grad clipping: verl default

Repro shape:

```bash
MODEL=Qwen/Qwen3.5-0.8B TEACHER_MODEL=Qwen/Qwen3.5-2B \
NGPUS=1 TRAIN_STEPS=50 SMOKE_N=64 LOG_PROB_TOP_K=0 TIMEOUT=35m \
bash local/bin/run_verl_full_vocab_opd_matrix.sh
```

Result: 8/8 methods rc0.

| case | steps | loss step 1 | loss step 50 | delta | max grad norm | mean tok/s |
|---|---:|---:|---:|---:|---:|---:|
| grpo | 50 | 0.000 | 0.000 | 0.000 | 0 | 40.2 |
| topk_opd | 50 | 0.045 | 0.121 | +0.076 | 14784 | 34.8 |
| full_reverse_kl | 50 | 0.172 | 0.125 | -0.046 | 1744 | 32.3 |
| full_forward_kl | 50 | 0.153 | 0.121 | -0.032 | 3088 | 31.7 |
| full_jsd | 50 | 0.032 | 0.024 | -0.009 | 500 | 32.1 |
| full_sym_kl | 50 | 0.162 | 0.119 | -0.043 | 7840 | 32.0 |
| full_topk_rkl | 50 | 0.159 | 0.195 | +0.036 | 5184 | 32.2 |
| full_entropy_aware | 50 | 0.176 | 0.130 | -0.047 | 2832 | 32.0 |

Last-step snapshot:

| case | loss | grad norm | tok/s |
|---|---:|---:|---:|
| grpo | 0.000 | 0.0 | 32.6 |
| topk_opd | 0.121 | 596.0 | 28.4 |
| full_reverse_kl | 0.125 | 94.5 | 25.7 |
| full_forward_kl | 0.121 | 53.0 | 25.3 |
| full_jsd | 0.024 | 49.0 | 25.9 |
| full_sym_kl | 0.119 | 46.5 | 26.5 |
| full_topk_rkl | 0.195 | 1600.0 | 26.1 |
| full_entropy_aware | 0.130 | 55.2 | 25.9 |

Tracked note:

- `agent_notes/experiments/003_verl_qwen35_50step_methods.md`

Ignored raw logs:

- `agent_notes/gcp_runs/opd-fullvocab-qwen35-50step-a100-c1b-0528-0941/gcp_runs/20260528_164534_verl_full_vocab_opd/`

## Interpretation

Full-vocab JSD is the safest method to try next.
It has the smallest loss, the cleanest gradient scale, and no obvious throughput penalty relative to the other full-vocab losses.

Reverse KL and entropy-aware are also reasonable.
Both decreased over 50 steps and ended with manageable last-step grad norms.

Forward KL decreased too, but it had larger early gradient spikes than JSD.
It remains useful as a comparison method, not the first default.

Symmetric KL decreased, but its max grad norm was large.
It should not be a default until LR or clipping checks are done.

Existing top-k OPD and full top-k reverse KL are not good next spends.
Both moved in the wrong loss direction on the 50-step run and had the largest gradient spikes.

GRPO is included only as an infrastructure baseline here.
Its reward signal is zero in this smoke, so its loss and grad norm are zero.

## Main caveats

This report does not claim model quality.
The run is too short, response length is too small, and validation is not meaningful.

Loss scales are objective-specific.
A lower absolute loss is not automatically a better model.

Gradient norms are pre-clip total norms from `clip_grad_norm_`.
The optimizer steps are clipped, but large pre-clip values still tell us about loss scale.

Full-vocab runs are memory-sensitive.
The current verified path is batch 1, response length 32, `use_remove_padding=False`.

## Decision

Carry forward these methods:

1. `full_jsd`
2. `full_reverse_kl`
3. `full_entropy_aware`
4. `full_forward_kl`

Do not spend more on `full_topk_rkl` or top-k OPD until LR, clipping, or reward normalization is revisited.

Next good run:

- Qwen3.5 `0.8B -> 2B`
- methods: `full_jsd`, `full_reverse_kl`, `full_entropy_aware`, `full_forward_kl`
- 200 to 500 train steps
- larger response length if budget allows
- at least one LR or grad-clip variant if using anything except JSD

## Experiment 004: held-out validation on the 50-step matrix

Purpose: check whether the 50-step methods move a held-out math metric.

Setup:
- same methods and 50-step run as experiment 003
- validation enabled with `VAL_BEFORE_TRAIN=True`, `TEST_FREQ=50`, `VAL_N=8`, `VAL_MAX_SAMPLES=64`
- held-out set: `datasets/test_data/MATH-500/test.parquet`

Observed held-out metrics at step 50:
- `val-core/math_dapo/acc/mean@8`: 0.0 for every method
- `val-core/math_dapo/acc/best@8/mean`: 0.0 for every method
- `val-core/math_dapo/acc/maj@8/mean`: 0.0 for every method
- `val-aux/math_dapo/reward/mean@8`: 0.0 for every method
- `val-aux/math_dapo/score/mean@8`: 0.0 for every method
- `val-aux/math_dapo/format_score/mean@8`: 0.0 for every method

Held-out loss:
- no dedicated PPO held-out loss metric was emitted by this validation path

Interpretation:
- no held-out math signal yet
- train-side loss scale differences did not translate into any held-out hits on this small validation slice

Tracked note:
- `agent_notes/experiments/004_verl_qwen35_heldout_eval.md`

## Budget and state

Conservative counted GCP spend after this phase: about $470 of the $500 phase cap.
No GCP GPU instances are running after the latest run.
