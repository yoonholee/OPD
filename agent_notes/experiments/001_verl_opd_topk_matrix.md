# 001: verl top-k OPD variant matrix

Status: complete for current verl top-k OPD surface.

## Question

Which current verl OPD code paths run end-to-end, and which support/weighting knobs have sane loss scale?

## Scope

Compared current verl-supported OPD knobs:
- `top_k_strategy`: `only_stu`, `only_tch`, `intersection`, `union`, `union-intersection`
- `reward_weight_mode`: `student_p`, `teacher_p`, `none`
- plus GRPO, `use_kl_loss=True`, and `entropy_coeff=0.001`

Not compared here:
- full-vocab `reverse_kl`, `forward_kl`, `jsd`, `topk_rkl`, `entropy_aware` from `lagrangian_prompts/modal_phase2.py`
- those were not verl trainer objectives at run time

## Setup

- Commit: `aec0862`, plus ignored temp runner `_tmp/distillation_spikes/run_verl_opd_variant_matrix.sh`
- GCP project: `soe-iris-gcp`
- Zone: `us-central1-b`
- Machine: `a2-highgpu-1g`, A100 spot
- Train steps: 2
- Smoke rows: 4
- Top-k: 4
- Response length: 32

Runs:
- Same-model sanity: `opd-variants-qwen3-a100-c1b-0527-2228`
- Stronger-teacher signal: `opd-variants-teacher17-a100-c1b-0527-2300`

Ignored raw logs:
- `agent_notes/gcp_runs/opd-variants-qwen3-a100-c1b-0527-2228/gcp_runs/20260528_053123_verl_opd_variants/`
- `agent_notes/gcp_runs/opd-variants-teacher17-a100-c1b-0527-2300/gcp_runs/20260528_060348_verl_opd_variants/`

Repro shape:

```bash
MODEL=Qwen/Qwen3-0.6B TEACHER_MODEL=Qwen/Qwen3-1.7B \
NGPUS=1 TRAIN_STEPS=2 SMOKE_N=4 LOG_PROB_TOP_K=4 TIMEOUT=35m \
bash _tmp/distillation_spikes/run_verl_opd_variant_matrix.sh
```

## Claim

All current verl top-k OPD variants ran end-to-end on GCP A100 for a 2-step smoke.

Axis: runtime correctness, reward/loss scale, and rough throughput.

Not an accuracy or hyperparameter claim.

## Results

Same-model sanity:
- 18/18 cases rc0
- student/teacher-weighted OPD losses stayed near zero, as expected when student and teacher are the same model

Stronger-teacher run:
- 18/18 cases rc0
- GRPO throughput: 106.6 tok/s
- OPD throughput: mostly 75 to 79 tok/s
- ref-KL OPD throughput: 70.7 tok/s

Key stronger-teacher rows:

| case | rc | last pg loss | grad norm | tok/s |
|---|---:|---:|---:|---:|
| grpo | 0 | 0.000 | 0.0 | 106.6 |
| opd_only_stu_student_p | 0 | 0.153 | 27.1 | 78.7 |
| opd_only_stu_teacher_p | 0 | -0.104 | 30.4 | 78.4 |
| opd_only_stu_none | 0 | 3.498 | 904.0 | 78.7 |
| opd_intersection_student_p | 0 | 0.066 | 20.4 | 76.0 |
| opd_intersection_teacher_p | 0 | -0.101 | 28.6 | 75.4 |
| opd_intersection_none | 0 | 2.288 | 580.0 | 75.9 |
| opd_union_student_p | 0 | 0.149 | 26.9 | 75.7 |
| opd_union_teacher_p | 0 | -0.080 | 17.9 | 75.1 |
| opd_union_none | 0 | 3.223 | 820.0 | 75.7 |
| opd_union_intersection_student_p | 0 | 0.026 | 6.9 | 75.3 |
| opd_union_intersection_teacher_p | 0 | 0.000 | 0.1 | 75.4 |
| opd_union_intersection_none | 0 | 9.177 | 2864.0 | 75.4 |
| opd_refkl_only_stu_student_p | 0 | 0.153 | 27.1 | 70.7 |
| opd_entropy_only_stu_student_p | 0 | 0.148 | 26.9 | 78.0 |

## Interpretation

`only_stu + student_p` is the best default smoke path.

`intersection + student_p` and `union + student_p` are reasonable comparison recipes.

`reward_weight_mode=none` is unstable-scale: grad norm 580 to 2864 and last pg loss 2.29 to 9.18 in the stronger-teacher run.

`union-intersection + teacher_p` can cancel almost all signal.

Ref-KL adds overhead in this smoke and does not change the 2-step loss row.

Entropy bonus runs and slightly increases entropy, but this smoke is too short to say anything about quality.

## Regressions and limits

- Two-step runs are too short for accuracy or hparam claims.
- Validation accuracy is 0 across this tiny math smoke because max response length is 32.
- Current result covers verl's top-k support/weight surface only.
- Full-vocab literature objectives are compared separately in `agent_notes/experiments/002_verl_full_vocab_opd.md`.

## Decision

Keep `only_stu + student_p` as the default OPD smoke path.

Do not default to `reward_weight_mode=none`.

Next objective work is the full-vocab matrix in `agent_notes/experiments/002_verl_full_vocab_opd.md`.
