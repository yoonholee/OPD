# verl OPD variant comparison

Status: complete for current verl top-k OPD surface.

Scope:
- Compared current verl-supported OPD knobs: `top_k_strategy` × `reward_weight_mode`, plus GRPO, ref-KL, and entropy bonus.
- Did not compare full-vocab `reverse_kl`, `forward_kl`, `jsd`, `topk_rkl`, or `entropy_aware` from `lagrangian_prompts/modal_phase2.py`. Those are not implemented as verl trainer objectives yet.

## EVAL: top-k OPD code-path matrix

Claim: all current verl top-k OPD variants ran end-to-end on GCP A100 for a 2-step smoke.

Axis: runtime correctness, reward/loss scale, and rough throughput. Not quality.

Eval:
- Commit: `142d020` plus ignored temp runner `_tmp/distillation_spikes/run_verl_opd_variant_matrix.sh`.
- Run A, same-model sanity: `opd-variants-qwen3-a100-c1b-0527-2228`, logs in ignored `agent_notes/gcp_runs/.../20260528_053123_verl_opd_variants/`.
- Run B, stronger-teacher signal: `opd-variants-teacher17-a100-c1b-0527-2300`, logs in ignored `agent_notes/gcp_runs/.../20260528_060348_verl_opd_variants/`.
- Both: GCP `soe-iris-gcp`, `us-central1-b`, `a2-highgpu-1g`, A100 spot, `TRAIN_STEPS=2`, `SMOKE_N=4`, `LOG_PROB_TOP_K=4`, response length 32.

Repro used:

```bash
MODEL=Qwen/Qwen3-0.6B TEACHER_MODEL=Qwen/Qwen3-1.7B \
NGPUS=1 TRAIN_STEPS=2 SMOKE_N=4 LOG_PROB_TOP_K=4 TIMEOUT=35m \
bash _tmp/distillation_spikes/run_verl_opd_variant_matrix.sh
```

Numbers:
- Same-model sanity: 18/18 cases rc0. Student/teacher-weighted OPD losses stayed near zero, as expected when student and teacher are the same model.
- Stronger-teacher run: 18/18 cases rc0.
- Stronger-teacher throughput: GRPO 106.6 tok/s; OPD variants mostly 75 to 79 tok/s; ref-KL variant 70.7 tok/s.
- Stronger-teacher `reward_weight_mode=none` is unstable-scale: grad norm 580 to 2864 and last pg loss 2.29 to 9.18.
- Stronger-teacher student-weighted variants are moderate-scale: last pg loss 0.026 to 0.153, grad norm 6.9 to 27.1.
- Stronger-teacher teacher-weighted variants are similar magnitude but opposite sign in this loss convention: last pg loss about -0.104 to -0.080, except `union-intersection` near zero.

Key rows, stronger-teacher run:

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

Regressions:
- `reward_weight_mode=none` produces huge gradients even in the same-model sanity run. Treat as diagnostic only, not a default.
- `union-intersection` can cancel almost all signal under teacher weighting. It is not a useful default without a reason.
- Two-step runs are too short for accuracy or hyperparameter claims. Validation accuracy is 0 across this tiny math smoke because max response length is 32.

Decision:
- Keep `only_stu + student_p` as the default OPD smoke path.
- Keep `intersection/union + student_p` as supported comparison recipes.
- Do not default to `reward_weight_mode=none`.
- Next real objective work is implementing the full-vocab/lit objectives in verl, not more top-k smoke tuning.
