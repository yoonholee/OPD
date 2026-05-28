# 004: Qwen3.5 held-out validation on the 50-step matrix

Status: complete.

## Question

Do the 50-step Qwen3.5 methods show any held-out math signal once validation is enabled?

## Setup

- Code commit: `59d4f90` for the report update, same training code as `d02c857`
- GCP project: `soe-iris-gcp`
- Zone: `us-central1-b`
- Machine: `a2-highgpu-1g`, A100 spot
- VM: `opd-fullvocab-qwen35-heldout-a100-c1b-0528-1151`
- Student: `Qwen/Qwen3.5-0.8B`
- Teacher: `Qwen/Qwen3.5-2B`
- Train steps: 50 per method
- Smoke rows: 64
- Validation: `VAL_BEFORE_TRAIN=True`, `TEST_FREQ=50`, `VAL_N=8`, `VAL_MAX_SAMPLES=64`
- Held-out set: `datasets/test_data/MATH-500/test.parquet`

Ignored raw logs:
- `agent_notes/gcp_runs/opd-fullvocab-qwen35-heldout-a100-c1b-0528-1151/gcp_runs/20260528_185447_verl_full_vocab_opd/`

Repro shape:

```bash
MODEL=Qwen/Qwen3.5-0.8B TEACHER_MODEL=Qwen/Qwen3.5-2B \
NGPUS=1 TRAIN_STEPS=50 SMOKE_N=64 LOG_PROB_TOP_K=0 \
VAL_BEFORE_TRAIN=True TEST_FREQ=50 VAL_MAX_SAMPLES=64 VAL_N=8 \
TIMEOUT=45m bash local/bin/run_verl_full_vocab_opd_matrix.sh
```

## Claim

Validation runs cleanly, but it gives no held-out math signal yet.

Axis: held-out accuracy and pass@k-style metrics.

This is not a quality result.

## Results

Step-50 held-out metrics were all 0.0 for every method:
- `val-core/math_dapo/acc/mean@8`
- `val-core/math_dapo/acc/best@8/mean`
- `val-core/math_dapo/acc/maj@8/mean`
- `val-aux/math_dapo/reward/mean@8`
- `val-aux/math_dapo/score/mean@8`
- `val-aux/math_dapo/format_score/mean@8`

No dedicated PPO held-out loss metric was emitted by this path.

## Interpretation

The 50-step training differences on the train-side loss did not move held-out math on this small validation slice.

The main useful output is negative: the validation plumbing works, and the answer is still zero across the board.

## Decision

Do not treat the current 50-step method ranking as a quality ranking.

If we want a real held-out signal, we need one of:
- a stronger dataset slice
- longer training
- a different reward/eval path that emits a true held-out loss
