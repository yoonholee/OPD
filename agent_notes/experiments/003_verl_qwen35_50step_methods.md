# 003: Qwen3.5 50-step OPD method matrix

Status: complete.

## Question

Do the Qwen3.5 OPD methods run for a less-trivial length, and which losses have sane scale over 50 updates?

## Setup

- Code commit: `d02c857`
- GCP project: `soe-iris-gcp`
- Zone: `us-central1-b`
- Machine: `a2-highgpu-1g`, A100 spot
- VM: `opd-fullvocab-qwen35-50step-a100-c1b-0528-0941`
- Student: `Qwen/Qwen3.5-0.8B`
- Teacher: `Qwen/Qwen3.5-2B`
- Train steps: 50 per method
- Smoke rows: 64
- Response length: 32
- Actor LR: 1e-6
- Grad clipping: verl default
- Full-vocab path: `use_remove_padding=False`
- Unit check before matrix: `PYTHONPATH=verl python verl/tests/trainer/ppo/test_full_vocab_distill.py`

Ignored raw logs:
- `agent_notes/gcp_runs/opd-fullvocab-qwen35-50step-a100-c1b-0528-0941/gcp_runs/20260528_164534_verl_full_vocab_opd/`

Repro shape:

```bash
MODEL=Qwen/Qwen3.5-0.8B TEACHER_MODEL=Qwen/Qwen3.5-2B \
NGPUS=1 TRAIN_STEPS=50 SMOKE_N=64 LOG_PROB_TOP_K=0 TIMEOUT=35m \
bash local/bin/run_verl_full_vocab_opd_matrix.sh
```

## Claim

All compared Qwen3.5 methods ran 50 updates end-to-end on 1xA100.

Axis: runtime correctness, divergence-loss trend, gradient scale, throughput.

This is still not a quality or accuracy claim.

## Results

All methods passed: 8/8 rc0.

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

## Interpretation

Full-vocab reverse KL, forward KL, JSD, symmetric KL, and entropy-aware all decreased over 50 steps.

JSD is the cleanest scale: lowest loss, lowest max grad norm among full-vocab objectives, and similar throughput.

Full top-k reverse KL and existing top-k OPD went the wrong direction on this 50-step smoke and had large gradient spikes.

Symmetric KL decreased but had a large early gradient spike. Treat it as usable only with LR or clipping checks.

## Regressions and limits

- Loss scales are objective-specific. Do not compare absolute loss across objectives as quality.
- Max response length 32 keeps validation meaningless.
- Grad norms are pre-clip total norms from `clip_grad_norm_`; optimizer steps are clipped.
- This uses one small synthetic/local smoke data slice, not DAPO-Math scale.

## Decision

For next longer Qwen3.5 pass, prioritize:
1. `full_jsd`
2. `full_reverse_kl`
3. `full_entropy_aware`
4. `full_forward_kl`

Do not spend more on `full_topk_rkl` or top-k OPD until LR/clipping or reward normalization is revisited.
