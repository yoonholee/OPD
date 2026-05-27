# H100 8x paper-pair results, 2026-05-27

VM: a3-highgpu-8g, us-east4-a, spot, ~25 min wall time, ~$10.

## GRPO real-reward (Qwen3-1.7B-Base, DAPO-Math, ttrl_math reward)

64 steps (epoch cap from SMOKE_N=2048 / batch 32).

Zero-reward problem fixed: 3.1% accuracy at step 1, real advantage variance.
New problem: mode collapse at step 60+ (acc=0, grad=0).

| step | acc | adv_std | grad_norm | entropy | resp_len |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.031 | 0.99 | 0.34 | 1.85 | 392 |
| 20 | 0.008 | 0.88 | 0.10 | 1.65 | 471 |
| 40 | 0.031 | 1.00 | 0.33 | 1.80 | 407 |
| 60 | 0 | 0 | 0 | 1.69 | 424 |
| 64 | 0 | 0 | 0 | 1.48 | 415 |

Mitigations to try next: enable `use_kl_loss=True` (currently False), lower LR (1e-6 → 5e-7), or bigger batch.

## OPD stronger-teacher (Qwen3-4B-Base teacher -> Qwen3-1.7B-Base student)

64 steps, same data.

Same-weight artifact eliminated: rewards strongly negative (~-5), gradient norms 3-5x larger than prior same-weight smoke. Real teacher-student gap.

| step | reward | adv_std | grad_norm | entropy | resp_len |
|---:|---:|---:|---:|---:|---:|
| 1 | -5.77 | 0.23 | 5.5 | 1.85 | 392 |
| 30 | -4.62 | 0.20 | 3.5 | 2.20 | 502 |
| 50 | -4.38 | 0.19 | 3.1 | 2.21 | 535 |
| 64 | -4.72 | 0.21 | 3.4 | 2.74 | 500 |

Reward trending up (-5.77 -> -4.38) then noisy. Entropy rises (1.85 -> 2.74) which is consistent with teacher pulling student into a higher-entropy distribution.

## LF SFT not run

NCCL 2.29 + torch 2.12 in LF venv refuses Socket fallback even with NCCL_NET=Socket exported. Same env works in verl venv (torch 2.11 / NCCL 2.27). Needs separate fix.

## Files

- Run dir: `agent_notes/gcp_runs/opd-paperpair-h100-east4-0527-0927/`
- GRPO log: `gcp_runs/20260527_163151_qwen35_2b/grpo_qwen35_2b.log`
- OPD log: `gcp_runs/20260527_164232_qwen35_2b/opd_qwen35_2b.log`

