# Native verl SFT smoke results

Status: first real smoke passed. Not yet promoted from temp scripts.

## 2026-05-27 GCP A100 SFT smoke

Claim: native verl SFT can train Qwen3.5-2B for a short off-policy SFT smoke after the Qwen3.5 patches in this working tree.
Axis: trainer viability and short-run loss direction.
Eval: `agent_notes/gcp_runs/distill-sft-a100-16step-c1b-0527-1904/gcp_runs/20260528_020802_verl_sft_spike/`.
Numbers: 16 train steps on 1x A100 spot, train/loss 3.0021 to 0.3784, val/loss 0.2381, train-step time total 28.99s, rc 0.
Regressions: not a quality result; targets are synthetic ground-truth answer strings, not teacher rollouts. 1x A100 smoke does not prove 4xL4 Qwen3.5-2B capacity or throughput.
Repro: from the current working tree, run `MODEL=Qwen/Qwen3.5-2B NGPUS=1 TRAIN_BATCH_SIZE=1 TRAIN_N=64 VAL_N=8 TRAIN_STEPS=16 LR=1e-5 MACHINE=a2-highgpu-1g ACCELERATOR='' PROVISIONING_MODEL=SPOT ZONE=us-central1-b bash _tmp/distillation_spikes/launch_gcp_verl_sft_spike.sh`.

## Bugs found before pass

- Native SFT did not use Qwen3.5 auto-class compatibility.
- Native SFT hardcoded FlashAttention 2; smoke uses SDPA.
- Hydra nested dict override needed `+data.apply_chat_template_kwargs.enable_thinking=False`.
- FSDP2 wrap policy needed to normalize set-valued `_no_split_modules`.
- SFT loss needed logits width instead of `config.vocab_size`.

## Capacity notes

- 4xL4 `g2-standard-48` was unavailable in `us-west1-a` and `us-west1-b` during this session.
- 1xL4 was unavailable in tested L4 zones or unsupported for the tried config.
- A3 H100 spot in `us-east4-a` was also stocked out.
- 1xA100 spot in `us-central1-a` and `us-central1-b` worked.
