# THUNLP/OPD Reproduction Scaffold

## Status: approved

Goal: reproduce the THUNLP/OPD paper's SFT, OPD, and RL/GRPO workflows on 4xH100 GCP, initially scaled down to Qwen/Qwen3.5-2B.

Constraints:
- Keep upstream code intact where possible.
- Add local scripts/config under `local/`.
- Treat Qwen3.5-2B VLM/text compatibility as an early smoke-test risk.
- Avoid full expensive runs until smoke passes and budget is explicit.
