# Plan: Qwen3.5 scale-up + throughput/stability pass

Status: in progress

1. Commit and push current Qwen3.5-2B smoke state on a branch.
2. Add exact-match / importance-sampling tests for trainer vs inference-server probability drift.
3. Research notable veRL forks for throughput/stability tricks and hparams.
4. Run 10-step Qwen3.5-2B GRPO/OPD with larger sane batch on GCP, measure throughput.
5. Try remove-padding/FA2 fixes and throughput interventions.
6. Fetch logs, delete all cloud resources, update report/notes.
