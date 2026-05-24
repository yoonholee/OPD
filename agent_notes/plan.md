# Plan: Qwen3.5 scale-up + throughput/stability pass

Status: completed

Result:

- Pinned Qwen3.5 smoke env moved to `local/qwen35_2b_smoke/pyproject.toml` + `uv.lock`.
- 10-step GRPO on G4 4-GPU passed, batch 8, response 64, steady throughput 148.7 tok/s.
- 10-step OPD on G4 4-GPU passed after `view` -> `reshape`, batch 8, response 64, steady throughput 138.8 tok/s.
- 3-step remove-padding GRPO probe passed, batch 4, response 32.
- G4/A100/H100 availability and boot-disk traps recorded in `agent_notes/gcp_runs/REPORT.md`.
- Exact-match / importance-sampling tests and non-contiguous reward entropy test added.
- All GCP VMs deleted. Final instance list empty.

Deferred:

- Apples-to-apples remove-padding throughput at batch 8.
- Larger real-reward GRPO task where rewards are nonzero.
- Stronger teacher OPD, EMA/later checkpoint teacher, or true self-improving loop.
