# GCP spend ledger

Status: active for distillation SFT exploration.

Budget:
- Phase cap: $500 credits.
- Conservative launch guard defaults to $25/hour for 4xL4 G2 jobs.
- Treat this as an estimate, not billing ground truth.

Rules:
- One GPU VM at a time unless explicitly overridden.
- Every launch writes `agent_notes/gcp_runs/<run>/spend_estimate.txt` locally.
- Stop new launches at $400 estimated cumulative spend until reviewed.
- Delete or auto-delete VMs after log sync.

Current conservative reserved estimate from created SFT + OPD variant VMs: $420.

Runs counted:
- `distill-sft-l4-0527-1825`: reserve $50, failed before model load, VM auto-deleted, logs synced.
- `distill-sft-l4-0527-retry1-1830`: reserve $50, loaded Qwen3.5, failed in FSDP2 wrap-policy normalization, VM auto-deleted, logs synced.
- `distill-sft-l4-0527-retry2-1836`: reserve $50, reached first SFT train step, failed on Qwen3.5 missing top-level `vocab_size`, VM auto-deleted, logs synced.
- `distill-sft-a100-1g-0527-1850`: reserve $50, 3-step native SFT rc0, loss 3.7648 to 1.1190, VM auto-deleted, logs synced.
- `distill-sft-a100-16step-0527-1857`: reserve $50, duplicate 3-step native SFT rc0 after a local launch-script path mistake, VM auto-deleted, logs synced.
- `distill-sft-a100-16step-c1b-0527-1904`: reserve $50, 16-step native SFT rc0, loss 3.0021 to 0.3784, VM auto-deleted, logs synced.

- `opd-variants-a100-c1b-0527-2213`: reserve $10, temp-runner root path bug, VM auto-deleted, logs synced.
- `opd-variants2-a100-c1b-0527-2219`: reserve $10, Qwen3.5-0.8B RL init too slow for smoke, manually deleted, no live GPU left.
- `opd-variants-qwen3-a100-c1b-0527-2228`: reserve $25, same-model Qwen3-0.6B top-k OPD matrix, 18/18 rc0, VM auto-deleted, logs synced.
- `opd-variants-teacher17-a100-c1b-0527-2300`: reserve $25, Qwen3-0.6B student / Qwen3-1.7B teacher top-k OPD matrix, 18/18 rc0, VM auto-deleted, logs synced.
- `opd-fullvocab-qwen3-a100-c1b-0528-0836`: reserve $25, Qwen3-0.6B student / Qwen3-1.7B teacher full-vocab OPD matrix, 8/8 rc0, VM auto-deleted, logs synced.
- `opd-fullvocab-qwen35-a100-c1b-0528-0858`: reserve $25, Qwen3.5-0.8B student / Qwen3.5-2B teacher full-vocab OPD matrix, 8/8 rc0, VM auto-deleted, logs synced.

Runs not counted as compute spend:
- L4/G4/H100 stockout attempts that failed before VM creation.

Known live state on 2026-05-28:
- No RUNNING GCP GPU instances in `soe-iris-gcp` after full-vocab matrices.
