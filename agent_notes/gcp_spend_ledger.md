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

Current conservative reserved estimate from created SFT exploration VMs: $300.

Runs counted:
- `distill-sft-l4-0527-1825`: reserve $50, failed before model load, VM auto-deleted, logs synced.
- `distill-sft-l4-0527-retry1-1830`: reserve $50, loaded Qwen3.5, failed in FSDP2 wrap-policy normalization, VM auto-deleted, logs synced.
- `distill-sft-l4-0527-retry2-1836`: reserve $50, reached first SFT train step, failed on Qwen3.5 missing top-level `vocab_size`, VM auto-deleted, logs synced.
- `distill-sft-a100-1g-0527-1850`: reserve $50, 3-step native SFT rc0, loss 3.7648 to 1.1190, VM auto-deleted, logs synced.
- `distill-sft-a100-16step-0527-1857`: reserve $50, duplicate 3-step native SFT rc0 after a local launch-script path mistake, VM auto-deleted, logs synced.
- `distill-sft-a100-16step-c1b-0527-1904`: reserve $50, 16-step native SFT rc0, loss 3.0021 to 0.3784, VM auto-deleted, logs synced.

Runs not counted as compute spend:
- L4/G4/H100 stockout attempts that failed before VM creation.

Known live state on 2026-05-27:
- No RUNNING GCP GPU instances in `soe-iris-gcp`.
- One TERMINATED spot H100 instance exists: `opd-tune3-passk-g4-0527-1148` in `us-west1-a`.
