# External patterns

- Upstream veRL import candidates: `vexact` zero-mismatch HF rollout, token-level rollout importance sampling, experimental `fully_async_policy`, `one_step_off_policy`, and transfer queue.
- EasyR1 import candidates: simpler config surface, LoRA-through-vLLM tensor adapter sync, DAPO online filtering, clip ratio low/high/dual, GSPO/CISPO examples, padding-free switch.
- Prime-RL import candidates: explicit `rl`/`opd`/`sft` training modes, separated orchestrator/trainer/inference services, local vLLM teacher requirement for OPD token logprobs, LoRA debug configs, async rollout/training overlap.
- Kernel-diff mitigations: use identical tokenizer/template/logit processors/dtype/backend for teacher and student logprobs; avoid non-default top-p/top-k unless the rollout mask is preserved; prefer temperature 1.0 and top_p 1.0/top_k disabled; use `processed_logprobs` when temperature is not 1.0; add same-weight calibration/deadband for OPD rewards.
- veRL throughput knobs worth preserving in THUNLP/OPD configs: `actor_rollout_ref.actor.use_dynamic_bsz=True`, `actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True`, rollout logprob dynamic batch, balanced batches, explicit `max_num_seqs` and `max_num_batched_tokens`.
- veRL correctness knobs worth importing before async OPD: rollout importance sampling/rejection diagnostics, processed-logprob mode when sampling processors are active, and same-weight drift calibration.
- EasyR1/OpenRLHF useful pattern: explicit FSDP/LoRA tensor sync into colocated vLLM plus sleep/wake to manage rollout memory. Not needed for this full-FSDP smoke, useful for LoRA later.
- Async OPD/RL candidates should come after the sync baseline: AReaL/SkyRL/NeMo-RL style staleness bounds and replay age need importance sampling or rejection before use.
