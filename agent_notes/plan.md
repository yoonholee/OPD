# Full-vocab OPD objectives

Status: complete for Qwen3.5 held-out validation.

Budget:
- Hard exploration budget: $500 GCP credits for this phase.
- Conservative counted spend after this phase: about $520.
- No RUNNING GCP GPU instances after held-out validation.

Completed:
1. Added verl actor losses for full-vocab reverse KL, forward KL, JSD, symmetric KL, top-k reverse KL, and entropy-aware switching.
2. Wired teacher full-vocab logprobs through reward worker into actor update.
3. Kept existing top-k OPD path intact.
4. Added a tracked matrix runner and summarizer.
5. Ran focused loss tests in the GCP torch env.
6. Ran A100 matrices:
   - Qwen3-0.6B student / Qwen3-1.7B teacher: 8/8 rc0.
   - Qwen3.5-0.8B student / Qwen3.5-2B teacher: 8/8 rc0.
7. Recorded results in `agent_notes/experiments/002_verl_full_vocab_opd.md`.

Current caveats:
- Full-vocab path currently requires `use_remove_padding=False`.
- Results are two-step wiring/scale smokes, not quality or hparam claims.
- Qwen3.5 grad norms are high for `sym_kl`, `forward_kl`, `topk_rkl`, and top-k OPD.

Completed longer runs:
- Qwen3.5-0.8B student / Qwen3.5-2B teacher, 50 train steps per method on 1xA100 spot.
- 8/8 methods rc0.
- Experiment note: `agent_notes/experiments/003_verl_qwen35_50step_methods.md`.

Held-out validation run:
- validation enabled on the same 50-step matrix with `VAL_BEFORE_TRAIN=True`, `TEST_FREQ=50`, `VAL_N=8`, `VAL_MAX_SAMPLES=64`.
- held-out math accuracy and pass@k-style metrics were 0.0 across methods.
- experiment note: `agent_notes/experiments/004_verl_qwen35_heldout_eval.md`.

Next:
- Do not read the current 50-step ranking as quality.
- Need either a stronger eval slice, longer training, or a different path that emits a true held-out loss.
