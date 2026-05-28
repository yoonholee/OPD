# Full-vocab OPD objectives

Status: complete for Qwen3.5 50-step method matrix.

Budget:
- Hard exploration budget: $500 GCP credits for this phase.
- Conservative counted spend after this phase: about $470.
- No RUNNING GCP GPU instances after the 50-step run.

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

Completed longer run:
- Qwen3.5-0.8B student / Qwen3.5-2B teacher.
- 50 train steps per method on 1xA100 spot.
- 8/8 methods rc0.
- Experiment note: `agent_notes/experiments/003_verl_qwen35_50step_methods.md`.

Next:
- Prioritize `full_jsd`, `full_reverse_kl`, `full_entropy_aware`, and `full_forward_kl` for longer runs.
- Do not spend more on `full_topk_rkl` or top-k OPD until LR/clipping or reward normalization is revisited.
