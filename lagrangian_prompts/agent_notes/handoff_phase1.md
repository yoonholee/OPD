# Handoff — 2026-05-14

## Goal of next session

Implement Phase 1 of the Lagrangian prompt-opt experiment: the **metric × teacher-context matrix** as redesigned in `lagrangian_prompts/SPEC.md` (commit 4c46fa1). Output: `(acc, fwd_kl, rev_kl, jsd, fwd_kl_first32, ylen)` for each of 19 conditions per (model, task) cell, 6 cells total.

## Where we are

- **Phase 0 is done**, fully written up at `lagrangian_prompts/experiments/001_phase0_kl_sanity.md`. Key finding: mean-per-token forward KL fails the 0.3 nats/tok gate on MATH for single-model distribution-level prompts. SPEC was redesigned in response.
- **No Phase 1 code yet.** SPEC.md is the only source of truth for the new design.
- **The Phase 0 stack works** (Modal H100 + vLLM 0.20.0 + CUDA 12.8.1 + Qwen3.5 GDN attention via flashinfer JIT). All friction is documented in the experiments/001 appendix — read it before writing infra code.

## Immediate next action

Open `lagrangian_prompts/SPEC.md` and read §Setup + §Phase 1 + §Modal infrastructure (lines ~40-200). Then start with `kl_metrics.py` (cheapest to test in isolation): given lists of per-token logprobs of y under both π_T and π_θ, compute fwd_kl / rev_kl / jsd / fwd_kl_first32. Unit-test it locally with a synthetic example before any Modal calls.

After that, in order:

1. `teacher_contexts.py` — pure-Python functions `make_context(family, problem, gold, demos_pool) -> chat_messages`. Test by printing for a sample problem.
2. `modal_phase1.py` — persistent vLLM sandbox via `modal/modal_h100_ssh.py` pattern (already in the repo). Multi-GPU (2-4 H100s), `gpu_memory_utilization=0.95`, prefix caching on.
3. `analyze_phase1.py` — 4 Pareto plots per (model, task) cell, one per KL flavor. Reuse `analyze_phase0.py`'s sharey="row" + Pareto-frontier-dashed-line conventions.

## Recent surprises (promote durable ones to friction/lessons after)

- `apply_chat_template(tokenize=True)` returns a **BatchEncoding dict** on Qwen3.5 tokenizers, not `list[int]`. Use `tokenize=False` then `tok.encode(text, add_special_tokens=False)` for stable list[int]. (See `modal_phase0.py:run_one`.)
- `vllm==0.20.0` requires `TokensPrompt(prompt_token_ids=...)` from `vllm.inputs`, not raw dict form. Dict form trips an int-vs-str comparison in input validation.
- Qwen3.5 needs **CUDA 12.8+ devel base** (12.4 fails: flashinfer GDN prefill uses CUDA 12.5+ PTX symbols).
- Qwen3.5 thinks by default; for response-distribution KL to be meaningful, pass `enable_thinking=False` to the chat template. With thinking on, the response is prompt-invariant in bulk and per-token KL washes out.
- Phase 0 ylen filter: drop rollouts with `|y| < 200` from KL aggregates. Short responses inflate per-token KL because the average is dominated by 1-2 decision points.
- Empty system prompt as π_θ is **broken on MATH** (~5-8% acc — verifier can't find `\boxed{}`). The new SPEC anchors π_θ to a minimal format prompt.

## Skills to load

- `coding` — for the implementation itself.
- `agent-notes` — to log new friction / lessons during Phase 1.
- `experiment-results` — for the Phase 1 writeup once results land.

## References

- Spec: `lagrangian_prompts/SPEC.md` (authoritative).
- Phase 0 writeup: `lagrangian_prompts/experiments/001_phase0_kl_sanity.md`.
- Phase 0 code (reuse where possible): `lagrangian_prompts/modal_phase0.py`, `data.py`, `verifier.py`, `prompts.py`, `analyze_phase0.py`.
- Brown post (motivation, §8): full text in chat history; not on disk. The Lagrangian is `max_p E[acc] − β·E[KL(π_T || π_θ)]`.
- TML On-Policy Distillation blog (uses reverse KL): saved in tool-results from previous session; the key claim is "OPD reaches RL-trained teacher in 7-10× fewer gradient steps."
- Three new baseline papers (referenced in SPEC §Teacher-context families):
  - SDFT — Shenfeld et al. 2026, arXiv:2601.19897
  - SDPO — Hübotter et al. 2026, arXiv:2601.20802
  - SDR — Zhao et al. 2026, arXiv:2601.18734
- Existing repo work referenced in SPEC: `explore/146_training_free_grpo.py` (prompt-opt without KL), `explore/150_prompt_gen_bounds/` (prompt-prior logprob pipeline), `explore/156_opd_trainer/` (OPD trainer for Phase 2), `modal/modal_h100_ssh.py` (persistent sandbox bringup).
- Last commit: `4c46fa1 lagrangian_prompts: redesign Phase 1 as metric x teacher-context matrix`.
- Pushed to: `https://github.com/yoonholee/sandbox` master.
