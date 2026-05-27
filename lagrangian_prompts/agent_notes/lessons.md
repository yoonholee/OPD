# Lessons — lagrangian_prompts

Durable, surprising findings about design, data, and interpretation. Friction goes in `friction.md`.

## SDR's "similar problem" framing is misleading by design

`teacher_contexts.py:_sdr_builder` feeds the model `problem["solution"]` — the gold solution to the **exact same problem** — under the prompt "Reference solution to a similar problem". The "similar" framing is a fig leaf inherited from SPEC §SDR; it's intentionally max leak ("upper-bound diagnostic"). When reporting SDR results, it is worth saying "full gold solution" rather than "similar-problem reference" so readers don't mistake this for a cross-problem transfer experiment.

## MATH-500 first-sentence extraction is leak-noisy

Programmatic "first sentence" / "first paragraph" of `problem["solution"]` does NOT produce a clean strategy hint:

- 2 of 5 sampled problems: first sentence **leaks the answer** ("r = 3", "70 yaps").
- 2 of 5: first sentence is a clean strategy ("side length of hexagon = side length of triangle", "each pair of circles has at most two intersection points").
- 1 of 5: first sentence sets up the key equation, partial leak.

First-paragraph extraction is worse — solutions are typically 1-3 paragraphs, so the first paragraph usually contains the answer computation. `\boxed{}` redaction catches the final answer but not intermediate numbers.

**How to apply:** For a clean partial-leak baseline, an LLM-based summary pass is required ("extract the high-level strategy in 1-2 sentences, no specific numbers"). Programmatic extraction produces a noisy mid-tier between OPSD and SDR-full, not a clean strategy-only condition. Report sdr_strategy and sdr_first_step with the noise caveat; do not interpret them as the Pareto's "strategy hint" point without verifying the extracted text per problem.

## SDPO ≠ SDFT ≠ SDR by information leak, but the line through them is what matters

When the SPEC compares SDFT vs SDPO vs SDR, the *quantity that varies* is what the teacher context contains beyond the bare problem:

- SDFT: k=2 other problems with their gold solutions. No leak about *this* problem.
- SDPO: this problem's draft (own) + this problem's gold answer.
- SDR:  this problem's full gold solution.

So SDPO and SDR both leak this problem's answer; SDFT does not. Phase 1 results show SDFT lifts ~5pp over anchor while SDR lifts ~10-15pp on MATH — the difference is the leak strength. For Pareto interpretation, **group conditions by what they know about the current problem**, not by family name.

## On Qwen3.5, response-distribution KL is dominated by formatting tokens, not reasoning

`fwd_kl` averaged over the full response is ~0.001-0.07 nats/tok across all conditions on every cell — about 2 orders of magnitude smaller than typical classifier KLs (~1 nat/tok). `fwd_kl_first32` (averaged over the first 32 tokens only) is 10-100x larger and tracks the family-level ordering more cleanly. This confirms Phase 0's "first-K-token forward KL" recommendation.

**How to apply:** Don't use mean-per-token forward KL as the Pareto x-axis for single-model prompt-opt on reasoning models. Use `fwd_kl_first32` or sum-KL with a hard length cap. Phase 1 results back this up across 3 model sizes × 2 tasks.

## MC-estimated rev_kl can be slightly negative for low-leak teacher contexts

D_KL(π_θ || π_T) is non-negative in expectation, but its single-sample MC estimator at finite K (4-8 rollouts × 50 problems) shows negative values of order 0.01-0.05 nats/tok when the true KL is near zero (e.g., SDPO under SDPO-no-answer at 2B, −0.091). This is consistent with sample-mean variance, not a sign bug — confirmed by checking that the forward KL on the same data is correctly positive and the clipping fraction is small. Treat |MC rev_kl| < SE as "not significantly distinct from zero", not as a violated lower bound.

**RESOLVED (v7, 2026-05-16):** the negatives were the k1 estimator (`log π−log q` at the sampled token), not the pipeline. The answer-leak pathology (π_T near-1 on the gold token, π_θ tiny → fat negative log-ratio tail) drags the k1 sample mean below zero even though true KL ≥ 0. Re-ran all 6 cells with `prompt_logprobs=20` and added two estimators in `kl_metrics.py`:
- **k3** (Schulman `(r−1)−log r`): same expectation as k1, ≥0 pointwise. Fixes the sign but NOT the heavy tail — k3 still explodes on the answer-leak token (the self-test shows k3≈75 on a synthetic 2% −8-nat tail). Use when you only have the realized-token logprob.
- **tk** (truncated analytic KL over the top-k next-token dist per position): lowest variance, only truncation bias, never samples the pathological token. This is the right default when `prompt_logprobs=k` is available. All negatives gone (0/cell under k3 and tk). Decision unchanged: Spearman(acc,KL) on distrib shifts ≤0.06 vs k1 per cell.

**How to apply:** default to **tk** for any KL-as-selection-axis work; it needs `prompt_logprobs≥8` from vLLM and the streaming extraction in `modal_phase1.py` (holding full-context top-k for all requests OOMs — extract per chunk, discard raw outputs). k1 is fine only as a sanity cross-check. Never report k1 rev_kl as a headline number on answer-leaking teacher contexts.

## Answer-first framing ≫ answer-then-reason for injecting a known answer (003)

Same tier-3 information (this problem's gold answer), two framings:
- OPSD: "The correct answer is X. Now produce the reasoning." → 0.8B/math 0.33, 0.8B/eq 0.54.
- answer-first: "State X in the required format on line 1, then justify." → 0.8B/math 0.55, 0.8B/eq 0.78, at equal or LOWER KL than OPSD.

The model commits to the correct answer before it can reason itself out of it, and the short anchored response keeps per-token KL low. This re-frames a 002 conclusion: 0.8B's Pareto-top was *not* really tier-5 (full solution); a re-framed tier-3 dominates it on both axes. **How to apply:** when a teacher context has the gold answer, put it in the verifier's exact format FIRST, then ask for a brief justification. Don't ask for reasoning that "arrives at" it.

Corollaries (0.8B, single call): (a) naive answer-copy ("output only the box") fails on MATH — the weak model mangles a complex boxed expression and the verifier misses; the justification step is load-bearing, only trivial on binary eq. (b) Long justification dilutes per-token KL (as the length effect predicts) but lets a weak model argue itself off the given answer (eq 0.85 → 0.68).

## Meta-Harness (propose→pick→execute→check→commit) fails on a weak model (003)

The connections-meta-harness scaffold, ported as a single-call tier-0 prompt, *hurt* Qwen3.5-0.8B: math 0.185 (vs anchor 0.340) at ~3× the KL; no better with the gold answer added. A 0.8B model spends its token budget narrating the protocol and derails. Evolved-harness gains do not transfer to a weak single-call inner model — scaffolding complexity has to match model capability.

## The 003 answer-leak advantage does not transfer to OPD (H3-null); the "collapse" is a trust-region artifact (004→005)

**Durable claim (post-ablation):** the 003 single-call Pareto does not predict OPD value, but as a **null, not an inversion**. Under a *matched, stabilized* recipe (reverse_kl + β·KL-to-base, 0.8B, same model student & teacher), the 003 up-left winner `t3_aj_concise` (answer-leak) only **ties** the no-leak `distrib:concise` on ID (0.42 vs 0.36, within n=50 SE) and **loses OOD-down** (GSM8K 0.45 vs 0.52). The +0.16 single-call accuracy edge of the answer-leak teacher buys nothing in OPD. SPEC H3 (Pareto rank → OPD value, ρ>0.6) fails because the leak advantage does not transfer — *not* because it inverts.

**Retraction of the first-pass claim:** 004 initially reported `aj_concise` "catastrophically collapses to 0 / structural inversion ρ≈−0.8". That was a **recipe artifact**, caught only by an ablation prompted on review (trust-calibration: 004 made a strong structural claim from a single un-stabilized recipe while listing recipe-sensitivity as its own caveat — the ablation should have preceded the claim):

- The collapse is a **no-trust-region instability**, not a teacher property. `reverse_kl` + β0.5·KL(π_θ‖π_base) removes it entirely (aj_concise ID 0.26→0.42, stable). The collapsed-finals Spearman is not a meaningful structural quantity.
- `forward_kl` is a **red herring**: it collapses *both* teachers (no-leak concise too: ID 0.36→0.20, GSM8K→0.06). So the collapse is not a mode-seeking-reverse-KL effect; reverse-KL is the *better* divergence here.
- A peaked answer-leak teacher (gold answer + "one concise line" → ~10-token near-deterministic π_T) is the *first to destabilize* under naive no-trust-region OPD, but a standard trust region fully tames it. Peakedness predicts *fragility without a stabilizer*, not a fundamental can't-teach property — weaker than the first-pass "one mechanism unifies Phase-1 KL pathology and Phase-2 collapse" claim, which is now only suggestive.

**How to apply:** (a) Select OPD teacher contexts by **no-leak distributional value, not 003/single-call rank** (it doesn't transfer) — this survives. (b) Always run OPD on this base with a **KL-to-base trust region**; without it, peaked answer-leak teachers collapse and even `forward_kl` collapses no-leak teachers. (c) Don't use `forward_kl` for OPD here. (d) `sdr`/stabilized-`concise` OOD-down GSM8K gains (0.52-0.53) from solution-style teachers are a real separate effect worth a follow-up. Ties [[004-opd-leak-vs-transfer]] + [[005-opd-recipe-ablation]]; the Phase-1 [[mc-rev-kl-answer-leak-pathology]] link is now suggestive, not load-bearing.

## OPD objective sophistication doesn't beat a simple stabilized baseline in the context/PI regime (006)

Tested the two most-cited literature OPD stabilizers on our setup: teacher
top-K + stop-grad reverse-KL (Revisiting-OPD/The Many Faces, +19.8% in-paper)
and entropy-aware reverse/forward mixing (Entropy-Aware OPD, +1.4–5pp in-paper).
**Neither helps; entropy_aware harms.** `topk_rkl` on no-leak `concise` reaches
the *same* 0.36 ID as plain reverse_kl but via a volatile 0.46→0.18→0.36 path
(reverse_kl was monotone-stable) → no gain, more variance; on answer-leak
`aj_concise` it collapses identically (→0). `entropy_aware` degrades `concise`
(ID 0.36→0.24, GSM8K 0.44→**0.07**) — any forward-KL admixture destabilizes
this base (matches 005's forward_kl-collapses-both). Nothing rescues the
answer-leak teacher under *any* objective — objective-agnostic, as The Many
Faces predicts for instance-specific PI.

**Durable lesson:** those stabilizers were designed for *capability*
distillation (strong→weak, genuine new skill). Our regime is *context / PI
distillation* (internalize a system prompt or an answer leak). The regime, not
the objective, determines what works. **How to apply:** for context/PI
distillation keep the simple recipe — per-token reverse_kl + KL-to-base trust
region (005); do not import capability-distillation loss tricks; spend effort on
the data/teacher side (OPCD shared-rule contexts; training-free teacher
selection) not the loss. Also: single-eval-point reads mislead (the topk_rkl
0.46@step-50 looked like a win; 0.18@step-100 killed it) — always read the full
curve. Ties [[004-opd-leak-vs-transfer]] [[005-opd-recipe-ablation]]
[[006-opd-lit-objectives]] and [[external-patterns]].
