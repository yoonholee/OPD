# Lagrangian Prompt Optimization

**Status:** Phase 0 complete ([writeup](experiments/001_phase0_kl_sanity.md)). Phase 1 redesigned post-Phase-0 findings: forward KL averaged per token is too washed out as a selection axis for distribution-level prompts on reasoning models, so Phase 1 now compares **multiple KL flavors × multiple teacher-context families** instead of running a GEPA β sweep.

**One-line:** Identify which KL definition and which teacher-context family produce a useful (acc, KL) Pareto, and whether the resulting ranking predicts on-policy distillation (OPD) efficiency on a larger student.

---

## Motivation

Will Brown's recent post (§8) sketches a Lagrangian objective that unifies several post-training methods:

```
max_p   E_x [ acc(x; p) ]  -  beta * E_x [ D_KL( pi_T( . | x, p)  ||  pi_theta( . | x) ) ]
```

where `pi_T` is the model conditioned on a hint `p`, `pi_theta` is the unconditioned base, and `acc` is per-task reward. Brown proposes several optimizers (GEPA over prompts, learned hint-writers, online co-evolution) but does not implement any.

This experiment runs the simplest instantiation: distribution-level prompt optimization, GEPA-style outer loop, single model on both sides of the KL.

The bet is that the Lagrangian's Pareto frontier picks out *useful* prompts (high reward at low KL) that downstream OPD can exploit cheaply. Specifically: a low-KL prompt should be a low-friction OPD teacher context, because OPD's compounding-error gap is bounded by the divergence between the teacher-induced state distribution and the student's own.

If true, this gives a cheap teacher-context selection procedure with no student training in the inner loop.

**Closest prior art:**

- Training-Free GRPO (Tencent, 2025; in this repo as `explore/146_training_free_grpo.py`) does prompt opt via GRPO-style updates with no KL regularizer. We add the KL term and test whether it changes which prompts win and whether the winning prompts transfer to OPD.
- TML's On-Policy Distillation blog (Lu et al., 2025) validates OPD itself (~10x compute savings vs RL on AIME). Our experiment is about *selecting the teacher context* for OPD, not about OPD itself.
- `explore/150_prompt_gen_bounds/` already implements prompt evaluation + base-model logprob scoring; we reuse the pattern, not the bound (different KL quantity).

---

## Hypotheses (revised)

- **H1 (Pareto exists under *some* KL flavor).** Across the four KL flavors we'll measure (forward, reverse, JSD, first-K-forward), at least one produces a meaningful (acc, KL) Pareto on Qwen3.5-{2B, 4B, 9B} × {MATH, EqTheories}. Phase 0 showed mean-per-token forward KL fails this on MATH; the bet is reverse, JSD, or first-K succeeds.
- **H2 (OPSD anchors the upper bound).** Per-instance answer-leaking prompts (OPSD-style) form a clear high-acc / high-KL corner that distribution-level prompts cannot reach. The gap between the Phase-0 prompt Pareto and the OPSD anchor quantifies "how much accuracy is left on the table by not knowing the answer." This is independently useful: it bounds what any same-family teacher context can contribute.
- **H3 (Cross-model predictive validity, load-bearing).** The same-model Pareto rank from H1's winning KL flavor predicts the prompt's effectiveness as a teacher context for OPD on a larger student (Qwen3.5-14B). Spearman correlation > 0.6 on 4-6 selected prompts spanning the frontier. *Publishable claim if it holds.*
- **H4 (Distillation methods stack on the Pareto in the expected order).** SDFT < SDPO < SDR on KL (and on accuracy). SDFT's same-task demos provide weak distributional pull without leaking; SDPO's retrospective feedback gives a more informative shift conditioned on the student's draft; SDR's full-solution leak is the strongest. If this ordering inverts on any axis (esp. accuracy), one of the methods doesn't work as advertised on this base model.

---

## Setup

### Model

- `pi_T`: `Qwen/Qwen3.5-{0.8B, 2B, 4B}` + teacher-context-specific system prompt (see below). Dropped 9B post-Phase-0 to focus compute on more conditions per cell rather than more model sizes.
- `pi_theta`: same model, **minimal format anchor** as system prompt (see "π_θ baseline" below). *Not* an empty system prompt — Phase 0 showed that gives 5-8% MATH accuracy because the model doesn't naturally emit `\boxed{}` and the verifier fails, making the KL anchor artificial.
- Non-thinking mode (`enable_thinking=False`). Phase 0 showed thinking-on collapses both accuracy (rollouts truncate mid-thought) and KL signal (thinking chain is prompt-invariant and dominates the response).
- vLLM 0.20.0 on Modal single H100, CUDA 12.8.1 devel base. See Phase 0 writeup for infra notes.

### Task

- MATH500 stratified across all 5 difficulty levels, **n=50 problems** (up from Phase 0's n=30). Split: 30 train (Pareto fitting + OPSD demo pool), 20 held-out test for reported Pareto.
- SAIR EqTheories `normal` config, n=50, 25/25 true/false balance, same split.
- Verifiers: `\boxed{}` extraction + sympy fallback (MATH), last `true`/`false` token match (EqTheories). Both already implemented in `verifier.py`.

### π_θ baseline (minimal format anchor)

We anchor π_θ to a prompt that establishes *only* the answer format the verifier expects, so that:
(a) baseline accuracy is non-zero and reflects model competence rather than verifier breakage;
(b) the "no hint" pole of the Pareto plot is interpretable;
(c) KL is measured relative to a real, format-functional reference, not an artificial one.

- **MATH π_θ:** `"Solve the problem. End your answer with \\boxed{answer}."`
- **EqTheories π_θ:** `"Decide whether E1 implies E2. End your answer with`true` or `false`."`

(Phase 0's `minimal_format` prompt is essentially this; we adopt it as the canonical baseline.)

### Teacher-context families

Four families of `pi_T`, all sharing the same minimal format anchor as a base, then layering additional context. The three "self-OPD" methods come from recent papers and operationalize the same `(student, teacher = student + extra context c)` template with different `c`:

1. **`distrib`** — Phase 0's 14 distribution-level prompts (one fixed system prompt across all problems). `concise`, `competition`, `magma_intro`, etc. See `prompts.py`.

2. **`sdft`** — SDFT (Shenfeld et al. 2026, arXiv:2601.19897, "Self-Distillation Enables Continual Learning"). `c` = k in-context demonstrations of the *target task*. For each problem `x`, teacher's user message is `"<demo_1>\n<solution_1>\n\n<demo_2>\n<solution_2>\n\n<x>"` (standard ICL prefix); student's user message is just `"<x>"`. Demos are sampled from the train pool with the same task family, excluding the current problem. We use **k=2 demos**; 3 random seeds per cell to estimate demo-choice variance.

3. **`sdpo`** — SDPO (Hübotter et al. 2026, arXiv:2601.20802, "Reinforcement Learning via Self-Distillation"). `c` = retrospective feedback on the student's own draft. Two passes:
   - Pass 1: sample `y_student ~ π_θ(·|x)` (minimal anchor only).
   - Pass 2: teacher's prompt is `"<x>\n\nYour previous answer: <y_student>\n\nFeedback: The correct answer is <gold>. Solve again carefully."` Teacher generates a corrected response.
   - We treat the second-pass response as `y_T`, score it under both π_T and π_θ contexts, and compute KL flavors as usual.

4. **`sdr`** — Self-Distilled Reasoner (Zhao et al. 2026, arXiv:2601.18734). `c` = the *full ground-truth solution* (not just the answer). For each problem `x`, teacher's prompt is `"Reference solution to a similar problem:\n<gold_solution>\n\nNow solve the following:\n<x>"`. Student sees only `<x>`. This is strictly more information than SDPO's feedback (full chain-of-thought vs final answer + draft critique). Expected upper bound on what same-family teacher contexts can achieve.

**Implementation note:** all four families produce `pi_T` as `(same model, augmented system or user message)`. No second model copy required. We implement each as a Python function `make_pT_context(problem, family, gold, demos_pool) -> chat_template_str` and route via a single `family` argument.

**Phase 0 sanity-check baselines `opsd` / `sdft_random` are replaced** by the above. The new `sdft` uses same-task demos (not random); SDR (was `opsd`) uses full solution (not answer-only).

### KL flavors (all per-token, mean-aggregated over the response)

For rollouts `y_T ~ pi_T(·|x)` and `y_θ ~ pi_θ(·|x)`, we compute per-token logprobs of each sequence under *both* π_T and π_θ via vLLM `prompt_logprobs=1`. From these we derive:

- **`fwd_kl`**: `mean_t [log pi_T(y_T_t) - log pi_theta(y_T_t)]`, sampled from π_T. *Same as Phase 0.* Brown §8's choice; predicts OPD-direction compounding error.
- **`rev_kl`**: `mean_t [log pi_theta(y_θ_t) - log pi_T(y_θ_t)]`, sampled from π_θ. TML OPD paper's choice; mode-seeking; predicts weight-training stability.
- **`jsd`**: `0.5 * E_{y~π_T}[log π_T(y_t) - log M(y_t)] + 0.5 * E_{y~π_θ}[log π_θ(y_t) - log M(y_t)]`, where `log M(y_t) = logsumexp(log π_T(y_t), log π_θ(y_t)) - log 2`. Symmetric, bounded in `[0, log 2]`. Robust to direction confounds.
- **`fwd_kl_first32`**: same as `fwd_kl` but averaged only over the first 32 response tokens. Phase 0 finding: most of the prompt-driven divergence lives in the early framing tokens; the rest of the chain is prompt-invariant. This isolates the framing decision.

Length filter (from Phase 0): drop any rollouts with `|y| < 200` from KL aggregates — those are length-confounded (1-2 token answers dominate the average). Report `(acc, KL)` per (prompt, problem) **only on rollouts that survive the length filter**.

K=4 rollouts/problem from each of π_T and π_θ (so 8 generations total per problem-prompt pair). Doubled from Phase 0 to reduce KL estimator variance, especially needed for reverse KL where we have a new sampling distribution to estimate over.

---

## Experimental plan

### Phase 0: sanity check — **DONE**

Scored 15-16 prompts × Qwen3.5-{2B, 4B, 9B} × {MATH, EqTheories} with **empty** π_θ baseline, forward KL only. Outcome: forward KL averaged per token failed the 0.3 nats/tok gate on MATH (length-filtered range 0.16-0.18 nats/tok across all model sizes), passed on EqTheories. `concise` won all six cells. Empty baseline broke the verifier on MATH (5-8% accuracy). Full writeup: [`experiments/001_phase0_kl_sanity.md`](experiments/001_phase0_kl_sanity.md).

### Phase 1: metric × teacher-context matrix (~$30-50, 1-2 days)

A full 4-way matrix of `(KL_flavor) × (teacher_context_family) × (model) × (task)`, with `(acc, KL_flavor, ylen)` measured for every cell.

**Conditions per (model, task):**

- π_θ baseline: minimal format anchor (one rollout set per problem; the reference distribution for all KL computations).
- `distrib`: 14 system prompts from Phase 0.
- `sdft`: 3 conditions (3 random demo-pair seeds, k=2 demos each).
- `sdpo`: 1 condition.
- `sdr`: 1 condition.

Total: **~19 conditions per cell**. Models {0.8B, 2B, 4B} × tasks {MATH, EqTheories} = **6 cells**.

**Per condition:** for each of n=50 problems, sample K=4 rollouts from π_T, K=4 from π_θ. Score every rollout under both π_T's and π_θ's context. Compute four KL flavors per rollout. Aggregate to per-condition `(mean acc, mean fwd_kl, mean rev_kl, mean jsd, mean fwd_kl_first32, mean ylen)`.

**vLLM parallelism budget.** All rollouts for a given (model, task) cell go into one vLLM offline batch on a single H100. Settings: `gpu_memory_utilization=0.95`, `max_num_seqs=512`, `max_num_batched_tokens=131072`, `enable_prefix_caching=True`, `enable_chunked_prefill=True`. The 19 conditions × 50 problems × K=8 = 7,600 rollouts per cell, ~5M generated tokens; at ~3-5k tok/s on H100 this is ~15-30 min/cell. SDPO's two-pass structure runs as two sequential batches (Pass 1 then Pass 2 dependent on Pass 1's drafts). All other families run in one batch each.

**Deliverable per cell:** four Pareto plots (one per KL flavor), each scattering all 19 conditions with the π_θ anchor at `(0, baseline_acc)`. SDR is the expected top-right anchor (highest KL, highest acc). Compute the convex hull per panel and report:

- which KL flavor produces the widest spread among `distrib` prompts (likely `rev_kl` or `fwd_kl_first32`);
- whether SDR strictly dominates the best `distrib` prompt (sanity check on the answer-leaking ceiling);
- where SDFT and SDPO sit on the Pareto vs `distrib` and SDR (tests H4 ordering);
- per-flavor Spearman(acc rank, KL rank) across `distrib`-only — is the Lagrangian's tradeoff knob real for this flavor, or is acc orthogonal to KL?

**Phase 1 success** = at least one `(KL flavor, model, task)` cell has a non-flat frontier across `distrib` prompts (the Lagrangian is *not* a free knob), AND the SDR point sits strictly above-and-right of the best `distrib` point (confirming the upper bound), AND the SDFT < SDPO < SDR ordering on KL is at least approximately preserved (H4).

### Phase 2: OPD validation (~$50, 3-5 days)

Using the winning KL flavor identified in Phase 1, pick 5 teacher contexts spanning the frontier (anchor, 2 mid-frontier `distrib`, best `distrib`, OPSD upper-bound). For each, run OPD on Qwen3.5-14B as student with that teacher context, 200 gradient steps on the same MATH train split.

Reuses `explore/156_opd_trainer/` directly (it already takes `(prompt, teacher_prompt)` pairs).

Measure per prompt:

- Accuracy lift on AIME-24 after OPD (single metric).
- Gradient steps to reach 50% of final lift (efficiency metric).
- Final student vs teacher per-token KL on held-out problems (compounding-error proxy).

**Test H3:** Spearman(Phase1_rank, Phase2_OPD_efficiency) > 0.6 across the 5 teacher contexts.

The OPSD anchor in this list is critical: if OPSD wins Phase 2 by a wide margin and `distrib` prompts barely lift, the story becomes "answer-knowing teachers matter; prompt-engineered teachers don't" — still publishable, but reframes the result.

---

## Baselines and controls

- **`minimal_format` (π_θ anchor)** — bottom-left of the Pareto. Establishes baseline accuracy with format-only system prompt; everything else is measured relative to this.
- **OPSD (answer-leak)** — top-right Pareto anchor. Quantifies the accuracy ceiling for any same-family teacher context. Sanity check: OPSD should approach 100% accuracy on MATH; if not, the model can't even reproduce a known answer, which would surface a separate bug.
- **SDFT (random demonstration)** — intermediate-KL controlled-noise baseline. Tests H4.
- **Random-prompt control** — sample 5 system prompts as random words / mojibake from the same length distribution as the `distrib` pool. Establishes a KL "noise floor": real prompts should be more KL-efficient (higher acc per nat) than random.
- **Best human-authored prompt** from the literature (Tencent training-free GRPO experiences, SAIR Stage 1 leaked entries if any are public). Sanity check that our seed pool isn't trivially beaten.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| All four KL flavors are washed out on MATH (extension of Phase 0 finding) | medium | OPSD condition guarantees at least *one* high-KL point; if even OPSD's KL is small, the framework breaks for this task family entirely → reframe writeup as a negative result. |
| OPSD doesn't lift accuracy (model can't use the leak) | low | Phase 0 showed format anchors lift acc 8×; an explicit answer-leak should approach 100% on easy levels. If not, switch to demoing the leak as `{answer} <reasoning that matches it>`. |
| Reverse KL estimator unstable (samples from π_θ that are very low-prob under π_T) | medium | Add per-token clip at \|log_ratio\| < 20 nats. Mask tokens beyond clip; report fraction masked. |
| JSD too small to be informative (bounded above by log 2) | low | Report JSD alongside fwd/rev so we see which direction dominates; not relied on solo. |
| Phase 2 cross-model effect dominates teacher-context effect | medium | Random-prompt control on 14B during OPD. If random matches selected, selection doesn't transfer. |
| Length confound (from Phase 0) | resolved | Hard `ylen >= 200` filter on KL aggregates. |
| K=4 rollouts insufficient for reverse KL stability (new sampling distribution) | medium | Start K=4, bump to K=8 if SE > 30% of mean KL. Cost doubles to ~$60-100. |
| OPSD answer-leak is too easy to game / model copies it verbatim | low | OPSD acc is reported but not used as a Pareto point we compete against — it's an upper-bound diagnostic. |

---

## Compute and cost

| Phase | Compute | $ |
|---|---|---|
| Phase 0 (sanity) — done | ~6 H100-hours actual | ~$30 actual |
| Phase 1 (metric × teacher matrix) | ~3-6 H100-hours (6 cells × 30 min, parallel on Modal) | ~$15-30 |
| Phase 2 (OPD validation on 14B) | ~40 H100-hours (5 contexts × 8 hr each) | ~$50 |
| **Total Phase 1+2** | | **~$65-80** |

Phase 1 cost has gone down vs the original GEPA design because there's no outer-loop optimizer to fund (no Claude mutator calls, no β sweep). The metric matrix is one batch per cell.

---

## Connections to existing repo work

- `explore/146_training_free_grpo.py`: same optimizer shape, no KL term. Our additive change is the KL regularizer + β sweep.
- `explore/150_prompt_gen_bounds/`: prompt-prior infrastructure (log p_M(prompt)). Different KL quantity (prompt-level vs response-distribution), but `eval_prompts.py` batched-eval pattern is reusable.
- `explore/147_on_policy_distillation.py`: SDFT/OPSD trainer with many KL variants. Reference for KL-direction choices in Phase 2.
- `explore/156_opd_trainer/`: minimal OPD trainer that takes `(prompt, teacher_prompt)` pairs. Phase 2 reuses directly.
- `modal/modal_h100_ssh.py`: H100 + vLLM bringup pattern. Reuse verbatim.

---

## Out of scope (v1)

- Equational theories as primary task. Planned as Phase 3 stretch if v1 works.
- Per-instance hint writers (Brown's §8 "hint-writing model" idea).
- Cross-task transfer (prompt trained on MATH, eval on AIME or GPQA).
- Training a hint-rewriter model via RL co-evolution.
- Comparison to vanilla RL at matched compute (separate project).
- Logit (top-k) distillation for the KL estimator.

---

## Stretch (Phase 3+)

- **Equational theories.** Re-run Phase 1+2 on the SAIR Davis/Tao competition task (`huggingface.co/datasets/SAIRfoundation/equational-theories-selected-problems`). Different domain, public leaderboard, validates method generalization beyond math reasoning.
- **Per-instance hint writers.** Train or prompt-optimize a `(problem, answer) -> hint` function. Closer to OPSD and to Brown's §8 hint-writer.
- **Cross-family transfer.** Pareto curves on Qwen, applied to Llama. Does the ranking transfer across architectures?
- **Prompt-prior vs response-KL.** Compare `150_prompt_gen_bounds`-style prompt-prior log p_M(p) against this experiment's response-distribution KL as predictors of OPD efficiency. The cheap one (prompt-prior) might be as good.

---

## File layout (planned)

```
lagrangian_prompts/
├── SPEC.md                    # this file
├── data.py                    # MATH-500 + EqTheories loaders (Phase 0)
├── verifier.py                # task verifiers (Phase 0)
├── prompts.py                 # distrib pool (Phase 0) + new minimal-anchor entries
├── modal_phase0.py            # Phase 0 driver
├── analyze_phase0.py          # Phase 0 Pareto plot + summary
├── teacher_contexts.py        # NEW: build sdft/sdpo/sdr π_T contexts per problem
├── modal_phase1.py            # NEW: metric × teacher matrix driver
├── kl_metrics.py              # NEW: forward/reverse/JSD/first-K KL from logprobs
├── analyze_phase1.py          # NEW: 4 Pareto plots per cell
├── experiments/
│   └── 001_phase0_kl_sanity.md
└── results/
    ├── phase0/                # done
    └── phase1/                # next
```

---

## Modal infrastructure

Switch from per-call Modal functions (Phase 0 pattern, one container per `(model, task)` call) to a **persistent multi-GPU H100 sandbox** running vLLM as a long-lived server. Rationale:

- Phase 0 paid the 90s vLLM warmup + flashinfer JIT cost on every cell. Persistent server eliminates that.
- 19 conditions × 50 problems × K=8 × 4 KL queries means many short batches; we want them all hitting a warm engine.
- Multi-GPU (2-4 H100s) lets the 4B model serve at higher concurrency, and gives room for SDPO's two-pass workload without queue contention.

Pattern: bring up the server via `modal/modal_h100_ssh.py`-style sandbox (already in the repo). Launch vLLM with the Qwen3.5 model and the parallelism settings above. Phase 1 driver pushes batched JSON requests over the forwarded port; results stream to a Modal Volume. Keep the sandbox alive across all six (model, task) cells, restarting vLLM with a different model when switching.

Compute budget assumes ~2 sandboxes (one warmed for 0.8B/2B, one for 4B; 2B and 0.8B fit fine together on shared GPU memory but separate processes for cache isolation).

## Open questions for review

1. **Phase 2 student model: Qwen3.5-9B or 14B?** SPEC's original Phase 2 used 14B for a wider capability gap. With Phase 1 dropped 9B, we still have 9B available as a "next size up" target. 9B is cheaper, 14B is a stronger test of cross-model transfer. Lean: 9B for v1, 14B if H3 holds and we want to harden the claim.

2. **K=2 demos for SDFT — too few?** Original SDFT paper used 5-10 demos. Going to k=2 saves context budget. Risk: too weak a distributional pull to differentiate from `distrib` family. Mitigation: if SDFT KL is indistinguishable from `distrib`, bump k to 4 in a follow-up.

3. **SDPO feedback format.** The Hübotter paper is centered on coding (stderr / failed tests as feedback). For math/EqTheories, our `"Your previous answer: X. The correct answer is Y."` is the natural translation, but is it the *right* translation? Alternative: provide a short "what was wrong" critique generated by a stronger model. Lean: start with the answer-comparison form; add a critique variant if KL ordering inverts.

4. **Phase 2 dependency on H1 winning KL flavor.** What if H1 doesn't decisively pick a winner (multiple flavors all marginal)? Plan: run Phase 2 selection under `rev_kl` by default (TML OPD's choice), report Pareto rank correlation under all four flavors as a robustness check.
