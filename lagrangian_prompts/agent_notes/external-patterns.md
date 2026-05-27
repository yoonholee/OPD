# External patterns — on-policy distillation (OPD)

Imported from the OPD literature (awesome-on-policy-distillation, 2026-05-17).
These are *other people's* recipes/findings, kept separate from our own
[[lessons]]. Each entry: the trick, the source, and how it maps to our
004/005 results.

## Canonical recipe (Thinking Machines/Tinker; MiniLLM 2306.08543; GKD 2306.13649)

Sample rollouts from the **student**; one **teacher forward pass** for logprobs
on the student's tokens; per-token **reverse-KL**; per-token advantage =
−reverse_KL; feed an importance-sampling policy-gradient loss. No reward model,
no teacher sampling. Discount = 0 (next-token only; future-reward coupling
*raises* gradient variance — Revisiting-OPD 2603.25562). 7–30× cheaper than
RL / large SFT. Partial/short rollouts are fine. LoRA closes most of its gap
under OPD (6% vs 13% for SFT).

**Hard precondition:** reverse-KL only mode-seeks *within the student's existing
support*. New behavior at ~0 student prob cannot be acquired by reverse-KL OPD —
**cold-start with SFT / forward-KL first** (forward-KL adds support; reverse-KL
refines). Our `modal_phase2.py` skips the cold start (student = bare base); OK
because our teachers shift *behavior*, not add new knowledge — but it bounds how
much any teacher can transfer.

## Stabilization tricks

| trick | source | what / why | our status |
|---|---|---|---|
| Reference (base) divergence constraint + teacher-rollout mixture | StableOPD 2604.08527 | KL-to-base anchor + mixing teacher rollouts into the student stream prevents repetition→length-inflation→truncation collapse; +7.2% avg | **independently rediscovered in 005** (KL-to-base β fixed the `aj_concise` collapse). Rollout-mixture lever untried. |
| Teacher top-K local support matching + **stop-gradient** on the top-K selection | Revisiting-OPD 2603.25562 (+19.8%); The Many Faces 2605.11182 | Don't reduce to the sampled-token log-ratio on drifting prefixes; compare student vs teacher over the teacher-supported top-K set per prefix; stop-grad the selection; + top-p rollouts + special-token masking | most-cited single fix; implemented as `topk_rkl`. **TESTED (006): no benefit in our regime** — same endpoint as reverse_kl on no-leak (0.36) with much higher variance; collapses identically on answer-leak. Capability-distillation trick; does not port to context/PI distillation. |
| Entropy-aware reverse/forward mixing | Entropy-Aware OPD 2603.07079 | reverse-KL's mode-seeking destabilizes / collapses diversity on **high-entropy teacher tokens**; use forward-KL there, reverse-KL elsewhere; +1.4/+2.4/+5.0pp Qwen3-0.6B/1.7B/4B math | implemented as `entropy_aware`. **TESTED (006): actively harmful** — no-leak `concise` ID 0.36→0.24, GSM8K 0.44→0.07. The forward-KL admixture degrades this base (consistent with 005 forward_kl-collapses-both). Do not use. |
| Logit-space intermediate target (skew/bridge) | Veto 2601.07155; DistiLLM skew-KL 2402.03898 | geometric bridge teacher↔student with tunable β; suppresses pathological low-confidence-token gradients; β doubles as diversity knob | same family as KL-to-base; not implemented |
| Contrastive, data-type-aware loss | DistiLLM-2 2503.07067 | different loss on teacher- vs student-generated data (push teacher up, student down); never one loss for both | not implemented |
| Control-variate per-token reverse-KL baseline | vOPD 2605.07865 | closed-form value baseline; unbiased lower-variance single-sample estimate, no critic | relevant at our K=4; not implemented |

## Teacher / context selection (this is what explains 004/005)

- **Instance-specific vs shared-latent privileged info** — The Many Faces
  (2605.11182), quoted: OPSD/context-distillation "is effective when PI
  represents a **shared latent rule, such as a system prompt or alignment
  preference**" but "fails ... due to test-time absence of **instance-specific**
  privileged information". → `concise` (shared system-prompt rule) transfers;
  `aj_concise` (per-problem gold answer = instance-specific PI) provably cannot.
  **Our 005 "null" is the predicted outcome, not a surprising finding.** Cite
  this; reframe 005 accordingly.
- **OPD success conditions** — Rethinking-OPD (2604.13016): needs (i) compatible
  student/teacher thinking patterns and (ii) teacher offering *genuinely new
  capability*; success = progressive alignment on a tiny (97–99% mass) shared
  high-prob token set. Recovery: off-policy cold start + teacher-aligned prompt
  selection.
- **OPCD** (2602.12275) — *our Phase-2 done right*: on-policy reverse-KL vs a
  **context-conditioned** teacher to internalize system prompts / experiential
  knowledge; reported to **preserve OOD** and enable cross-size. Align
  `modal_phase2` framing to this; the productive lever is the shared-rule
  `distrib`/`concise` family, not `aj_concise`/`sdr`.
- **Privileged context ⇒ entropy collapse + overconfidence** — Illusion of
  Certainty (2604.16830): helpful privileged context "induces entropy collapse
  and a systematic optimism bias"; teacher-conditioned success is not a valid
  deployment-time confidence target (CaOPD: use student-grounded empirical
  success). Explains our entropy collapse under the privileged teacher; warns
  even *successful* OPD miscalibrates.
- **Training-free teacher/context picker** — Unmasking OPD (2605.10889):
  per-token/per-question cosine between the distillation gradient and an "ideal
  success gradient" selects teacher/context **without training runs**; guidance
  is more useful on *incorrect* rollouts than correct ones. Replaces our
  $20 four-run sweeps with a cheap diagnostic. Strong follow-up candidate.
- Competence-frontier weighting — PACED 2603.11178 (pass-rate weighting);
  AOPD 2605.06387 (localized teacher matching in non-positive-advantage regions).

## Implications for our work

1. 005's KL-to-base fix and "leak doesn't transfer" are both *named, predicted*
   results (StableOPD; The Many Faces). Reframe 005: not a surprising null — the
   expected consequence of instance-specific PI.
2. ~~Highest-value upgrades: topk_rkl, entropy_aware~~ **FALSIFIED in 006.**
   Both literature objectives gave no benefit / harm in our context-PI regime
   (topk_rkl = same endpoint, more variance; entropy_aware = degrades, GSM8K
   0.44→0.07). They are *capability-distillation* tricks; they do not port to
   context/PI distillation. **Keep the simple recipe: per-token `reverse_kl` +
   KL-to-base trust region (005).** The remaining untried loss-side idea is
   rollout-mixture (StableOPD); deprioritized — the loss is not the bottleneck.
3. The real lever is the **data/teacher side**, not the objective: OPCD-style
   shared-latent-rule (system-prompt / `distrib`) internalization with OOD
   preservation, + Unmasking-OPD's training-free gradient-alignment diagnostic
   to pick teachers/contexts without $20 sweeps. Answer-leak teachers remain a
   proven dead end (instance-specific PI, objective-agnostic — 006 confirms).

## Capability-matched / down-leveled teacher (the Phase-3 lever; lit-search 2026-05-19)

The exact Phase-3 framing — prompt-only, explicit (reward × KL-to-student) frontier, no teacher finetune — is **underexplored**. The adjacent established line is *capability-matched CoT distillation*, which repeatedly finds the down-leveled teacher distills better despite lower raw quality:

- **Small Models Struggle to Learn from Strong Reasoners** (2502.12143, ACL Findings 2025). Small students learn *better* from shorter, simpler reasoning chains aligned with their intrinsic capacity than from a strong reasoner's long chains; proposes **Mix Distillation** (mix long/short or large/small-model CoT). Directly corroborates our H3 null + OpenThoughts "better model ≠ better teacher". This is the headline prior for "prompt the teacher down".
- **Tailoring Instructions to Student's Learning Levels Boosts KD** (LGTM, 2305.09651, ACL 2023). Adapts the teacher's *instruction* to student level; grounds the gap in Cho-Hariharan.
- **Cho & Hariharan 2019, On the Efficacy of KD.** Root cause: a bigger/more-accurate teacher is harder for the student to emulate; an early-stopped (weaker) teacher distills better. Theoretical seed of both the down-level idea and our null.
- **Learning Student-Friendly Teacher Networks** (Park et al., NeurIPS 2021, 2102.07650) and **Teachers That Listen: Adaptive Student-Aware Distillation** (OpenReview, 2025-09). Make the teacher distillable, but via *teacher finetuning*, not a prompt. The prompt-only frontier is the gap Phase 3 targets.
- **TAKD** (Mirzadeh, teacher-assistant) bridges the gap with an intermediate-size model — the "no single prompt closes a large gap; you may need a mid model" caution.

How it maps: our Phase-3 `downlevel` prompt is the prompt-only instantiation of this line; the **style-vs-capability KL decomposition** (plan.md) is the testable mechanism the line implies but does not measure. (arXiv IDs verified via web search 2026-05-19; pre-2026 ones are established, not model memory.)

## Sources

awesome-on-policy-distillation (chrisliu298). Key: TM blog
`thinkingmachines.ai/blog/on-policy-distillation`; OPD Survey 2604.00626;
MiniLLM 2306.08543; GKD 2306.13649; StableOPD 2604.08527; Revisiting-OPD
2603.25562; The Many Faces 2605.11182; Entropy-Aware 2603.07079; Veto
2601.07155; DistiLLM-2 2503.07067; OPCD 2602.12275; Rethinking-OPD 2604.13016;
Illusion of Certainty 2604.16830; Unmasking OPD 2605.10889. (arXiv IDs are 2026,
post knowledge-cutoff — from fetched abstracts, not model memory.)
