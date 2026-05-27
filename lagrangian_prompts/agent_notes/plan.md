# Plan — Phase 3: prompt the teacher *down* toward the student (OPD warm-start substitute)

## Status: IMPLEMENTED, Modal dry-run in flight (2026-05-19). Local-laptop runs forbidden by user; Modal-only.

User decisions (2026-05-19): teacher = **Qwen3.5-4B** first (4B→0.8B, cheap before 14B); code path = **new `modal_crossmodel.py`** (phase1 path untouched).
Built `modal_crossmodel.py`: two co-resident vLLM engines on one H100, 4 conditions (`anchor`/`concise`/`downlevel`/`verbose` as system-prompt suffixes on `ANCHOR[task]`), student π_θ = bare anchor on the 0.8B engine, cross-scored both directions, `kl_metrics.aggregate`.
`py_compile` clean; Modal CLI+token present; Modal `--dry-run` (n=5,k=2,maxtok=96) launched detached to validate engine co-residence / chat-template / cross-score shapes (laptop forbidden, so the dry-run is on Modal).
**Load-bearing methodological decision:** `aggregate(ylen_floor=0)` — the default 200-token length filter would discard the short `downlevel`/`concise` rollouts, but response length IS the Phase-3 independent variable. Primary frontier axis = `fwd_kl_first32` (length-invariant); per-token `fwd_kl_tk` reported but length differs across conditions by design.
Next: on dry-run green, launch the n=50 MATH run (`modal run lagrangian_prompts/modal_crossmodel.py --teacher Qwen/Qwen3.5-4B --student Qwen/Qwen3.5-0.8B --task math --n 50 --k 4 --max-tokens 1024`, ~1-2 H100-hr, ~$10). Pull → `experiments/007_*.md`; finding → [[lessons]].

Origin: Mujin raised the OPD **warm-start problem** — a big teacher is far off-policy from a small student, so on-policy reverse-KL only mode-seeks within the student's support (see [[external-patterns]] "hard precondition"), and the standard recipe needs an SFT warm-start first.
The Phase 3 question: can prompt-optimizing the *teacher* substitute for (part of) that SFT warm-start by pulling π_T toward the small student's support while holding task reward?

This phase is a **pre-OPD measurement only — no OPD training yet.**
We measure the achievable (teacher reward, KL-to-small-student) frontier as a function of the teacher's prompt.
Cross-model: π_T = big model + prompt; π_θ = small student + the bare 002/003 anchor.
Verified 2026-05-19 (cheap tokenizer check): Qwen3.5-{0.8B,4B} share an identical tokenizer (vocab 248077, encode-identical); Qwen2.5-{0.5B,1.5B} likewise (vocab 151665).
So cross-model token-level KL is well-defined within a Qwen size family — the load-bearing assumption holds.

**Fundamental hypothesis (the thing to test).** KL(π_T‖π_θ) decomposes into a **style** component (length / format / register — prompt-closable) and a **capability** component (reasoning depth the small model cannot represent — needs SFT).
The Phase-1 washout finding (response KL dominated by formatting tokens, ~0.001–0.07 nats/tok) is prior evidence the style component is large, so a prompt could close most of the warm-start-relevant distance.
Win condition: some prompt sits at high teacher reward **and** low cross-model KL; the residual KL at max prompt-reward is the irreducible capability gap that SFT would still have to cover.

**Guardrail (from 004/005).** On-policy ≠ useful: the answer-leak teacher was low-KL, high single-call reward, and transferred nothing.
Selection must be reward + coverage floored, not KL alone (OpenThoughts coverage lesson). A degenerate near-deterministic teacher is the failure corner.

**Faithful design (Modal H100).** Extend `modal_phase1.py` to two models (it is currently single-model: π_T and π_θ are the same `model_id` with different prompts).
Needed change: load a second student vLLM engine; route Phase-D `pT_under_anchor` cross-scoring through the student model and `anchor_under_pT` through the teacher (both models fit one H100 in bf16: 4B+0.8B). Either a contained edit or a new `modal_crossmodel.py`. Modal dry-run (1 problem) before any paid run.
- Teacher = Qwen3.5-4B first (modest real gap, cheap); Qwen3.5-14B is the true warm-start regime but costlier — **decision point, ask user**.
- Student π_θ = Qwen3.5-0.8B + bare anchor (consistent with 002–006). Task = MATH, n=50, `fwd_kl_tk`/`rev_kl_tk`.
- Prompts: `anchor` (bare student prompt on the big teacher — measures the unprompted off-policy distance + reward), `concise` (002/003 no-leak winner), `downlevel` (explicit "answer as a small/young model would: few short steps, simple arithmetic, match the format exactly" — the down-leveling lever, motivated by Small-Models-Struggle / LGTM in [[external-patterns]]), `verbose` (full strong CoT — the high-reward/high-KL corner to avoid).
- Output: (reward, fwd_kl_tk, rev_kl_tk, ylen) per prompt → frontier plot. Result → `experiments/007_*.md`; a finding → [[lessons]].

**Status detail.** Local POC (Qwen2.5-1.5B→0.5B, 8 GSM-style problems) was written and started on the laptop, then **killed unrun** per user "don't run it on this laptop" (2026-05-19) — no POC numbers exist. Faithful Modal run awaits user go on (a) teacher size (4B vs 14B), (b) launch + ~$ spend.

**Lit anchor.** Capability-matched / down-leveled-teacher distillation line — see [[external-patterns]] new section. Consistent external finding: matching teacher output to student capacity beats raw teacher strength; nobody has measured the pure prompt-induced (reward, KL-to-student) frontier — that is this phase's contribution.

---

# Plan — Phase 2: OPD from selected teacher contexts (experiment 004)

## Status: COMPLETE (2026-05-16) — see experiments/004_opd_leak_vs_transfer.md

Outcome (corrected after the 005 recipe ablation): SPEC H3 fails as a
**null, not an inversion**. The 003 answer-leak Pareto advantage does **not
transfer to OPD**: under a matched stabilized recipe (reverse_kl + KL-to-base)
`aj_concise` only ties no-leak `concise` on ID and loses OOD-down. 004's
"catastrophic collapse / inversion" was a **trust-region artifact** (KL-to-base
fixes it; forward_kl collapses both teachers). Surviving rule: pick OPD teachers
by no-leak distributional value, not 003 rank; always use a KL-to-base trust
region. See experiments/004 (raw runs, corrected) + 005 (ablation, conclusion).
Compile infeasible on Qwen3.5 (friction.md) → MAX_NEW/N_STEPS lowered per
step-6. Below is the original (executed) design.

Decisions confirmed with user 2026-05-16:
- **Compute: Modal H100**, adapt `explore/147_on_policy_distillation.py` (HF
  trainer; trains AND samples the same model → no vLLM weight-sync). NOT Tinker
  (its model list is Qwen3 ≥4B, not Qwen3.5-0.8B; using it breaks the
  base-model consistency with 002/003). GCP only as fallback if Modal H100 scarce.
- **Student/base: Qwen/Qwen3.5-0.8B** (exact 002/003 base — required so this
  tests "does the 003 Pareto position predict OPD transfer", SPEC H3).
- **4 OPD runs**, each student starts from base Qwen3.5-0.8B, teacher = same base
  + a context:
  | run | teacher context | tier | 0.8B/math (acc, tk-KL) from 003 |
  |---|---|---|---|
  | `anchor` | none (self-distill control) | — | (0.36, 0) |
  | `concise` | `distrib:concise` (no leak) | t0 | (0.38, .009) |
  | `aj_concise` | `t3_aj_concise` (003 winner, answer-leak) | t3 | (0.54, .042) |
  | `sdr` | full gold solution (max leak) | t5 | (0.44, .075) |

## Central hypothesis (label = guess, to be tested)

No-leak teacher (`concise`) → genuine OOD-transferable accuracy lift (student can
reproduce the behavior). Leak teachers (`aj_concise`, `sdr`) → student learns the
teacher's *format/calibration* but not correctness (can't recover an answer it
isn't given), so accuracy gain ≪ teacher's and may not survive OOD. `anchor` →
≈ no-op (noise floor). The leak-vs-transfer gap IS the result; ties to SPEC H3.

## Implementation (new file: `lagrangian_prompts/modal_phase2.py`)

Reuse 147's proven pieces; do NOT rewrite the loss/loop from scratch.

1. **Modal app**: H100, HF image (torch+transformers+datasets+math_eval; NO vLLM
   — training is HF). Mount an `phase2-results` Volume. ~1 run per Modal function
   call, `--teacher {anchor,concise,aj_concise,sdr}`; launch 4 detached in
   parallel like Phase 1.
2. **Port from 147**: `compute_distill_loss` (use `reverse_kl` — TML OPD default),
   `tokenize_batch`, `comp_mask`, the on-policy train step (student.generate →
   teacher fwd on context+rollout, student fwd on no-context+rollout, KL on
   response-token positions, optimizer step), `evaluate_model`.
3. **Replace `format_teacher_prompt`** with a dispatch over the 4 conditions,
   built from `teacher_contexts.py`:
   - `anchor`: teacher prompt == student prompt (no context) → KL≈0 control.
   - `concise`: `_distrib_builder(task,"concise")` system layering.
   - `aj_concise`: `_t3_aj_concise` (needs gold answer — 147 already loads it).
   - `sdr`: `_sdr_builder` (full `solution` in context — 147 already has it).
   Student prompt = the minimal anchor (matches 002/003 π_θ), NOT 147's default
   wording — keep consistent with the Pareto measurement.
4. **Datasets**: train = MATH train split (n≈2-4k, level-stratified). Eval sets:
   - ID: held-out MATH-500 test (same as 002/003) — n=50-200.
   - OOD-down: GSM8K test — n=200 (distribution shift easier).
   - OOD-up: MATH level-5 held-out — n≈100 (shift harder). (AIME-24 n=30 too
     noisy for 0.8B; skip or report separately as n=30 cavefocused.)
5. **Logging**: per-eval-step JSON `{step, train_loss, entropy, acc_id,
   acc_gsm8k, acc_math_l5}` → results Volume. EVAL_EVERY≈25-50, N_STEPS≈200-300,
   LR 1e-6 (147 default; 0.8B), BATCH 4, MAX_NEW 1024, temp 0.7. Eval at step 0
   (base) for the before/after delta.
6. **Hyperparams to sanity-check on a dry-run**: 0.8B fits one H100 in bf16 for
   train+generate easily. HF `model.generate` for on-policy rollouts is the
   slow part (~all the wall time); 200 steps × bs4 × 1024 tok ≈ feasible in
   1-3 H100-hr/run. If too slow, lower MAX_NEW to 512 or N_STEPS to 150.

## Run + report

- Local `--dry-run` first (1 step, tiny eval) to catch shape/template bugs.
- Launch 4 detached Modal runs; ~$10-20 total (4 × 1-3 H100-hr).
- Pull → `experiments/004_<slug>.md` (decision-first style): Takeaway, Setup,
  master table (per teacher: base→final acc on ID/GSM8K/MATH-L5, Δ, step-to-50%
  -of-final-lift), **learning-curve figure** (acc vs step, all 4, 3 eval sets →
  `experiments/figs/004_*.png`, non-gitignored path), Key findings tying back to
  003 / SPEC H3. Update lessons.md if the leak-vs-transfer hypothesis resolves.

## Risks / watch

- 147 has cluster `/scr`,`/iris` paths + typer CLI — strip for Modal (use Volume
  paths, plain function args), don't import 147 directly.
- `math_eval.get_answer_expr` / `is_correct` are repo-local — `add_local_python_source`.
- Teacher fwd needs the gold answer/solution in context for aj_concise/sdr; the
  student deploy-time eval must NOT (that's the whole point) — keep the two
  prompt paths strictly separate (a leak here would silently inflate results;
  high-severity correctness risk — assert teacher-only fields never reach eval).
- Reverse-KL on response tokens only (mask the context); 147's `comp_mask` is
  per-rollout EOS-based — verify it masks the prompt too in this two-prompt setup.

## Pointers

- `explore/147_on_policy_distillation.py` — source trainer (loss/loop/eval).
- `lagrangian_prompts/teacher_contexts.py` — `_distrib_builder`, `_t3_aj_concise`,
  `_sdr_builder`, `ANCHOR`.
- `experiments/003_0p8b_answer_first_upleft.md` — teacher Pareto positions.
- `experiments/002_phase1_metric_matrix.md` — full tier matrix + SPEC H3.
- `agent_notes/lessons.md` — answer-first ≫ answer-then-reason; KL estimators.
- SPEC.md §Phase 2 — original OPD-validation intent.
