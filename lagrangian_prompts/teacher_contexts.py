"""
Teacher-context constructors for Phase 1.

Four families, all sharing a minimal-format anchor as π_θ baseline:

  distrib(prompt_name)   — Phase 0's 14 system prompts.
  sdft(seed, k=2)        — k same-task ICL demos in user message.
  sdpo()                 — two-pass: π_T sees student's draft + gold answer + critique.
  sdr()                  — full gold solution leak in user message.

All builders return chat-message lists; caller applies chat template.

π_θ baseline (the anchor for KL) is one minimal-format system prompt per task,
shared across all conditions:

  MATH:        "Solve the problem. End your answer with \\boxed{answer}."
  EqTheories:  "Decide whether E1 implies E2. End your answer with `true` or `false`."
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Optional

from prompts import EQ_PROMPTS, MATH_PROMPTS

ANCHOR = {
    "math": "Solve the problem. End your answer with \\boxed{answer}.",
    "eqtheories": "Decide whether E1 implies E2. End your answer with `true` or `false`.",
}


@dataclass
class Condition:
    """One condition in the metric × teacher-context matrix."""
    name: str
    family: str
    needs_draft: bool = False     # uses π_θ-sampled draft per problem
    needs_critique: bool = False  # uses π_θ-generated critique of draft (3-pass)
    # builder(problem, gold, anchor, demos_pool, draft_text=None, critique_text=None)
    builder: Optional[Callable] = None
    # Information tier — what extra info beyond the bare problem this condition uses.
    # "none": problem only (anchor, distrib)
    # "self": problem + model's own draft/critique (sdpo_no_answer, sdpo_critique)
    # "other_solutions": problem + other problems' gold (sdft)
    # "answer": problem + this problem's gold answer (opsd, sdpo)
    # "partial_solution": problem + slice of this problem's gold solution (sdr_strategy, sdr_first_step)
    # "full_solution": problem + this problem's full gold solution (sdr)
    info_tier: str = "none"


def anchor_messages(task: str, problem: dict) -> list[dict]:
    """π_θ messages: minimal format anchor as system, raw problem as user."""
    return [
        {"role": "system", "content": ANCHOR[task]},
        {"role": "user", "content": problem["problem"]},
    ]


def _distrib_builder(task: str, prompt_name: str) -> Callable:
    """π_T = anchor + candidate system prompt + raw problem.

    SPEC §Teacher-context families: "all sharing the same minimal format anchor
    as a base, then layering additional context". Without prepending the anchor,
    prompts that don't mention `\\boxed{}` (cot, expert, show_work) tank MATH
    accuracy because the verifier can't find a boxed answer — confounding
    reasoning quality with formatting.
    """
    prompts = MATH_PROMPTS if task == "math" else EQ_PROMPTS
    candidate = prompts[prompt_name]

    def build(problem, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        sys_msg = f"{anchor}\n\n{candidate}" if candidate else anchor
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": problem["problem"]},
        ]
    return build


def _sdft_builder(task: str, seed: int, k: int = 2) -> Callable:
    """π_T = anchor system + ICL prefix (k demos from train pool, current problem excluded)."""
    rng = random.Random(seed)

    def build(problem, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        # Pick k demos from pool that aren't the current problem.
        candidates = [p for p in demos_pool if p.get("id") != problem.get("id")]
        if len(candidates) < k:
            raise ValueError(f"sdft: need {k} demos, pool has only {len(candidates)} eligible")
        # Per-problem-deterministic shuffle: stable across rollouts, varies across problems.
        local_rng = random.Random(f"sdft|seed={seed}|pid={problem.get('id', '')}")
        chosen = local_rng.sample(candidates, k)

        # Build the demo prefix. For MATH the gold solution is the worked-out chain;
        # for eqtheories it's just true/false (we synthesize a one-line "rationale").
        demo_blocks = []
        for d in chosen:
            if task == "math":
                demo_blocks.append(f"Problem:\n{d['problem']}\n\nSolution:\n{d['solution']}")
            else:
                ans = d["answer"]
                demo_blocks.append(
                    f"Problem:\n{d['problem']}\n\nAnswer: {ans}"
                )
        user_text = "\n\n---\n\n".join(demo_blocks) + f"\n\n---\n\nProblem:\n{problem['problem']}"
        return [
            {"role": "system", "content": anchor},
            {"role": "user", "content": user_text},
        ]
    return build


def _sdpo_builder(task: str) -> Callable:
    """π_T = anchor + (problem + draft + gold answer + 'solve again' instruction)."""
    def build(problem, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        if draft_text is None:
            raise ValueError("sdpo: draft_text required (sample π_θ first)")
        user_text = (
            f"{problem['problem']}\n\n"
            f"Your previous answer:\n{draft_text}\n\n"
            f"Feedback: The correct answer is {gold}. "
            f"Solve again carefully, showing your reasoning."
        )
        return [
            {"role": "system", "content": anchor},
            {"role": "user", "content": user_text},
        ]
    return build


def _sdpo_no_answer_builder(task: str) -> Callable:
    """π_T = anchor + (problem + draft + 'review your work' instruction). No gold."""
    def build(problem, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        if draft_text is None:
            raise ValueError("sdpo_no_answer: draft_text required")
        user_text = (
            f"{problem['problem']}\n\n"
            f"Your previous answer:\n{draft_text}\n\n"
            f"Review your work and solve again carefully. "
            f"If your previous reasoning was correct, refine it; if not, fix the mistakes."
        )
        return [
            {"role": "system", "content": anchor},
            {"role": "user", "content": user_text},
        ]
    return build


def _sdpo_critique_builder(task: str) -> Callable:
    """π_T = anchor + (problem + draft + model's own critique + 'solve again')."""
    def build(problem, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        if draft_text is None or critique_text is None:
            raise ValueError("sdpo_critique: draft + critique required")
        user_text = (
            f"{problem['problem']}\n\n"
            f"Your previous answer:\n{draft_text}\n\n"
            f"Critique of the previous answer:\n{critique_text}\n\n"
            f"Now solve again carefully, addressing the points raised in the critique."
        )
        return [
            {"role": "system", "content": anchor},
            {"role": "user", "content": user_text},
        ]
    return build


def _opsd_builder(task: str) -> Callable:
    """π_T = anchor + (problem + 'the answer is X. show reasoning')."""
    def build(problem, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        user_text = (
            f"{problem['problem']}\n\n"
            f"The correct answer is {gold}. "
            f"Now produce the reasoning that arrives at this answer."
        )
        return [
            {"role": "system", "content": anchor},
            {"role": "user", "content": user_text},
        ]
    return build


def _redact_boxed(s: str) -> str:
    """Replace \\boxed{...} content with [answer] to hide the bottom-line leak."""
    return re.sub(r'\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '[answer]', s)


def _first_sentence(text: str) -> str:
    """Best-effort first sentence. Falls back to first 200 chars."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return sentences[0] if sentences else text[:200]


def _first_paragraph(text: str) -> str:
    """First non-empty paragraph (split on blank lines). Falls back to first 400 chars."""
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    return paragraphs[0] if paragraphs else text[:400]


def _sdr_strategy_builder(task: str) -> Callable:
    """π_T = anchor + (first sentence of gold solution as 'hint' + problem).

    MATH only — eqtheories has no worked solution. The first sentence is a
    BEST-EFFORT strategy extract; for some problems it leaks intermediates
    or the final value. Use with caution; the variable leak strength is a
    known limitation flagged in the writeup.
    """
    if task != "math":
        return None
    def build(problem, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        sol = problem.get("solution", "")
        hint = _first_sentence(_redact_boxed(sol))
        user_text = (
            f"Strategy hint: {hint}\n\n"
            f"Problem: {problem['problem']}"
        )
        return [
            {"role": "system", "content": anchor},
            {"role": "user", "content": user_text},
        ]
    return build


def _sdr_first_step_builder(task: str) -> Callable:
    """π_T = anchor + (first paragraph of gold solution + problem).

    MATH only. First paragraph often contains the answer computation; this is
    a strong-leak condition expected to sit between SDR-strategy and SDR-full.
    """
    if task != "math":
        return None
    def build(problem, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        sol = problem.get("solution", "")
        hint = _first_paragraph(_redact_boxed(sol))
        user_text = (
            f"Worked first step:\n{hint}\n\n"
            f"Continue and complete this solution:\n{problem['problem']}"
        )
        return [
            {"role": "system", "content": anchor},
            {"role": "user", "content": user_text},
        ]
    return build


def _sdr_builder(task: str) -> Callable:
    """π_T = anchor + (full gold solution as 'reference' + raw problem).

    For MATH: feeds problem["solution"] under "Reference solution to a similar
    problem" framing. Note: this is the EXACT problem's gold solution; the
    "similar" framing is a fig leaf per SPEC §SDR ("upper-bound diagnostic").

    For eqtheories: no worked solution in the dataset; we synthesize a generic
    rationale from the gold label. Expect SDR-eq to be ineffective.
    """
    def build(problem, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        if task == "math":
            ref = problem.get("solution") or f"The answer is {gold}."
        else:
            ref = (
                f"Reasoning: {'There is a derivation' if gold == 'true' else 'A counterexample exists in a small finite magma'}.\n"
                f"Therefore the answer is {gold}."
            )
        user_text = (
            f"Reference solution to a similar problem:\n{ref}\n\n"
            f"Now solve the following:\n{problem['problem']}"
        )
        return [
            {"role": "system", "content": anchor},
            {"role": "user", "content": user_text},
        ]
    return build


# ============================================================================
# Experiment 003: hand-designed variants to push the 0.8B Pareto up-and-left.
# Tier-0 = problem only (no leak). Tier-3 = this problem's gold answer.
# Design intent for a weak model: (a) lock the answer format so verifier misses
# don't cost accuracy, (b) bound reasoning length so the small model doesn't
# derail (verbose tanked 0.8B/math to 0.155), (c) answer-first to avoid
# truncation past the answer, (d) a single-call "meta-harness" protocol
# (propose→pick→execute→check→commit) ported from connections-meta-harness.
# ============================================================================

def _sys_layer(anchor: str, extra: str) -> str:
    return f"{anchor}\n\n{extra}" if extra else anchor


def _t0_terse_cot(task):
    def build(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": _sys_layer(
            anchor, "Think in at most 3 short steps, one line each. Then give the "
                    "final answer in the required format. Do not write anything "
                    "after the answer.")},
                {"role": "user", "content": p["problem"]}]
    return build


def _t0_answer_first(task):
    fmt = "\\boxed{ANSWER}" if task == "math" else "`true` or `false`"
    def build(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": _sys_layer(
            anchor, f"On the FIRST line output only {fmt}. Then on the next "
                    "lines give a brief justification (<=3 sentences).")},
                {"role": "user", "content": p["problem"]}]
    return build


def _t0_format_strict(task):
    fmt = "\\boxed{ANSWER}" if task == "math" else "exactly one word: true or false"
    def build(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": _sys_layer(
            anchor, f"Output ONLY {fmt}. No reasoning, no words, nothing else.")},
                {"role": "user", "content": p["problem"]}]
    return build


_META_HARNESS = (
    "Use this protocol exactly:\n"
    "1. Briefly list 2 candidate approaches (one phrase each).\n"
    "2. Pick the one most likely to be correct for this problem.\n"
    "3. Execute it concisely.\n"
    "4. Independently check the result with a different quick method.\n"
    "5. If the check disagrees, redo step 3 once.\n"
    "Then give the final answer in the required format."
)


def _t0_meta_harness(task):
    def build(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": _sys_layer(anchor, _META_HARNESS)},
                {"role": "user", "content": p["problem"]}]
    return build


def _t3_answer_then_justify(task):
    fmt = "\\boxed{}" if task == "math" else "`true`/`false`"
    def build(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": anchor},
                {"role": "user", "content": (
                    f"{p['problem']}\n\nThe correct answer is {gold}. "
                    f"Put it in {fmt} on the first line, then justify it in "
                    f"<=2 sentences.")}]
    return build


def _t3_confirm(task):
    def build(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": anchor},
                {"role": "user", "content": (
                    f"{p['problem']}\n\nA candidate answer is {gold}. Verify it "
                    f"with a quick check and present the correct final answer in "
                    f"the required format.")}]
    return build


def _t3_meta_harness_gold(task):
    def build(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": _sys_layer(anchor, _META_HARNESS)},
                {"role": "user", "content": (
                    f"{p['problem']}\n\n(The intended answer is {gold}; your "
                    f"step-4 check should confirm you reach it.)")}]
    return build


def _t03_combo(task):
    """Best tier-0 prompt (concise + format lock) stacked with the gold answer."""
    fmt = "\\boxed{}" if task == "math" else "`true`/`false`"
    def build(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": _sys_layer(
            anchor, "Solve concisely. Show only the key steps.")},
                {"role": "user", "content": (
                    f"{p['problem']}\n\nThe correct answer is {gold}. End with "
                    f"it in {fmt}.")}]
    return build


def _t3_box_only(task):
    """Ceiling/min-KL anchor of the answer-leak corner: format the gold, no prose."""
    fmt = "\\boxed{ANSWER}" if task == "math" else "exactly one word: true or false"
    def build(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": anchor},
                {"role": "user", "content": (
                    f"{p['problem']}\n\nThe answer is {gold}. Output it as "
                    f"{fmt} and nothing else.")}]
    return build


def _t3_aj_long(task):
    """t3_answer_then_justify but a fuller justification: raises ylen → lowers
    per-token KL (Phase-0 length effect) without giving up the committed answer."""
    fmt = "\\boxed{}" if task == "math" else "`true`/`false`"
    def build(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": anchor},
                {"role": "user", "content": (
                    f"{p['problem']}\n\nThe correct answer is {gold}. Put it in "
                    f"{fmt} on the first line, then give a thorough step-by-step "
                    f"justification (5+ sentences) showing why it is correct.")}]
    return build


def _t3_aj_concise(task):
    """Answer-first from gold, layered on the 'solve concisely' system prompt
    (the best tier-0 prompt) — the low-KL refinement of t03_combo."""
    fmt = "\\boxed{}" if task == "math" else "`true`/`false`"
    def build(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": _sys_layer(
            anchor, "Solve concisely. Show only the key steps.")},
                {"role": "user", "content": (
                    f"{p['problem']}\n\nThe correct answer is {gold}. State it "
                    f"first in {fmt}, then one concise line of justification.")}]
    return build


_EXP003_BUILDERS = {
    "t0_terse_cot": ("none", _t0_terse_cot),
    "t0_answer_first": ("none", _t0_answer_first),
    "t0_format_strict": ("none", _t0_format_strict),
    "t0_meta_harness": ("none", _t0_meta_harness),
    "t3_answer_then_justify": ("answer", _t3_answer_then_justify),
    "t3_confirm": ("answer", _t3_confirm),
    "t3_meta_harness_gold": ("answer", _t3_meta_harness_gold),
    "t03_combo": ("answer", _t03_combo),
}

# Round 2: refine the winner (answer-first gold injection) across the KL axis.
_EXP003_R2_BUILDERS = {
    "t3_box_only": ("answer", _t3_box_only),
    "t3_aj_long": ("answer", _t3_aj_long),
    "t3_aj_concise": ("answer", _t3_aj_concise),
}


# ============================================================================
# Round 3: 10 variants, each a distinct hypothesis about WHY answer-first wins
# and whether the up-left frontier can be pushed further.
# ============================================================================
_FMT = lambda task: "\\boxed{}" if task == "math" else "`true`/`false`"
_FMT_STRICT = lambda task: ("\\boxed{ANSWER}" if task == "math"
                            else "exactly one word: true or false")


def _t3_aj_verify(task):
    """answer-first, then a VERIFICATION (not a justification): does framing the
    second step as 'check it' beat 'justify it'?"""
    def b(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": anchor},
                {"role": "user", "content": (
                    f"{p['problem']}\n\nThe correct answer is {gold}. Put it in "
                    f"{_FMT(task)} on the first line, then do ONE quick "
                    f"independent check that confirms it.")}]
    return b


def _t3_aj_1sent(task):
    """answer-first + exactly one sentence: push KL lower than t3_aj_concise,
    map the acc/KL knee."""
    def b(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": anchor},
                {"role": "user", "content": (
                    f"{p['problem']}\n\nThe correct answer is {gold}. State it "
                    f"in {_FMT(task)} on line 1, then exactly ONE sentence of "
                    f"justification. Nothing more.")}]
    return b


def _t3_restate(task):
    """answer-first preceded by a one-line problem restatement: does grounding
    the model in the problem before justifying improve fidelity (esp. math)?"""
    def b(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": anchor},
                {"role": "user", "content": (
                    f"{p['problem']}\n\nThe correct answer is {gold}. First "
                    f"restate what is being asked in one line. Then show the 2-3 "
                    f"key steps that reach it, ending with it in {_FMT(task)}.")}]
    return b


def _t3_answer_in_system(task):
    """Position ablation: gold in the SYSTEM prompt rather than the user turn."""
    def b(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": _sys_layer(
            anchor, f"The correct answer to the user's problem is {gold}. "
                    f"Present it first in the required format, then justify "
                    f"briefly.")},
                {"role": "user", "content": p["problem"]}]
    return b


def _t3_confidence_gate(task):
    """answer-first + self-confidence-gated single re-derivation."""
    def b(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": anchor},
                {"role": "user", "content": (
                    f"{p['problem']}\n\nThe correct answer is {gold}. Put it in "
                    f"{_FMT(task)} on line 1. State your confidence (high/low). "
                    f"If low, re-derive once; otherwise give one line of "
                    f"justification.")}]
    return b


def _t3_discriminate(task):
    """Discrimination, not assertion: gold vs a plausible distractor. Does
    making the model *choose* improve reasoning fidelity over copying?"""
    def b(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        if task == "eqtheories":
            distract = "false" if str(gold).strip().lower() == "true" else "true"
        else:
            distract = "0"  # generic plausible-wrong numeric for MATH
        return [{"role": "system", "content": anchor},
                {"role": "user", "content": (
                    f"{p['problem']}\n\nTwo candidate answers: ({gold}) and "
                    f"({distract}). Decide which is correct, put it in "
                    f"{_FMT(task)} on line 1, then justify the choice briefly.")}]
    return b


def _t3_fill_in(task):
    """Guided derivation: gold + an explicit skeleton to complete."""
    def b(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": anchor},
                {"role": "user", "content": (
                    f"{p['problem']}\n\nThe answer is {gold}. Complete this:\n"
                    f"Setup: ...\nKey step: ...\nTherefore the answer is "
                    f"{_FMT(task)}.")}]
    return b


def _t0_answer_first_selfgen(task):
    """Structure-vs-leak ablation: answer-first with the model's OWN answer (no
    gold), softer than R1's t0_answer_first (which forced box-only line 1)."""
    def b(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": _sys_layer(
            anchor, "Decide your final answer, write it in the required format "
                    "on line 1, then justify it. Revise the line-1 answer only "
                    "if your justification contradicts it.")},
                {"role": "user", "content": p["problem"]}]
    return b


def _t0_min_scaffold(task):
    """Minimal scaffold (no gold): does a 2-step self-consistency-lite work
    where the full Meta-Harness protocol failed?"""
    def b(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": _sys_layer(
            anchor, "Solve it. Then write 'CHECK:' and re-derive the answer a "
                    "different way. Output the more reliable of the two in the "
                    "required format.")},
                {"role": "user", "content": p["problem"]}]
    return b


def _t0_commit(task):
    """concise + forced commitment (small models hedge / leave open)."""
    def b(p, gold, anchor, demos_pool, draft_text=None, critique_text=None):
        return [{"role": "system", "content": _sys_layer(
            anchor, "Solve concisely. If unsure, commit to the single most "
                    "likely standard answer rather than hedging or leaving it "
                    "open.")},
                {"role": "user", "content": p["problem"]}]
    return b


_EXP003_R3_BUILDERS = {
    "t3_aj_verify": ("answer", _t3_aj_verify),
    "t3_aj_1sent": ("answer", _t3_aj_1sent),
    "t3_restate": ("answer", _t3_restate),
    "t3_answer_in_system": ("answer", _t3_answer_in_system),
    "t3_confidence_gate": ("answer", _t3_confidence_gate),
    "t3_discriminate": ("answer", _t3_discriminate),
    "t3_fill_in": ("answer", _t3_fill_in),
    "t0_answer_first_selfgen": ("none", _t0_answer_first_selfgen),
    "t0_min_scaffold": ("none", _t0_min_scaffold),
    "t0_commit": ("none", _t0_commit),
}


def build_conditions_exp003r3(task: str) -> list[Condition]:
    """Round 3: 10 new variants + carried frontier (refs + R1/R2 winners)."""
    conds = [
        Condition("distrib:concise", "distrib", info_tier="none",
                  builder=_distrib_builder(task, "concise")),
        Condition("opsd", "opsd", info_tier="answer", builder=_opsd_builder(task)),
        Condition("sdr", "sdr", info_tier="full_solution", builder=_sdr_builder(task)),
        Condition("t3_answer_then_justify", "exp003_t3", info_tier="answer",
                  builder=_t3_answer_then_justify(task)),
        Condition("t3_aj_concise", "exp003_t3", info_tier="answer",
                  builder=_t3_aj_concise(task)),
    ]
    for name, (tier, fac) in _EXP003_R3_BUILDERS.items():
        fam = "exp003_t0" if tier == "none" else "exp003_t3"
        conds.append(Condition(name, fam, info_tier=tier, builder=fac(task)))
    return conds


def build_conditions_exp003(task: str) -> list[Condition]:
    """Curated small set: 3 reference frontier points + the new variants.

    References (existing winners, for visual frontier context):
      distrib:concise (best tier-0), opsd (tier-3 answer), sdr (tier-5 ceiling).
    """
    conds = [
        Condition("distrib:concise", "distrib", info_tier="none",
                  builder=_distrib_builder(task, "concise")),
        Condition("opsd", "opsd", info_tier="answer", builder=_opsd_builder(task)),
        Condition("sdr", "sdr", info_tier="full_solution", builder=_sdr_builder(task)),
    ]
    for name, (tier, fac) in _EXP003_BUILDERS.items():
        fam = "exp003_t0" if tier == "none" else "exp003_t3"
        conds.append(Condition(name, fam, info_tier=tier, builder=fac(task)))
    return conds


def build_conditions_exp003r2(task: str) -> list[Condition]:
    """Round 2: the Round-1 winner + its KL-axis refinements + frontier refs."""
    conds = [
        Condition("distrib:concise", "distrib", info_tier="none",
                  builder=_distrib_builder(task, "concise")),
        Condition("opsd", "opsd", info_tier="answer", builder=_opsd_builder(task)),
        Condition("sdr", "sdr", info_tier="full_solution", builder=_sdr_builder(task)),
        # Round-1 winner, carried for direct comparison.
        Condition("t3_answer_then_justify", "exp003_t3", info_tier="answer",
                  builder=_t3_answer_then_justify(task)),
        Condition("t03_combo", "exp003_t3", info_tier="answer",
                  builder=_t03_combo(task)),
    ]
    for name, (tier, fac) in _EXP003_R2_BUILDERS.items():
        conds.append(Condition(name, "exp003_t3", info_tier=tier,
                               builder=fac(task)))
    return conds


def build_conditions(
    task: str,
    distrib_prompt_names: Optional[list[str]] = None,
    sdft_seeds: tuple[int, ...] = (0, 1, 2),
    sdft_k: int = 2,
) -> list[Condition]:
    """Construct the full condition list for one (model, task) cell.

    Tiered by what extra information beyond the bare problem each condition uses:
      tier=none:             anchor, distrib:* (14-15)
      tier=self:             sdpo_no_answer, sdpo_critique
      tier=other_solutions:  sdft:* (3 seeds)
      tier=answer:           opsd, sdpo
      tier=partial_solution: sdr_strategy, sdr_first_step  (MATH only)
      tier=full_solution:    sdr

    Default totals: math → 14 + 2 + 3 + 2 + 2 + 1 = 24 conditions
                    eq   → 15 + 2 + 3 + 2 + 0 + 1 = 23 conditions
    """
    prompts = MATH_PROMPTS if task == "math" else EQ_PROMPTS
    if distrib_prompt_names is None:
        distrib_prompt_names = [n for n in prompts if n != "empty"]

    conditions: list[Condition] = []

    # Tier: none — problem only
    for pname in distrib_prompt_names:
        conditions.append(Condition(
            name=f"distrib:{pname}",
            family="distrib",
            info_tier="none",
            builder=_distrib_builder(task, pname),
        ))

    # Tier: self — model's own resources (no gold)
    conditions.append(Condition(
        name="sdpo_no_answer",
        family="sdpo_v",
        needs_draft=True,
        info_tier="self",
        builder=_sdpo_no_answer_builder(task),
    ))
    conditions.append(Condition(
        name="sdpo_critique",
        family="sdpo_v",
        needs_draft=True,
        needs_critique=True,
        info_tier="self",
        builder=_sdpo_critique_builder(task),
    ))

    # Tier: other_solutions — k=2 demos from train pool
    for seed in sdft_seeds:
        conditions.append(Condition(
            name=f"sdft:seed{seed}",
            family="sdft",
            info_tier="other_solutions",
            builder=_sdft_builder(task, seed=seed, k=sdft_k),
        ))

    # Tier: answer — current-problem gold answer
    conditions.append(Condition(
        name="opsd",
        family="opsd",
        info_tier="answer",
        builder=_opsd_builder(task),
    ))
    conditions.append(Condition(
        name="sdpo",
        family="sdpo",
        needs_draft=True,
        info_tier="answer",
        builder=_sdpo_builder(task),
    ))

    # Tier: partial_solution — slice of current-problem gold solution (MATH only)
    if task == "math":
        conditions.append(Condition(
            name="sdr_strategy",
            family="sdr_v",
            info_tier="partial_solution",
            builder=_sdr_strategy_builder(task),
        ))
        conditions.append(Condition(
            name="sdr_first_step",
            family="sdr_v",
            info_tier="partial_solution",
            builder=_sdr_first_step_builder(task),
        ))

    # Tier: full_solution — full gold solution
    conditions.append(Condition(
        name="sdr",
        family="sdr",
        info_tier="full_solution",
        builder=_sdr_builder(task),
    ))

    return conditions


def _self_test() -> None:
    """Print sample messages for one MATH and one EqTheories problem."""
    # Synthetic problems (avoid network in self-test).
    math_pool = [
        {"id": "p0", "problem": "Find x: 2x+3=11.", "solution": "2x=8, so x=4. \\boxed{4}", "answer": "4"},
        {"id": "p1", "problem": "Sum 1 to 10.", "solution": "55. \\boxed{55}", "answer": "55"},
        {"id": "p2", "problem": "What is 7*8?", "solution": "56. \\boxed{56}", "answer": "56"},
    ]
    eq_pool = [
        {"id": "e0", "problem": "Does (x*y)*z = x*(y*z) imply x*x = x?", "answer": "false"},
        {"id": "e1", "problem": "Does x*y = y*x imply (x*y)*z = x*(y*z)?", "answer": "false"},
        {"id": "e2", "problem": "Does x*y = y imply x*x = x?", "answer": "true"},
    ]

    for task, pool in [("math", math_pool), ("eqtheories", eq_pool)]:
        problem = pool[0]
        gold = problem["answer"]
        anchor = ANCHOR[task]

        conds = build_conditions(task, sdft_seeds=(0,), sdft_k=2)
        print(f"\n{'='*60}\nTASK: {task}  PROBLEM: {problem['problem'][:60]}...\n{'='*60}")
        print(f"\n[ANCHOR / π_θ]")
        for m in anchor_messages(task, problem):
            print(f"  {m['role']}: {m['content'][:120]}")

        # Print one example per (family, tier) combo.
        seen = set()
        for c in conds:
            key = (c.family, c.info_tier)
            if key in seen:
                continue
            seen.add(key)
            tags = f"family={c.family}, tier={c.info_tier}"
            if c.needs_draft:
                tags += ", needs_draft"
            if c.needs_critique:
                tags += ", needs_critique"
            print(f"\n[{c.name}] ({tags})")
            try:
                msgs = c.builder(
                    problem, gold, anchor, demos_pool=pool,
                    draft_text="(student's wrong attempt; got the wrong answer.)" if c.needs_draft else None,
                    critique_text="(the previous attempt missed the key insight that...)" if c.needs_critique else None,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            for m in msgs:
                content_preview = m["content"][:300].replace("\n", " ⏎ ")
                print(f"  {m['role']}: {content_preview}")

    # Sanity: count conditions per tier.
    for task in ["math", "eqtheories"]:
        conds = build_conditions(task)
        by_tier = {}
        for c in conds:
            by_tier.setdefault(c.info_tier, []).append(c.name)
        print(f"\nCondition counts ({task}):")
        for tier, names in by_tier.items():
            print(f"  tier={tier:<18} n={len(names):>2}  {names[:4]}...")
        print(f"  TOTAL: {len(conds)}")


if __name__ == "__main__":
    _self_test()
