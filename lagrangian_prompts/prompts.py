"""
Phase 0 candidate system prompts.

We want enough spread (short vs long, generic vs specific, helpful vs adversarial)
to get a real KL spread across the pool. Council's gate is >=0.3 nats/token spread.
"""

# === MATH-500 candidate system prompts ===

MATH_PROMPTS = {
    "empty": "",
    "cot": "Let's think step by step.",
    "expert": "You are an expert mathematician. Reason carefully and show all work.",
    "show_work": "Show your work step by step. Be precise about arithmetic.",
    "notation_first": (
        "First, restate the problem in clean mathematical notation. "
        "Then identify the technique. Then solve. End with the answer in \\boxed{}."
    ),
    "verify": (
        "Solve the problem step by step. After arriving at an answer, "
        "verify it by substitution or an independent check. "
        "Place the final answer in \\boxed{}."
    ),
    "technique_first": (
        "Begin by identifying which standard technique applies "
        "(algebraic manipulation, casework, induction, generating functions, etc.). "
        "Apply it carefully. Show every step. Final answer in \\boxed{}."
    ),
    "concise": "Solve concisely. Show only the key steps. Final answer in \\boxed{}.",
    "double_check": (
        "Solve carefully. Double-check every arithmetic step. "
        "If you spot a possible error, redo that step. "
        "Final answer in \\boxed{}."
    ),
    "structured": (
        "Use this structure:\n"
        "1. Restate the problem.\n"
        "2. Identify what is asked.\n"
        "3. Plan an approach.\n"
        "4. Execute the plan, showing all steps.\n"
        "5. Verify the answer.\n"
        "Final answer in \\boxed{}."
    ),
    "competition": (
        "This is a competition math problem. Be rigorous. Check special cases. "
        "Watch for off-by-one errors. Place the final answer in \\boxed{}."
    ),
    "one_line": "Answer in one line. No explanation. Just the final answer in \\boxed{}.",
    "trap_check": (
        "Solve the problem step by step. "
        "Before committing to an answer, ask: 'Is there a hidden case or constraint I missed?' "
        "Final answer in \\boxed{}."
    ),
    "minimal_format": "Final answer in \\boxed{}.",
    "verbose": (
        "Solve this problem with maximum detail. "
        "Explain every algebraic manipulation, every choice of substitution, "
        "and every case analysis. Justify each step with a sentence. "
        "Verify the final answer by plugging it back into the original constraints. "
        "Format the final answer as \\boxed{answer}."
    ),
}


# === SAIR equational theories candidate system prompts ===
#
# Task: given E1 (hypothesis) and E2 (conclusion) over magmas, does E1 imply E2?
# Output should be "true" or "false". Verifier accepts either case.

EQ_PROMPTS = {
    "empty": "",
    "direct": (
        "You will be given two equational laws over a magma. "
        "Decide whether the first equation implies the second. "
        "End your answer with the single word `true` or `false`."
    ),
    "cot": (
        "Reason step by step about whether the hypothesis implies the conclusion. "
        "End with `true` or `false`."
    ),
    "either_or": (
        "To decide whether E1 implies E2, you have two options:\n"
        "1. Derive E2 from E1 using equational reasoning (substitution, transitivity).\n"
        "2. Construct a finite magma where E1 holds but E2 fails (a counterexample).\n"
        "Try option 2 first for likely-false implications; option 1 for likely-true. "
        "End with `true` or `false`."
    ),
    "counterexample_first": (
        "Equational implications over magmas are usually disprovable by small finite "
        "counterexamples. First, try to find a counterexample by considering small magmas "
        "(2-element, 3-element). If you cannot, attempt a derivation. "
        "End your answer with `true` or `false`."
    ),
    "free_magma": (
        "An equation E1 implies E2 iff E2 holds in every magma satisfying E1, "
        "equivalently, E2 holds in the free magma quotiented by E1. "
        "Reason about which models satisfy E1 and whether they all satisfy E2. "
        "End with `true` or `false`."
    ),
    "syntactic_check": (
        "First, compare the variable sets and structure of E1 and E2. "
        "If E2 has variables not in E1, the implication is usually false. "
        "If E2 is structurally simpler than E1 in obvious ways, look for a derivation. "
        "End with `true` or `false`."
    ),
    "verify_both": (
        "Decide the implication. State the answer, then state your confidence and "
        "your reasoning (derivation or counterexample). "
        "Final answer on the last line: `true` or `false`."
    ),
    "tao_hint": (
        "This is a question from the Equational Theories Project (Tao et al). "
        "Most implications are false; counterexamples typically live in small finite magmas. "
        "If you suspect the implication is true, look for a short equational derivation. "
        "End with `true` or `false`."
    ),
    "concise": "true or false?",
    "structured": (
        "Format:\n"
        "Hypothesis: <restate E1>\n"
        "Conclusion: <restate E2>\n"
        "Candidate counterexample: <try to construct one>\n"
        "Or candidate derivation: <try to derive>\n"
        "Answer: true|false"
    ),
    "extra_careful": (
        "Read E1 and E2 carefully. Note the variables. "
        "Try to construct a magma where E1 holds but E2 fails. "
        "If after trying 2-element, 3-element, and 4-element magmas you find no counterexample, "
        "attempt to derive E2 from E1. "
        "End with `true` or `false`."
    ),
    "one_line": "Answer with exactly one word: true or false.",
    "minimal_format": "End your answer with `true` or `false`.",
    "magma_intro": (
        "A magma is a set with a binary operation `*`. An equational law is a universally "
        "quantified equation. E1 implies E2 means every magma satisfying E1 also satisfies E2. "
        "Reason carefully. End your answer with `true` or `false`."
    ),
    "false_default": (
        "In the Equational Theories project, implications between random pairs of laws "
        "are usually false. Bias toward false unless you have a clear derivation. "
        "End with `true` or `false`."
    ),
}


def get_prompts(task: str) -> dict[str, str]:
    if task == "math":
        return MATH_PROMPTS
    elif task == "eqtheories":
        return EQ_PROMPTS
    else:
        raise ValueError(f"Unknown task: {task}")
