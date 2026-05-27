"""
Task-specific verifiers. Pure-Python, no GPU.
"""

from __future__ import annotations

import re


# === MATH ===


def _extract_boxed(text: str) -> str | None:
    """Find the last \\boxed{...} expression, handling one level of nesting."""
    matches = re.findall(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def _normalize_math_answer(s: str) -> str:
    s = s.strip()
    s = s.replace("\\!", "")
    s = s.replace("\\,", "")
    s = s.replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.lstrip("$").rstrip("$")
    return s


def verify_math(output: str, gold_answer: str) -> bool:
    """Best-effort MATH-style verifier. String match after normalization, then sympy fallback."""
    extracted = _extract_boxed(output)
    if extracted is None:
        # Fallback: try last line.
        lines = [ln for ln in output.strip().split("\n") if ln.strip()]
        if not lines:
            return False
        extracted = lines[-1].strip()

    pred = _normalize_math_answer(extracted)
    gold = _normalize_math_answer(gold_answer)

    if pred == gold:
        return True

    # Try sympy equivalence as a last resort. Best-effort; many MATH answers don't parse.
    try:
        from sympy import Eq, simplify
        from sympy.parsing.latex import parse_latex

        a = parse_latex(pred)
        b = parse_latex(gold)
        return bool(simplify(a - b) == 0)
    except Exception:
        return False


# === Equational theories ===


_TRUE_TOKENS = {"true", "yes", "implies", "holds", "valid"}
_FALSE_TOKENS = {"false", "no", "does not imply", "doesn't imply", "counterexample"}


def verify_eqtheories(output: str, gold_answer: str) -> bool:
    """Extract last `true`/`false` from output and compare to gold."""
    text = output.lower().strip()

    # Find the LAST mention of true/false in the text.
    matches = list(re.finditer(r"\b(true|false)\b", text))
    if matches:
        pred = matches[-1].group(1)
    else:
        # Fallback: look for affirmative/negative tokens near the end.
        last_chunk = text[-200:]
        if any(t in last_chunk for t in _TRUE_TOKENS):
            pred = "true"
        elif any(t in last_chunk for t in _FALSE_TOKENS):
            pred = "false"
        else:
            return False

    return pred == gold_answer.strip().lower()


# === Dispatch ===


def verify(task: str, output: str, gold_answer: str) -> bool:
    if task == "math":
        return verify_math(output, gold_answer)
    elif task == "eqtheories":
        return verify_eqtheories(output, gold_answer)
    else:
        raise ValueError(f"Unknown task: {task}")
