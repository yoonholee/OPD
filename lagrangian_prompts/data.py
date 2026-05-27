"""
Phase 0 data loaders.

MATH-500: stratified across all 5 difficulty levels.
SAIR equational theories: `normal` config (~1000 problems with true/false labels).

Both loaders return list[dict] with task-specific keys + a unified `problem` field
containing the prompt-ready user message text.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Optional

from datasets import load_dataset


# === MATH ===


def _parse_math_level(lv) -> int:
    if isinstance(lv, int):
        return lv
    if isinstance(lv, str):
        m = re.search(r"\d+", lv)
        return int(m.group()) if m else 3
    return 3


def _extract_answer(solution: str) -> str:
    """Extract \\boxed{} answer from a MATH solution. Falls back to last line."""
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    if m:
        return m[-1]
    return solution.strip().split("\n")[-1].strip()


def load_math_500(n_per_level: int = 10, seed: int = 42) -> list[dict]:
    """Stratified MATH subset: n_per_level problems per difficulty level (1..5).

    Uses HuggingFaceH4/MATH-500 (the standard 500-problem subset).
    """
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    by_level = defaultdict(list)
    for row in ds:
        lvl = _parse_math_level(row["level"])
        by_level[lvl].append(
            {
                "problem": row["problem"],
                "solution": row["solution"],
                "answer": row.get("answer") or _extract_answer(row["solution"]),
                "level": lvl,
                "subject": row.get("subject", ""),
                "id": f"math500_lvl{lvl}_{len(by_level[lvl]):03d}",
            }
        )

    rng = random.Random(seed)
    out: list[dict] = []
    for lvl in sorted(by_level.keys()):
        bucket = by_level[lvl]
        rng.shuffle(bucket)
        out.extend(bucket[:n_per_level])
    return out


# === Equational theories ===


def load_eqtheories(n: int = 50, config: str = "normal", seed: int = 42) -> list[dict]:
    """Load SAIR equational theories problems.

    Configs: normal (1000), hard1 (69), hard2 (200), hard3 (400).
    Stage 1 eval set is balanced 50/50 true/false; the public configs are also
    expected to be roughly balanced.

    Each problem becomes: "Does E1 imply E2?" with a true/false ground truth.
    """
    ds = load_dataset(
        "SAIRfoundation/equational-theories-selected-problems",
        config,
        split="train",
    )
    rows = list(ds)

    # Try to balance true/false in the sample.
    rng = random.Random(seed)
    rng.shuffle(rows)

    by_label = defaultdict(list)
    for row in rows:
        # Label field may be `label` (bool/str) or `is_implication_true` etc.
        lbl = _coerce_eq_label(row)
        by_label[lbl].append(row)

    half = n // 2
    picked: list = []
    picked.extend(by_label[True][:half])
    picked.extend(by_label[False][: n - half])
    rng.shuffle(picked)

    out: list[dict] = []
    for i, row in enumerate(picked):
        hyp = _get_eq_field(row, ["equation1", "hypothesis", "hyp", "premise", "e1"])
        conc = _get_eq_field(row, ["equation2", "conclusion", "conc", "goal", "e2"])
        label = _coerce_eq_label(row)
        problem_text = (
            f"Does the following equational implication hold over magmas?\n\n"
            f"Hypothesis (E1): {hyp}\n"
            f"Conclusion (E2): {conc}\n\n"
            f"Does E1 imply E2?"
        )
        out.append(
            {
                "problem": problem_text,
                "hypothesis": hyp,
                "conclusion": conc,
                "answer": "true" if label else "false",
                "label_bool": label,
                "id": row.get("id", f"eq_{config}_{i:04d}"),
            }
        )
    return out


def _get_eq_field(row: dict, candidates: list[str]) -> str:
    for k in candidates:
        if k in row and row[k] is not None:
            return str(row[k])
    raise KeyError(f"None of {candidates} found in row with keys {list(row.keys())}")


def _coerce_eq_label(row: dict) -> bool:
    for k in ["answer", "label", "is_implication_true", "implies", "truth"]:
        if k in row and row[k] is not None:
            v = row[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.strip().lower() in ("true", "yes", "1", "t")
    raise KeyError(f"No label field found in row with keys {list(row.keys())}")


def load(task: str, **kwargs) -> list[dict]:
    if task == "math":
        return load_math_500(**kwargs)
    elif task == "eqtheories":
        return load_eqtheories(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")
