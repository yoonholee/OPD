#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import math
import re
import sys
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
DICT_RE = re.compile(r"\{.*\}")
STEP_RE = re.compile(r"(?:^|\s)step:(\d+)\s*-\s*(.*)")
NUM_RE = re.compile(
    r"^(?:np\.float64\()?([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|inf|-inf)\)?"
)


def clean(text: str) -> str:
    return ANSI_RE.sub("", text.replace("\r", "\n"))


def parse_value(raw: str):
    raw = raw.strip().rstrip(",")
    match = NUM_RE.match(raw)
    if not match:
        if raw in {"True", "False"}:
            return raw == "True"
        return raw
    try:
        return float(match.group(1))
    except ValueError:
        return math.nan


def parse_step_lines(text: str):
    for line in clean(text).splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        row = {"training/global_step": float(match.group(1))}
        for part in match.group(2).split(" - "):
            if ":" not in part:
                continue
            key, raw = part.split(":", 1)
            row[key.strip()] = parse_value(raw)
        yield row


def parse_dict_payloads(text: str):
    for line in text.splitlines():
        if not any(
            x in line
            for x in (
                "actor/",
                "critic/",
                "train/",
                "training/",
                "reward",
                "teacher/",
                "val-",
            )
        ):
            continue
        match = DICT_RE.search(line)
        if not match:
            continue
        blob = match.group(0)
        try:
            obj = ast.literal_eval(blob)
        except Exception:
            try:
                obj = json.loads(blob)
            except Exception:
                continue
        if isinstance(obj, dict):
            yield obj


def status_rc(path: Path):
    for candidate in (
        path / "matrix.status",
        path / "opd_qwen35_2b.status",
        path / "grpo_qwen35_2b.status",
    ):
        if not candidate.exists():
            continue
        match = re.search(r"rc=(\d+)", candidate.read_text(errors="replace"))
        if match:
            return int(match.group(1))
    return None


def case_status(path: Path) -> str:
    status = path / "matrix.status"
    if not status.exists():
        return "missing"
    text = status.read_text(errors="replace")
    if " END " in text and "rc=" in text:
        return "done"
    if " START " in text:
        return "running"
    return "unknown"


def last_metrics(path: Path) -> tuple[dict, int]:
    text = ""
    for name in ("opd_qwen35_2b.log", "grpo_qwen35_2b.log", "matrix.log"):
        p = path / name
        if p.exists():
            text += "\n" + p.read_text(errors="replace")[-4_000_000:]
    metrics = list(parse_dict_payloads(text)) + list(parse_step_lines(text))
    metrics = [
        m
        for m in metrics
        if any(
            k.startswith(
                (
                    "actor/",
                    "critic/",
                    "train/",
                    "training/",
                    "teacher/",
                    "val-",
                    "perf/",
                )
            )
            for k in m
        )
    ]
    return (metrics[-1] if metrics else {}, len(metrics))


def main() -> int:
    root = Path(sys.argv[1])
    rows = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        last, n_metrics = last_metrics(case_dir)
        row = {
            "case": case_dir.name,
            "status": case_status(case_dir),
            "rc": status_rc(case_dir),
            "n_metric_payloads": n_metrics,
            "last_step": last.get("training/global_step"),
            "first_or_last_loss": last.get("actor/pg_loss", last.get("actor/loss")),
            "full_vocab_loss": last.get("actor/full_vocab_distill_loss"),
            "reverse_kl": last.get("actor/full_vocab_reverse_kl"),
            "forward_kl": last.get("actor/full_vocab_forward_kl"),
            "jsd": last.get("actor/full_vocab_jsd"),
            "student_entropy": last.get("actor/full_vocab_student_entropy"),
            "teacher_entropy": last.get(
                "teacher/full_vocab_entropy", last.get("teacher/entropy_mean")
            ),
            "grad_norm": last.get("actor/grad_norm"),
            "reward": last.get(
                "critic/rewards/mean", last.get("reward_model/reward_mean")
            ),
            "step_s": last.get("timing_s/step"),
            "throughput": last.get("perf/throughput"),
        }
        rows.append(row)

    (root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    cols = list(rows[0]) if rows else ["case", "status", "rc"]
    lines = ["\t".join(cols)]
    for row in rows:
        lines.append(
            "\t".join("" if row.get(col) is None else str(row.get(col)) for col in cols)
        )
    (root / "summary.tsv").write_text("\n".join(lines) + "\n")
    print((root / "summary.tsv").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
