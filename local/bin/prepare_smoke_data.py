#!/usr/bin/env python3
"""Create tiny local datasets for OPD/GRPO/SFT smoke tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def write_subset(src: Path, dst: Path, n: int) -> int:
    df = pd.read_parquet(src).head(n).copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return len(df)


def write_sft_jsonl(src: Path, dst: Path, n: int) -> int:
    df = pd.read_parquet(src).head(n).copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            prompt = row.prompt[0]["content"]
            gt = row.reward_model["ground_truth"]
            # Smoke only: cheap supervised target so the LlamaFactory path can step.
            # Paper SFT should instead use lllyx/OpenThought3-Qwen3-4B or local teacher rollout.
            answer = f"We need solve the problem. For this smoke record, the known final answer is \\boxed{{{gt}}}."
            f.write(json.dumps({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}, ensure_ascii=False) + "\n")
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()
    root = args.root
    out = root / "datasets" / "local_smoke"
    n1 = write_subset(root / "datasets" / "dapo-math-17k-processed.parquet", out / f"dapo_math_train_{args.n}.parquet", args.n)
    n2 = write_subset(root / "datasets" / "OpenThoughts3_opd.parquet", out / f"openthought_train_{args.n}.parquet", args.n)
    n3 = write_sft_jsonl(root / "datasets" / "dapo-math-17k-processed.parquet", root / "LlamaFactory" / "data" / f"opd_smoke_sft_{args.n}.jsonl", args.n)
    print({"dapo": n1, "openthought": n2, "sft": n3, "out": str(out)})


if __name__ == "__main__":
    main()
