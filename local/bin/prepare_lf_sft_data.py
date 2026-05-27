#!/usr/bin/env python3
"""Download a slice of lllyx/OpenThought3-Qwen3-4B and write LF sharegpt JSONL.

Run this from the repo root after activating .venv-lf-sft (datasets is installed there).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="number of samples to slice")
    ap.add_argument("--out", type=Path, default=Path("LlamaFactory/data/OpenThought3-4B.jsonl"))
    args = ap.parse_args()

    ds = load_dataset("lllyx/OpenThought3-Qwen3-4B", split=f"train[:{args.n}]")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in ds:
            msgs = row.get("messages") or row.get("conversations")
            if not msgs:
                continue
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print({"wrote": str(args.out), "n_rows": len(ds), "fields": list(ds.features.keys())})


if __name__ == "__main__":
    main()
