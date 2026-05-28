#!/usr/bin/env python3
"""Create tiny local datasets for OPD/GRPO smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def write_subset(src: Path, dst: Path, n: int) -> int:
    df = pd.read_parquet(src).head(n).copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
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
    print({"dapo": n1, "openthought": n2, "out": str(out)})


if __name__ == "__main__":
    main()
