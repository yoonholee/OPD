#!/usr/bin/env python3
"""Measure same-weight next-token logprob drift across HF attention kernels."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Question: What is 17 + 29? Answer:",
    "Solve x^2 - 5x + 6 = 0. Final answer:",
    "Write a short proof that sqrt(2) is irrational. Proof:",
    "Compute the derivative of x^3 sin x. Answer:",
]


def topk_logprobs(model, tok, prompt: str, k: int) -> dict[int, float]:
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**ids).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)[0]
        vals, idx = torch.topk(lp, k)
    return {int(i): float(v) for i, v in zip(idx.cpu(), vals.cpu())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("prob_diff.json"))
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = []
    impls = ["sdpa", "flash_attention_2"]
    models = {}
    for impl in impls:
        t0 = time.time()
        models[impl] = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation=impl,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        rows.append({"event": "loaded", "impl": impl, "seconds": round(time.time() - t0, 2)})
    for prompt in PROMPTS:
        a = topk_logprobs(models["sdpa"], tok, prompt, args.k)
        b = topk_logprobs(models["flash_attention_2"], tok, prompt, args.k)
        union = sorted(set(a) | set(b))
        diffs = [abs(a.get(i, -math.inf) - b.get(i, -math.inf)) for i in union if math.isfinite(a.get(i, -math.inf)) and math.isfinite(b.get(i, -math.inf))]
        same_top1 = max(a, key=a.get) == max(b, key=b.get)
        rows.append({
            "prompt": prompt,
            "overlap_at_k": len(set(a) & set(b)) / args.k,
            "same_top1": same_top1,
            "max_abs_logprob_diff_shared_topk": max(diffs) if diffs else None,
            "mean_abs_logprob_diff_shared_topk": sum(diffs) / len(diffs) if diffs else None,
            "sdpa_top1": tok.decode([max(a, key=a.get)]),
            "flash_top1": tok.decode([max(b, key=b.get)]),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
