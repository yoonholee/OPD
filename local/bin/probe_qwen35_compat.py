#!/usr/bin/env python3
"""Probe Qwen3.5-2B text-only loading in Transformers and vLLM."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def transformers_probe(model: str) -> dict:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    m = AutoModelForImageTextToText.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Compute 2+2. Answer only the number."}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(m.device)
    with torch.inference_mode():
        out = m.generate(**inputs, max_new_tokens=8, do_sample=False)
    text = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return {"ok": True, "backend": "transformers", "seconds": round(time.time() - t0, 2), "text": text}


def vllm_probe(model: str) -> dict:
    from vllm import LLM, SamplingParams

    t0 = time.time()
    kwargs = dict(
        model=model,
        trust_remote_code=True,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    kwargs["gdn_prefill_backend"] = "triton"
    try:
        llm = LLM(**kwargs)
    except TypeError:
        kwargs.pop("gdn_prefill_backend", None)
        llm = LLM(**kwargs)
    out = llm.generate(["Compute 2+2. Answer only the number."], SamplingParams(max_tokens=8, temperature=0.0))
    return {"ok": True, "backend": "vllm", "seconds": round(time.time() - t0, 2), "text": out[0].outputs[0].text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--backend", choices=["transformers", "vllm"], required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    try:
        result = transformers_probe(args.model) if args.backend == "transformers" else vllm_probe(args.model)
    except Exception as exc:
        result = {"ok": False, "backend": args.backend, "error_type": type(exc).__name__, "error": str(exc)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
