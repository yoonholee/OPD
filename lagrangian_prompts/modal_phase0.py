"""
Modal app for Phase 0: KL/accuracy scan over prompts × models × tasks.

Per (model, task), we:
  1. Load vLLM with the model on one H100.
  2. For each candidate prompt p, for each problem x:
       - Sample K rollouts y_i ~ pi(.|x, p)              [pi_T]
       - For each y_i, compute log pi(y_i | x, empty)    [pi_theta]
       - Compute acc(p, x) and KL_bar(p, x).
  3. Aggregate per-prompt (acc, KL_bar) and return.

Result is a dict of (prompt_name -> per-problem records + aggregates).

Usage (local):
    modal run lagrangian_prompts/modal_phase0.py \\
        --model Qwen/Qwen3.5-4B --task math --n 30 --k 2 --max-tokens 1024 --no-enable-thinking

The local script loads problems + prompts and calls run_one.remote().
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

APP_NAME = "lagrangian-prompts-phase0"
PROJECT_ROOT = Path(__file__).parent

# Qwen3.5 uses Gated Delta Net (mamba-style) attention; flashinfer JIT-compiles
# the kernel at runtime, which requires nvcc -> use a CUDA devel base image.
IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "ninja-build")
    .pip_install(
        "vllm==0.20.0",
        "transformers>=4.45",
        "datasets",
        "numpy",
        "huggingface_hub",
        "hf_transfer",
        "sympy",
        "antlr4-python3-runtime==4.11",  # for sympy.parsing.latex
    )
    .env(
        {
            "VLLM_USE_DEEP_GEMM": "0",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_HOME": "/usr/local/cuda",
        }
    )
    .add_local_python_source("verifier", "data", "prompts")
)

HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)
VLLM_CACHE = modal.Volume.from_name("vllm-compile-cache", create_if_missing=True)
FLASHINFER_CACHE = modal.Volume.from_name("flashinfer-cache", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("phase0-results", create_if_missing=True)

app = modal.App(APP_NAME, image=IMAGE)


@app.function(
    gpu="H100!",
    timeout=60 * 60,
    volumes={
        "/root/.cache/huggingface": HF_CACHE,
        "/root/.cache/vllm": VLLM_CACHE,
        "/root/.cache/flashinfer": FLASHINFER_CACHE,
        "/results": RESULTS_VOL,
    },
)
def run_one(
    model_id: str,
    task: str,
    n: int = 30,
    k: int = 4,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    enable_thinking: bool = False,
) -> dict:
    """Score every (prompt, problem) pair for one (model, task) combo.

    Loads problems + prompts inside the function so we don't need datasets locally.
    """
    import os

    # Ensure deep_gemm guard is set before vllm imports.
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"

    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from data import load
    from prompts import get_prompts
    from verifier import verify

    if task == "math":
        n_per_level = max(1, n // 5)
        problems = load("math", n_per_level=n_per_level)
    elif task == "eqtheories":
        problems = load("eqtheories", n=n)
    else:
        raise ValueError(f"Unknown task: {task}")
    prompts = get_prompts(task)
    print(f"[{model_id} / {task}] loaded {len(problems)} problems, {len(prompts)} prompts", flush=True)

    t0 = time.time()
    print(f"[{model_id} / {task}] loading vLLM...", flush=True)
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    tok = llm.get_tokenizer()
    print(f"[{model_id} / {task}] loaded in {time.time() - t0:.1f}s", flush=True)

    # === Phase A: build chat prompts ===
    # For each (prompt_name, problem), construct two chat-template strings:
    #   pT_text = template(system=prompt, user=problem)
    #   pTheta_ids = tokens of template(system="", user=problem)
    # Track positions so we can match rollouts back to the originating pair.

    prompt_names = list(prompts.keys())
    gen_inputs: list[str] = []  # one entry per (prompt_name, problem)
    pair_meta: list[dict] = []  # parallel to gen_inputs
    pTheta_ids_cache: list[list[int]] = []

    for problem_idx, prob in enumerate(problems):
        problem_text = prob["problem"]
        # Empty-system version (pi_theta context) -- compute once per problem.
        msgs_theta = [
            {"role": "system", "content": ""},
            {"role": "user", "content": problem_text},
        ]
        # apply_chat_template(tokenize=True) may return list[int], list[list[int]],
        # or a BatchEncoding dict depending on tokenizer/config. Normalize to list[int].
        theta_text = tok.apply_chat_template(
            msgs_theta, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        theta_ids = tok.encode(theta_text, add_special_tokens=False)
        pTheta_ids_cache.append(list(theta_ids))

        for pname in prompt_names:
            sys_msg = prompts[pname]
            msgs_T = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": problem_text},
            ]
            pT_text = tok.apply_chat_template(
                msgs_T, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            gen_inputs.append(pT_text)
            pair_meta.append(
                {
                    "problem_idx": problem_idx,
                    "prompt_name": pname,
                }
            )

    n_pairs = len(gen_inputs)
    print(
        f"[{model_id} / {task}] generating {n_pairs} pairs x K={k} rollouts...",
        flush=True,
    )

    # === Phase B: batched generation under pi_T ===
    sp_gen = SamplingParams(
        n=k,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        logprobs=1,
    )
    t1 = time.time()
    gen_outputs = llm.generate(gen_inputs, sp_gen)
    print(
        f"[{model_id} / {task}] generation done in {time.time() - t1:.1f}s",
        flush=True,
    )

    # === Phase C: assemble pi_theta logprob queries ===
    # For each rollout y_i, build prompt_token_ids = pTheta_ids + y_ids and
    # request prompt_logprobs to recover log pi_theta(y_i | empty_ctx).
    lp_inputs: list[dict] = []  # one entry per rollout
    lp_meta: list[dict] = []  # parallel; needed to align logprobs back to rollouts

    rollout_records: list[dict] = []  # final per-rollout records

    for pair_idx, gen_out in enumerate(gen_outputs):
        meta = pair_meta[pair_idx]
        problem_idx = meta["problem_idx"]
        theta_ids = pTheta_ids_cache[problem_idx]

        for k_idx, completion in enumerate(gen_out.outputs):
            y_ids = list(completion.token_ids)
            y_text = completion.text

            # Sum log pi_T(y_t | ...) from the generation output's per-token logprobs.
            # vLLM: completion.logprobs is list[dict[int, Logprob] | None] of length len(y_ids).
            log_T = 0.0
            for t, tid in enumerate(y_ids):
                lp_at_t = completion.logprobs[t] if completion.logprobs else None
                if lp_at_t is None or tid not in lp_at_t:
                    # Defensive: rare edge case where the sampled token isn't in returned top-k.
                    # Skip; will fall through with whatever sum we have. Mark as suspect.
                    continue
                log_T += float(lp_at_t[tid].logprob)

            # Defensive: cast all to plain Python int. vllm 0.20's input validator
            # does max(prompt_ids), which barfs if anything is a str (or non-int).
            full_ids = [int(t) for t in theta_ids] + [int(t) for t in y_ids]
            if pair_idx == 0 and k_idx == 0:
                print(
                    f"  [debug] theta_ids type={type(theta_ids).__name__} "
                    f"len={len(theta_ids)} first3={list(theta_ids)[:3]} "
                    f"y_ids type={type(completion.token_ids).__name__} "
                    f"len={len(y_ids)} first3={y_ids[:3]} "
                    f"full_ids[0]={full_ids[0]!r} type={type(full_ids[0]).__name__}",
                    flush=True,
                )
            lp_inputs.append(TokensPrompt(prompt_token_ids=full_ids))
            lp_meta.append(
                {
                    "pair_idx": pair_idx,
                    "k_idx": k_idx,
                    "y_len": len(y_ids),
                    "theta_len": len(theta_ids),
                    "log_T": log_T,
                    "y_text": y_text,
                    "y_ids": y_ids,
                }
            )

    # === Phase D: batched pi_theta logprob computation ===
    print(
        f"[{model_id} / {task}] scoring {len(lp_inputs)} rollouts under pi_theta...",
        flush=True,
    )
    sp_lp = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
    )
    t2 = time.time()
    lp_outputs = llm.generate(lp_inputs, sp_lp)
    print(
        f"[{model_id} / {task}] logprob pass done in {time.time() - t2:.1f}s",
        flush=True,
    )

    # === Phase E: per-rollout KL + acc ===
    for lp_meta_i, lp_out in zip(lp_meta, lp_outputs):
        full_ids = lp_out.prompt_token_ids
        prompt_logprobs = lp_out.prompt_logprobs  # list[dict[int, Logprob] | None]
        theta_len = lp_meta_i["theta_len"]
        y_len = lp_meta_i["y_len"]

        if prompt_logprobs is None or len(prompt_logprobs) != len(full_ids):
            print(
                f"  WARN: prompt_logprobs length mismatch at pair {lp_meta_i['pair_idx']}",
                flush=True,
            )
            log_theta = float("nan")
        else:
            log_theta = 0.0
            log_theta_n_valid = 0
            for i in range(theta_len, theta_len + y_len):
                lp_at_i = prompt_logprobs[i]
                tid = full_ids[i]
                if lp_at_i is None or tid not in lp_at_i:
                    continue
                log_theta += float(lp_at_i[tid].logprob)
                log_theta_n_valid += 1
            if log_theta_n_valid < y_len * 0.9:
                # Suspect: too many positions missing.
                print(
                    f"  WARN: pair {lp_meta_i['pair_idx']} k {lp_meta_i['k_idx']}: "
                    f"only {log_theta_n_valid}/{y_len} positions had logprob.",
                    flush=True,
                )

        log_T = lp_meta_i["log_T"]
        kl_sum = log_T - log_theta
        kl_per_tok = kl_sum / max(y_len, 1)

        pair_meta_i = pair_meta[lp_meta_i["pair_idx"]]
        prob = problems[pair_meta_i["problem_idx"]]
        is_correct = verify(task, lp_meta_i["y_text"], prob["answer"])

        rollout_records.append(
            {
                "prompt_name": pair_meta_i["prompt_name"],
                "problem_idx": pair_meta_i["problem_idx"],
                "problem_id": prob.get("id", str(pair_meta_i["problem_idx"])),
                "k_idx": lp_meta_i["k_idx"],
                "y_len": y_len,
                "log_T": log_T,
                "log_theta": log_theta,
                "kl_sum": kl_sum,
                "kl_per_tok": kl_per_tok,
                "is_correct": bool(is_correct),
                "y_text": lp_meta_i["y_text"][:2000],  # truncate for storage
            }
        )

    # === Aggregate per prompt ===
    by_prompt: dict[str, dict] = {}
    for pname in prompt_names:
        recs = [r for r in rollout_records if r["prompt_name"] == pname]
        if not recs:
            continue
        kls = np.array([r["kl_per_tok"] for r in recs if r["kl_per_tok"] == r["kl_per_tok"]])  # filter nan
        accs = np.array([1.0 if r["is_correct"] else 0.0 for r in recs])
        ylens = np.array([r["y_len"] for r in recs])
        by_prompt[pname] = {
            "n_rollouts": len(recs),
            "acc_mean": float(accs.mean()),
            "acc_se": float(accs.std(ddof=1) / np.sqrt(len(accs))) if len(accs) > 1 else 0.0,
            "kl_mean": float(kls.mean()) if len(kls) else float("nan"),
            "kl_std": float(kls.std(ddof=1)) if len(kls) > 1 else 0.0,
            "kl_se": float(kls.std(ddof=1) / np.sqrt(len(kls))) if len(kls) > 1 else 0.0,
            "ylen_mean": float(ylens.mean()),
            "ylen_p90": float(np.percentile(ylens, 90)),
        }

    result = {
        "model_id": model_id,
        "task": task,
        "k": k,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n_problems": len(problems),
        "n_prompts": len(prompt_names),
        "by_prompt": by_prompt,
        "rollouts": rollout_records,
        "wall_time_s": time.time() - t0,
    }

    # Save to volume for persistence.
    safe_model = model_id.replace("/", "__")
    out_path = f"/results/phase0__{safe_model}__{task}__{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{model_id} / {task}] saved to {out_path}", flush=True)
    RESULTS_VOL.commit()

    # Print quick summary.
    print(f"\n=== {model_id} / {task} ===", flush=True)
    print(f"{'prompt':<24} {'acc':>6} {'kl/tok':>8} {'ylen':>6}", flush=True)
    for pname, agg in sorted(by_prompt.items(), key=lambda kv: -kv[1]["acc_mean"]):
        print(
            f"{pname:<24} {agg['acc_mean']:>6.3f} {agg['kl_mean']:>8.3f} {agg['ylen_mean']:>6.0f}",
            flush=True,
        )

    return result


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3.5-4B",
    task: str = "math",
    n: int = 30,
    k: int = 4,
    max_tokens: int = 1024,
    enable_thinking: bool = False,
):
    """Convenience entry: dispatch one (model, task) config to Modal H100."""
    print(
        f"Dispatching model={model} task={task} n={n} k={k} "
        f"max_tokens={max_tokens} enable_thinking={enable_thinking}"
    )
    result = run_one.remote(
        model_id=model,
        task=task,
        n=n,
        k=k,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )

    # Print summary locally.
    print("\n=== Final summary ===")
    print(f"{'prompt':<24} {'acc':>6} {'kl/tok':>8} {'ylen':>6}")
    for pname, agg in sorted(result["by_prompt"].items(), key=lambda kv: -kv[1]["acc_mean"]):
        print(
            f"{pname:<24} {agg['acc_mean']:>6.3f} {agg['kl_mean']:>8.3f} {agg['ylen_mean']:>6.0f}"
        )

    # Compute spread + correlation diagnostics.
    import statistics

    accs = [agg["acc_mean"] for agg in result["by_prompt"].values()]
    kls = [agg["kl_mean"] for agg in result["by_prompt"].values()]
    if len(kls) > 1:
        kl_range = max(kls) - min(kls)
        acc_range = max(accs) - min(accs)
        # Pearson correlation
        mean_a, mean_k = statistics.mean(accs), statistics.mean(kls)
        cov = sum((a - mean_a) * (kk - mean_k) for a, kk in zip(accs, kls))
        var_a = sum((a - mean_a) ** 2 for a in accs) ** 0.5
        var_k = sum((kk - mean_k) ** 2 for kk in kls) ** 0.5
        rho = cov / (var_a * var_k) if var_a * var_k > 0 else float("nan")
        print(f"\nKL range (nats/tok): {kl_range:.3f}")
        print(f"Acc range          : {acc_range:.3f}")
        print(f"Pearson(acc, kl)   : {rho:.3f}")
        print(f"\nCouncil gate: KL range >= 0.3 nats/tok? "
              f"{'PASS' if kl_range >= 0.3 else 'FAIL'}")
