"""
Phase 1 driver: metric × teacher-context matrix.

For each (model, task) cell, runs all 19 conditions through one warm vLLM session
on a single H100. Per condition × problem: K=4 π_T rollouts + K=4 shared π_θ
anchor rollouts (the anchor rollouts are sampled once per problem and reused
across conditions). Computes fwd_kl, rev_kl, jsd, fwd_kl_first32 per condition
via kl_metrics, length-filtered at ylen >= 200.

SDPO is two-pass: its π_T context depends on a π_θ-sampled draft. We reuse the
first anchor rollout per problem as that draft, so SDPO doesn't need an extra
generation pass.

Usage:
    modal run lagrangian_prompts/modal_phase1.py \\
        --model Qwen/Qwen3.5-4B --task math --n 50 --k 4 --max-tokens 1024
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "lagrangian-prompts-phase1"
PROJECT_ROOT = Path(__file__).parent

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
        "antlr4-python3-runtime==4.11",
    )
    .env(
        {
            "VLLM_USE_DEEP_GEMM": "0",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_HOME": "/usr/local/cuda",
        }
    )
    .add_local_python_source(
        "verifier", "data", "prompts", "kl_metrics", "teacher_contexts",
    )
)

HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)
VLLM_CACHE = modal.Volume.from_name("vllm-compile-cache", create_if_missing=True)
FLASHINFER_CACHE = modal.Volume.from_name("flashinfer-cache", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("phase1-results", create_if_missing=True)

app = modal.App(APP_NAME, image=IMAGE)


@app.function(
    gpu="H100!",
    timeout=60 * 120,
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
    n: int = 50,
    k: int = 4,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    gpu_memory_utilization: float = 0.80,
    max_model_len: int = 4096,
    topk: int = 20,  # logprobs/prompt_logprobs depth for the truncated KL estimator
    cond_set: str = "full",  # "full" | "exp003" (curated 0.8B up-left sweep)
    enable_thinking: bool = False,
) -> dict:
    """Score every (condition, problem) pair for one (model, task) combo."""
    import os

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from data import load
    from kl_metrics import Rollout, aggregate
    from teacher_contexts import (
        ANCHOR, anchor_messages, build_conditions, build_conditions_exp003,
        build_conditions_exp003r2, build_conditions_exp003r3,
    )
    from verifier import verify

    # === Load problems ===
    if task == "math":
        n_per_level = max(1, n // 5)
        problems = load("math", n_per_level=n_per_level)
    elif task == "eqtheories":
        problems = load("eqtheories", n=n)
    else:
        raise ValueError(f"Unknown task: {task}")
    print(f"[{model_id}/{task}] loaded {len(problems)} problems", flush=True)

    conditions = (
        build_conditions_exp003(task) if cond_set == "exp003"
        else build_conditions_exp003r2(task) if cond_set == "exp003r2"
        else build_conditions_exp003r3(task) if cond_set == "exp003r3"
        else build_conditions(task))
    print(f"[{model_id}/{task}] cond_set={cond_set}, built {len(conditions)} "
          f"conditions: {[c.name for c in conditions]}", flush=True)

    # === Load vLLM ===
    t0 = time.time()
    print(f"[{model_id}/{task}] loading vLLM...", flush=True)
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        # v5 stack of workarounds for vllm-project/vllm#37729 (Qwen3.5 V1 deadlock):
        # - enforce_eager=True: bypass cuda graphs (necessary, not sufficient)
        # - enable_prefix_caching=False: skip the LRU path (helped 3/6 cells)
        # - gdn_prefill_backend="triton": dodge the flashinfer GDN kernel path,
        #   which #37729 specifically names as a deadlock trigger for Qwen3.5
        # - max_num_seqs=64: bound concurrency
        enforce_eager=True,
        max_num_seqs=64,
        trust_remote_code=True,
        enable_prefix_caching=False,
        gdn_prefill_backend="triton",
    )
    tok = llm.get_tokenizer()
    print(f"[{model_id}/{task}] loaded in {time.time() - t0:.1f}s", flush=True)

    def render(messages: list[dict]) -> tuple[str, list[int]]:
        """Apply chat template and tokenize. Returns (text, ids)."""
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        ids = tok.encode(text, add_special_tokens=False)
        return text, [int(t) for t in ids]

    def _lp_dict(d) -> dict:
        """vLLM {tid: Logprob} → {int tid: float logp}. Empty if None."""
        if not d:
            return {}
        return {int(tid): float(o.logprob) for tid, o in d.items()}

    def _extract_gen(comp):
        """From a sampling completion: (y_ids, logp_self[t], topk_self[t])."""
        y_ids = [int(t) for t in comp.token_ids]
        logp_self, topk_self = [], []
        for t, tid in enumerate(y_ids):
            d = comp.logprobs[t] if comp.logprobs else None
            topk_self.append(_lp_dict(d))
            logp_self.append(float(d[tid].logprob) if d and tid in d else 0.0)
        return y_ids, logp_self, topk_self

    def chunked_generate(inputs, sp, chunk_size: int, label: str):
        """Submit inputs in chunks to keep activation memory bounded.

        vLLM's scheduler tries to admit every queued request and overruns activation
        memory when 1000s of long-prompt prompt_logprobs queries arrive at once.
        Chunking serializes those waves while still letting each chunk fully
        saturate the GPU.
        """
        all_outputs = []
        n = len(inputs)
        for i in range(0, n, chunk_size):
            sub = inputs[i:i + chunk_size]
            t_chunk = time.time()
            outs = llm.generate(sub, sp)
            all_outputs.extend(outs)
            print(f"  [{label}] chunk {i // chunk_size + 1}/"
                  f"{(n + chunk_size - 1) // chunk_size}: "
                  f"{len(sub)} inputs in {time.time() - t_chunk:.1f}s", flush=True)
        return all_outputs

    def chunked_apply(inputs, sp, chunk_size: int, label: str, fn):
        """Like chunked_generate but stream: call fn(global_idx, output) per
        output and DROP raw outputs after each chunk. Required for
        prompt_logprobs=topk, where holding every request's full-context top-k
        objects at once OOMs the container (topk × every prompt position ×
        thousands of requests). fn must extract only what it needs.
        """
        n = len(inputs)
        nchunks = (n + chunk_size - 1) // chunk_size
        for i in range(0, n, chunk_size):
            sub = inputs[i:i + chunk_size]
            t_chunk = time.time()
            outs = llm.generate(sub, sp)
            for j, o in enumerate(outs):
                fn(i + j, o)
            del outs
            print(f"  [{label}] chunk {i // chunk_size + 1}/{nchunks}: "
                  f"{len(sub)} inputs in {time.time() - t_chunk:.1f}s", flush=True)

    # === Phase A: sample K anchor (π_θ) rollouts per problem ===
    print(f"[{model_id}/{task}] === Phase A: anchor rollouts ===", flush=True)
    anchor_texts: list[str] = []
    anchor_ids_per_prob: list[list[int]] = []
    for prob in problems:
        text, ids = render(anchor_messages(task, prob))
        anchor_texts.append(text)
        anchor_ids_per_prob.append(ids)

    sp_gen = SamplingParams(
        n=k, temperature=temperature, top_p=0.95,
        max_tokens=max_tokens, logprobs=topk,
    )
    t1 = time.time()
    anchor_rollouts: list[list[dict]] = [None] * len(anchor_texts)

    def _stash_anchor(p_idx, out):
        per_prob = []
        for comp in out.outputs:
            y_ids, logp_self, topk_self = _extract_gen(comp)
            per_prob.append({
                "y_ids": y_ids,
                "y_text": comp.text,
                "logp_self": logp_self,    # sampled under anchor → logp_θ
                "topk_self": topk_self,    # π_θ top-k per position
            })
        anchor_rollouts[p_idx] = per_prob

    chunked_apply(anchor_texts, sp_gen, 200, "A.gen", _stash_anchor)
    print(f"[{model_id}/{task}] anchor gen done in {time.time() - t1:.1f}s "
          f"({len(anchor_rollouts)} prompts × n={k})", flush=True)

    # === Phase A.5: critique generation (only if any condition needs it) ===
    # sdpo_critique uses a same-model critique of the first anchor draft. One
    # extra forward pass per problem, n=1, modestly truncated.
    critique_per_problem: list[str] = [""] * len(problems)
    if any(c.needs_critique for c in conditions):
        print(f"[{model_id}/{task}] === Phase A.5: critique gen ===", flush=True)
        crit_texts: list[str] = []
        for p_idx, prob in enumerate(problems):
            draft = anchor_rollouts[p_idx][0]["y_text"]
            crit_msgs = [
                {"role": "system", "content": ANCHOR[task]},
                {"role": "user", "content": (
                    f"Look at this attempt at the problem and critique it. "
                    f"Be specific about any mistakes or missed approaches; be concise.\n\n"
                    f"Problem:\n{prob['problem']}\n\n"
                    f"Attempt:\n{draft}\n\n"
                    f"Critique:"
                )},
            ]
            text, _ = render(crit_msgs)
            crit_texts.append(text)
        sp_crit = SamplingParams(
            n=1, temperature=0.7, top_p=0.95, max_tokens=400,
        )
        crit_outputs = chunked_generate(crit_texts, sp_crit, chunk_size=200, label="A5.crit")
        for p_idx, out in enumerate(crit_outputs):
            if out.outputs:
                critique_per_problem[p_idx] = out.outputs[0].text

    # === Phase B: build π_T chat prompts for each (condition, problem) ===
    # SDPO needs per-problem draft text → use anchor_rollouts[p][0]["y_text"].
    # sdpo_critique additionally uses critique_per_problem[p_idx].
    print(f"[{model_id}/{task}] === Phase B: build pi_T prompts ===", flush=True)
    pT_texts: list[str] = []
    pT_ids_per_pair: list[list[int]] = []
    pT_meta: list[dict] = []
    for ci, cond in enumerate(conditions):
        for p_idx, prob in enumerate(problems):
            draft = anchor_rollouts[p_idx][0]["y_text"] if cond.needs_draft else None
            critique = critique_per_problem[p_idx] if cond.needs_critique else None
            msgs = cond.builder(
                prob, prob["answer"], ANCHOR[task],
                demos_pool=problems, draft_text=draft, critique_text=critique,
            )
            text, ids = render(msgs)
            pT_texts.append(text)
            pT_ids_per_pair.append(ids)
            pT_meta.append({"cond_idx": ci, "prob_idx": p_idx})

    # === Phase C: sample K π_T rollouts per (condition, problem) ===
    print(f"[{model_id}/{task}] === Phase C: pi_T gen "
          f"({len(pT_texts)} prompts × n={k}) ===", flush=True)
    t2 = time.time()
    pT_rollouts: list[list[dict]] = [None] * len(pT_texts)

    def _stash_pT(pair_idx, out):
        per_pair = []
        for comp in out.outputs:
            y_ids, logp_self, topk_self = _extract_gen(comp)
            per_pair.append({
                "y_ids": y_ids,
                "y_text": comp.text,
                "logp_self": logp_self,   # sampled under condition → logp_T
                "topk_self": topk_self,   # π_T top-k per position
            })
        pT_rollouts[pair_idx] = per_pair

    chunked_apply(pT_texts, sp_gen, 200, "C.gen", _stash_pT)
    print(f"[{model_id}/{task}] pi_T gen done in {time.time() - t2:.1f}s", flush=True)

    # === Phase D: cross-scoring via prompt_logprobs ===
    # For each π_T rollout (sampled from condition c, problem p), score under
    # anchor context to get logp_θ at the rollout's tokens.
    # For each anchor rollout (problem p), score under condition c's π_T context
    # to get logp_T at the rollout's tokens. (Done once per (c, p, k_idx).)
    sp_lp = SamplingParams(
        n=1, temperature=0.0, max_tokens=1, prompt_logprobs=topk,
    )

    # ---- Build cross-score requests ----
    # Each entry: TokensPrompt + meta tag describing what it scores.
    cross_inputs = []
    cross_meta = []  # parallel list

    # (1) π_T rollouts under anchor context.
    for pair_idx, per_pair in enumerate(pT_rollouts):
        ci = pT_meta[pair_idx]["cond_idx"]
        pi = pT_meta[pair_idx]["prob_idx"]
        anchor_ids = anchor_ids_per_prob[pi]
        for ki, r in enumerate(per_pair):
            full_ids = anchor_ids + r["y_ids"]
            cross_inputs.append(TokensPrompt(prompt_token_ids=full_ids))
            cross_meta.append({
                "kind": "pT_under_anchor",
                "cond_idx": ci,
                "prob_idx": pi,
                "k_idx": ki,
                "ctx_len": len(anchor_ids),
                "y_len": len(r["y_ids"]),
            })

    # (2) Anchor rollouts under each condition's π_T context.
    for ci, cond in enumerate(conditions):
        for pi in range(len(problems)):
            # condition's π_T context for this problem is at pT_ids_per_pair[ci*P + pi]
            pair_idx = ci * len(problems) + pi
            ctx_ids = pT_ids_per_pair[pair_idx]
            for ki, r in enumerate(anchor_rollouts[pi]):
                full_ids = ctx_ids + r["y_ids"]
                cross_inputs.append(TokensPrompt(prompt_token_ids=full_ids))
                cross_meta.append({
                    "kind": "anchor_under_pT",
                    "cond_idx": ci,
                    "prob_idx": pi,
                    "k_idx": ki,
                    "ctx_len": len(ctx_ids),
                    "y_len": len(r["y_ids"]),
                })

    print(f"[{model_id}/{task}] === Phase D: cross-scoring "
          f"({len(cross_inputs)} requests) ===", flush=True)
    t3 = time.time()

    def _extract_logp_topk(prompt_logprobs, full_ids, ctx_len, y_len):
        """Slice (per-token logp, per-token top-k dict) for the y portion."""
        if prompt_logprobs is None:
            return [0.0] * y_len, [{} for _ in range(y_len)]
        logp, tk = [], []
        for i in range(ctx_len, ctx_len + y_len):
            if i >= len(prompt_logprobs) or i >= len(full_ids):
                logp.append(0.0)
                tk.append({})
                continue
            d = prompt_logprobs[i]
            tid = full_ids[i]
            tk.append(_lp_dict(d))
            logp.append(float(d[tid].logprob) if d and tid in d else 0.0)
        return logp, tk

    # ---- Stash cross-scored (logp, topk) into nested structures ----
    # pT_rollouts[pair_idx][ki]["logp_anchor"|"topk_anchor"]  ← pT_under_anchor
    # anchor_under_cond[ci][pi][ki] = {"logp":[...], "topk":[...]}  ← anchor_under_pT
    anchor_under_cond: list[list[list[dict]]] = [
        [[{} for _ in range(k)] for _ in range(len(problems))]
        for _ in range(len(conditions))
    ]
    def _stash_cross(idx, lp_out):
        meta_i = cross_meta[idx]
        logp_at_y, topk_at_y = _extract_logp_topk(
            lp_out.prompt_logprobs, lp_out.prompt_token_ids,
            meta_i["ctx_len"], meta_i["y_len"])
        ci, pi, ki = meta_i["cond_idx"], meta_i["prob_idx"], meta_i["k_idx"]
        if meta_i["kind"] == "pT_under_anchor":
            pair_idx = ci * len(problems) + pi
            pT_rollouts[pair_idx][ki]["logp_anchor"] = logp_at_y
            pT_rollouts[pair_idx][ki]["topk_anchor"] = topk_at_y
        else:  # anchor_under_pT
            anchor_under_cond[ci][pi][ki] = {"logp": logp_at_y, "topk": topk_at_y}

    chunked_apply(cross_inputs, sp_lp, 200, "D.lp", _stash_cross)
    print(f"[{model_id}/{task}] cross-scoring done in {time.time() - t3:.1f}s", flush=True)

    # === Phase E: per-condition aggregation ===
    print(f"[{model_id}/{task}] === Phase E: aggregating ===", flush=True)
    by_condition: dict[str, dict] = {}
    rollout_records: list[dict] = []  # for raw dump

    for ci, cond in enumerate(conditions):
        # Gather rollouts across all problems.
        rollouts_for_kl: list[Rollout] = []
        accs: list[float] = []
        ylens_T: list[int] = []
        for pi, prob in enumerate(problems):
            pair_idx = ci * len(problems) + pi

            # T-rollouts: sampled under condition, scored under (self=π_T, anchor=π_θ).
            for ki, r in enumerate(pT_rollouts[pair_idx]):
                logp_T = r["logp_self"]
                logp_th = r.get("logp_anchor", [0.0] * len(r["y_ids"]))
                rollouts_for_kl.append(Rollout(
                    source="T", logp_T=logp_T, logp_theta=logp_th,
                    topk_T=r.get("topk_self"),
                    topk_theta=r.get("topk_anchor"),
                ))
                is_correct = verify(task, r["y_text"], prob["answer"])
                accs.append(1.0 if is_correct else 0.0)
                ylens_T.append(len(r["y_ids"]))
                rollout_records.append({
                    "cond": cond.name,
                    "family": cond.family,
                    "prob_idx": pi,
                    "k_idx": ki,
                    "source": "T",
                    "y_len": len(r["y_ids"]),
                    "is_correct": bool(is_correct),
                    "y_text": r["y_text"][:1000],
                })

            # θ-rollouts: anchor-sampled, scored under (anchor=π_θ from sampling, condition's π_T).
            for ki, r_anchor in enumerate(anchor_rollouts[pi]):
                xs = anchor_under_cond[ci][pi][ki]
                logp_T_at_anchor = xs.get("logp") or [0.0] * len(r_anchor["y_ids"])
                topk_T_at_anchor = xs.get("topk")
                rollouts_for_kl.append(Rollout(
                    source="theta",
                    logp_T=logp_T_at_anchor,
                    logp_theta=r_anchor["logp_self"],
                    topk_T=topk_T_at_anchor,
                    topk_theta=r_anchor.get("topk_self"),
                ))

        agg = aggregate(rollouts_for_kl)
        # Accuracy: only T-rollouts (each condition's own π_T, what we'd deploy).
        import statistics
        acc_mean = statistics.mean(accs) if accs else 0.0
        acc_se = (statistics.stdev(accs) / (len(accs) ** 0.5)) if len(accs) > 1 else 0.0
        ylen_mean = statistics.mean(ylens_T) if ylens_T else 0.0

        by_condition[cond.name] = {
            "family": cond.family,
            "info_tier": cond.info_tier,
            "n_T_kept": agg.n_T_kept,
            "n_theta_kept": agg.n_theta_kept,
            "acc_mean": acc_mean,
            "acc_se": acc_se,
            "ylen_mean_T": ylen_mean,
            "ylen_mean_T_kept": agg.ylen_mean_T,
            "ylen_mean_theta_kept": agg.ylen_mean_theta,
            "fwd_kl": agg.fwd_kl,
            "rev_kl": agg.rev_kl,
            "jsd": agg.jsd,
            "fwd_kl_first32": agg.fwd_kl_first32,
            "fwd_kl_k3": agg.fwd_kl_k3,
            "rev_kl_k3": agg.rev_kl_k3,
            "fwd_kl_tk": agg.fwd_kl_tk,
            "rev_kl_tk": agg.rev_kl_tk,
            "tk_avail": agg.tk_avail,
            "fwd_clip_frac": agg.fwd_clip_frac,
            "rev_clip_frac": agg.rev_clip_frac,
        }

    # Anchor-only baseline accuracy + ylen for context.
    anchor_accs: list[float] = []
    anchor_ylens: list[int] = []
    for pi, prob in enumerate(problems):
        for r in anchor_rollouts[pi]:
            is_correct = verify(task, r["y_text"], prob["answer"])
            anchor_accs.append(1.0 if is_correct else 0.0)
            anchor_ylens.append(len(r["y_ids"]))
            rollout_records.append({
                "cond": "anchor",
                "family": "anchor",
                "prob_idx": pi,
                "k_idx": -1,
                "source": "theta",
                "y_len": len(r["y_ids"]),
                "is_correct": bool(is_correct),
                "y_text": r["y_text"][:1000],
            })
    import statistics
    anchor_summary = {
        "family": "anchor",
        "acc_mean": statistics.mean(anchor_accs) if anchor_accs else 0.0,
        "acc_se": (statistics.stdev(anchor_accs) / (len(anchor_accs) ** 0.5))
                  if len(anchor_accs) > 1 else 0.0,
        "ylen_mean": statistics.mean(anchor_ylens) if anchor_ylens else 0.0,
        "fwd_kl": 0.0, "rev_kl": 0.0, "jsd": 0.0, "fwd_kl_first32": 0.0,
    }

    result = {
        "model_id": model_id,
        "task": task,
        "n_problems": len(problems),
        "n_conditions": len(conditions),
        "k": k,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "anchor": anchor_summary,
        "by_condition": by_condition,
        "rollouts": rollout_records,  # truncated y_text
        "wall_time_s": time.time() - t0,
    }

    safe_model = model_id.replace("/", "__")
    out_path = f"/results/phase1__{safe_model}__{task}__{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"[{model_id}/{task}] saved to {out_path}", flush=True)
    RESULTS_VOL.commit()

    # Quick summary print, sorted by accuracy.
    print(f"\n=== {model_id} / {task} ===", flush=True)
    print(f"{'condition':<26} {'fam':<8} {'acc':>6} {'fwd_k1':>7} {'fwd_k3':>7} "
          f"{'fwd_tk':>7} {'rev_tk':>7} {'ylen':>6}", flush=True)
    print(f"{'anchor':<26} {'anchor':<8} {anchor_summary['acc_mean']:>6.3f} "
          f"{0:>7.3f} {0:>7.3f} {0:>7.3f} {0:>7.3f} {anchor_summary['ylen_mean']:>6.0f}",
          flush=True)
    for cname, agg in sorted(by_condition.items(), key=lambda kv: -kv[1]["acc_mean"]):
        print(
            f"{cname:<26} {agg['family']:<8} {agg['acc_mean']:>6.3f} "
            f"{agg['fwd_kl']:>7.3f} {agg['fwd_kl_k3']:>7.3f} "
            f"{agg['fwd_kl_tk']:>7.3f} {agg['rev_kl_tk']:>7.3f} "
            f"{agg['ylen_mean_T']:>6.0f}",
            flush=True,
        )

    return result


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3.5-4B",
    task: str = "math",
    n: int = 50,
    k: int = 4,
    max_tokens: int = 1024,
    topk: int = 20,
    cond_set: str = "full",
    enable_thinking: bool = False,
):
    print(f"Dispatching model={model} task={task} n={n} k={k} "
          f"max_tokens={max_tokens} topk={topk} cond_set={cond_set}")
    result = run_one.remote(
        model_id=model, task=task, n=n, k=k, max_tokens=max_tokens,
        topk=topk, cond_set=cond_set, enable_thinking=enable_thinking,
    )
    print(f"\nDone. wall_time={result.get('wall_time_s', 0):.1f}s")
