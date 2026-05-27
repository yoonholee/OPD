"""Phase 3 driver: cross-model (reward, KL-to-small-student) frontier.

Pre-OPD measurement. NO training. Question: can a teacher *prompt* pull a bigger
model's output distribution toward a smaller same-family student's on-policy
distribution while keeping task reward (the OPD warm-start substitute)?

  pi_T     = TEACHER model (Qwen3.5-4B) + a condition system prompt
  pi_theta = STUDENT model (Qwen3.5-0.8B) + the bare 002/003 anchor

Tokenizer is identical within a Qwen3.5 size family (verified 2026-05-19:
vocab 248077, encode-identical), so token-level KL across the two models is
well-defined. Two vLLM engines co-resident on one H100 (4B+0.8B fit easily).

Conditions (system-prompt suffix layered on ANCHOR[task]):
  anchor    -- teacher gets the bare student prompt: the UNPROMPTED-teacher
               off-policy distance + reward (the baseline the prompt must beat).
  concise   -- 002/003 no-leak winner family (short, format-anchored).
  downlevel -- explicit down-leveling (the Phase-3 lever; Small-Models-Struggle
               / LGTM): answer as a small model would, few tiny steps.
  verbose   -- full strong CoT: the high-reward / high-KL corner to AVOID.

Frontier read: is reward monotone in KL-to-student (you only get reward by going
off-policy -> a prompt cannot substitute for SFT warm-start), or does some prompt
sit high-reward AND low-KL (the style component is prompt-closable)?

Usage:
    modal run lagrangian_prompts/modal_crossmodel.py \\
        --teacher Qwen/Qwen3.5-4B --student Qwen/Qwen3.5-0.8B \\
        --task math --n 50 --k 4 --max-tokens 1024
    # dry run (Modal, NOT laptop) to catch shape/template/memory bugs:
    modal run lagrangian_prompts/modal_crossmodel.py --dry-run
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "lagrangian-prompts-crossmodel"
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

# Condition system-prompt suffixes, layered on teacher_contexts.ANCHOR[task].
# Student pi_theta NEVER gets a suffix (bare anchor only).
COND_SUFFIX = {
    "anchor": "",
    "concise": " Be extremely concise: at most two short sentences of working before the final answer.",
    "downlevel": (
        " You are a small, weak model with a very short attention span. "
        "Answer the way such a model would: at most two tiny steps, only "
        "simple arithmetic, no meta-commentary, no exploring alternatives. Keep it very short."
    ),
    "verbose": (
        " Think step by step in full detail. Explore multiple solution "
        "approaches, double-check every calculation, and explain your reasoning thoroughly before concluding."
    ),
}


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
    teacher_id: str,
    student_id: str,
    task: str = "math",
    n: int = 50,
    k: int = 4,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    max_model_len: int = 4096,
    topk: int = 20,
    teacher_gmu: float = 0.45,
    student_gmu: float = 0.35,
    enable_thinking: bool = False,
) -> dict:
    import os
    import statistics

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from data import load
    from kl_metrics import Rollout, aggregate
    from teacher_contexts import ANCHOR, anchor_messages
    from verifier import verify

    # === Problems ===
    if task == "math":
        problems = load("math", n_per_level=max(1, n // 5))
    elif task == "eqtheories":
        problems = load("eqtheories", n=n)
    else:
        raise ValueError(f"Unknown task: {task}")
    P = len(problems)
    conds = list(COND_SUFFIX)
    print(f"[xmodel] T={teacher_id} S={student_id} task={task} "
          f"P={P} conds={conds}", flush=True)

    # === Two co-resident vLLM engines on one H100 ===
    # Qwen3.5 V1 deadlock workaround stack (vllm#37729): enforce_eager,
    # no prefix caching, triton GDN. Same as modal_phase1.
    common = dict(
        dtype="bfloat16", max_model_len=max_model_len, enforce_eager=True,
        max_num_seqs=64, trust_remote_code=True, enable_prefix_caching=False,
        gdn_prefill_backend="triton",
    )
    t0 = time.time()
    print("[xmodel] loading TEACHER engine...", flush=True)
    teacher = LLM(model=teacher_id, gpu_memory_utilization=teacher_gmu, **common)
    tok_T = teacher.get_tokenizer()
    print(f"[xmodel] teacher up ({time.time() - t0:.0f}s); loading STUDENT...", flush=True)
    student = LLM(model=student_id, gpu_memory_utilization=student_gmu, **common)
    tok_S = student.get_tokenizer()
    print(f"[xmodel] both engines up ({time.time() - t0:.0f}s)", flush=True)

    def render(tok, messages):
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        ids = tok.encode(text, add_special_tokens=False)
        return text, [int(t) for t in ids]

    def _lp_dict(d):
        return {int(tid): float(o.logprob) for tid, o in d.items()} if d else {}

    def _extract_gen(comp):
        y_ids = [int(t) for t in comp.token_ids]
        logp, tk = [], []
        for t, tid in enumerate(y_ids):
            d = comp.logprobs[t] if comp.logprobs else None
            tk.append(_lp_dict(d))
            logp.append(float(d[tid].logprob) if d and tid in d else 0.0)
        return y_ids, logp, tk

    def chunked_apply(engine, inputs, sp, label, fn, chunk=200):
        n_in = len(inputs)
        nch = (n_in + chunk - 1) // chunk
        for i in range(0, n_in, chunk):
            tc = time.time()
            outs = engine.generate(inputs[i:i + chunk], sp)
            for j, o in enumerate(outs):
                fn(i + j, o)
            del outs
            print(f"  [{label}] chunk {i // chunk + 1}/{nch} "
                  f"({time.time() - tc:.0f}s)", flush=True)

    def _extract_logp_topk(prompt_logprobs, full_ids, ctx_len, y_len):
        if prompt_logprobs is None:
            return [0.0] * y_len, [{} for _ in range(y_len)]
        logp, tk = [], []
        for i in range(ctx_len, ctx_len + y_len):
            if i >= len(prompt_logprobs) or i >= len(full_ids):
                logp.append(0.0); tk.append({}); continue
            d = prompt_logprobs[i]
            tid = full_ids[i]
            tk.append(_lp_dict(d))
            logp.append(float(d[tid].logprob) if d and tid in d else 0.0)
        return logp, tk

    sp_gen = SamplingParams(n=k, temperature=temperature, top_p=0.95,
                            max_tokens=max_tokens, logprobs=topk)
    sp_lp = SamplingParams(n=1, temperature=0.0, max_tokens=1,
                           prompt_logprobs=topk)

    # === Phase A: STUDENT anchor rollouts (pi_theta) ===
    print("[xmodel] Phase A: student anchor rollouts", flush=True)
    s_anchor_text, s_anchor_ids = [], []
    for prob in problems:
        txt, ids = render(tok_S, anchor_messages(task, prob))
        s_anchor_text.append(txt); s_anchor_ids.append(ids)
    anchor_roll: list = [None] * P

    def _stash_anchor(pi, out):
        rr = []
        for comp in out.outputs:
            y, lp, tk = _extract_gen(comp)
            rr.append({"y_ids": y, "y_text": comp.text,
                       "logp_self": lp, "topk_self": tk})
        anchor_roll[pi] = rr

    chunked_apply(student, s_anchor_text, sp_gen, "A.student", _stash_anchor)

    # === Phase B/C: TEACHER rollouts per (cond, prob) (pi_T) ===
    print("[xmodel] Phase C: teacher rollouts", flush=True)
    pT_text, pT_cond_ids, pT_meta = [], [], []
    for ci, c in enumerate(conds):
        sysmsg = ANCHOR[task] + COND_SUFFIX[c]
        for pi, prob in enumerate(problems):
            msgs = [{"role": "system", "content": sysmsg},
                    {"role": "user", "content": prob["problem"]}]
            txt, ids = render(tok_T, msgs)
            pT_text.append(txt); pT_cond_ids.append(ids)
            pT_meta.append({"ci": ci, "pi": pi})
    pT_roll: list = [None] * len(pT_text)

    def _stash_pT(idx, out):
        rr = []
        for comp in out.outputs:
            y, lp, tk = _extract_gen(comp)
            rr.append({"y_ids": y, "y_text": comp.text,
                       "logp_self": lp, "topk_self": tk})
        pT_roll[idx] = rr

    chunked_apply(teacher, pT_text, sp_gen, "C.teacher", _stash_pT)

    # === Phase D: cross-scoring ===
    # (1) teacher rollouts under STUDENT+anchor  -> logp_theta at y
    # (2) student anchor rollouts under TEACHER+cond -> logp_T at y
    print("[xmodel] Phase D1: teacher rollouts under student", flush=True)
    d1_inputs, d1_meta = [], []
    for pair_idx, rr in enumerate(pT_roll):
        pi = pT_meta[pair_idx]["pi"]
        a_ids = s_anchor_ids[pi]
        for ki, r in enumerate(rr):
            d1_inputs.append(TokensPrompt(prompt_token_ids=a_ids + r["y_ids"]))
            d1_meta.append({"pair_idx": pair_idx, "ki": ki,
                            "ctx": len(a_ids), "ylen": len(r["y_ids"])})

    def _stash_d1(idx, lp_out):
        m = d1_meta[idx]
        logp, tk = _extract_logp_topk(lp_out.prompt_logprobs,
                                      lp_out.prompt_token_ids, m["ctx"], m["ylen"])
        r = pT_roll[m["pair_idx"]][m["ki"]]
        r["logp_theta"] = logp
        r["topk_theta"] = tk

    chunked_apply(student, d1_inputs, sp_lp, "D1.student", _stash_d1)

    print("[xmodel] Phase D2: student rollouts under teacher+cond", flush=True)
    d2_inputs, d2_meta = [], []
    for ci, c in enumerate(conds):
        for pi in range(P):
            ctx_ids = pT_cond_ids[ci * P + pi]
            for ki, r in enumerate(anchor_roll[pi]):
                d2_inputs.append(TokensPrompt(prompt_token_ids=ctx_ids + r["y_ids"]))
                d2_meta.append({"ci": ci, "pi": pi, "ki": ki,
                                "ctx": len(ctx_ids), "ylen": len(r["y_ids"])})
    anchor_under_T = [[[{} for _ in range(k)] for _ in range(P)]
                      for _ in range(len(conds))]

    def _stash_d2(idx, lp_out):
        m = d2_meta[idx]
        logp, tk = _extract_logp_topk(lp_out.prompt_logprobs,
                                      lp_out.prompt_token_ids, m["ctx"], m["ylen"])
        anchor_under_T[m["ci"]][m["pi"]][m["ki"]] = {"logp": logp, "topk": tk}

    chunked_apply(teacher, d2_inputs, sp_lp, "D2.teacher", _stash_d2)

    # === Phase E: aggregate per condition ===
    print("[xmodel] Phase E: aggregate", flush=True)
    by_cond = {}
    for ci, c in enumerate(conds):
        rolls, accs, ylens = [], [], []
        for pi, prob in enumerate(problems):
            pair_idx = ci * P + pi
            for r in pT_roll[pair_idx]:
                rolls.append(Rollout(
                    source="T", logp_T=r["logp_self"],
                    logp_theta=r.get("logp_theta", [0.0] * len(r["y_ids"])),
                    topk_T=r.get("topk_self"), topk_theta=r.get("topk_theta")))
                accs.append(1.0 if verify(task, r["y_text"], prob["answer"]) else 0.0)
                ylens.append(len(r["y_ids"]))
            for ki, ra in enumerate(anchor_roll[pi]):
                xs = anchor_under_T[ci][pi][ki]
                rolls.append(Rollout(
                    source="theta",
                    logp_T=xs.get("logp") or [0.0] * len(ra["y_ids"]),
                    logp_theta=ra["logp_self"],
                    topk_T=xs.get("topk"), topk_theta=ra.get("topk_self")))
        # ylen_floor=0: Phase 3's independent variable IS response length
        # (`downlevel` is designed short). The default 200-token length filter
        # would discard exactly the conditions under study. Lean on
        # fwd_kl_first32 (length-invariant) as the primary frontier axis;
        # per-token tk-KL is reported but length differs across conds by design.
        agg = aggregate(rolls, ylen_floor=0)
        by_cond[c] = {
            "acc_mean": statistics.mean(accs) if accs else 0.0,
            "acc_se": (statistics.stdev(accs) / len(accs) ** 0.5)
                      if len(accs) > 1 else 0.0,
            "ylen_mean_T": statistics.mean(ylens) if ylens else 0.0,
            "fwd_kl_tk": agg.fwd_kl_tk, "rev_kl_tk": agg.rev_kl_tk,
            "fwd_kl": agg.fwd_kl, "rev_kl": agg.rev_kl, "jsd": agg.jsd,
            "fwd_kl_first32": agg.fwd_kl_first32, "tk_avail": agg.tk_avail,
            "n_T_kept": agg.n_T_kept, "n_theta_kept": agg.n_theta_kept,
        }

    # Student's own anchor accuracy = the floor (pi_theta task reward).
    s_accs = [1.0 if verify(task, r["y_text"], problems[pi]["answer"]) else 0.0
              for pi in range(P) for r in anchor_roll[pi]]
    student_floor = statistics.mean(s_accs) if s_accs else 0.0

    result = {
        "teacher_id": teacher_id, "student_id": student_id, "task": task,
        "n_problems": P, "k": k, "max_tokens": max_tokens,
        "temperature": temperature, "student_anchor_acc": student_floor,
        "by_condition": by_cond, "wall_time_s": time.time() - t0,
    }
    safe = f"{teacher_id.replace('/', '_')}__to__{student_id.replace('/', '_')}"
    out_path = f"/results/crossmodel__{safe}__{task}__{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    RESULTS_VOL.commit()
    print(f"[xmodel] saved {out_path}", flush=True)

    # Frontier print, sorted by KL.
    print(f"\n=== {teacher_id} -> {student_id} / {task} ===", flush=True)
    print(f"student anchor acc (floor) = {student_floor:.3f}", flush=True)
    print("primary axis = fwd_first32 (length-invariant); fwd_tk = full "
          "per-token (length differs by design)", flush=True)
    print(f"{'cond':<10} {'reward':>7} {'acc_se':>7} {'fwd_f32':>8} "
          f"{'fwd_tk':>7} {'rev_tk':>7} {'jsd':>7} {'ylen':>6}", flush=True)
    for c, a in sorted(by_cond.items(), key=lambda kv: kv[1]["fwd_kl_first32"]):
        print(f"{c:<10} {a['acc_mean']:>7.3f} {a['acc_se']:>7.3f} "
              f"{a['fwd_kl_first32']:>8.3f} {a['fwd_kl_tk']:>7.3f} "
              f"{a['rev_kl_tk']:>7.3f} {a['jsd']:>7.3f} "
              f"{a['ylen_mean_T']:>6.0f}", flush=True)
    return result


@app.local_entrypoint()
def main(
    teacher: str = "Qwen/Qwen3.5-4B",
    student: str = "Qwen/Qwen3.5-0.8B",
    task: str = "math",
    n: int = 50,
    k: int = 4,
    max_tokens: int = 1024,
    topk: int = 20,
    dry_run: bool = False,
):
    if dry_run:
        n, k, max_tokens = 5, 2, 96
        print("[DRY RUN] n=5 k=2 max_tokens=96 (Modal, shape/memory check)")
    print(f"Dispatching xmodel teacher={teacher} student={student} "
          f"task={task} n={n} k={k} max_tokens={max_tokens}")
    res = run_one.remote(
        teacher_id=teacher, student_id=student, task=task, n=n, k=k,
        max_tokens=max_tokens, topk=topk,
    )
    print(f"\nDone. wall_time={res.get('wall_time_s', 0):.0f}s "
          f"student_floor={res.get('student_anchor_acc', 0):.3f}")
