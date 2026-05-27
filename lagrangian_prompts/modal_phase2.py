"""
Phase 2 driver: on-policy distillation (OPD) from selected teacher contexts.

Tests SPEC H3: does a teacher context's 003 Pareto position predict whether
on-policy distillation from it transfers to a context-free student?

Same model is student and teacher (no vLLM weight-sync). Student samples from
the bare anchor prompt (the exact 002/003 pi_theta). Teacher = same weights
seeing a privileged context. Reverse-KL on the response tokens only, on-policy.

Four runs, each student starts from Qwen3.5-0.8B:
  anchor      teacher == student (no context)         tier --  KL~=0 control
  concise     distrib:concise (no leak)               tier 0
  aj_concise  t3_aj_concise (003 winner, answer-leak)  tier 3
  sdr         full gold solution (max leak)            tier 5

Hypothesis: no-leak teacher (concise) -> transferable accuracy lift; leak
teachers (aj_concise, sdr) -> student learns format/calibration but can't
recover an answer it isn't given, so the gain is small / non-OOD. anchor ~ no-op.

Loop/loss/eval ported from explore/147_on_policy_distillation.py (proven). The
only swaps: chat-template rendering + a 4-way teacher dispatch built from
teacher_contexts.py, and a context-free anchor student prompt (NOT 147's
wording) so this stays consistent with the 002/003 Pareto measurement.

Usage:
    # local shape/template smoke test (CPU/MPS ok, tiny):
    uv run lagrangian_prompts/modal_phase2.py --dry-run

    # one run on Modal H100:
    modal run --detach lagrangian_prompts/modal_phase2.py --teacher concise

    # all four (launch detached in parallel, like Phase 1):
    for t in anchor concise aj_concise sdr; do
      modal run --detach lagrangian_prompts/modal_phase2.py --teacher $t &
    done
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import modal

APP_NAME = "lagrangian-prompts-phase2"
PROJECT_ROOT = Path(__file__).parent

TEACHERS = ("anchor", "concise", "aj_concise", "sdr")

# Hyperparams (plan.md §Implementation step 5; 147 defaults for 0.8B).
MODEL = "Qwen/Qwen3.5-0.8B"
TASK = "math"
LR = 1e-6
# N_STEPS/MAX_NEW lowered from the plan's 250/1024 per plan.md step-6 fallback:
# torch.compile is infeasible on Qwen3.5 (linear-attention static-cache crash,
# see friction.md), so uncompiled HF generate is the wall-time bottleneck
# (~31s/step @1024). 512/200 holds 4 runs to ~$20-30. All 4 teachers eval
# identically so cross-teacher deltas (the result) are preserved; only
# absolute acc shifts vs a 1024 cap (caveated in the writeup).
N_STEPS = 200
BATCH_SIZE = 4
EVAL_EVERY = 50  # evals at 0,50,100,150,200 -> 5 learning-curve points
MAX_NEW_TOKENS = 512
MAX_PROMPT_LEN = 512
MAX_TEACHER_PROMPT_LEN = 2048
TEMPERATURE = 0.7
WARMUP_STEPS = 10
SEED = 42
LOSS_METHOD = "reverse_kl"  # TML OPD default

# Eval-set sizes. ID = held-out MATH-500, OOD-down = GSM8K, OOD-up = MATH L5.
# Shrunk from the plan's upper range (200 total, SE ~0.05) — fine for the
# cross-teacher deltas this experiment measures, ~3x cheaper than full sizes
# on HF generate. Paired with torch.compile (below) to hit the ~$10-20 budget.
N_ID_PER_LEVEL = 10   # MATH-500 stratified -> 50
N_GSM8K = 100
N_MATH_L5 = 50

MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.45",
        "datasets",
        "numpy",
        "huggingface_hub",
        "hf_transfer",
        "sympy",
        "antlr4-python3-runtime==4.11",
        "accelerate",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("verifier", "prompts", "teacher_contexts", "data")
)

HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("phase2-results", create_if_missing=True)

app = modal.App(APP_NAME, image=IMAGE)


# ── Datasets ───────────────────────────────────────────────────────────────


def _parse_level(lv) -> int:
    if isinstance(lv, int):
        return lv
    if isinstance(lv, str):
        m = re.search(r"\d+", lv)
        return int(m.group()) if m else 3
    return 3


def _extract_boxed_answer(solution: str) -> str:
    """Last \\boxed{...} in a MATH solution; fall back to last line."""
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    if m:
        return m[-1].strip()
    return solution.strip().split("\n")[-1].strip()


def load_math_train(n: int, seed: int = SEED) -> list[dict]:
    """MATH train split (hendrycks_math), level-stratified to n.

    Disjoint from HuggingFaceH4/MATH-500 (test) and from the level-5 test
    eval set below, so train/eval do not overlap.
    """
    from datasets import load_dataset

    sources = ["EleutherAI/hendrycks_math", "competition_math"]
    examples: list[dict] = []
    for source in sources:
        try:
            for subj in MATH_SUBJECTS:
                ds = load_dataset(source, subj, split="train")
                for row in ds:
                    sol = row["solution"]
                    examples.append({
                        "problem": row["problem"],
                        "solution": sol,
                        "answer": _extract_boxed_answer(sol),
                        "level": _parse_level(row.get("level", "Level 3")),
                        "id": f"mathtrain_{len(examples):05d}",
                    })
            if examples:
                break
        except Exception as e:
            print(f"  math-train {source}: {e}", flush=True)
            examples = []
    if not examples:
        raise ValueError("Could not load MATH train split.")

    by_level: dict[int, list] = defaultdict(list)
    for ex in examples:
        by_level[ex["level"]].append(ex)
    rng = random.Random(seed)
    per_level = max(1, n // 5)
    out: list[dict] = []
    for lvl in sorted(by_level):
        bucket = by_level[lvl]
        rng.shuffle(bucket)
        out.extend(bucket[:per_level])
    rng.shuffle(out)
    levels = defaultdict(int)
    for ex in out:
        levels[ex["level"]] += 1
    print(f"MATH train: {len(out)} problems, levels={dict(sorted(levels.items()))}",
          flush=True)
    return out


def load_math_l5_eval(n: int, seed: int = SEED) -> list[dict]:
    """OOD-up: held-out MATH level-5 from the test split (disjoint from train)."""
    from datasets import load_dataset

    rows: list[dict] = []
    for subj in MATH_SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split="test")
        for row in ds:
            if _parse_level(row.get("level", "Level 3")) != 5:
                continue
            rows.append({
                "problem": row["problem"],
                "answer": _extract_boxed_answer(row["solution"]),
                "level": 5,
            })
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def load_gsm8k_eval(n: int, seed: int = SEED) -> list[dict]:
    """OOD-down: GSM8K test (distribution shift toward easier grade-school)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for row in ds:
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"problem": row["question"], "answer": gold})
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


# ── Prompt rendering ───────────────────────────────────────────────────────


def _render(tok, messages: list[dict], enable_thinking: bool = False) -> str:
    """Chat-template a message list. Non-thinking, matching 002/003."""
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def make_student_prompt(tok, task: str):
    """Student = bare anchor (problem only). The exact 002/003 pi_theta."""
    from teacher_contexts import anchor_messages

    def fn(prob: dict) -> str:
        return _render(tok, anchor_messages(task, prob))

    return fn


def make_teacher_prompt(tok, task: str, teacher: str):
    """4-way dispatch over teacher contexts (plan.md step 3)."""
    from teacher_contexts import (
        ANCHOR, _distrib_builder, _sdr_builder, _t3_aj_concise,
        anchor_messages,
    )

    if teacher == "anchor":
        # Teacher prompt == student prompt -> KL ~= 0 control.
        def fn(prob):
            return _render(tok, anchor_messages(task, prob))
    else:
        if teacher == "concise":
            builder = _distrib_builder(task, "concise")
        elif teacher == "aj_concise":
            builder = _t3_aj_concise(task)
        elif teacher == "sdr":
            builder = _sdr_builder(task)
        else:
            raise ValueError(f"Unknown teacher: {teacher}")

        def fn(prob):
            msgs = builder(
                prob, prob["answer"], ANCHOR[task],
                demos_pool=None, draft_text=None, critique_text=None,
            )
            return _render(tok, msgs)

    return fn


# ── Loss (147's reverse_kl + 004-ablation: forward_kl/jsd + KL-to-base) ─────


def distill_loss(s_logits, t_logits, mask, method="reverse_kl",
                 ref_logits=None, kl_base_beta=0.0, topk_k=20, ent_q=0.5):
    """Student↔teacher divergence on response tokens, + optional trust region.

    method:
      reverse_kl    KL(s||t), mode-seeking (147 / Phase-2 default)
      forward_kl    KL(t||s), mean-seeking (004 ablation)
      jsd           Jensen-Shannon (β=0.5)
      topk_rkl      teacher top-K local support matching + stop-grad on the
                    top-K selection (Revisiting-OPD 2603.25562 / The Many Faces
                    2605.11182). reverse-KL over the teacher-supported top-`k`
                    set, locally renormalized. The selection comes from
                    t_logits, which opd_step computes under no_grad → the index
                    set is naturally detached (the stop-gradient).
      entropy_aware reverse-KL on low-teacher-entropy tokens, forward-KL on
                    high-entropy ones (Entropy-Aware OPD 2603.07079). Split at
                    the per-batch teacher-entropy quantile `ent_q` over valid
                    response tokens.
    ref_logits/kl_base_beta: add β·KL(s||base) — frozen-base trust region
        (005; StableOPD 2604.08527 reference constraint). Stacks with any method.
    """
    import torch
    import torch.nn.functional as F

    log_s = F.log_softmax(s_logits, dim=-1)
    log_t = F.log_softmax(t_logits, dim=-1)
    if method == "reverse_kl":
        div = (log_s.exp() * (log_s - log_t)).sum(dim=-1)
    elif method == "forward_kl":
        div = (log_t.exp() * (log_t - log_s)).sum(dim=-1)
    elif method == "jsd":
        p_s, p_t = log_s.exp(), log_t.exp()
        m = (0.5 * p_s + 0.5 * p_t).clamp(min=1e-8)
        log_m = m.log()
        div = 0.5 * (p_s * (log_s - log_m)).sum(-1) \
            + 0.5 * (p_t * (log_t - log_m)).sum(-1)
    elif method == "topk_rkl":
        # Teacher-supported top-K; idx from t_logits (no_grad in opd_step) so
        # the selection is detached. Renormalize both over the K set, reverse-KL.
        idx = t_logits.topk(topk_k, dim=-1).indices          # (B,T,k)
        ls = F.log_softmax(torch.gather(s_logits, -1, idx), dim=-1)
        lt = F.log_softmax(torch.gather(t_logits, -1, idx), dim=-1)
        div = (ls.exp() * (ls - lt)).sum(dim=-1)
    elif method == "entropy_aware":
        rkl = (log_s.exp() * (log_s - log_t)).sum(dim=-1)
        fkl = (log_t.exp() * (log_t - log_s)).sum(dim=-1)
        with torch.no_grad():
            t_ent = -(log_t.exp() * log_t).sum(dim=-1)        # (B,T)
            # torch.quantile rejects bf16 (the Modal model dtype) → compute the
            # split threshold in fp32.
            valid = t_ent[mask > 0].float()
            thr = (torch.quantile(valid, ent_q)
                   if valid.numel() > 0 else valid.new_zeros(()))
            hi = (t_ent.float() >= thr).float()               # high-entropy → fwd
        div = hi * fkl + (1.0 - hi) * rkl
    else:
        raise ValueError(f"Unknown loss method: {method}")
    loss = (div * mask).sum() / mask.sum().clamp(min=1)

    klb_val = 0.0
    if ref_logits is not None and kl_base_beta > 0:
        log_r = F.log_softmax(ref_logits, dim=-1)
        klb = (log_s.exp() * (log_s - log_r)).sum(dim=-1)
        klb_term = (klb * mask).sum() / mask.sum().clamp(min=1)
        loss = loss + kl_base_beta * klb_term
        klb_val = klb_term.item()

    with torch.no_grad():
        ent = -(F.softmax(s_logits, dim=-1) * log_s).sum(-1)
        avg_ent = (ent * mask).sum() / mask.sum().clamp(min=1)
    return loss, {"loss": loss.item(), "entropy": avg_ent.item(),
                  "kl_base": klb_val}


# ── Tokenize / mask helpers (147) ──────────────────────────────────────────


def tokenize_batch(tok, texts, max_length, device):
    import torch  # noqa: F401

    tok.padding_side = "left"
    enc = tok(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length, add_special_tokens=False,
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def comp_mask(ids, eos_id, pad_id):
    import torch

    B, T = ids.shape
    m = torch.ones(B, T, device=ids.device, dtype=torch.float32)
    for i in range(B):
        eos = (ids[i] == eos_id).nonzero(as_tuple=True)[0]
        if len(eos) > 0:
            m[i, eos[0].item() + 1:] = 0
        if pad_id != eos_id:
            m[i] *= (ids[i] != pad_id).float()
    return m


# ── Eval ───────────────────────────────────────────────────────────────────


def evaluate(model, tok, eval_data, task, device, max_new, student_prompt):
    """Greedy pass@1 on a labeled eval set, student (anchor) prompt only."""
    import torch

    from verifier import verify

    raw = getattr(model, "_orig_mod", model)
    was_training = model.training
    model.eval()
    raw.config.use_cache = True

    correct = total = 0
    bs = 4
    for i in range(0, len(eval_data), bs):
        batch = eval_data[i:i + bs]
        prompts = [student_prompt(ex) for ex in batch]
        ids, mask = tokenize_batch(tok, prompts, MAX_PROMPT_LEN, device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids, attention_mask=mask,
                max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        plen = ids.shape[1]
        for j, ex in enumerate(batch):
            text = tok.decode(gen[j, plen:], skip_special_tokens=True)
            correct += int(verify(task, text, ex["answer"]))
            total += 1

    raw.config.use_cache = False
    if was_training:
        model.train()
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


# ── One OPD step (ported from 147 _step_distill) ───────────────────────────


def opd_step(model, raw, optimizer, tok, batch, device, task, max_new,
             student_prompt, teacher_prompt, method="reverse_kl",
             ref_model=None, kl_base_beta=0.0, topk_k=20, ent_q=0.5):
    import torch

    from verifier import verify

    B = len(batch)
    s_prompts = [student_prompt(ex) for ex in batch]
    s_ids, s_mask = tokenize_batch(tok, s_prompts, MAX_PROMPT_LEN, device)
    s_plen = s_ids.shape[1]

    model.eval()
    raw.config.use_cache = True
    with torch.no_grad():
        gen = model.generate(
            input_ids=s_ids, attention_mask=s_mask,
            max_new_tokens=max_new, do_sample=True,
            temperature=TEMPERATURE, top_p=0.95,
            pad_token_id=tok.pad_token_id,
        )
    comp_ids = gen[:, s_plen:]
    c_len = comp_ids.shape[1]
    if c_len == 0:
        return None, {"loss": 0.0, "entropy": 0.0, "acc": 0.0}

    texts = [tok.decode(comp_ids[i], skip_special_tokens=True) for i in range(B)]
    acc = sum(verify(task, texts[i], batch[i]["answer"]) for i in range(B)) / B

    c_m = comp_mask(comp_ids, tok.eos_token_id, tok.pad_token_id)
    if c_m.sum() == 0:
        return None, {"loss": 0.0, "entropy": 0.0, "acc": acc}

    # Teacher prompt (privileged context). For teacher=="anchor" this equals the
    # student prompt -> KL ~= 0 control.
    t_prompts = [teacher_prompt(ex) for ex in batch]
    t_ids, t_mask = tokenize_batch(tok, t_prompts, MAX_TEACHER_PROMPT_LEN, device)
    t_plen = t_ids.shape[1]

    t_full = torch.cat([t_ids, comp_ids], dim=1)
    t_full_mask = torch.cat([t_mask, c_m.long()], dim=1)
    s_full = gen
    s_full_mask = torch.cat([s_mask, c_m.long()], dim=1)

    with torch.no_grad():
        raw.config.use_cache = False
        t_out = model(input_ids=t_full, attention_mask=t_full_mask)
        t_logits = t_out.logits[:, t_plen - 1: t_plen - 1 + c_len].clone()
    del t_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ref_logits = None
    if ref_model is not None and kl_base_beta > 0:
        with torch.no_grad():
            ref_out = ref_model(input_ids=s_full, attention_mask=s_full_mask)
            ref_logits = ref_out.logits[
                :, s_plen - 1: s_plen - 1 + c_len].clone()
        del ref_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.train()
    raw.config.use_cache = False
    s_out = model(input_ids=s_full, attention_mask=s_full_mask)
    s_logits = s_out.logits[:, s_plen - 1: s_plen - 1 + c_len]

    loss, stats = distill_loss(
        s_logits, t_logits, c_m, method=method,
        ref_logits=ref_logits, kl_base_beta=kl_base_beta,
        topk_k=topk_k, ent_q=ent_q)
    stats["acc"] = acc

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss, stats


# ── Train driver (shared local + Modal) ────────────────────────────────────


def _train(teacher: str, model_name: str, n_steps: int, n_train: int,
           dry_run: bool, results_dir: Path, no_compile: bool = True,
           method: str = "reverse_kl", kl_base_beta: float = 0.0,
           tag: str = "", topk_k: int = 20, ent_q: float = 0.5) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert teacher in TEACHERS, f"Unknown teacher {teacher}; pick {TEACHERS}"
    sys.stdout.reconfigure(line_buffering=True)
    task = TASK

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    use_cuda = device.type == "cuda"

    if dry_run:
        n_steps, batch_size, max_new = 2, 2, 64
        eval_every = 1
        id_pl, gsm_n, l5_n = 1, 4, 4
        n_train = 64
    else:
        batch_size, max_new = BATCH_SIZE, MAX_NEW_TOKENS
        eval_every = EVAL_EVERY
        id_pl, gsm_n, l5_n = N_ID_PER_LEVEL, N_GSM8K, N_MATH_L5

    if use_cuda:
        torch.set_float32_matmul_precision("high")

    print(f"Loading {model_name} (teacher={teacher})...", flush=True)
    kw = {"dtype": dtype}
    if use_cuda:
        kw["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if use_cuda and not dry_run:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # torch.compile + static KV cache for faster generation. 147's path, but
    # it is INCOMPATIBLE with Qwen3.5: the static-cache mask builder in
    # transformers raises KeyError('linear_attention') for Qwen3.5's GDN
    # linear-attention layers (same GDN-path fragility as the vLLM deadlock in
    # friction.md). 147 was proven on DeepSeek-Qwen2 (full attention). So
    # compile is OFF by default here; uncompiled generate works (dry-run +
    # canary validated). Opt in only on a full-attention model.
    use_compile = use_cuda and not dry_run and not no_compile
    if use_compile:
        model.generation_config.cache_implementation = "static"
        model = torch.compile(model, dynamic=True)
        print("Compiled with static KV cache + gradient checkpointing",
              flush=True)
    raw = getattr(model, "_orig_mod", model)

    # 004 ablation: frozen base policy for the KL-to-base trust region.
    ref_model = None
    if kl_base_beta > 0:
        print(f"Loading frozen ref (KL-to-base β={kl_base_beta})...",
              flush=True)
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, **kw).to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    student_prompt = make_student_prompt(tok, task)
    teacher_prompt = make_teacher_prompt(tok, task, teacher)

    # === Data ===
    train_data = load_math_train(n_train)
    from data import load_math_500

    eval_sets = {
        "id": load_math_500(n_per_level=id_pl),
        "gsm8k": load_gsm8k_eval(gsm_n),
        "math_l5": load_math_l5_eval(l5_n),
    }
    for name, s in eval_sets.items():
        print(f"eval[{name}]: {len(s)} problems", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01, fused=use_cuda
    )
    rng = np.random.RandomState(SEED)

    def get_lr(step):
        if step < WARMUP_STEPS:
            return LR * step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, n_steps - WARMUP_STEPS)
        return LR * 0.5 * (1 + np.cos(np.pi * progress))

    results_dir.mkdir(parents=True, exist_ok=True)
    config = dict(
        teacher=teacher, model=model_name, task=task, loss=method, lr=LR,
        kl_base_beta=kl_base_beta, tag=tag, topk_k=topk_k, ent_q=ent_q,
        n_steps=n_steps, batch_size=batch_size, n_train=len(train_data),
        max_new_tokens=max_new, temperature=TEMPERATURE,
        eval_sizes={k: len(v) for k, v in eval_sets.items()},
    )
    suffix = f"__{tag}" if tag else ""
    out_path = results_dir / f"phase2__{teacher}{suffix}.json"

    def run_evals(step):
        row = {"step": step}
        for name, s in eval_sets.items():
            ev = evaluate(model, tok, s, task, device, max_new, student_prompt)
            row[f"acc_{name}"] = ev["accuracy"]
            print(f"  EVAL[{name}] step={step} acc={ev['accuracy']:.3f} "
                  f"({ev['correct']}/{ev['total']})", flush=True)
        return row

    print("\nBaseline eval (step 0, before training)...", flush=True)
    history = [{**run_evals(0), "train_loss": None, "entropy": None}]

    def flush():
        with open(out_path, "w") as f:
            json.dump({"config": config, "history": history}, f, indent=2)

    flush()

    print(f"\n{teacher} | {n_steps} steps | bs={batch_size} | {method}"
          f"{f' | klβ={kl_base_beta}' if kl_base_beta else ''}"
          f"{f' | tag={tag}' if tag else ''}", flush=True)
    header = f"{'step':>5} | {'loss':>9} | {'ent':>6} | {'acc':>5} | {'t':>5}"
    print(header)
    print("-" * len(header))

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    for step in range(n_steps):
        t0 = time.time()
        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(step)
        idx = rng.choice(len(train_data), size=batch_size, replace=False)
        batch = [train_data[i] for i in idx]
        loss, stats = opd_step(
            model, raw, optimizer, tok, batch, device, task, max_new,
            student_prompt, teacher_prompt, method=method,
            ref_model=ref_model, kl_base_beta=kl_base_beta,
            topk_k=topk_k, ent_q=ent_q,
        )
        elapsed = time.time() - t0
        print(f"{step:5d} | {stats['loss']:9.5f} | {stats['entropy']:6.3f} | "
              f"{stats['acc']:5.1%} | {elapsed:5.1f}s", flush=True)

        if (step + 1) % eval_every == 0 or step == n_steps - 1:
            row = run_evals(step + 1)
            row["train_loss"] = stats["loss"]
            row["entropy"] = stats["entropy"]
            history.append(row)
            flush()

    peak = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
    print(f"\nPeak VRAM: {peak:.1f} GB | saved to {out_path}", flush=True)
    flush()
    return {"config": config, "history": history, "out_path": str(out_path)}


@app.function(
    gpu="H100!",
    timeout=60 * 180,
    volumes={
        "/root/.cache/huggingface": HF_CACHE,
        "/results": RESULTS_VOL,
    },
)
def run_one(teacher: str, model_name: str = MODEL, n_steps: int = N_STEPS,
            n_train: int = 3000, dry_run: bool = False,
            no_compile: bool = True, method: str = "reverse_kl",
            kl_base_beta: float = 0.0, tag: str = "",
            topk_k: int = 20, ent_q: float = 0.5) -> dict:
    res = _train(teacher, model_name, n_steps, n_train, dry_run,
                 Path("/results"), no_compile=no_compile, method=method,
                 kl_base_beta=kl_base_beta, tag=tag,
                 topk_k=topk_k, ent_q=ent_q)
    RESULTS_VOL.commit()
    return res


@app.local_entrypoint()
def main(teacher: str = "concise", model: str = MODEL, n_steps: int = N_STEPS,
         n_train: int = 3000, dry_run: bool = False, no_compile: bool = True,
         method: str = "reverse_kl", kl_base_beta: float = 0.0,
         tag: str = "", topk_k: int = 20, ent_q: float = 0.5):
    print(f"Dispatching teacher={teacher} model={model} n_steps={n_steps} "
          f"n_train={n_train} dry_run={dry_run} no_compile={no_compile} "
          f"method={method} kl_base_beta={kl_base_beta} tag={tag} "
          f"topk_k={topk_k} ent_q={ent_q}")
    res = run_one.remote(
        teacher=teacher, model_name=model, n_steps=n_steps,
        n_train=n_train, dry_run=dry_run, no_compile=no_compile,
        method=method, kl_base_beta=kl_base_beta, tag=tag,
        topk_k=topk_k, ent_q=ent_q,
    )
    final = res["history"][-1]
    accs = {k: v for k, v in final.items() if k.startswith("acc_")}
    print(f"\nDone. teacher={teacher} final={accs}")


if __name__ == "__main__":
    # Local smoke test (no Modal): tiny, CPU/MPS-ok shape + template check.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--teacher", default="aj_concise", choices=TEACHERS)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    if not args.dry_run:
        print("Local invocation is smoke-test only; pass --dry-run "
              "(use `modal run` for real runs).")
        sys.exit(1)
    _train(args.teacher, args.model, n_steps=2, n_train=64, dry_run=True,
           results_dir=PROJECT_ROOT / "results" / "phase2_dryrun")
