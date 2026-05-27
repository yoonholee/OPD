# Friction log — lagrangian_prompts

## 2026-05-13: vLLM OOM at engine init when cranking SPEC settings

**Symptom:** All Phase 1 cells with `gpu_memory_utilization=0.95` + `max_num_seqs=512` + `max_num_batched_tokens=131072` (the values literally copied from SPEC.md §Modal infrastructure) OOMed during vLLM engine warmup, specifically inductor-compiled cuda graph capture. Smaller models (0.8B, 2B) survived because more H100 headroom remained after KV-cache pre-allocation; 4B blew up:

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 4.50 GiB. GPU 0 has a total capacity of 79.18 GiB
of which 4.39 GiB is free. Process 1 has 74.77 GiB memory in use.
```

**What I tried first:** Bumped to SPEC's aspirational settings on the assumption Phase 1's higher batch parallelism justified them.

**What worked:** Reverted to Phase 0's settings — `gpu_memory_utilization=0.85`, no `max_num_seqs` or `max_num_batched_tokens` overrides. vLLM's defaults handle the actual concurrency Phase 1 needs (≤200 simultaneous requests in our largest batch). The aggressive max_num_batched_tokens=131072 in particular is what blows compile-time buffers.

**Suggested fix to SPEC:** §Modal infrastructure currently reads:
> Settings: gpu_memory_utilization=0.95, max_num_seqs=512, max_num_batched_tokens=131072, ...

This is wrong for the 4B model on a single H100 with the v0.20 vLLM compile path. Either:
- Drop the explicit numbers and just say "vLLM defaults + prefix caching".
- Caveat: "0.95 only fits if the model is smaller and the cuda graph capture buffer is bounded; verify per model size."

**Cost of this lesson:** ~2 minutes of compute on each of two failed 4B cells before they died at warmup. <$1.

## 2026-05-14: vLLM 0.20 V1 + Qwen3.5 + prefix caching = deadlock (vllm-project/vllm#37729)

**Symptom:** Phase D's cross-scoring (`generate(prompt_logprobs=1)` on ~8000 long-prompt requests, all sharing 70%+ prefix overlap) hangs at random chunk for Qwen3.5-{0.8B, 2B}. 4B-eq survived once, 4B-math survived once. Smaller models hang reliably. "Processed prompts" bar freezes mid-batch, no error, deadlock until Modal timeout.

**What I tried:**
1. Chunked the loop (chunk_size=400 → 200). Helped on 4B/eq but smaller models still hung.
2. `enforce_eager=True` + `max_num_seqs=64`. 4B/math succeeded; smaller models still hung at different chunks (5/38, 8/40, 17/38). Random position rules out a specific bad request.

**What worked:** `enable_prefix_caching=False`. The upstream issue 37729 specifically identifies "all sharing ~70% cached prefix" as the trigger; the LRU eviction in the V1 scheduler's prefix-cache path interacts badly with sustained concurrent load. Cost: ~30% slower for distrib (which has shared anchor prefix). Acceptable.

**Suspected root cause:** V1 engine prefix-cache + chunked prefill + Qwen3.5 GDN attention path has a scheduler state-machine bug. Pinning to vLLM 0.16 fixes it but is not available in the Modal image.

**Suggested SPEC update:** §Modal infrastructure says `enable_prefix_caching=True`. Add a caveat: "for response-distribution KL on Qwen3.5, set False — vllm-project/vllm#37729 deadlocks the scheduler when many requests share the same teacher-context prefix."

## 2026-05-13: `modal run --detach` still streams logs locally

Expected `--detach` to fire-and-forget. It actually keeps the local stdout connected to log streaming until either the job finishes OR the user ctrl-c's. Running 6 detached jobs in parallel from `Bash run_in_background=true` works (each tail process is independent), but the local terminal task only "completes" when the Modal job ends, ~20-30 min later — not when the job is *dispatched*.

**Workaround for status checks:** `modal app list` and `modal app logs <app_id>` give the actual job state independently of the local CLI's stream.

## 2026-05-16: torch.compile + static KV cache breaks on Qwen3.5 (linear attention)

**Symptom:** Phase 2 (`modal_phase2.py`) ported 147's compile path
(`generation_config.cache_implementation="static"` + `torch.compile`). The
*dry-run* (compile disabled on dry_run) passed; the first real run failed at
the step-0 baseline eval inside `model.generate`:

```
File ".../transformers/masking_utils.py", line 1479, in create_masks_for_generate
  causal_masks[layer_pattern] = LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING[layer_pattern](**mask_kwargs)
KeyError: 'linear_attention'
```

**Root cause:** Qwen3.5 is a hybrid model with GDN *linear-attention* layers.
The static-cache generate path asks transformers to build per-layer-pattern
causal masks; this transformers version's `LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING`
has no `'linear_attention'` entry, so static cache + Qwen3.5 = hard crash. 147's
compile path was validated on DeepSeek-R1-Distill-Qwen-1.5B (full attention),
which never exercises that mapping. Same GDN-path fragility as the vLLM Qwen3.5
deadlock above — Qwen3.5's linear attention is a recurring breakage source.

**What worked:** Drop compile entirely (default `no_compile=True` in
`modal_phase2.py`). Uncompiled dynamic-cache `model.generate` works on Qwen3.5
(dry-run + the pre-crash part of the canary both confirm). Control cost via the
eval-set shrink + N_STEPS/MAX_NEW instead of compile.

**Suggested rule:** 147's static-cache+compile speedup is full-attention-only.
For Qwen3.x-hybrid (GDN) models, do not set `cache_implementation="static"`;
budget for uncompiled HF generate (it is the wall-time bottleneck) or use vLLM
for the rollout/eval generation instead of HF `.generate`.

**Cost of this lesson:** caught by a 1-run canary (~10 min H100 to the failing
baseline eval, <$1) before launching all 4 — staged rollout paid for itself.

## 2026-05-17: local loss unit-test passed in fp32 but bf16 crashed on Modal

**Symptom:** `entropy_aware` OPD objective unit-tested fine locally (random
fp32 tensors, grad checks passed), but both `entropy_aware` Modal runs crashed
at step 0: `RuntimeError: quantile() input tensor must be either float or
double dtype`. The Modal model loads in **bfloat16**; `torch.quantile` rejects
bf16. The fp32 unit test never exercised the prod dtype.

**Fix:** cast to fp32 inside the op (`t_ent[mask>0].float()`); retest harness
now loops `dtype in {bfloat16, float32}` for every loss branch.

**Cost:** ~2 × 25 min wasted H100 (~$1-2) before the completion notification +
monitor surfaced it. topk_rkl runs (no quantile) were unaffected and continued.

**Rule:** any numerical kernel unit test must run in the *production dtype*
(bf16 for these Modal runs), not just the float32 default. A green fp32 test is
not evidence the bf16 path works — torch ops differ in dtype support
(quantile, some reductions, fused kernels).
