# Distillation stack design

Status: draft, user-approved direction, not implemented yet.

## Locked decisions

- Rename local nested repo concept from OPD to **distillation**. OPD is one objective, not the repo identity.
- Keep upstream THUNLP/OPD/verl code as the core training stack, with local fixes incorporated in-place.
- Keep Qwen3.5 compatibility shims as ordinary training-code support.
- Do not optimize for rebasing onto upstream. Maintain small diffs and an inspectable patch audit instead.
- Add a maintenance tool that clones upstream into a temp dir and reports local deviations tied to known issues.
- Raw logs stay local and ignored. Tracked artifacts are summaries, compact evidence, and selected figures.
- LlamaFactory is not core. Delete vendored copy. If SFT warm-start is needed, use LlamaFactory externally.
- SFT-style distillation is still in scope as a **recipe/objective**, not as vendored LlamaFactory code.
- Rename `lagrangian_prompts` to **teacher-context selection**. It is a first-class research module separate from trainer internals.
- Compute providers are adapters. Schmidt/GCP launchers route jobs, but do not own training semantics.
- Split on-policy and off-policy recipes up front. They share model/data/eval/teacher-context/compute schemas, but not the execution loop.
- Start off-policy/SFT-style distillation on verl native SFT code. Treat LlamaFactory as an optional external exporter only if verl SFT blocks progress.

## Proposed module shape

```text
distillation/
  README.md
  PATCHES.md
  tools/audit_upstream_diff.py
  verl/
  recipes/
    shared/
      qwen35_2b.yaml
      math_eval.yaml
      schmidt.yaml
      gcp.yaml
    on_policy/
      grpo_qwen35.yaml
      opd_reverse_kl.yaml
      opd_jsd.yaml
      opd_topk_rkl.yaml
      opd_entropy_aware.yaml
      opd_stable.yaml
      vopd.yaml
      opcd.yaml
    off_policy/
      sft_distill.yaml
      skew_kl.yaml
      distillm2_contrastive.yaml
  launch/
    schmidt.sh
    gcp.sh
  teacher_contexts/
  archive/summaries/
```

## Objective families to support as recipes

Known locally: `modal_phase2.py` already contains `reverse_kl`, `forward_kl`, `jsd`, `topk_rkl`, `entropy_aware`, and KL-to-base.
External papers to encode as recipe families, not bespoke scripts:

- Canonical OPD: student rollouts + teacher forward + reverse KL.
- StableOPD: KL-to-base/reference constraint + rollout mixture.
- Revisiting OPD / Many Faces: teacher top-K local support matching, stop-grad selection, top-p rollouts, special-token masking.
- Entropy-aware OPD: mix reverse KL and forward KL on high teacher entropy tokens.
- vOPD: control-variate reverse-KL baseline, optionally top-k approximation.
- OPCD: context-conditioned teacher on student trajectories, the closest match to teacher-context selection.
- DistiLLM/skew-KL and DistiLLM-2 contrastive: off-policy/SFT-style distillation objectives worth keeping recipe slots for.

## Load-bearing assumption

Teacher-context selection, model/data/eval schemas, and compute adapters are common across on-policy and off-policy distillation.
Only trajectory source and trainer loop split.
If false, teacher contexts or evals are objective-family-specific and the schema should split further.

## Recipe seam

Known from local tree:
- verl already has native SFT entry points: `verl/verl/trainer/fsdp_sft_trainer.py`, `verl/verl/trainer/sft_trainer.py`, `verl/verl/trainer/config/sft_trainer.yaml`, plus `verl/examples/sft/*`.
- vendored LlamaFactory is large: 526 tracked files, 513 text files, about 107k loc-ish.
- native verl SFT core is about 1.6k loc across trainer, dataset, and config.
- `lagrangian_prompts/modal_phase2.py` already carries `reverse_kl`, `forward_kl`, `jsd`, `topk_rkl`, `entropy_aware`, and KL-to-base.

Inference:
- Keeping LlamaFactory vendored would create a second training stack, second dependency solver, second cluster/debug surface.
- The stable path is to use verl for both on-policy and first off-policy recipes, then add a LlamaFactory YAML/dataset exporter only if verl SFT misses a needed feature.
- Objective registry should be pure loss/math plus metadata. It should not own rollout, sampling, or launch semantics.

Minimal schema sketch:

```yaml
name: opd_reverse_kl_qwen35_2b
family: on_policy
model: shared/qwen35_2b.yaml
data: shared/math_train.yaml
eval: shared/math_eval.yaml
compute: shared/schmidt.yaml
teacher_context:
  source: teacher_contexts/pareto.yaml
  selector: kl_budget
  kl_bits_max: 0.5
rollout:
  source: student
  n: 4
objective:
  name: reverse_kl
  trust_region:
    name: kl_to_base
    beta: 0.5
```

```yaml
name: sft_distill_qwen35_2b
family: off_policy
model: shared/qwen35_2b.yaml
data: generated/teacher_samples.parquet
eval: shared/math_eval.yaml
compute: shared/schmidt.yaml
teacher_context:
  source: teacher_contexts/pareto.yaml
  selector: kl_budget
  kl_bits_max: 0.5
sampling:
  source: teacher
  n: 1
objective:
  name: sft_ce
```
