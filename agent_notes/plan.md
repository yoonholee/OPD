# Distillation cleanup and GCP SFT exploration

Status: in progress, first native SFT smoke passed

Budget:
- Hard exploration budget: $500 GCP credits for this phase.
- Start with cheap compile/dry-run/local rendering, then one L4 smoke, then small sweeps only after loss decreases.
- Stop rule: stop launching new jobs if spend estimate reaches $400, any single job looks likely to exceed $100, or repeated setup failures consume 3 attempts on the same root cause.

Current decisions:
- Project identity is `distillation`; OPD is one objective family.
- Delete vendored `LlamaFactory`. Use verl native SFT first.
- Keep SFT-style distillation in scope as an off-policy recipe family.
- Split recipes up front: `recipes/on_policy`, `recipes/off_policy`, `recipes/shared`.
- Shared schemas: model, data, eval, teacher-context, compute.
- Split executor loops: on-policy student rollout loop vs off-policy cached teacher/sample loop.
- Keep Qwen3.5 shims in main training code.
- Keep raw logs local and ignored. Track summaries only.

Execution plan:
1. Remove `LlamaFactory` from the nested repo.
2. Make ignored temp spike dir for architecture prototypes.
3. Prototype recipe rendering with throwaway scripts before moving anything into clean code.
4. Prototype real verl SFT launch on GCP using the existing Qwen3.5 environment.
5. Run smallest real SFT smoke: setup, data load, 1 to 3 train steps, verify finite loss and nonincreasing trend if enough steps.
6. If smoke passes, run tiny hparam sweep over lr, batch, sequence length, and remove-padding/offload knobs.
7. Summarize results in `agent_notes/experiments/` or `archive/summaries/`; leave raw logs ignored.
8. Promote only the winning temp scripts into clean `recipes/`, `launch/`, and `tools/`.

Smoke matrix, ordered:
- local: recipe parser and Hydra override renderer.
- local: inspect verl SFT config surface and render Qwen3.5 SFT overrides.
- GCP L4: 1 step Qwen3.5-2B SFT, tiny parquet, `use_remove_padding=False`.
- GCP L4: 10 to 25 steps, verify loss decreases.
- GCP L4/G4: small sweep if L4 throughput or memory blocks.

Architectural spike questions:
- Is the recipe schema deep enough to hide Hydra boilerplate without hiding objective knobs?
- Does off-policy SFT need anything beyond verl native SFT for first experiments?
- Can the same teacher-context selector feed both on-policy and off-policy jobs?
- Which launch bits are compute adapters vs training semantics?

Verification gates:
- Parser tests pass locally.
- Rendered Hydra overrides point to existing verl config keys.
- GCP job writes a loss curve and exits cleanly.
- Loss is finite on every logged step.
- For multi-step smoke, final smoothed loss is below first smoothed loss, or the run is marked inconclusive with evidence.

Do not yet:
- Build the final framework.
- Vendor LlamaFactory.
- Move temp scripts into clean dirs before a real smoke passes.
- Treat one noisy short loss curve as a hyperparameter conclusion.


## Current result, 2026-05-27

- Deleted vendored LlamaFactory and LF-only helper scripts.
- Added ignored temp spike dir `_tmp/distillation_spikes/`.
- Native verl SFT smoke passed on GCP 1xA100 spot with Qwen3.5-2B.
- Best smoke: `distill-sft-a100-16step-c1b-0527-1904`, 16 steps, train/loss 3.0021 to 0.3784, val/loss 0.2381, rc 0.
- 4xL4 Qwen3.5-2B smoke is still blocked by stockout, not by known code errors.

Next:
- implement full-vocab/lit OPD objectives in verl if we want `reverse_kl`, `forward_kl`, `jsd`, `topk_rkl`, and `entropy_aware` comparisons inside the main stack
- turn the temp SFT runner into a clean off-policy recipe only after one provider-stable run
- add a real teacher-rollout SFT dataset instead of synthetic answer-string targets
- promote only the clean parts of the variant runner into recipe/launch code

## OPD variant matrix, 2026-05-28

- Ran current verl top-k OPD surface on GCP A100: GRPO, 15 `top_k_strategy` × `reward_weight_mode` OPD cases, ref-KL, entropy.
- Same-model Qwen3-0.6B sanity: 18/18 rc0.
- Qwen3-0.6B student / Qwen3-1.7B teacher: 18/18 rc0.
- Result note: `agent_notes/verl_opd_variant_results.md`.
- Full-vocab lagrangian/lit objectives are not yet verl trainer objectives.
