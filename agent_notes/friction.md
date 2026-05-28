# Friction

- `zsh` wipe trap: `rm -rf ./* ./.??*` failed with `no matches found`. Workaround: `bash -lc 'shopt -s dotglob nullglob; rm -rf -- /path/*'`.
- `uv run` in OPD root can solve vendored `verl/pyproject.toml` and fail on heavy/vLLM conflicts. Workaround for data probes: `uv run --no-project --with ...`.
- GCP 4xH100 spot had stockout across tried zones. 1xH100 spot was preempted. For serious OPD reproduction, use on-demand or reservation.
- GCP DLVM needed `build-essential` and `pip` inside uv venv for THUNLP/vLLM installs. `uv venv` alone did not provide `pip`.
- THUNLP install script has unbounded `transformers>=4.51`; in 2026 it pulled Transformers 5.9, which broke THUNLP imports. Pin `transformers==4.57.3`, `tokenizers>=0.22,<0.23` for the old stack.
- EasyR1 self-build from source is fragile. Clean env failed on `flash-attn` metadata until torch was installed, then failed on missing `psutil`; previous non-clean env imported a flash-attn wheel built for another torch. Prefer EasyR1's published Docker/Apptainer image.
- vLLM config trap: `max_num_batched_tokens` must be >= default `max_num_seqs` 1024. Tiny smokes with 512 fail during SchedulerConfig validation unless `max_num_seqs` is also lowered.

- Qwen3.5 vLLM init at `gpu_memory_utilization=0.22` failed with no cache blocks. Fix: 0.45 on 4xL4 for 2B smoke.
- Qwen3.5 vLLM init with default `max_num_seqs=1024` failed: available Mamba cache blocks 463. Fix: set `max_num_seqs=32` for tiny smoke.
- Qwen3.5 + activation offload failed during backward with `assert not isinstance(tensor, tuple)` in `activation_offload.py`. Fix: disable `actor_rollout_ref.model.enable_activation_offload`.
- Local `date -Is` is GNU-only; macOS BSD `date` rejected it while writing delete proof. Use `date -u +%Y-%m-%dT%H:%M:%SZ`.
- `numpy<2` is wrong for the stable Qwen3.5/vLLM 0.21 smoke. The locked env uses NumPy 2.3.5; forcing NumPy 1.x conflicts with the current OpenCV/vLLM stack.
- G4 creation with `pd-balanced` fails. Use `BOOT_DISK_TYPE=hyperdisk-balanced` for `g4-standard-*`.
- Hardcoded `NCCL_SOCKET_IFNAME=ens7` fails off G2/L4. G4 used `ens3`; A2/A100 used `ens8`. Route-detect the NIC.
- Qwen3.5 VLM text-only batch >1 failed in `extract_multi_modal_inputs -> torch.cat` because `multi_modal_inputs` shapes varied. Use `data.return_multi_modal_inputs=False`.
- vLLM FlashInfer sampler JIT stalled under colocated 4-worker Ray on Qwen3.5. Use `VLLM_USE_FLASHINFER_SAMPLER=0`; expect separate GDN/FLA Triton warmup.
- OPD 10-step first failed in `RewardModelWorker._compute_entropy_safe` with non-contiguous logits and `.view`. Fix: `.reshape`, covered by `test_reward_entropy_accepts_non_contiguous_logits`.

- verl `update_policy` has an **asymmetric-branch bug class** in `dp_actor.py` (two instances found, may be more).
  - **A1**: 2D-advantages (vanilla GRPO) branch at L843 dropped `entropy` via `_, log_prob, *_ = self._forward_micro_batch(...)`. Used at L907 when `entropy_coeff > 0` → UnboundLocalError. Fix: `entropy, log_prob, *_ = ...`.
  - **A2**: 3D-advantages (top-k / OPD) branch at L837 dropped `log_prob` via `entropy, _, _, topk_log_probs = ...`. Used at L921 in `kl_penalty(logprob=log_prob, ...)` when `use_kl_loss=True` → UnboundLocalError. Fix: `entropy, log_prob, _, topk_log_probs = ...`.
  - Why both went undetected: verl/514 changed `entropy_coeff` default 0.001 → 0.0 in v0.3.x, so neither bonus path was exercised in CI. KL-loss + OPD is rarely tested as a combo because each papers tend to use one or the other. Generalized regression test covers both branches binding entropy at position 0 AND log_prob at position 1.

- Ray head init race when two SLURM jobs land on the same node (B300 b301): three layers of bug.
  (1) `/tmp/ray/session_*` collision → `_write_cluster_info_to_kv` assert. Fix: `--temp-dir=$TMPDIR/ray-$SLURM_JOB_ID`.
  (2) Default GCS port 6379 collision across jobs on same node. Fix: `--port=<job-derived>`.
  (3) Chosen GCS port can land inside Ray's default worker_ports range (10002-19999). `--port=10031` triggered `ValueError: Ray component worker_ports is trying to use a port number 10031 that is used by other components.` Fix: keep custom ports below 10000 AND override `--min-worker-port`/`--max-worker-port` to a per-job slice (1000 ports starting at 20000+slot*1000). Also override `--ray-client-server-port` (default 10001 collides). Also gate `ray stop --force` behind non-SLURM (kills neighbor's Ray, exit 0:15). All four pieces wired in `BASE_ENV`; regression test covers all of them.
  (4) `SLURM_JOB_ID % 8` is not enough: jobs 53679 and 53687 collided on the same port slice. Use `% 40` slices.
  (5) Parallel jobs launched in the same second collided on `LOGDIR`. Include `_slurm$SLURM_JOB_ID` in default run dirs.

- `ulimit -n 1048576` in sbatch script body fails with "Operation not permitted" on Schmidt; SLURM enforces hard limits. Non-fatal with `|| true`. Need to set `LimitNOFILE` in slurm.conf or via prolog if higher is required.

- Four parallel B300 OPD jobs on one node can survive Ray startup but die near final validation/shutdown with `FAILED 0:15` and/or `DataLoader worker ... killed by signal: Killed`. Observed MaxRSS ~65 GB per batch process for jobs 53688-53691; only one of four completed rc0. Workaround: serialize baseline/comparison jobs or reduce validation/data-loader host-memory pressure before trusting Slurm exit state.

- rsync `--exclude='checkpoint/'` (no leading slash) matches **every** `checkpoint/` directory in the tree, not just the root one. This silently nuked `verl/verl/utils/checkpoint/`, breaking `from verl.utils.checkpoint.checkpoint_manager import ...` on the Schmidt copy. Symptom: `ModuleNotFoundError: No module named 'verl.utils.checkpoint'` only on Schmidt. Same gotcha applies to `--exclude='model/'` matching `LlamaFactory/src/llamafactory/model/`, `verl/verl/trainer/config/model/`. Fix: use `--exclude='/checkpoint/'` (leading slash anchors to source root) or specifically `--exclude='./checkpoint'`. Always grep `find . -type d -name <excluded>` before running an rsync.

## 2026-05-27: native verl SFT missed Qwen3.5 compat path

- Symptom caught before GCP launch: `fsdp_sft_trainer.py` used `AutoModelForCausalLM` and hardcoded `flash_attention_2`, while Qwen3.5 config advertises `Qwen3_5ForConditionalGeneration`.
- Root cause: Qwen3.5 compat shims existed in PPO/FSDP worker paths, not in the native SFT trainer path.
- Fix: mirror the FSDP worker auto-class selection in `fsdp_sft_trainer.py` and allow `+model.attn_implementation=sdpa`.
- Follow-up: real GCP SFT smoke must prove text-only forward works for `AutoModelForImageTextToText` with the native SFT batch.

## 2026-05-27: SFT Hydra append syntax for nested dict defaults

- Symptom: first GCP native SFT smoke failed before model load with `Could not override 'data.apply_chat_template_kwargs.enable_thinking'`.
- Root cause: `apply_chat_template_kwargs` exists as an empty dict under struct mode, but `enable_thinking` does not exist inside it.
- Fix: use `+data.apply_chat_template_kwargs.enable_thinking=False` in rendered SFT overrides.

## 2026-05-27: Qwen3.5 `_no_split_modules` can be a set under FSDP2

- Symptom: second GCP native SFT smoke loaded Qwen3.5, then failed in `apply_fsdp2` with `TypeError: 'set' object is not subscriptable`.
- Root cause: FSDP2 assumed `model._no_split_modules` was list-like. Qwen3.5 returned a set of layer names.
- Fix: normalize string, set, tuple, and ListConfig wrap policies to a list before indexing.

## 2026-05-27: Qwen3.5 config has no top-level vocab size

- Symptom: third native SFT smoke reached the first train step, then failed on `self.model.config.vocab_size`.
- Root cause: Qwen3.5 conditional-generation config stores vocabulary metadata differently from plain CausalLM configs.
- Fix: infer vocabulary width from `shift_logits.size(-1)` and use `reshape` instead of `view`.

## 2026-05-27: GCP L4 stockout in us-west1-a and us-west1-b

- Symptom: retry3 create failed in `us-west1-a`; retry3b failed in `us-west1-b`, both for `g2-standard-48` plus 4 L4.
- Root cause: zonal L4 stockout, not code.
- Workaround: use a smaller one-L4 Qwen3.5-0.8B smoke to keep debugging native SFT, then rerun 4xL4 for Qwen3.5-2B when stock returns.

## 2026-05-28: local full-vocab unit tests need a torch env

- Symptom: local `pytest` path lacked `pygments`, and standalone test import lacked `torch`.
- Root cause: OPD checkout has no local torch venv.
- Workaround: run the standalone tests inside the managed GCP env: `source .venv-qwen35-2b/bin/activate; PYTHONPATH=verl python verl/tests/trainer/ppo/test_full_vocab_distill.py`.
