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
