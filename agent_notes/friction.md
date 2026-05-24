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
