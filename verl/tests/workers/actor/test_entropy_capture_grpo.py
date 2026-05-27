"""Regression tests for two bugs hit during paper-pair tuning.

Bug A (verl entropy capture):
  `dp_actor.py` had a buggy `_, log_prob, *_ = self._forward_micro_batch(...)` in the
  2D-advantages branch (vanilla GRPO), which dropped entropy. The 3D-advantages branch
  (top-k, used by OPD) captured it correctly. With `entropy_coeff > 0`, the 2D path
  hit `UnboundLocalError: cannot access local variable 'entropy'` at
  `agg_loss(loss_mat=entropy, ...)`. Repro signature was:

      configs: USE_KL_LOSS=True, ENTROPY_COEFF=0.001, algorithm.adv_estimator=grpo
      mode:    plain GRPO (no top-k, advantages.dim() == 2)

Bug B (Ray head race on shared SLURM nodes):
  Submitting two single-GPU SLURM jobs that landed on the same node had both
  call `ray start --head` against the default `/tmp/ray/...` session dir.
  The second job's `_write_cluster_info_to_kv()` asserted on session-name mismatch.
  Fix: pass `--temp-dir=$TMPDIR/ray-$SLURM_JOB_ID` so each job owns its session.

We don't spin up GPUs / Ray for these tests; we statically inspect the source
to ensure the regression patterns can't reappear.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DP_ACTOR = REPO_ROOT / "verl" / "workers" / "actor" / "dp_actor.py"
RUNNER = REPO_ROOT.parent / "local" / "bin" / "run_qwen35_2b_smoke.sh"
SYNC_SCRIPT = REPO_ROOT.parent / "local" / "bin" / "sync_to_schmidt.sh"


def _exported_int(src: str, name: str) -> int | None:
    patterns = (
        rf"(?:export\s+)?{name}=(\d+)",
        rf"(?:export\s+)?{name}=\$\{{{name}:-(\d+)\}}",
    )
    for pattern in patterns:
        match = re.search(pattern, src)
        if match is not None:
            return int(match.group(1))
    return None


def _update_policy_function() -> ast.FunctionDef:
    tree = ast.parse(DP_ACTOR.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "update_policy":
            return node
    raise AssertionError("update_policy() not found in dp_actor.py")


def test_forward_micro_batch_binds_entropy_and_log_prob_in_both_branches():
    """Both 3D-advantages (top-k / OPD) and 2D-advantages (vanilla GRPO) branches of
    `update_policy` must bind BOTH `entropy` (position 0) AND `log_prob` (position 1)
    from `_forward_micro_batch`, which returns `(entropy, log_probs, topk_ids, topk_log_probs)`.

    Asymmetric bug class (two seen, likely more lurking):
    - Bug A1: 2D branch had `_, log_prob, *_ = ...`, dropped entropy. UnboundLocalError at
      `agg_loss(loss_mat=entropy, ...)` when `entropy_coeff > 0`.
    - Bug A2: 3D branch had `entropy, _, _, topk_log_probs = ...`, dropped log_prob.
      UnboundLocalError at `kl_penalty(logprob=log_prob, ...)` when `use_kl_loss=True` (OPD path).

    Both fixed by capturing entropy AND log_prob at positions 0 and 1 in BOTH branches.
    """
    fn = _update_policy_function()
    calls = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn_attr = node.value.func
            if (
                isinstance(fn_attr, ast.Attribute)
                and fn_attr.attr == "_forward_micro_batch"
                and isinstance(fn_attr.value, ast.Name)
                and fn_attr.value.id == "self"
            ):
                calls.append(node)
    assert len(calls) >= 2, f"expected >=2 calls to self._forward_micro_batch inside update_policy(); got {len(calls)}"
    for call in calls:
        target = call.targets[0]
        assert isinstance(target, ast.Tuple), (
            f"line {call.lineno}: assignment from _forward_micro_batch must unpack a tuple"
        )
        # Position 0 must bind `entropy` (used in `entropy_coeff != 0` branch).
        first = target.elts[0]
        assert isinstance(first, ast.Name) and first.id == "entropy", (
            f"line {call.lineno}: position 0 must be `entropy` (got `{ast.unparse(first)}`). "
            "Bug class A1 returns: entropy_coeff > 0 crashes."
        )
        # Position 1 must bind `log_prob` (used in `use_kl_loss=True` branch via kl_penalty).
        if len(target.elts) > 1:
            second = target.elts[1]
            assert isinstance(second, ast.Name) and second.id == "log_prob", (
                f"line {call.lineno}: position 1 must be `log_prob` (got `{ast.unparse(second)}`). "
                "Bug class A2 returns: use_kl_loss=True crashes (OPD/top-k path)."
            )


def test_entropy_and_logprob_capture_via_source_regex():
    """Belt-and-suspenders source-regex check against the two known bad unpacking patterns."""
    src = DP_ACTOR.read_text()
    # A1: 2D branch dropping entropy
    bad_a1 = re.search(r"_, log_prob, \*_ = self\._forward_micro_batch", src)
    assert bad_a1 is None, (
        "Bug A1 returned: `_, log_prob, *_ = self._forward_micro_batch(...)`. Replace `_,` with `entropy,`."
    )
    # A2: 3D branch dropping log_prob (verl/OPD-specific)
    bad_a2 = re.search(r"entropy, _, _, topk_log_probs = self\._forward_micro_batch", src)
    assert bad_a2 is None, (
        "Bug A2 returned: `entropy, _, _, topk_log_probs = self._forward_micro_batch(...)`. "
        "Replace position 1 `_` with `log_prob`."
    )


def test_runner_unsets_amd_visible_devices():
    """Bug D: Schmidt's SLURM (and any cluster with mixed AMD/NVIDIA) sets
    ROCR_VISIBLE_DEVICES alongside CUDA_VISIBLE_DEVICES when allocating GPUs.
    verl's `worker.py:_setup_env_cuda_visible_devices` rejects this with
    `ValueError: Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set.`
    Fix: `unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES` in BASE_ENV before launching Ray workers.
    """
    src = RUNNER.read_text()
    assert "unset ROCR_VISIBLE_DEVICES" in src, (
        "runner must `unset ROCR_VISIBLE_DEVICES` in BASE_ENV; otherwise verl crashes on mixed-vendor clusters"
    )
    assert "HIP_VISIBLE_DEVICES" in src, "runner must also unset HIP_VISIBLE_DEVICES (same verl guardrail covers HIP)"


def test_runner_raises_fd_limit():
    """Bug from H100 8x bring-up: Ray on 8 GPUs hit `File descriptor limit reached`
    in raylet (SIGABRT). Default fd ulimit (1024) is too low. Fix: `ulimit -n 1048576`
    at runner top. On Schmidt this fails with EPERM but is non-fatal due to `|| true`.
    """
    src = RUNNER.read_text()
    assert re.search(r"ulimit\s+-n\s+\d{6,}", src), "runner must raise the fd limit (ulimit -n 1048576) before Ray init"


def test_runner_exports_nccl_socket_fallback():
    """Bug from H100 a3-highgpu-8g (NCCL 2.29): default NCCL configuration fails with
    `Failed to initialize any NET plugin` because the GCP DLVM doesn't ship a NET plugin.
    Fix: export NCCL_NET=Socket + NCCL_IB_DISABLE=1 + NCCL_SOCKET_IFNAME at the runner top
    (NOT just inside BASE_ENV) so all subprocesses inherit (e.g. LF's torchrun).
    """
    src = RUNNER.read_text()
    # Two requirements: (1) global export at top, (2) Socket fallback configured.
    assert re.search(r"^export NCCL_NET=", src, re.MULTILINE), (
        "NCCL_NET must be exported at the top of the script, not just inside BASE_ENV"
    )
    assert "NCCL_IB_DISABLE" in src, "NCCL_IB_DISABLE must be set to fall back from InfiniBand to Socket"
    assert "NCCL_SOCKET_IFNAME" in src, "NCCL_SOCKET_IFNAME must be set so the Socket transport knows which NIC to use"


def test_sync_to_schmidt_uses_anchored_excludes():
    """Bug C: rsync `--exclude='checkpoint/'` (no leading slash) matches every
    directory named `checkpoint` in the tree, not just the root. This silently
    nuked `verl/verl/utils/checkpoint/` and `verl/verl/third_party/torch/distributed/checkpoint/`,
    breaking `from verl.utils.checkpoint.checkpoint_manager import ...` on the Schmidt copy.

    Symptom on Schmidt: `ModuleNotFoundError: No module named 'verl.utils.checkpoint'`.
    Fix: use leading '/' (`--exclude='/checkpoint/'`) to anchor the pattern to the source root.

    This test asserts the canonical sync script uses anchored excludes for these traps.
    """
    assert SYNC_SCRIPT.exists(), (
        f"sync_to_schmidt.sh missing at {SYNC_SCRIPT}; create a canonical sync script "
        "so the rsync exclude rules are version-controlled and testable"
    )
    # Inspect actual --exclude arguments (skip comments).
    exclude_args = []
    for line in SYNC_SCRIPT.read_text().splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        for m in re.finditer(r"--exclude=(['\"])([^'\"]+)\1", line):
            exclude_args.append(m.group(2))
    assert exclude_args, "sync_to_schmidt.sh has no --exclude args at all (expected at least /checkpoint/ and /model/)"
    for trap in ("checkpoint/", "model/"):
        bad = trap
        good = "/" + trap
        # Forbid the unanchored form; require the anchored form to be present.
        assert bad not in exclude_args, (
            f"sync_to_schmidt.sh uses unanchored `--exclude='{bad}'`; matches nested dirs too. "
            f"Replace with `--exclude='{good}'` to anchor to source root."
        )
        assert good in exclude_args, f"sync_to_schmidt.sh should exclude root-level `{good}` via anchored pattern"
    # Verify the verl/utils/checkpoint module would be synced (sanity).
    nested_module = REPO_ROOT / "verl" / "utils" / "checkpoint" / "__init__.py"
    assert nested_module.exists(), (
        f"verl.utils.checkpoint is missing locally at {nested_module}; can't verify rsync would include it"
    )


def test_rollout_overrides_generation_config_sampling_params():
    """verl/702 (OPEN, hiyouga, Mar 2025): vLLM 0.8.0+ silently loads `generation_config.json`
    from the HF model dir during rollout. If you don't explicitly set sampling params, the model's
    config silently overrides (e.g., Qwen3 ships `temperature=0.7, top_p=0.8, top_k=20`, your
    rollout becomes sub-uniform without you knowing). This is a SUBTLE off-policy bug because
    train logprob recomputation uses the *intended* params while rollout used the silent ones.

    Mitigation: explicitly set temperature, top_p, top_k for both train rollout and val rollout
    so the verl→vLLM bridge passes them through.
    """
    src = RUNNER.read_text()
    # Train rollout: must set temperature, top_p, top_k.
    assert re.search(r"actor_rollout_ref\.rollout\.temperature=", src), "train rollout temperature must be explicit"
    # Val rollout: must set temperature AND top_p so val pass@K isn't silently re-shaped by generation_config.
    assert re.search(r"actor_rollout_ref\.rollout\.val_kwargs\.temperature=", src), (
        "val_kwargs.temperature must be explicit"
    )
    assert re.search(r"actor_rollout_ref\.rollout\.val_kwargs\.top_p=", src), "val_kwargs.top_p must be explicit"


def test_train_and_rollout_dtypes_match():
    """Off-policy kernel drift: if train and rollout use different dtypes (e.g., bf16 train + fp16
    rollout), token logprobs disagree even with identical weights. Manifests as nonzero gradient
    on a same-weight teacher/student (prior session measured max prob diff 0.75 at n=4).

    The mitigation is structural: keep both FSDP train and vLLM rollout in bf16. Test asserts the
    runner sets `actor.fsdp_config.model_dtype=bfloat16`. vLLM bf16 is default for Qwen3.
    """
    src = RUNNER.read_text()
    assert "fsdp_config.model_dtype=bfloat16" in src, (
        "actor.fsdp_config.model_dtype must be bfloat16 to match vLLM bf16 rollout default. "
        "Mismatched dtypes silently introduce off-policy drift."
    )
    # ref model dtype also matters: it computes ref_log_prob used in kl_penalty.
    assert src.count("fsdp_config.model_dtype=bfloat16") >= 2, (
        "Both actor.fsdp_config.model_dtype AND ref.fsdp_config.model_dtype must be bfloat16; "
        "ref_log_prob in different precision creates drift in the kl_penalty term."
    )


def test_response_budget_supports_boxed_answer_emit():
    """Original zero-reward bug: MAX_RESPONSE_LENGTH=64 (the legacy default) is too short for a
    student to emit `\\boxed{...}`, which `ttrl_math` requires for scoring. Result: every reward
    is 0, GRPO grad is 0, training is a no-op.

    Test asserts the runner's default MAX_RESPONSE_LENGTH (and any pass through gcp_paper_pair.sh
    or schmidt_opd.sbatch) is >= 256. The launcher overrides typically set 1024; this guard catches
    regressions to the toy default.
    """
    src = RUNNER.read_text()
    assert _exported_int(src, "MAX_RESPONSE_LENGTH") is not None, (
        "MAX_RESPONSE_LENGTH default-assignment must exist in runner"
    )
    # 32-64 was the smoke-only default that ate the zero-reward bug. Launchers MUST override.
    for launcher_rel in ("local/bin/gcp_paper_pair.sh", "local/bin/schmidt_opd.sbatch"):
        path = RUNNER.parent.parent.parent / launcher_rel
        if not path.exists():
            continue
        launcher_src = path.read_text()
        launcher_default = _exported_int(launcher_src, "MAX_RESPONSE_LENGTH")
        assert launcher_default is not None, f"{launcher_rel} must override MAX_RESPONSE_LENGTH explicitly"
        assert launcher_default >= 256, (
            f"{launcher_rel} sets MAX_RESPONSE_LENGTH={launcher_default}; must be >= 256 so the "
            "student can emit \\boxed{...} for ttrl_math reward (orig zero-reward bug)."
        )


def test_val_prompts_fit_under_max_prompt_length():
    """verl/4975 (OPEN, Jan 2026): `data.filter_overlong_prompts=True` filters TRAIN prompts but
    does NOT filter the val set. Long val prompts (e.g., MATH-500 has prompts up to ~700 tokens)
    crash with `NotImplementedError: sequence_length=X is larger than max_length=Y` at val time.

    We hit this when MAX_PROMPT_LENGTH=512 and MATH-500 had a 694-token prompt. Mitigation:
    set MAX_PROMPT_LENGTH at least as large as the longest val prompt + headroom.

    Test asserts launchers set MAX_PROMPT_LENGTH >= 1024 (covers MATH-500). For non-MATH val sets
    (AIME24 fits in 512; Olympiad-Bench may need more), users should validate per-set.
    """
    for launcher_rel in ("local/bin/gcp_paper_pair.sh", "local/bin/schmidt_opd.sbatch"):
        path = RUNNER.parent.parent.parent / launcher_rel
        if not path.exists():
            continue
        launcher_src = path.read_text()
        n = _exported_int(launcher_src, "MAX_PROMPT_LENGTH")
        if n is None:
            continue
        # If val_files is set to MATH-500, prompt budget must be >= 1024 (MATH-500 max ~694 tokens).
        if "MATH-500" in launcher_src:
            assert n >= 1024, (
                f"{launcher_rel} sets MAX_PROMPT_LENGTH={n} but uses MATH-500 val set "
                f"(prompts up to ~700 tokens). verl/4975: filter_overlong_prompts does NOT filter val. "
                "Bump to >= 1024."
            )


def test_runner_pins_all_ray_ports_on_shared_node():
    """Ray opens MORE ports than just `--port`/`--ray-client-server-port`/`--dashboard-port`.
    The default `--dashboard-agent-listen-port=52365` is FIXED across Ray instances and silently
    collides on shared SLURM nodes (discuss.ray.io thread "agent couldn't be started due to port conflict").
    Same for `--metrics-export-port`, `--runtime-env-agent-port`, `--node-manager-port`,
    `--object-manager-port`, `--dashboard-agent-grpc-port`. All must be pinned per-job.
    """
    src = RUNNER.read_text()
    required = [
        "--dashboard-agent-listen-port=",  # default 52365 (FIXED), the silent footgun
        "--dashboard-agent-grpc-port=",
        "--runtime-env-agent-port=",
        "--metrics-export-port=",
        "--node-manager-port=",
        "--object-manager-port=",
    ]
    missing = [flag for flag in required if flag not in src]
    assert not missing, (
        f"runner must pin these Ray ports per-job on shared SLURM nodes: {missing}. "
        "Default values can collide between neighbor jobs."
    )


def test_runner_unsets_ray_address_and_sets_ray_tmpdir():
    """Bare `ray.init()` (no address arg) consults `RAY_ADDRESS` env var, then
    `/tmp/ray/ray_current_cluster`, both shared across jobs on a node. Job B's driver code can
    accidentally connect to Job A's cluster. Mitigations: (1) `unset RAY_ADDRESS` at top of runner
    so it doesn't leak from submission env; (2) `export RAY_TMPDIR=<per-job>` so Python-side
    `ray.init()` cluster discovery uses the per-job temp dir, not /tmp/ray.
    """
    src = RUNNER.read_text()
    assert "unset RAY_ADDRESS" in src, (
        "runner must `unset RAY_ADDRESS` to prevent driver code from connecting to a neighbor's cluster"
    )
    assert re.search(r"export\s+RAY_TMPDIR=", src), (
        "runner must `export RAY_TMPDIR=<per-job-dir>` so Python ray.init() discovery is per-job-scoped"
    )


def test_runner_passes_num_gpus_to_ray():
    """Ray's GPU autodetect via `nvidia-smi` can over-report GPUs vs the SLURM grant
    (Ray #13607, ray-discuss "stop Ray from managing CUDA_VISIBLE_DEVICES").
    Symptom: actor lands on a neighbor job's GPU → OOM or silent contention.
    Fix: pass `--num-gpus=$NGPUS` to `ray start --head` explicitly so Ray's accounting matches
    the SLURM allocation regardless of what nvidia-smi reports.
    """
    src = RUNNER.read_text()
    assert re.search(r"--num-gpus=", src), (
        "ray start --head must pass --num-gpus explicitly; default autodetect can see neighbor GPUs"
    )


def test_runner_caps_omp_threads():
    """OpenMP default thread count = CPU count. On shared HPC nodes (B300 = 224 cores) that's
    >200 threads per process → ulimit -u explosion + fd exhaustion in Ray workers
    (Ray #54225, #36936). Cap at 1; verl/vllm rely on torch internals, not OpenMP.
    """
    src = RUNNER.read_text()
    assert re.search(r"OMP_NUM_THREADS=\$\{OMP_NUM_THREADS:-1\}", src) or "export OMP_NUM_THREADS=1" in src, (
        "runner must set OMP_NUM_THREADS=1 (or default-assign to 1) to prevent thread explosion"
    )


def test_vllm_rollout_overrides_prefix_cache_and_logprob_mode():
    """Three vLLM 0.21 correctness traps that need explicit `engine_kwargs.vllm.*` overrides:

    (1) `enable_prefix_caching` defaults TRUE in vLLM V1. For verl OPD/GRPO with sleep/wake weight
        updates each step, cached KV blocks from the OLD weights can be reused after wake with NEW
        weights → silently wrong rollouts. (vllm #18055 measured 78%→60% GSM8K acc.)
    (2) `logprobs_mode` defaults to `raw_logprobs` (pre-temperature, pre-top-k/top-p). Verl's
        rollout-returned logprobs are then in a different distribution than the sampled tokens.
        Set `processed_logprobs` to match the sampling distribution.
        (ServiceNow "Correctness Before Corrections" + vllm forum #1616.)
    (3) `generation_config` defaults to load HF `generation_config.json`, which silently overrides
        sampling params via `min(...)`. (vllm #34005). Set to `"vllm"` to skip the HF config entirely.
    """
    src = RUNNER.read_text()
    assert "actor_rollout_ref.rollout.enable_prefix_caching=False" in src, (
        "vLLM rollout config must set enable_prefix_caching=False; default True breaks RL sleep/wake"
    )
    assert "engine_kwargs.vllm.logprobs_mode=processed_logprobs" in src, (
        "vLLM engine_kwargs must set logprobs_mode=processed_logprobs; default raw_logprobs "
        "is pre-temperature and creates off-policy drift"
    )
    assert "engine_kwargs.vllm.generation_config=vllm" in src, (
        "vLLM engine_kwargs must set generation_config=vllm; default loads HF generation_config.json "
        "and silently overrides sampling params (vllm #34005)"
    )


def test_runner_isolates_ray_per_slurm_job():
    """When `SLURM_JOB_ID` is set, the runner must isolate ALL Ray ports per job.

    Three independent failure modes seen on Schmidt B300 node b301:
    1. Shared /tmp/ray session_name → `_write_cluster_info_to_kv` assertion.
       Fix: `--temp-dir=$TMPDIR/ray-$SLURM_JOB_ID`.
    2. Default GCS port 6379 shared across jobs. Even with --temp-dir, two ray-start-head
       on same port can't both bind.
       Fix: `--port` derived from SLURM_JOB_ID.
    3. Chosen GCS port landing inside Ray default worker_ports range (10002-19999).
       `ValueError: Ray component worker_ports is trying to use a port number 10031 that
       is used by other components.` (--port=10031 was inside the worker range).
       Fix: keep --port below 10000 AND override --min-worker-port/--max-worker-port to a
       per-job slice.
    4. `ray stop --force` kills neighbor jobs' Ray instances on the shared node (SIGTERM,
       exit 0:15).
       Fix: gate `ray stop` behind non-SLURM context.

    This test enforces all four pieces stay wired in the runner.
    """
    src = RUNNER.read_text()
    assert "SLURM_JOB_ID" in src, "runner must check SLURM_JOB_ID before invoking ray start"
    assert "--temp-dir=" in src, "runner must pass --temp-dir (Bug 1)"
    assert "--port=" in src, "runner must pass --port (Bug 2: default GCS port 6379 collides)"
    assert "--min-worker-port=" in src and "--max-worker-port=" in src, (
        "runner must pass --min-worker-port/--max-worker-port (Bug 3: --port can land in default worker range)"
    )
    assert "--ray-client-server-port=" in src, (
        "runner must pass --ray-client-server-port (Ray client port 10001 collides across jobs)"
    )
    assert "RAY_STOP_CMD" in src, "runner must gate `ray stop --force` (Bug 4: kills neighbor jobs)"
    assert re.search(r"ray\s+start\s+--head\s+\\?\$RAY_FLAGS", src), "ray start --head must consume $RAY_FLAGS"


if __name__ == "__main__":
    test_forward_micro_batch_binds_entropy_and_log_prob_in_both_branches()
    test_entropy_and_logprob_capture_via_source_regex()
    test_runner_unsets_amd_visible_devices()
    test_runner_raises_fd_limit()
    test_runner_exports_nccl_socket_fallback()
    test_sync_to_schmidt_uses_anchored_excludes()
    test_runner_isolates_ray_per_slurm_job()
    test_rollout_overrides_generation_config_sampling_params()
    test_train_and_rollout_dtypes_match()
    test_response_budget_supports_boxed_answer_emit()
    test_val_prompts_fit_under_max_prompt_length()
    test_runner_pins_all_ray_ports_on_shared_node()
    test_runner_unsets_ray_address_and_sets_ray_tmpdir()
    test_runner_passes_num_gpus_to_ray()
    test_runner_caps_omp_threads()
    test_vllm_rollout_overrides_prefix_cache_and_logprob_mode()
    print("OK: all 16 tests passed")
