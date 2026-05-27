# Pedagogical RL / OPD progress

Updated: 2026-05-27

## Where we are

We have a first working reimplementation of the Noah Ziems / Chakraborty et al. Pedagogical RL idea inside the local OPD/verl stack.
It is not a full two-stage teacher-GRPO plus student-assimilation pipeline yet.
It is the smallest OPD-native approximation: scale OPD token rewards by a spike-aware learnability weight, optionally add a surprisal gate, then train with existing OPD machinery.

The idle GCP GPU is stopped.
Current active compute is done: no Schmidt jobs are in queue.

## Implemented locally

Files:
- `verl/verl/trainer/ppo/pedagogical.py`
  - `spike_weight(...)`: implements the paper's spike penalty using student top-1 vs sampled-token logprob gap.
  - `surprisal_gate(...)`: optional token-level imitation/assimilation gate.
  - `apply_pedagogical_scaling(...)`: applies product-form reward scaling, supports OPD 3D `[B,T,K]` rewards.
- `verl/verl/trainer/ppo/ray_trainer.py`
  - applies pedagogical scaling after `true_reward_score` is available and logs `pedagogy/*` metrics.
- `local/bin/run_qwen35_2b_smoke.sh`
  - added `PEDAGOGICAL_*` env knobs.
  - fixed script exit-code propagation.
  - avoids `ray stop --force` under Slurm.
  - adds per-Slurm-job Ray ports/temp dirs and `_slurm$SLURM_JOB_ID` log dirs.
- `local/bin/schmidt_opd.sbatch`
  - hparams now overridable from env for sweeps.
- `verl/tests/trainer/ppo/test_pedagogical.py`
  - unit tests for spike penalty, gate, and 3D scaling.

Important non-pedagogy fix:
- `verl/verl/workers/actor/dp_actor.py`
  - fixed branch-asymmetric `entropy` / `log_prob` capture bugs that crash GRPO/OPD with entropy or KL knobs enabled.

## Verification

Local:
- `python3 -m py_compile verl/verl/trainer/ppo/pedagogical.py verl/verl/trainer/ppo/ray_trainer.py verl/tests/trainer/ppo/test_pedagogical.py`
- `bash -n local/bin/run_qwen35_2b_smoke.sh local/bin/schmidt_opd.sbatch`

Schmidt:
- remote syntax check passed.
- remote manual torch checks for `pedagogical.py` passed.
- 3-step B300 smoke passed: job `53678`, run `agent_notes/gcp_runs/20260527_171226_qwen35_2b`, rc 0.

GCP:
- stopped `opd-tune3-passk-g4-0527-1148` in `us-west1-a`; last seen `TERMINATED`.

## Schmidt sweep results

Common config:
- model: `Qwen/Qwen3-1.7B-Base`
- teacher: `Qwen/Qwen3-4B-Base`
- B300, 1 GPU per job
- `TRAIN_STEPS=50`, `SMOKE_N=2048`, `VAL_N=8`, MATH-500 validation
- initial val for these jobs: mean@8 `0.07825`, best@8 `0.312578`, maj@8 `0.025418`

| job | name | state | final mean@8 | final best@8 | final maj@8 | notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 53688 | `opd-base-fix` | FAILED `0:15` | n/a | n/a | n/a | reached step 49, no final val; Slurm killed batch |
| 53689 | `opd-ped-l01` | FAILED `0:15` | 0.08000 | 0.30806 | 0.03082 | final metrics logged; DataLoader worker killed after final step |
| 53690 | `opd-ped-l02` | FAILED `0:15` | 0.07525 | 0.29881 | 0.028378 | final metrics logged, but batch did not write rc |
| 53691 | `opd-ped-l02-tg` | COMPLETED rc 0 | 0.08100 | 0.31470 | 0.029052 | clean run, best of this sweep |

Clean run deltas vs initial val:
- `opd-ped-l02-tg`: mean@8 `0.07825 -> 0.08100`, delta `+0.00275`.
- `opd-ped-l02-tg`: best@8 `0.31258 -> 0.31470`, delta `+0.00212`.
- `opd-ped-l02-tg`: maj@8 `0.02542 -> 0.02905`, delta `+0.00363`.

Interpretation:
- The pedagogical code path works and logs sane metrics.
- The token-gated `lambda=0.2` setting is the only fully clean 50-step run and is slightly positive on all three MATH-500 summary metrics.
- These are tiny deltas on one seed. Treat as smoke evidence, not a claim.
- `lambda=0.1` had better maj@8 but failed after final metrics, so it is useful but not clean.
- Plain base run did not finish final eval, so we still lack a same-run OPD baseline for this exact config.

## Failure / infra status

Known current blockers:
- Parallel B300 jobs can still get killed near shutdown/final eval. `sacct` showed jobs 53688-53690 as `FAILED 0:15`, while 53691 completed.
- `opd-ped-l01` logged `RuntimeError: DataLoader worker ... killed by signal: Killed` after final metrics.
- CPU RSS was around 65 GB per batch process. Four parallel jobs on one node likely pushed host memory too hard during validation/shutdown.
- Base run reached step 49 but was killed before final validation.

Recent infra fixes already made:
- per-job Ray temp dirs
- per-job Ray GCS/client/dashboard/object/worker ports
- `_slurm$SLURM_JOB_ID` log dirs
- no `ray stop --force` under Slurm
- nonzero shell exit propagation

Next likely fix:
- run the next sweeps serialized, or lower parallelism/validation memory, before trusting comparisons.

## Next experiments

1. Rerun a clean 50-step baseline serialized:
   - `PEDAGOGICAL_ENABLED=False`
   - same hparams as 53691
2. Rerun best pedagogical setting serialized:
   - `PEDAGOGICAL_ENABLED=True`
   - `PEDAGOGICAL_SPIKE_LAMBDA=0.2`
   - `PEDAGOGICAL_TOKEN_GATE=True`
3. If both are clean, expand to 100-200 steps and 2-3 seeds.
4. Implement closer-to-paper two-stage version only after the reward-scaling variant clears baseline:
   - stage A: train privileged teacher on `R * G_spike`
   - stage B: sample teacher rollouts and student-train with normalized surprisal-gated SFT/OPD loss.
