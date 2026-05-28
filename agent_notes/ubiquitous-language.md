# Ubiquitous language

- **Distillation stack**: upstream THUNLP/OPD + local verl fixes used for OPD-style and SFT-style distillation experiments.
- **Patch set**: fixes discovered during runs that should live directly in the training code, not as optional overlays. Keep diffs small and tied to a concrete issue.
- **Patch audit**: maintenance script that clones upstream to a temp dir and emits inspectable line-by-line deviations plus the issue each deviation exists to fix.
- **Teacher-context selection**: Lagrangian prompt work. Separate research module that selects/constructs teacher contexts for distillation jobs.
- **Distillation recipe**: a provider-independent training recipe that renders to verl/Hydra overrides. Split into on-policy and off-policy families, while sharing model, data, eval, teacher-context, and compute schemas.
- **On-policy recipe**: student samples from the current policy, then a teacher/objective scores the student trajectory. OPD, StableOPD, vOPD, OPCD, and GRPO-style probes live here.
- **Off-policy recipe**: examples are sampled or cached before training, usually from a teacher or dataset, then the student trains by SFT-style or distillation loss. SFT distill, skew-KL, and contrastive distill live here.
- **Compute adapter**: thin launch layer for a provider such as Schmidt or GCP. It should not own training semantics.
- **Evidence archive**: raw logs and run artifacts kept locally/ignored for forensics. Summaries only are tracked in git.
