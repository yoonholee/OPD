# OPD cleanup

Status: in progress

Scope:
- clean only OPD code touched in the pedagogical-RL thread
- do not touch unrelated dirty files in sandbox root

Plan:
- audit OPD diffs for accidental/debuggy changes
- simplify pedagogical reward integration and runner env plumbing
- verify syntax/tests that are available locally
- report remaining non-clean artifacts and next baseline run

# Lagrangian prompts merge

Status: complete

Result:
- imported sandbox `lagrangian_prompts/` as an OPD-adjacent experiment directory
- kept scripts runnable from OPD root as `lagrangian_prompts/*.py`
- ignored raw Modal `lagrangian_prompts/results/`; preserved tracked Phase 0 pareto figure
- kept writeups and experiment figures tracked
- verified Python syntax and pure imports locally
