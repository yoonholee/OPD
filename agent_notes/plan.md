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
