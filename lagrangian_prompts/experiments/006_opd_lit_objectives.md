# 006: Literature OPD stabilizers (top-K stop-grad, entropy-aware) do not transfer to the context/PI-distillation regime

**Takeaway:** The two most-cited OPD stabilization objectives from the
literature — **teacher top-K local support matching with stop-gradient**
(Revisiting-OPD 2603.25562 / The Many Faces 2605.11182, reported +19.8%) and
**entropy-aware reverse/forward mixing** (Entropy-Aware OPD 2603.07079, reported
+1.4–5pp) — give **no benefit and some harm** in our setup. They were developed
for *capability* distillation (strong→weak, genuine new capability); our regime
is *context / privileged-info distillation* (internalize a system prompt or an
answer leak). The recipe that still wins is plain `reverse_kl` + the KL-to-base
trust region from [005](005_opd_recipe_ablation.md).

2×2 vs the 004 `reverse_kl` baselines (0.8B, MATH, n=50 ID / 100 GSM8K, step-200):

| teacher | `reverse_kl` (004) | `topk_rkl` | `entropy_aware` |
|---|---|---|---|
| **concise** (no-leak, shared rule) | ID **0.36** stable [.40/.34/.34/.36], GSM 0.44 | ID 0.36 but **volatile** [.46/.18/.36/.36], GSM 0.43 | ID **0.24** [.20/.20/.24], GSM **0.07** |
| **aj_concise** (answer-leak, instance PI) | ID 0.00 collapse | ID **0.00** collapse [.10/.06/.00/.00] | ID 0.10 collapse [.02/.08/.10/.10] |

(base ID 0.26 / GSM 0.42 all cells. ID SE ≈ 0.07, n=1 seed.)

## Findings

1. **`topk_rkl` gives zero benefit.** On no-leak `concise` it lands at the
   *same* endpoint as plain reverse_kl (0.36 ID) but via a wildly volatile path
   (0.26→0.46→0.18→0.36) where reverse_kl was monotone-stable
   (0.40/0.34/0.34/0.36). Same result, strictly more variance → worse. On
   answer-leak `aj_concise` it collapses **identically** to reverse_kl (→0.00).
   The teacher-top-K + stop-grad selection does not change the outcome in either
   regime here.
2. **`entropy_aware` is actively harmful.** No-leak `concise`: ID 0.36→**0.24**
   and GSM8K 0.44→**0.07** (catastrophic OOD-down loss). Answer-leak: collapses
   (0.10). Its forward-KL-on-high-entropy-tokens component degrades everything —
   consistent with 005's "forward_kl collapses both teachers": any forward-KL
   admixture is destabilizing on this base.
3. **No objective rescues instance-specific PI.** `aj_concise` collapses under
   *every* objective tried (reverse_kl, forward_kl, topk_rkl, entropy_aware).
   Only the KL-base trust region (005) prevented collapse — and even then
   without making the leak transfer. This is the objective-agnostic
   impossibility predicted by The Many Faces (2605.11182): an instance-specific
   privileged signal cannot be internalized by a PI-free deployed policy,
   regardless of loss formulation. Not a tuning problem.
4. **Regime mismatch is the headline.** The literature's stabilizers target
   capability transfer (a strong teacher with genuinely new skill, per
   Rethinking-OPD 2604.13016's success conditions). Our teachers shift
   *behavior/format* (shared rule) or leak *instance answers* (PI) — neither is
   "new capability". The tricks don't port; the regime determines what works.

## Practical conclusion

- **Best recipe for our (context/PI) regime, unchanged:** plain per-token
  `reverse_kl` + a KL-to-base trust region (005). `topk_rkl` adds variance for
  no gain; `entropy_aware` (any forward-KL admixture) degrades. Do **not** adopt
  them in `modal_phase2.py` as defaults.
- **The only productive lever remains shared-latent-rule context distillation**
  (`concise`/`distrib`), not answer leaks — and even there, objective
  sophistication doesn't beat the simple stabilized baseline. The open lever is
  the *data/teacher* side (OPCD-style, training-free teacher selection via
  Unmasking-OPD's gradient-alignment diagnostic), not the loss.

## Caveats

- n=50 ID / n=1 seed. The across-the-board null (no objective beats stabilized
  reverse_kl) and the instance-PI impossibility are robust (theory-backed: The
  Many Faces); the exact volatile-`concise_topkrkl` endpoint (0.36) is within
  SE of the reverse_kl 0.36 — the claim is "no gain + more variance", not a
  precise number.
- `topk_k=20`, `ent_q=0.5` are single un-tuned settings; a sweep could shift
  magnitudes but cannot make instance-PI transferable (objective-agnostic) and
  is unlikely to make a forward-KL admixture stop degrading this base (005
  showed pure forward_kl collapses both teachers).
- `concise_entaware` is the step-0→150 trajectory: its Modal job was infra-
  retried and the redundant restart was stopped (the step-50/100/150 points
  0.20/0.20/0.24 are monotone-degraded vs the 0.36 baseline — conclusion
  unaffected; saved ~$6).
- `entropy_aware` had a bf16 bug (`torch.quantile` rejects bf16) caught at
  step 0 and fixed before the reported runs ([friction.md](../agent_notes/friction.md));
  the fp32 unit test missed it — retest harness now loops the prod dtype.
- 004/005's MAX_NEW-512, floor-L5 caveats carry.

## Navigation

- [`004`](004_opd_leak_vs_transfer.md) raw runs; [`005`](005_opd_recipe_ablation.md)
  collapse=trust-region-artifact + leak-doesn't-transfer; **006 = literature
  objectives don't port to this regime**.
- [`agent_notes/external-patterns.md`](../agent_notes/external-patterns.md) —
  implications section updated with this regime-mismatch result.
- `modal_phase2.py` `--method {reverse_kl,forward_kl,jsd,topk_rkl,entropy_aware}
  --kl-base-beta --topk-k --ent-q`;
  `results/phase2/phase2__{aj_concise,concise}__{topkrkl,entaware}.json`.

## Appendix

Date: 2026-05-17. Status: complete. 4 runs ×1 seed (+1 infra-retry stopped),
~1.6 H100-hr each, ≈ $22. Command: `modal run --detach modal_phase2.py
--teacher {aj_concise,concise} --method {topk_rkl,entropy_aware} --tag
{topkrkl,entaware}`. Objectives implemented in `distill_loss`; unit-tested
fp32+bf16, grad→student only. Trust-calibration: the `topk_rkl` step-50 ID 0.46
spike on `concise` looked like a win mid-run; the full trajectory (0.18 by
step-100) showed it was transient — single-eval-point reads are unreliable here,
the full curve is load-bearing (same lesson as 004→005).
