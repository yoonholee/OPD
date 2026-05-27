"""
KL flavor estimators from per-token logprobs.

Inputs: per-rollout records giving per-token log π_T(y_t) and log π_θ(y_t)
for sequences sampled from either π_T (source="T") or π_θ (source="theta").

Estimators (all per-token, mean-aggregated):

  fwd_kl   = E_{y~π_T}    [ log π_T(y_t) - log π_θ(y_t) ]              (Brown §8)
  rev_kl   = E_{y~π_θ}    [ log π_θ(y_t) - log π_T(y_t) ]              (TML OPD)
  jsd      = 0.5 * E_{y~π_T}[log π_T(y_t) - log M(y_t)]
           + 0.5 * E_{y~π_θ}[log π_θ(y_t) - log M(y_t)]
             where log M(y_t) = logsumexp(log π_T, log π_θ) - log 2     (symmetric)
  fwd_kl_first32 = same as fwd_kl, first 32 response tokens only       (framing only)

Length filter: rollouts with y_len < ylen_floor (default 200) are dropped from KL
aggregates per Phase 0 finding (length-confounded).

Reverse KL clipping: per-token |log_ratio| can blow up when sampling from π_θ
hits a token assigned tiny prob under π_T. We clip per-token contributions at
±KL_CLIP_NATS and report the masked fraction. SPEC §Risks: "Add per-token clip at
|log_ratio| < 20 nats. Mask tokens beyond clip; report fraction masked."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

YLEN_FLOOR_DEFAULT = 200
KL_CLIP_NATS = 20.0
FIRST_K_DEFAULT = 32


@dataclass
class Rollout:
    """A single sampled response with per-token logprobs under both contexts.

    topk_T / topk_theta are optional per-position {token_id: logprob} dicts (the
    top-k next-token distribution under each context, conditioned on the realized
    prefix). When present, the truncated analytic estimator (`*_kl_tk`) uses them;
    when absent those fields are NaN. k1 and k3 only need the realized-token
    logprobs (logp_T / logp_theta).
    """
    source: str  # "T" or "theta" — which distribution generated this rollout
    logp_T: list[float]  # per-token log π_T(y_t | x, c_T), len = y_len
    logp_theta: list[float]  # per-token log π_θ(y_t | x), len = y_len
    topk_T: list[dict] | None = None      # per-token {tid: logp} under π_T context
    topk_theta: list[dict] | None = None  # per-token {tid: logp} under π_θ context

    @property
    def y_len(self) -> int:
        assert len(self.logp_T) == len(self.logp_theta), (
            f"length mismatch: T={len(self.logp_T)} theta={len(self.logp_theta)}"
        )
        return len(self.logp_T)


def _logsumexp2(a: float, b: float) -> float:
    """Numerically stable logsumexp for two scalars."""
    m = max(a, b)
    if math.isinf(m) and m < 0:
        return m
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _clip_log_ratio(diff: float) -> tuple[float, bool]:
    """Clip |diff| at KL_CLIP_NATS. Returns (clipped_value, was_clipped)."""
    if diff > KL_CLIP_NATS:
        return KL_CLIP_NATS, True
    if diff < -KL_CLIP_NATS:
        return -KL_CLIP_NATS, True
    return diff, False


def _k3(log_ratio: float) -> float:
    """Schulman k3 estimator of KL from a single sample.

    For KL(P||Q) with x~P and log_ratio = log P(x) - log Q(x):
    r = Q/P = exp(-log_ratio); k3 = (r - 1) - log r = (r - 1) + log_ratio.
    E[k3] = KL (same as k1 = log_ratio), but k3 >= 0 pointwise and lower variance.
    log_ratio is pre-clipped to ±KL_CLIP_NATS so exp() can't overflow.
    """
    lr = max(-KL_CLIP_NATS, min(KL_CLIP_NATS, log_ratio))
    r = math.exp(-lr)
    return (r - 1.0) + lr


def _trunc_kl_at_pos(p_topk: dict, q_topk: dict) -> float | None:
    """Truncated analytic KL(P||Q) at one position from top-k logprob dicts.

    P = the distribution we're taking expectation under (its top-k carries the
    weight). For tokens in P's top-k but missing from Q's top-k, Q's prob is
    floored at Q's smallest observed top-k prob (an upper bound on the true tail
    prob → slightly conservative). Tail mass of P beyond its top-k is dropped
    (typically <1% for peaky LM next-token dists); this is the only bias and is
    documented as truncation error. Returns None if inputs are unusable.
    """
    if not p_topk or not q_topk:
        return None
    q_floor = min(q_topk.values())  # log-prob; smallest seen under Q
    kl = 0.0
    for tid, lp in p_topk.items():
        p = math.exp(lp)
        lq = q_topk.get(tid, q_floor)
        kl += p * (lp - lq)
    return max(0.0, kl)  # clamp the tiny negative from float / floor slack


@dataclass
class KLAggregate:
    # k1 (naive log-ratio; unbiased, sign-indefinite, high variance) — original.
    fwd_kl: float = float("nan")
    rev_kl: float = float("nan")
    jsd: float = float("nan")
    fwd_kl_first32: float = float("nan")
    # k3 (Schulman; same expectation, >=0 pointwise, lower variance).
    fwd_kl_k3: float = float("nan")
    rev_kl_k3: float = float("nan")
    # tk (truncated top-k analytic; lowest variance, tiny truncation bias).
    fwd_kl_tk: float = float("nan")
    rev_kl_tk: float = float("nan")

    n_T_kept: int = 0
    n_theta_kept: int = 0
    ylen_mean_T: float = 0.0
    ylen_mean_theta: float = 0.0

    fwd_clip_frac: float = 0.0
    rev_clip_frac: float = 0.0
    tk_avail: bool = False  # were top-k dicts present (tk fields meaningful)?

    diagnostics: dict = field(default_factory=dict)


def aggregate(
    rollouts: list[Rollout],
    ylen_floor: int = YLEN_FLOOR_DEFAULT,
    first_k: int = FIRST_K_DEFAULT,
) -> KLAggregate:
    """Compute the four KL flavors from a bag of rollouts for one (problem, condition).

    Length filter: drop rollouts with y_len < ylen_floor from all KL means.
    fwd_kl_first32 is computed even on rollouts shorter than first_k; in that case
    it averages over min(first_k, y_len) tokens.
    """
    rT = [r for r in rollouts if r.source == "T" and r.y_len >= ylen_floor]
    rTh = [r for r in rollouts if r.source == "theta" and r.y_len >= ylen_floor]

    agg = KLAggregate(
        n_T_kept=len(rT),
        n_theta_kept=len(rTh),
        ylen_mean_T=_mean([r.y_len for r in rT]) if rT else 0.0,
        ylen_mean_theta=_mean([r.y_len for r in rTh]) if rTh else 0.0,
    )

    # === fwd_kl from T-rollouts: log π_T(y_T_t) - log π_θ(y_T_t) ===
    if rT:
        fwd_per_tok: list[float] = []
        fwd_k3: list[float] = []
        fwd_clip_n = 0
        fwd_total = 0
        for r in rT:
            for lt, le in zip(r.logp_T, r.logp_theta):
                d, clipped = _clip_log_ratio(lt - le)
                fwd_per_tok.append(d)
                fwd_k3.append(_k3(lt - le))  # KL(π_T||π_θ), sampled from π_T
                fwd_clip_n += int(clipped)
                fwd_total += 1
        agg.fwd_kl = _mean(fwd_per_tok)
        agg.fwd_kl_k3 = _mean(fwd_k3)
        agg.fwd_clip_frac = fwd_clip_n / max(fwd_total, 1)

        # tk: truncated analytic KL(π_T||π_θ) summed over π_T's top-k per position.
        if all(r.topk_T is not None and r.topk_theta is not None for r in rT):
            fwd_tk: list[float] = []
            for r in rT:
                for pk, qk in zip(r.topk_T, r.topk_theta):
                    v = _trunc_kl_at_pos(pk, qk)
                    if v is not None:
                        fwd_tk.append(v)
            if fwd_tk:
                agg.fwd_kl_tk = _mean(fwd_tk)
                agg.tk_avail = True

        # fwd_kl_first32: only first_k tokens per rollout (over T-rollouts).
        fwd_first_k: list[float] = []
        for r in rT:
            n_use = min(first_k, r.y_len)
            for lt, le in zip(r.logp_T[:n_use], r.logp_theta[:n_use]):
                d, _ = _clip_log_ratio(lt - le)
                fwd_first_k.append(d)
        agg.fwd_kl_first32 = _mean(fwd_first_k)

    # === rev_kl from θ-rollouts: log π_θ(y_θ_t) - log π_T(y_θ_t) ===
    if rTh:
        rev_per_tok: list[float] = []
        rev_k3: list[float] = []
        rev_clip_n = 0
        rev_total = 0
        for r in rTh:
            for lt, le in zip(r.logp_T, r.logp_theta):
                d, clipped = _clip_log_ratio(le - lt)
                rev_per_tok.append(d)
                rev_k3.append(_k3(le - lt))  # KL(π_θ||π_T), sampled from π_θ
                rev_clip_n += int(clipped)
                rev_total += 1
        agg.rev_kl = _mean(rev_per_tok)
        agg.rev_kl_k3 = _mean(rev_k3)
        agg.rev_clip_frac = rev_clip_n / max(rev_total, 1)

        # tk: truncated analytic KL(π_θ||π_T) summed over π_θ's top-k per position.
        if all(r.topk_T is not None and r.topk_theta is not None for r in rTh):
            rev_tk: list[float] = []
            for r in rTh:
                for pk, qk in zip(r.topk_theta, r.topk_T):
                    v = _trunc_kl_at_pos(pk, qk)
                    if v is not None:
                        rev_tk.append(v)
            if rev_tk:
                agg.rev_kl_tk = _mean(rev_tk)
                agg.tk_avail = True

    # === jsd: needs both directions ===
    if rT and rTh:
        log2 = math.log(2.0)
        # Half-1: from T-rollouts, log π_T - log M
        h1: list[float] = []
        for r in rT:
            for lt, le in zip(r.logp_T, r.logp_theta):
                logM = _logsumexp2(lt, le) - log2
                h1.append(lt - logM)  # >= 0 (log π_T >= log M when log π_T >= log π_θ; on average yes)
        # Half-2: from θ-rollouts, log π_θ - log M
        h2: list[float] = []
        for r in rTh:
            for lt, le in zip(r.logp_T, r.logp_theta):
                logM = _logsumexp2(lt, le) - log2
                h2.append(le - logM)
        agg.jsd = 0.5 * _mean(h1) + 0.5 * _mean(h2)

    return agg


def _self_test() -> None:
    """Synthetic sanity tests. Run with `python kl_metrics.py`."""
    import random
    random.seed(0)

    # === Test 1: identical π_T and π_θ → all KL flavors ≈ 0 ===
    n_tok = 250
    same_logps = [-2.0 + random.gauss(0, 0.1) for _ in range(n_tok)]
    rollouts = [
        Rollout(source="T", logp_T=same_logps, logp_theta=same_logps),
        Rollout(source="theta", logp_T=same_logps, logp_theta=same_logps),
    ]
    agg = aggregate(rollouts)
    assert abs(agg.fwd_kl) < 1e-9, f"fwd_kl on identical: {agg.fwd_kl}"
    assert abs(agg.rev_kl) < 1e-9, f"rev_kl on identical: {agg.rev_kl}"
    assert abs(agg.jsd) < 1e-9, f"jsd on identical: {agg.jsd}"
    assert abs(agg.fwd_kl_first32) < 1e-9, f"fwd_kl_first32 on identical: {agg.fwd_kl_first32}"
    print("PASS: identical distributions → all KL ≈ 0")

    # === Test 2: π_T concentrated, π_θ diffuse → fwd_kl > 0, rev_kl > 0, jsd in [0, log2] ===
    # Synthesize a clean per-token gap of 1 nat (T assigns higher prob to its samples
    # than theta does) for the T-rollouts, and same gap reversed for theta-rollouts.
    logp_high = [-1.0] * n_tok
    logp_low = [-2.0] * n_tok
    rollouts = [
        Rollout(source="T", logp_T=logp_high, logp_theta=logp_low),
        Rollout(source="theta", logp_T=logp_low, logp_theta=logp_high),
    ]
    agg = aggregate(rollouts)
    assert abs(agg.fwd_kl - 1.0) < 1e-9, f"fwd_kl: {agg.fwd_kl}"
    assert abs(agg.rev_kl - 1.0) < 1e-9, f"rev_kl: {agg.rev_kl}"
    # JSD upper-bounded by log 2 ≈ 0.693
    assert 0 <= agg.jsd <= math.log(2) + 1e-9, f"jsd out of bounds: {agg.jsd}"
    # For symmetric 1-nat gap, JSD analytically:
    # log M = logsumexp(-1, -2) - log 2; JSD = 0.5*[(-1)-logM] + 0.5*[(-1)-logM] -- but
    # h1 uses T-rollouts where log π_T = -1, log π_θ = -2 → log M = lse(-1,-2)-log 2 ≈ -1.6868
    # h1 entry: -1 - (-1.6868) = 0.6868 ... avg.
    # h2 uses θ-rollouts where (in our synthetic) log π_T = -2, log π_θ = -1 → log M same → 0.6868.
    # So JSD ≈ 0.5*0.6868 + 0.5*0.6868 = 0.6868 ≈ log(2)·0.991 (close to upper bound).
    expected_jsd_ish = -1.0 - (math.log(math.exp(-1.0) + math.exp(-2.0)) - math.log(2))
    assert abs(agg.jsd - expected_jsd_ish) < 1e-9, f"jsd analytic mismatch: got {agg.jsd} expected {expected_jsd_ish}"
    print(f"PASS: T-favored vs theta-favored → fwd_kl=1.0, rev_kl=1.0, jsd={agg.jsd:.4f}")

    # === Test 3: length filter drops short rollouts ===
    short = [-2.0] * 50  # below ylen_floor=200
    long_high = [-1.0] * 250
    long_low = [-2.0] * 250
    rollouts = [
        Rollout(source="T", logp_T=short, logp_theta=short),  # dropped
        Rollout(source="T", logp_T=long_high, logp_theta=long_low),
        Rollout(source="theta", logp_T=long_low, logp_theta=long_high),
    ]
    agg = aggregate(rollouts)
    assert agg.n_T_kept == 1, f"n_T_kept={agg.n_T_kept}"
    assert agg.n_theta_kept == 1, f"n_theta_kept={agg.n_theta_kept}"
    assert abs(agg.fwd_kl - 1.0) < 1e-9
    print("PASS: ylen<200 filtered out")

    # === Test 4: clipping kicks in for extreme log-ratios ===
    extreme_T = [0.0] * 250  # log π_T = 0 (prob 1)
    extreme_th = [-100.0] * 250  # log π_θ = -100 (vanishingly small)
    rollouts = [
        Rollout(source="T", logp_T=extreme_T, logp_theta=extreme_th),
    ]
    agg = aggregate(rollouts)
    assert abs(agg.fwd_kl - KL_CLIP_NATS) < 1e-9, f"fwd_kl clip: {agg.fwd_kl}"
    assert agg.fwd_clip_frac == 1.0, f"fwd_clip_frac: {agg.fwd_clip_frac}"
    print(f"PASS: clipping at {KL_CLIP_NATS} nats")

    # === Test 5: fwd_kl_first32 uses only first 32 tokens ===
    # T-rollout where first 32 tokens have 1-nat gap, remaining tokens have 0 gap.
    logp_T_mixed = [-1.0] * 32 + [-2.0] * 218
    logp_th_mixed = [-2.0] * 32 + [-2.0] * 218
    rollouts = [
        Rollout(source="T", logp_T=logp_T_mixed, logp_theta=logp_th_mixed),
    ]
    agg = aggregate(rollouts)
    # full fwd_kl: (32*1.0 + 218*0.0) / 250 = 0.128
    # fwd_kl_first32: 32*1.0 / 32 = 1.0
    assert abs(agg.fwd_kl - 32 / 250) < 1e-9, f"fwd_kl: {agg.fwd_kl}"
    assert abs(agg.fwd_kl_first32 - 1.0) < 1e-9, f"fwd_kl_first32: {agg.fwd_kl_first32}"
    print(f"PASS: fwd_kl_first32 isolates framing (fwd_kl={agg.fwd_kl:.3f}, first32={agg.fwd_kl_first32:.3f})")

    # === Test 6: empty rollouts → NaN, no crash ===
    agg = aggregate([])
    assert math.isnan(agg.fwd_kl)
    assert math.isnan(agg.rev_kl)
    assert math.isnan(agg.jsd)
    print("PASS: empty rollouts → NaN")

    # === Test 7: only one direction provided → that flavor populated, jsd NaN ===
    rollouts = [Rollout(source="T", logp_T=logp_high, logp_theta=logp_low)]
    agg = aggregate(rollouts)
    assert abs(agg.fwd_kl - 1.0) < 1e-9
    assert math.isnan(agg.rev_kl), f"rev_kl should be NaN: {agg.rev_kl}"
    assert math.isnan(agg.jsd), f"jsd should be NaN: {agg.jsd}"
    print("PASS: one-direction-only → fwd_kl OK, rev/jsd NaN")

    # === Test 8: k3 — same expectation as k1, non-negative, lower variance ===
    # Build a noisy log-ratio with true per-token KL ≈ 0 but heavy tail (the
    # answer-leak pathology): mostly 0, occasionally a large negative log-ratio
    # under reverse (π_T assigns ~1 to a token π_θ barely covers).
    random.seed(1)
    lr = []  # log π_θ - log π_T per token (reverse direction, sampled from π_θ)
    for _ in range(4000):
        lr.append(-8.0 if random.random() < 0.02 else random.gauss(0.0, 0.05))
    logp_th = [0.0] * len(lr)
    logp_T_ = [logp_th[i] - lr[i] for i in range(len(lr))]  # le - lt = lr
    agg = aggregate([Rollout(source="theta", logp_T=logp_T_, logp_theta=logp_th)])
    # k1 is the raw mean of lr; the 2% −8 tail drags it strongly negative.
    assert agg.rev_kl < 0, f"expected k1 driven negative by tail, got {agg.rev_kl}"
    # k3 has the SAME expectation but is >= 0 pointwise → aggregate >= 0.
    assert agg.rev_kl_k3 >= 0, f"k3 must be >=0, got {agg.rev_kl_k3}"
    print(f"PASS: k3 fixes the sign (k1 rev={agg.rev_kl:.3f} < 0, k3 rev={agg.rev_kl_k3:.3f} ≥ 0)")

    # === Test 9: truncated top-k analytic KL ===
    # Two clean categorical dists over 4 tokens; compare tk to the exact KL.
    import math as _m
    P = {0: 0.7, 1: 0.2, 2: 0.07, 3: 0.03}
    Q = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    exact_fwd = sum(P[v] * _m.log(P[v] / Q[v]) for v in P)  # KL(P||Q)
    pk = {v: _m.log(p) for v, p in P.items()}
    qk = {v: _m.log(q) for v, q in Q.items()}
    n = 250
    roll = Rollout(
        source="T",
        logp_T=[pk[0]] * n, logp_theta=[qk[0]] * n,
        topk_T=[dict(pk) for _ in range(n)],
        topk_theta=[dict(qk) for _ in range(n)],
    )
    roll2 = Rollout(
        source="theta",
        logp_T=[qk[0]] * n, logp_theta=[pk[0]] * n,
        topk_T=[dict(qk) for _ in range(n)],
        topk_theta=[dict(pk) for _ in range(n)],
    )
    agg = aggregate([roll, roll2])
    assert agg.tk_avail, "tk_avail should be True when top-k present"
    assert abs(agg.fwd_kl_tk - exact_fwd) < 1e-9, f"tk fwd {agg.fwd_kl_tk} vs exact {exact_fwd}"
    assert agg.fwd_kl_tk >= 0 and agg.rev_kl_tk >= 0
    print(f"PASS: truncated KL matches exact (tk fwd={agg.fwd_kl_tk:.4f}, exact={exact_fwd:.4f})")

    # tk absent → tk fields NaN, k1/k3 still computed.
    agg = aggregate([Rollout(source="T", logp_T=logp_high, logp_theta=logp_low)])
    assert math.isnan(agg.fwd_kl_tk) and not agg.tk_avail
    assert not math.isnan(agg.fwd_kl_k3)
    print("PASS: tk degrades to NaN without top-k; k1/k3 unaffected")

    print("\nAll kl_metrics self-tests passed.")


if __name__ == "__main__":
    _self_test()
