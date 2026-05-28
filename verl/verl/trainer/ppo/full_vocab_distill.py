# Copyright 2026 Individual Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import math
from typing import Any

import torch


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype, device=x.device)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def _kl_reverse(log_s: torch.Tensor, log_t: torch.Tensor) -> torch.Tensor:
    p_s = log_s.exp()
    return (p_s * (log_s - log_t)).sum(dim=-1)


def _kl_forward(log_s: torch.Tensor, log_t: torch.Tensor) -> torch.Tensor:
    p_t = log_t.exp()
    return (p_t * (log_t - log_s)).sum(dim=-1)


def _jsd(log_s: torch.Tensor, log_t: torch.Tensor) -> torch.Tensor:
    p_s = log_s.exp()
    p_t = log_t.exp()
    log_m = torch.logaddexp(log_s, log_t) - math.log(2.0)
    return 0.5 * (p_s * (log_s - log_m)).sum(dim=-1) + 0.5 * (p_t * (log_t - log_m)).sum(dim=-1)


def _topk_rkl(log_s: torch.Tensor, log_t: torch.Tensor, topk: int) -> torch.Tensor:
    if topk <= 0:
        raise ValueError(f"topk_rkl needs topk > 0, got {topk}")
    k = min(topk, log_t.shape[-1])
    idx = torch.topk(log_t, k=k, dim=-1).indices.detach()
    s_k = torch.gather(log_s, dim=-1, index=idx)
    t_k = torch.gather(log_t, dim=-1, index=idx)
    s_k = torch.log_softmax(s_k, dim=-1)
    t_k = torch.log_softmax(t_k, dim=-1)
    return (s_k.exp() * (s_k - t_k)).sum(dim=-1)


def compute_full_vocab_distill_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    objective: str,
    *,
    topk: int = 20,
    entropy_quantile: float = 0.5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute full-vocab student/teacher divergence on response positions.

    Args:
        student_log_probs: ``[batch, response_len, vocab]`` current actor log-probs.
        teacher_log_probs: ``[batch, response_len, vocab]`` frozen teacher log-probs.
        response_mask: ``[batch, response_len]`` valid response positions.
        objective: one of ``reverse_kl``, ``forward_kl``, ``jsd``, ``sym_kl``,
            ``topk_rkl``, or ``entropy_aware``.
    """
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError(f"student/teacher shape mismatch: {student_log_probs.shape} vs {teacher_log_probs.shape}")
    if student_log_probs.shape[:2] != response_mask.shape:
        raise ValueError(f"mask shape mismatch: {student_log_probs.shape[:2]} vs {response_mask.shape}")

    log_s = student_log_probs.float()
    log_t = teacher_log_probs.to(device=log_s.device, dtype=torch.float32).detach()
    mask = response_mask.to(device=log_s.device, dtype=torch.float32)
    objective = objective.lower()

    rkl = None
    fkl = None
    jsd = None
    if objective == "reverse_kl":
        rkl = _kl_reverse(log_s, log_t)
        div = rkl
    elif objective == "forward_kl":
        fkl = _kl_forward(log_s, log_t)
        div = fkl
    elif objective == "jsd":
        jsd = _jsd(log_s, log_t)
        div = jsd
    elif objective == "sym_kl":
        rkl = _kl_reverse(log_s, log_t)
        fkl = _kl_forward(log_s, log_t)
        div = 0.5 * (rkl + fkl)
    elif objective == "topk_rkl":
        div = _topk_rkl(log_s, log_t, topk=topk)
    elif objective == "entropy_aware":
        rkl = _kl_reverse(log_s, log_t)
        fkl = _kl_forward(log_s, log_t)
        with torch.no_grad():
            teacher_entropy = -(log_t.exp() * log_t).sum(dim=-1)
            valid = teacher_entropy[mask > 0].float()
            threshold = torch.quantile(valid, entropy_quantile) if valid.numel() else valid.new_zeros(())
            high_entropy = (teacher_entropy.float() >= threshold).to(dtype=log_s.dtype)
        div = high_entropy * fkl + (1.0 - high_entropy) * rkl
    else:
        raise ValueError(f"unknown full-vocab distillation objective: {objective}")

    loss = _masked_mean(div, mask)

    with torch.no_grad():
        student_entropy = -(log_s.exp() * log_s).sum(dim=-1)
        teacher_entropy = -(log_t.exp() * log_t).sum(dim=-1)
        metrics: dict[str, Any] = {
            "actor/full_vocab_distill_loss": loss.detach().item(),
            "actor/full_vocab_student_entropy": _masked_mean(student_entropy, mask).detach().item(),
            "teacher/full_vocab_entropy": _masked_mean(teacher_entropy, mask).detach().item(),
        }
        if rkl is not None:
            metrics["actor/full_vocab_reverse_kl"] = _masked_mean(rkl, mask).detach().item()
        if fkl is not None:
            metrics["actor/full_vocab_forward_kl"] = _masked_mean(fkl, mask).detach().item()
        if jsd is not None:
            metrics["actor/full_vocab_jsd"] = _masked_mean(jsd, mask).detach().item()
        if objective == "topk_rkl":
            metrics["actor/full_vocab_topk"] = min(topk, log_s.shape[-1])
        if objective == "entropy_aware":
            metrics["actor/full_vocab_entropy_quantile"] = entropy_quantile

    return loss, metrics
