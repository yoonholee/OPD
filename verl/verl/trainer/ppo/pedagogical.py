from __future__ import annotations

from typing import Any

import torch


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _expand_to(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while tensor.dim() < target.dim():
        tensor = tensor.unsqueeze(-1)
    return tensor


def spike_weight(
    old_log_probs: torch.Tensor,
    student_top_k_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float = 5.0,
    lam: float = 0.2,
    max_gap: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Spike-aware learnability score from Pedagogical RL."""

    top1_log_probs = student_top_k_log_probs[..., 0]
    gaps = (top1_log_probs - old_log_probs).clamp_min(0)
    if max_gap is not None:
        gaps = gaps.clamp_max(float(max_gap))

    mask = response_mask.bool()
    lengths = mask.sum(dim=-1).clamp_min(1).to(gaps.dtype)

    if beta <= 0:
        masked_gaps = torch.where(mask, gaps, torch.zeros_like(gaps))
        penalty = masked_gaps.sum(dim=-1) / lengths
        weights = torch.exp(-float(lam) * penalty)
    else:
        neg_inf = torch.full_like(gaps, -torch.inf)
        scaled = torch.where(mask, float(beta) * gaps, neg_inf)
        penalty = torch.logsumexp(scaled, dim=-1) - torch.log(lengths)
        penalty = torch.where(mask.any(dim=-1), penalty, torch.zeros_like(penalty))
        weights = torch.exp(-(float(lam) / float(beta)) * penalty)

    masked_gaps = torch.where(mask, gaps, torch.zeros_like(gaps))
    metrics = {
        "pedagogy/spike_weight_mean": weights.mean(),
        "pedagogy/spike_weight_min": weights.min(),
        "pedagogy/surprise_gap_mean": masked_gaps.sum() / lengths.sum().clamp_min(1),
        "pedagogy/surprise_gap_max": masked_gaps.max(),
    }
    return weights, metrics


def surprisal_gate(
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = -8.0,
    kappa: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = torch.sigmoid(float(kappa) * (old_log_probs - float(gamma)))
    weights = torch.where(response_mask.bool(), weights, torch.zeros_like(weights))
    valid = response_mask.bool()
    valid_weights = weights[valid]
    if valid_weights.numel() == 0:
        mean = torch.zeros((), device=weights.device, dtype=weights.dtype)
        min_val = mean
    else:
        mean = valid_weights.mean()
        min_val = valid_weights.min()
    return weights, {
        "pedagogy/token_gate_mean": mean,
        "pedagogy/token_gate_min": min_val,
    }


def apply_pedagogical_scaling(
    token_level_scores: torch.Tensor,
    batch: Any,
    config: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply product-form pedagogy reward and optional assimilation gate."""

    if not _cfg_get(config, "enabled", False):
        return token_level_scores, {}

    tensors = batch.batch
    required = {"old_log_probs", "student_top_k_log_probs", "response_mask"}
    if not required.issubset(tensors.keys()):
        return token_level_scores, {}

    response_mask = tensors["response_mask"].to(token_level_scores.device)
    old_log_probs = tensors["old_log_probs"].to(token_level_scores.device)
    student_top_k_log_probs = tensors["student_top_k_log_probs"].to(token_level_scores.device)

    seq_weights, tensor_metrics = spike_weight(
        old_log_probs=old_log_probs,
        student_top_k_log_probs=student_top_k_log_probs,
        response_mask=response_mask,
        beta=float(_cfg_get(config, "spike_beta", 5.0)),
        lam=float(_cfg_get(config, "spike_lambda", 0.2)),
        max_gap=_cfg_get(config, "spike_max_gap", None),
    )

    scale = seq_weights
    if _cfg_get(config, "reward_gate", True) and "true_reward_score" in tensors:
        outcome = tensors["true_reward_score"].to(token_level_scores.device).sum(dim=-1)
        if _cfg_get(config, "reward_gate_mode", "raw") == "positive":
            outcome = (outcome > 0).to(token_level_scores.dtype)
        else:
            outcome = outcome.clamp_min(0).to(token_level_scores.dtype)
        scale = scale * outcome
        tensor_metrics["pedagogy/outcome_mean"] = outcome.mean()

    scaled_scores = token_level_scores * _expand_to(scale, token_level_scores)

    if _cfg_get(config, "token_gate", False):
        gate, gate_metrics = surprisal_gate(
            old_log_probs=old_log_probs,
            response_mask=response_mask,
            gamma=float(_cfg_get(config, "gate_gamma", -8.0)),
            kappa=float(_cfg_get(config, "gate_kappa", 1.0)),
        )
        scaled_scores = scaled_scores * _expand_to(gate, scaled_scores)
        tensor_metrics.update(gate_metrics)

    metrics = {key: val.detach().float().mean().item() for key, val in tensor_metrics.items()}
    return scaled_scores, metrics
