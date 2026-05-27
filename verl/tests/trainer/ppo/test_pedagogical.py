import torch

from verl.trainer.ppo.pedagogical import apply_pedagogical_scaling, spike_weight, surprisal_gate


class _Batch:
    def __init__(self, tensors):
        self.batch = tensors


def test_spike_weight_penalizes_single_large_jump():
    old_log_probs = torch.tensor([[-0.2, -10.0, -0.2], [-0.5, -0.5, -0.5]])
    top1 = torch.tensor([[-0.1, -0.1, -0.1], [-0.1, -0.1, -0.1]])
    student_top_k_log_probs = torch.stack([top1, top1 - 1], dim=-1)
    response_mask = torch.ones_like(old_log_probs)

    weights, metrics = spike_weight(old_log_probs, student_top_k_log_probs, response_mask, beta=5.0, lam=0.2)

    assert weights[0] < weights[1]
    assert metrics["pedagogy/surprise_gap_max"] > 9


def test_surprisal_gate_downweights_unlikely_tokens():
    old_log_probs = torch.tensor([[-2.0, -12.0]])
    response_mask = torch.ones_like(old_log_probs)

    weights, _ = surprisal_gate(old_log_probs, response_mask, gamma=-8.0, kappa=1.0)

    assert weights[0, 0] > 0.99
    assert weights[0, 1] < 0.02


def test_apply_pedagogical_scaling_supports_3d_opd_rewards():
    scores = torch.ones(2, 3, 4)
    old_log_probs = torch.tensor([[-0.2, -10.0, -0.2], [-0.5, -0.5, -0.5]])
    top1 = torch.tensor([[-0.1, -0.1, -0.1], [-0.1, -0.1, -0.1]])
    batch = _Batch(
        {
            "old_log_probs": old_log_probs,
            "student_top_k_log_probs": torch.stack([top1, top1 - 1], dim=-1),
            "response_mask": torch.ones_like(old_log_probs),
            "true_reward_score": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        }
    )

    scaled, metrics = apply_pedagogical_scaling(
        scores,
        batch,
        {
            "enabled": True,
            "spike_beta": 5.0,
            "spike_lambda": 0.2,
            "reward_gate": True,
            "token_gate": False,
        },
    )

    assert scaled.shape == scores.shape
    assert scaled[0].sum() > 0
    assert scaled[1].sum() == 0
    assert "pedagogy/spike_weight_mean" in metrics
