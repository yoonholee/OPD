import math

import torch

from verl.trainer.ppo.full_vocab_distill import compute_full_vocab_distill_loss


def _logp(x):
    return torch.log_softmax(torch.tensor(x, dtype=torch.float64), dim=-1).float()


def test_reverse_forward_and_jsd_match_manual_values():
    s = _logp([[[1.0, -1.0, 0.5], [0.3, -0.2, 0.0]]])
    t = _logp([[[0.1, 0.2, -0.4], [-0.3, 0.7, 0.2]]])
    mask = torch.tensor([[1.0, 1.0]])

    p_s = s.exp()
    p_t = t.exp()
    manual_rkl = (p_s * (s - t)).sum(-1).mean()
    manual_fkl = (p_t * (t - s)).sum(-1).mean()
    log_m = torch.logaddexp(s, t) - math.log(2.0)
    manual_jsd = (0.5 * (p_s * (s - log_m)).sum(-1) + 0.5 * (p_t * (t - log_m)).sum(-1)).mean()

    rkl, _ = compute_full_vocab_distill_loss(s, t, mask, "reverse_kl")
    fkl, _ = compute_full_vocab_distill_loss(s, t, mask, "forward_kl")
    jsd, _ = compute_full_vocab_distill_loss(s, t, mask, "jsd")

    torch.testing.assert_close(rkl, manual_rkl)
    torch.testing.assert_close(fkl, manual_fkl)
    torch.testing.assert_close(jsd, manual_jsd)


def test_mask_excludes_padding_and_gradient_flows_to_student_only():
    s = _logp([[[2.0, 0.0], [0.0, 2.0]]]).requires_grad_(True)
    t = _logp([[[0.0, 2.0], [2.0, 0.0]]])
    mask = torch.tensor([[1.0, 0.0]])

    loss, _ = compute_full_vocab_distill_loss(s, t, mask, "forward_kl")
    loss.backward()

    assert torch.isfinite(s.grad).all()
    assert s.grad[0, 0].abs().sum() > 0
    assert s.grad[0, 1].abs().sum() == 0


def test_topk_rkl_uses_teacher_supported_subset():
    s = _logp([[[3.0, 2.0, 1.0, 0.0]]])
    t = _logp([[[0.0, 3.0, 2.0, 1.0]]])
    mask = torch.ones(1, 1)

    loss, metrics = compute_full_vocab_distill_loss(s, t, mask, "topk_rkl", topk=2)
    idx = torch.topk(t, k=2, dim=-1).indices
    s_k = torch.log_softmax(torch.gather(s, -1, idx), dim=-1)
    t_k = torch.log_softmax(torch.gather(t, -1, idx), dim=-1)
    manual = (s_k.exp() * (s_k - t_k)).sum(-1).mean()

    torch.testing.assert_close(loss, manual)
    assert metrics["actor/full_vocab_topk"] == 2


if __name__ == "__main__":
    test_reverse_forward_and_jsd_match_manual_values()
    test_mask_excludes_padding_and_gradient_flows_to_student_only()
    test_topk_rkl_uses_teacher_supported_subset()
