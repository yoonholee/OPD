# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask
from verl.workers.config.actor import ActorConfig


def _actor_config():
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size=2,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
    )


def test_exact_matching_train_and_rollout_logprobs_give_identity_is_weights():
    train_log_prob = torch.tensor(
        [[-1.0, -2.0, -3.0, -9.0], [-0.5, -0.25, -4.0, -8.0]],
        dtype=torch.float32,
    )
    rollout_log_prob = train_log_prob.clone()
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)

    weights_proto, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=train_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is="token",
        rollout_is_threshold=10.0,
        rollout_rs=None,
    )

    weights = weights_proto.batch["rollout_is_weights"]
    assert torch.equal(modified_mask, response_mask)
    assert torch.allclose(weights, response_mask)
    assert metrics["rollout_corr/kl"] == 0.0
    assert metrics["rollout_corr/log_ppl_abs_diff"] == 0.0

    shifted_rollout = rollout_log_prob.clone()
    shifted_rollout[0, 1] -= torch.log(torch.tensor(2.0))
    shifted_weights_proto, _, _ = compute_rollout_correction_and_rejection_mask(
        old_log_prob=train_log_prob,
        rollout_log_prob=shifted_rollout,
        response_mask=response_mask,
        rollout_is="token",
        rollout_is_threshold=10.0,
        rollout_rs=None,
    )
    shifted_weights = shifted_weights_proto.batch["rollout_is_weights"]
    assert not torch.allclose(shifted_weights, response_mask)
    assert torch.allclose(shifted_weights[0, 1], torch.tensor(2.0))


def test_token_importance_sampling_matches_known_train_over_rollout_ratios():
    ratios = torch.tensor([[2.0, 0.5, 1.25, 1.0], [1.5, 0.25, 4.0, 1.0]], dtype=torch.float32)
    rollout_log_prob = torch.full_like(ratios, -2.0).log_softmax(dim=-1)
    train_log_prob = rollout_log_prob + torch.log(ratios)
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    expected = ratios * response_mask

    weights_proto, _, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=train_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is="token",
        rollout_is_threshold=10.0,
        rollout_rs=None,
    )

    weights = weights_proto.batch["rollout_is_weights"]
    assert torch.allclose(weights, expected, atol=1e-6)
    assert metrics["rollout_corr/rollout_is_ratio_fraction_high"] == 0.0


def test_sequence_importance_sampling_broadcasts_product_over_valid_tokens():
    token_ratios = torch.tensor([[2.0, 0.5, 1.25, 9.0], [1.5, 0.25, 4.0, 9.0]], dtype=torch.float32)
    rollout_log_prob = torch.full_like(token_ratios, -1.0)
    train_log_prob = rollout_log_prob + torch.log(token_ratios)
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
    expected_seq = torch.tensor([[1.25, 1.25, 1.25, 0.0], [0.375, 0.375, 0.0, 0.0]])

    weights_proto, _, _ = compute_rollout_correction_and_rejection_mask(
        old_log_prob=train_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is="sequence",
        rollout_is_threshold=10.0,
        rollout_rs=None,
    )

    weights = weights_proto.batch["rollout_is_weights"]
    assert torch.allclose(weights, expected_seq, atol=1e-6)


def test_policy_loss_applies_precomputed_rollout_is_weights():
    old_log_prob = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float32)
    log_prob = old_log_prob.clone()
    advantages = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    response_mask = torch.ones_like(old_log_prob)
    rollout_is_weights = torch.tensor([[2.0, 0.5, 1.0]], dtype=torch.float32)

    pg_loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=_actor_config(),
        rollout_is_weights=rollout_is_weights,
    )

    expected = (-advantages * rollout_is_weights).mean()
    assert torch.allclose(pg_loss, expected)
    assert metrics["actor/ppo_kl"] == 0.0

    unweighted_loss, _ = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=_actor_config(),
        rollout_is_weights=None,
    )
    assert not torch.allclose(pg_loss, unweighted_loss)
