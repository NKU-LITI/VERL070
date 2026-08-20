import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_token_on_off_policy_loss
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.scaf_grpo_utils import (
    build_hinted_gen_batch,
    find_failed_group_representatives,
    inject_expert_trajectories,
    replace_rollout_trajectories,
    select_minimal_successful_hints,
)
from verl.workers.actor.dp_actor import DataParallelPPOActor, _compute_micro_batch_loss_scale


class _ActorConfig(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _failed_problem_batch():
    return DataProto.from_single_dict(
        {
            "dummy_tensor": torch.zeros((1, 1), dtype=torch.uint8),
            "question": np.asarray(["What is 1 + 1?"], dtype=object),
            "uid": np.asarray(["problem-1"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "2"}], dtype=object),
            "data_source": np.asarray(["math"], dtype=object),
            "difficulty_bucket": np.asarray(["hard"], dtype=object),
            "knowledge_components_parts": np.asarray([["addition", "integers", "arithmetic", "sum"]], dtype=object),
            "planning_skeleton_parts": np.asarray([["identify", "combine", "simplify", "check"]], dtype=object),
            "solution_breakdown_parts": np.asarray([["one", "plus one", "equals", "two"]], dtype=object),
        }
    )


def test_build_hinted_gen_batch_expands_three_progressive_stages():
    hinted = build_hinted_gen_batch(_failed_problem_batch())

    assert len(hinted) == 12
    assert hinted.non_tensor_batch["raw_prompt"].shape == (12,)
    assert hinted.non_tensor_batch["hint_level"].tolist() == list(range(1, 13))
    assert hinted.non_tensor_batch["hint_stage"].tolist() == [1] * 4 + [2] * 4 + [3] * 4
    assert hinted.non_tensor_batch["difficulty_bucket"].tolist() == ["hard"] * 12
    assert "addition integers" in hinted.non_tensor_batch["raw_prompt"][1][1]["content"]
    assert "Planning Hints" in hinted.non_tensor_batch["raw_prompt"][4][1]["content"]
    assert "Solution Hints" in hinted.non_tensor_batch["raw_prompt"][8][1]["content"]


def test_find_failed_groups_and_select_minimal_successful_hint():
    original_uids = np.asarray(["a", "a", "b", "b"], dtype=object)
    original_rewards = torch.tensor([[0.0], [0.0], [0.0], [1.0]])
    representatives, stats = find_failed_group_representatives(original_uids, original_rewards)

    assert representatives == [0]
    assert stats.all_wrong == 1
    assert stats.any_correct == 1

    hint_uids = np.asarray(["a"] * 12, dtype=object)
    hint_levels = np.arange(1, 13)
    hint_rewards = torch.zeros((12, 1))
    hint_rewards[6] = 1
    hint_rewards[10] = 1
    selected, fully_failed, stage_counts = select_minimal_successful_hints(hint_uids, hint_levels, hint_rewards)

    assert selected == {"a": 6}
    assert fully_failed == set()
    assert stage_counts == {1: 0, 2: 1, 3: 0}


def test_replace_rollout_trajectory_can_keep_original_prompt():
    batch = DataProto.from_single_dict(
        {
            "prompts": torch.tensor([[0, 5, 6], [0, 5, 6]]),
            "responses": torch.tensor([[7, 0], [8, 0]]),
            "input_ids": torch.tensor([[0, 5, 6, 7, 0], [0, 5, 6, 8, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0]]),
            "position_ids": torch.tensor([[0, 0, 1, 2, 0], [0, 0, 1, 2, 0]]),
            "response_mask": torch.tensor([[1, 0], [1, 0]]),
            "uid": np.asarray(["a", "a"], dtype=object),
        }
    )
    hinted = DataProto.from_single_dict(
        {
            "prompts": torch.tensor([[0, 9, 9]]),
            "responses": torch.tensor([[4, 3]]),
            "input_ids": torch.tensor([[0, 9, 9, 4, 3]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1]]),
            "position_ids": torch.tensor([[0, 0, 1, 2, 3]]),
            "response_mask": torch.tensor([[1, 1]]),
            "uid": np.asarray(["a"], dtype=object),
        }
    )

    replaced = replace_rollout_trajectories(batch, hinted, {"a": 0}, replace_num=1, keep_original_prompt=True)

    assert replaced == [0]
    assert batch.batch["prompts"][0].tolist() == [0, 5, 6]
    assert batch.batch["responses"][0].tolist() == [4, 3]
    assert batch.batch["input_ids"][0].tolist() == [0, 5, 6, 4, 3]
    assert batch.batch["attention_mask"][0].tolist() == [0, 1, 1, 1, 1]


def test_build_hinted_gen_batch_rejects_missing_dataset_fields():
    batch = DataProto.from_single_dict({"dummy_tensor": torch.zeros((1, 1), dtype=torch.uint8)})
    with pytest.raises(KeyError, match="missing required fields"):
        build_hinted_gen_batch(batch)


def test_scaf_source_grad_norms_separate_rollout_and_hint_tokens():
    actor = object.__new__(DataParallelPPOActor)
    actor.config = _ActorConfig(
        clip_ratio=0.2,
        clip_ratio_low=None,
        clip_ratio_high=None,
        loss_agg_mode="token-mean",
        global_batch_info={},
    )
    log_prob = torch.tensor([[0.01, 0.02], [0.03, 0.04]], requires_grad=True)

    metrics = actor._compute_scaf_source_grad_norms(
        old_log_prob=torch.zeros_like(log_prob),
        log_prob=log_prob,
        advantages=torch.ones_like(log_prob),
        response_mask=torch.ones_like(log_prob),
        hint_source_mask=torch.tensor([[0, 0], [1, 1]]),
        loss_scale_factor=1.0,
    )

    assert metrics["actor/source_grad_norm/rollout"] > 0
    assert metrics["actor/source_grad_norm/hint"] > 0


def test_log_rollout_data_includes_difficulty_bucket(tmp_path):
    trainer = object.__new__(RayPPOTrainer)
    trainer.tokenizer = MagicMock()
    trainer.tokenizer.batch_decode.side_effect = [["question"], ["answer"]]
    trainer._dump_generations = MagicMock()
    batch = DataProto.from_single_dict(
        {
            "prompts": torch.tensor([[1, 2]]),
            "responses": torch.tensor([[3, 4]]),
            "token_level_scores": torch.tensor([[0.0, 1.0]]),
            "reward_model": np.asarray([{"ground_truth": "42"}], dtype=object),
            "difficulty_bucket": np.asarray(["hard"], dtype=object),
            "trajectory_source": np.asarray(["expert"], dtype=object),
        }
    )

    trainer._log_rollout_data(batch, {}, {}, str(tmp_path))

    dump_kwargs = trainer._dump_generations.call_args.kwargs
    assert dump_kwargs["reward_extra_infos_dict"]["difficulty_bucket"] == ["hard"]
    assert dump_kwargs["reward_extra_infos_dict"]["trajectory_source"] == ["expert"]


def test_inject_expert_trajectory_replaces_one_group_member():
    tokenizer = MagicMock(pad_token_id=0, eos_token_id=2)
    tokenizer.return_value = {"input_ids": [7, 8]}
    batch = DataProto.from_single_dict(
        {
            "prompts": torch.tensor([[0, 5], [0, 5], [0, 6]]),
            "responses": torch.tensor([[3, 0, 0, 0], [4, 0, 0, 0], [9, 0, 0, 0]]),
            "input_ids": torch.tensor([[0, 5, 3, 0, 0, 0], [0, 5, 4, 0, 0, 0], [0, 6, 9, 0, 0, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0]]),
            "position_ids": torch.tensor([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]]),
            "response_mask": torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
            "uid": np.asarray(["a", "a", "b"], dtype=object),
            "qwen_expert_trajectory": np.asarray(["expert a", "expert a", "expert b"], dtype=object),
        }
    )

    injected = inject_expert_trajectories(batch, {"a"}, tokenizer)

    assert injected == [0]
    assert batch.batch["responses"][0].tolist() == [7, 8, 2, 0]
    assert batch.batch["luffy_expert_mask"].tolist() == [
        [True, True, True, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
    assert batch.non_tensor_batch["trajectory_source"].tolist() == ["expert", "rollout", "rollout"]


def _expert_mode_batch():
    return DataProto.from_single_dict(
        {
            "prompts": torch.tensor([[0, 5], [0, 5], [0, 6], [0, 6]]),
            "responses": torch.tensor([[3, 0, 0, 0], [4, 0, 0, 0], [9, 0, 0, 0], [10, 0, 0, 0]]),
            "input_ids": torch.tensor(
                [
                    [0, 5, 3, 0, 0, 0],
                    [0, 5, 4, 0, 0, 0],
                    [0, 6, 9, 0, 0, 0],
                    [0, 6, 10, 0, 0, 0],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                ]
            ),
            "position_ids": torch.tensor(
                [
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                ]
            ),
            "response_mask": torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
            "uid": np.asarray(["failed", "failed", "solved", "solved"], dtype=object),
            "qwen_expert_trajectory": np.asarray(
                ["expert failed", "expert failed", "expert solved", "expert solved"], dtype=object
            ),
        }
    )


def _expert_mode_trainer(*, every_group: bool):
    trainer = object.__new__(RayPPOTrainer)
    trainer.with_hint = False
    trainer.with_luffy_expert = True
    trainer.luffy_expert_every_group = every_group
    trainer.luffy_expert_key = "qwen_expert_trajectory"
    trainer.global_steps = 0
    trainer.warmup_steps = 50
    trainer.replace_num = 1
    trainer.replace_hint_prompt = False
    trainer.scaf_hint_stages = 3
    trainer.use_rm = False
    trainer.reward_fn = MagicMock()
    trainer.tokenizer = MagicMock(pad_token_id=0, eos_token_id=2)
    trainer.tokenizer.return_value = {"input_ids": [7, 8]}
    trainer._generate_scaf_hints = MagicMock()
    return trainer


def _score_injected_experts(expert_batch, _reward_fn):
    return torch.ones_like(expert_batch.batch["response_mask"], dtype=torch.float32), {}


def test_expert_only_mode_injects_only_pass_at_k_zero_group():
    trainer = _expert_mode_trainer(every_group=False)
    batch = _expert_mode_batch()
    initial_rewards = torch.zeros((4, 4))
    initial_rewards[2, 0] = 1
    metrics = {}

    with patch("verl.trainer.ppo.ray_trainer.compute_reward", side_effect=_score_injected_experts):
        trainer._apply_scaf_grpo(batch, initial_rewards, {}, metrics)

    assert batch.non_tensor_batch["trajectory_source"].tolist() == ["expert", "rollout", "rollout", "rollout"]
    assert metrics["batch/luffy_expert_injected_uid_count"] == 1
    assert metrics["batch/luffy_expert_target_uid_count"] == 1
    trainer._generate_scaf_hints.assert_not_called()


def test_every_group_mode_injects_one_expert_per_group():
    trainer = _expert_mode_trainer(every_group=True)
    trainer.with_hint = True
    batch = _expert_mode_batch()
    initial_rewards = torch.zeros((4, 4))
    initial_rewards[2, 0] = 1
    metrics = {}

    with patch("verl.trainer.ppo.ray_trainer.compute_reward", side_effect=_score_injected_experts):
        trainer._apply_scaf_grpo(batch, initial_rewards, {}, metrics)

    assert batch.non_tensor_batch["trajectory_source"].tolist() == ["expert", "rollout", "expert", "rollout"]
    assert metrics["batch/luffy_expert_injected_uid_count"] == 2
    assert metrics["batch/luffy_expert_target_uid_count"] == 2
    assert metrics["batch/luffy_expert_every_group"] == 1.0
    trainer._generate_scaf_hints.assert_not_called()


def test_get_gen_batch_keeps_expert_trajectory_on_training_batch():
    trainer = object.__new__(RayPPOTrainer)
    trainer.with_luffy_expert = True
    trainer.luffy_expert_key = "qwen_expert_trajectory"
    trainer.async_rollout_mode = False
    batch = DataProto.from_single_dict(
        {
            "dummy_tensor": torch.zeros((2, 1), dtype=torch.uint8),
            "data_source": np.asarray(["math", "math"], dtype=object),
            "qwen_expert_trajectory": np.asarray(["expert a", "expert b"], dtype=object),
            "raw_prompt": np.asarray(
                [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]], dtype=object
            ),
        }
    )

    gen_batch = trainer._get_gen_batch(batch)

    assert batch.non_tensor_batch["qwen_expert_trajectory"].tolist() == ["expert a", "expert b"]
    assert "qwen_expert_trajectory" not in gen_batch.non_tensor_batch
    assert "raw_prompt" in gen_batch.non_tensor_batch


def _load_luffy_source_loss():
    source_file = Path(__file__).resolve().parents[4] / "LUFFY-source/luffy/verl/verl/mix_src/mix_core_alg.py"
    if not source_file.exists():
        pytest.skip("LUFFY-source is not available beside verl070-tmp")
    spec = importlib.util.spec_from_file_location("luffy_source_mix_core_alg", source_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_token_on_off_policy_loss


def test_luffy_token_loss_matches_source_and_gradient():
    source_loss = _load_luffy_source_loss()
    old_log_prob = torch.tensor([[-2.0, -1.0, -0.5], [-1.5, -2.5, -3.0]], dtype=torch.float64)
    target_log_prob = (old_log_prob + torch.tensor([[0.3, -0.4, 0.2], [-0.2, 0.1, 0.4]])).requires_grad_()
    source_log_prob = target_log_prob.detach().clone().requires_grad_()
    advantages = torch.tensor([[1.25, 1.25, 1.25], [-0.75, -0.75, -0.75]], dtype=torch.float64)
    eos_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    prefix_mask = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)
    kwargs = {
        "old_log_prob": old_log_prob,
        "advantages": advantages,
        "eos_mask": eos_mask,
        "cliprange": 0.2,
        "clip_upper_bound": 1.0,
        "prefix_mask": prefix_mask,
        "off_cliprange": 0.2,
        "off_normalize": False,
        "off_max_clip": None,
        "off_min_clip": None,
        "all_max_clip": None,
        "off_policy_reshape": "p_div_p_0.1",
        "off_policy_reshape_weight": 0.1,
        "off_policy_reshape_pow_exp": 0.5,
        "on_policy_reshape": "no_reshape",
        "on_policy_reshape_weight": 0.1,
        "on_policy_reshape_pow_exp": 0.5,
        "target_probs": None,
        "loss_remove_token_mean": True,
        "loss_remove_clip": True,
    }

    actual = compute_token_on_off_policy_loss(log_prob=target_log_prob, **kwargs)
    expected = source_loss(log_prob=source_log_prob, **kwargs)

    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.allclose(actual[key], expected[key]), key

    actual["pg_loss"].backward()
    expected["pg_loss"].backward()
    assert torch.allclose(target_log_prob.grad, source_log_prob.grad)


def test_luffy_expert_probability_uses_released_p_div_p_point_one_transform():
    log_prob = torch.log(torch.tensor([[0.01]], dtype=torch.float64)).requires_grad_()
    output = compute_token_on_off_policy_loss(
        old_log_prob=log_prob.detach(),
        log_prob=log_prob,
        advantages=torch.ones_like(log_prob),
        eos_mask=torch.ones_like(log_prob, dtype=torch.bool),
        cliprange=0.2,
        clip_upper_bound=1.0,
        prefix_mask=torch.ones_like(log_prob, dtype=torch.bool),
        off_cliprange=0.2,
        off_policy_reshape="p_div_p_0.1",
        loss_remove_token_mean=True,
        loss_remove_clip=True,
    )

    assert output["pg_loss"].item() == pytest.approx(-(0.01 / 0.11))
    output["pg_loss"].backward()
    assert log_prob.grad.item() < 0


def test_luffy_dynamic_token_sum_loss_uses_declared_micro_batch_scaling():
    config = _ActorConfig(
        use_dynamic_bsz=True,
        use_off_policy_loss=True,
        loss_remove_token_mean=True,
        ppo_mini_batch_size=32,
    )

    scale = _compute_micro_batch_loss_scale(config, micro_batch_size=6, gradient_accumulation=32)

    assert scale == pytest.approx(1 / 32)
