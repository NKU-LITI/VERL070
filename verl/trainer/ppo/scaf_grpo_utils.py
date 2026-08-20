# Copyright 2026 Scaf-GRPO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Utilities for constructing and selecting Scaf-GRPO hinted rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto

HINT_STAGES = (
    ("knowledge_components_parts", "Knowledge Hints"),
    ("planning_skeleton_parts", "Planning Hints"),
    ("solution_breakdown_parts", "Solution Hints"),
)

DEFAULT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


@dataclass(frozen=True)
class RewardGroupStats:
    """Outcome counts for the original rollout groups."""

    total: int
    all_wrong: int
    all_correct: int
    any_correct: int


def _normalise_hint_parts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(part).strip() for part in value if part is not None and str(part).strip()]


def find_failed_group_representatives(
    uids: np.ndarray, token_level_rewards: torch.Tensor
) -> tuple[list[int], RewardGroupStats]:
    """Return one row for every UID whose complete rollout group received zero reward."""
    sequence_rewards = token_level_rewards.sum(dim=-1).detach().cpu().numpy()
    representative_indices: list[int] = []
    all_correct = 0
    any_correct = 0

    unique_uids = np.unique(uids)
    for uid in unique_uids:
        indices = np.flatnonzero(uids == uid)
        rewards = sequence_rewards[indices]
        if np.all(rewards <= 0):
            representative_indices.append(int(indices[0]))
        else:
            any_correct += 1
            if np.all(rewards > 0):
                all_correct += 1

    return representative_indices, RewardGroupStats(
        total=len(unique_uids),
        all_wrong=len(representative_indices),
        all_correct=all_correct,
        any_correct=any_correct,
    )


def build_hinted_gen_batch(
    base_batch: DataProto,
    *,
    stage_count: int = 3,
    hints_per_stage: int = 4,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> DataProto:
    """Expand one row per failed problem into progressive hierarchical hint prompts.

    verl 0.7 applies the chat template inside the agent loop, so this function
    deliberately emits message lists instead of the already-tokenized prompts
    used by the verl 0.4.1 implementation.
    """
    if not 1 <= stage_count <= len(HINT_STAGES):
        raise ValueError(f"stage_count must be between 1 and {len(HINT_STAGES)}, got {stage_count}")
    if hints_per_stage < 1:
        raise ValueError(f"hints_per_stage must be positive, got {hints_per_stage}")

    required_keys = {"question", "uid", "reward_model", "data_source"}
    missing = required_keys.difference(base_batch.non_tensor_batch)
    if missing:
        raise KeyError(f"Scaf-GRPO dataset is missing required fields: {sorted(missing)}")

    questions = base_batch.non_tensor_batch["question"]
    original_uids = base_batch.non_tensor_batch["uid"]
    reward_models = base_batch.non_tensor_batch["reward_model"]
    data_sources = base_batch.non_tensor_batch["data_source"]

    raw_prompts: list[list[dict[str, str]]] = []
    uids: list[Any] = []
    hint_levels: list[int] = []
    hint_stages: list[int] = []
    reward_model_out: list[Any] = []
    data_source_out: list[Any] = []
    difficulty_out: list[Any] = []
    difficulties = base_batch.non_tensor_batch.get("difficulty_bucket")

    for row_idx, question in enumerate(questions):
        hint_level = 1
        for stage_idx, (stage_key, hint_label) in enumerate(HINT_STAGES[:stage_count], start=1):
            stage_values = base_batch.non_tensor_batch.get(stage_key)
            parts = _normalise_hint_parts(stage_values[row_idx] if stage_values is not None else None)
            for hint_count in range(1, hints_per_stage + 1):
                combined_hints = " ".join(parts[:hint_count])
                raw_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Question: {question}\n{hint_label}: {combined_hints}"},
                    ]
                )
                uids.append(original_uids[row_idx])
                hint_levels.append(hint_level)
                hint_stages.append(stage_idx)
                reward_model_out.append(reward_models[row_idx])
                data_source_out.append(data_sources[row_idx])
                if difficulties is not None:
                    difficulty_out.append(difficulties[row_idx])
                hint_level += 1

    data = {
        # DataProto still requires a non-empty TensorDict in verl 0.7.
        "dummy_tensor": torch.zeros((len(raw_prompts), 1), dtype=torch.uint8),
        "raw_prompt": np.fromiter(raw_prompts, dtype=object, count=len(raw_prompts)),
        "uid": np.asarray(uids, dtype=object),
        "hint_level": np.asarray(hint_levels),
        "hint_stage": np.asarray(hint_stages),
        "reward_model": np.asarray(reward_model_out, dtype=object),
        "data_source": np.asarray(data_source_out, dtype=object),
    }
    if difficulties is not None:
        data["difficulty_bucket"] = np.asarray(difficulty_out, dtype=object)
    return DataProto.from_single_dict(data, meta_info={"new_gen": True})


def select_minimal_successful_hints(
    uids: np.ndarray, hint_levels: np.ndarray, token_level_rewards: torch.Tensor
) -> tuple[dict[Any, int], set[Any], dict[int, int]]:
    """Select the lowest successful hint level for each UID."""
    sequence_rewards = token_level_rewards.sum(dim=-1).detach().cpu().numpy()
    selected: dict[Any, int] = {}
    fully_failed: set[Any] = set()
    selected_stage_counts = {1: 0, 2: 0, 3: 0}

    for uid in np.unique(uids):
        indices = np.flatnonzero(uids == uid)
        successful = indices[sequence_rewards[indices] > 0]
        if len(successful) == 0:
            fully_failed.add(uid)
            continue
        selected_idx = int(successful[np.argmin(hint_levels[successful])])
        selected[uid] = selected_idx
        stage = (int(hint_levels[selected_idx]) - 1) // 4 + 1
        selected_stage_counts[stage] += 1

    return selected, fully_failed, selected_stage_counts


def replace_rollout_trajectories(
    batch: DataProto,
    hinted_batch: DataProto,
    selected_hints: dict[Any, int],
    *,
    replace_num: int = 1,
    keep_original_prompt: bool = False,
) -> list[int]:
    """Replace up to ``replace_num`` rollout rows per rescued UID in-place."""
    if replace_num < 1:
        raise ValueError(f"replace_num must be positive, got {replace_num}")

    replaced: list[int] = []
    replacement_counts: dict[Any, int] = {}
    for row_idx, uid in enumerate(batch.non_tensor_batch["uid"]):
        if uid not in selected_hints or replacement_counts.get(uid, 0) >= replace_num:
            continue

        hinted_idx = selected_hints[uid]
        replacement_counts[uid] = replacement_counts.get(uid, 0) + 1
        replaced.append(row_idx)

        if keep_original_prompt:
            responses = hinted_batch.batch["responses"][hinted_idx].to(batch.batch["responses"].device)
            response_mask = hinted_batch.batch["response_mask"][hinted_idx].to(batch.batch["response_mask"].device)
            prompt_length = batch.batch["prompts"].shape[-1]
            prompt_mask = batch.batch["attention_mask"][row_idx, :prompt_length]
            attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
            batch.batch["responses"][row_idx] = responses
            batch.batch["input_ids"][row_idx] = torch.cat((batch.batch["prompts"][row_idx], responses), dim=-1)
            batch.batch["attention_mask"][row_idx] = attention_mask
            batch.batch["position_ids"][row_idx] = torch.clamp(attention_mask.cumsum(dim=-1) - 1, min=0)
            batch.batch["response_mask"][row_idx] = response_mask
        else:
            for tensor_key in batch.batch.keys():
                if (
                    tensor_key in hinted_batch.batch
                    and batch.batch[tensor_key][row_idx].shape == hinted_batch.batch[tensor_key][hinted_idx].shape
                ):
                    batch.batch[tensor_key][row_idx] = hinted_batch.batch[tensor_key][hinted_idx].to(
                        batch.batch[tensor_key].device
                    )

    return replaced


def _extract_expert_text(value: Any) -> str:
    """Normalize the expert trajectory formats used by Scaf/LUFFY datasets."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("content", "")).strip()
    if isinstance(value, (list, tuple)):
        assistant_messages = [
            item.get("content", "") for item in value if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        if assistant_messages:
            return str(assistant_messages[-1]).strip()
        if len(value) == 1:
            return _extract_expert_text(value[0])
    return ""


def inject_expert_trajectories(
    batch: DataProto,
    expert_uids: set[Any],
    tokenizer: Any,
    *,
    expert_key: str = "qwen_expert_trajectory",
) -> list[int]:
    """Replace one rollout for each requested UID with a complete expert response.

    The prompt remains on-policy and only response tokens are marked as expert
    tokens. The actor uses this mask to apply LUFFY's off-policy loss.
    """
    expert_mask = torch.zeros_like(batch.batch["response_mask"], dtype=torch.bool)
    batch.batch["luffy_expert_mask"] = expert_mask
    if not expert_uids:
        return []
    if expert_key not in batch.non_tensor_batch:
        raise KeyError(f"LUFFY expert injection requires dataset field {expert_key!r}")
    if batch.batch["position_ids"].ndim != 2:
        raise ValueError("LUFFY expert injection currently supports text-only position_ids")

    response_length = batch.batch["responses"].shape[-1]
    if response_length < 1:
        raise ValueError("LUFFY expert injection requires a positive response length")

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None or eos_token_id is None:
        raise ValueError("LUFFY expert injection requires tokenizer pad_token_id and eos_token_id")

    trajectory_sources = batch.non_tensor_batch.get("trajectory_source")
    if trajectory_sources is None:
        trajectory_sources = np.asarray(["rollout"] * len(batch), dtype=object)
        batch.non_tensor_batch["trajectory_source"] = trajectory_sources

    injected: list[int] = []
    injected_uids: set[Any] = set()
    for row_idx, uid in enumerate(batch.non_tensor_batch["uid"]):
        if uid not in expert_uids or uid in injected_uids:
            continue

        expert_text = _extract_expert_text(batch.non_tensor_batch[expert_key][row_idx])
        if not expert_text:
            raise ValueError(f"Empty expert trajectory for uid={uid!r} in field {expert_key!r}")
        token_ids = tokenizer(expert_text, add_special_tokens=False)["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        token_ids = list(token_ids)
        if not token_ids or token_ids[-1] != eos_token_id:
            token_ids.append(eos_token_id)
        if len(token_ids) > response_length:
            raise ValueError(
                f"Expert trajectory for uid={uid!r} has {len(token_ids)} tokens, exceeding "
                f"data.max_response_length={response_length}"
            )

        response = torch.full_like(batch.batch["responses"][row_idx], pad_token_id)
        response[: len(token_ids)] = torch.as_tensor(token_ids, dtype=response.dtype, device=response.device)
        response_mask = torch.zeros_like(batch.batch["response_mask"][row_idx])
        response_mask[: len(token_ids)] = 1

        prompt = batch.batch["prompts"][row_idx]
        prompt_length = prompt.shape[-1]
        prompt_mask = batch.batch["attention_mask"][row_idx, :prompt_length]
        attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)

        batch.batch["responses"][row_idx] = response
        batch.batch["response_mask"][row_idx] = response_mask
        batch.batch["input_ids"][row_idx] = torch.cat((prompt, response), dim=-1)
        batch.batch["attention_mask"][row_idx] = attention_mask
        batch.batch["position_ids"][row_idx] = torch.clamp(attention_mask.cumsum(dim=-1) - 1, min=0)
        batch.batch["luffy_expert_mask"][row_idx] = response_mask.bool()
        if "rollout_log_probs" in batch.batch:
            batch.batch["rollout_log_probs"][row_idx].zero_()
        trajectory_sources[row_idx] = "expert"

        injected.append(row_idx)
        injected_uids.add(uid)

    missing_uids = expert_uids.difference(injected_uids)
    if missing_uids:
        raise ValueError(f"Failed to inject expert trajectories for UIDs: {sorted(map(str, missing_uids))}")
    return injected
