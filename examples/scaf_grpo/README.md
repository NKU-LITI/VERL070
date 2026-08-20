# Scaf-GRPO on verl 0.7

This directory contains the verl 0.7 entry point for the Scaf-GRPO algorithm
migrated from `Scaf-GRPO-source` (verl 0.4.1.dev).

The training parquet must contain the normal verl RL fields plus:

- `question`
- `knowledge_components_parts`
- `planning_skeleton_parts`
- `solution_breakdown_parts`
- `qwen_expert_trajectory`

Each hint-parts field should contain four progressively more informative text
segments. When every rollout for a problem receives zero reward, the trainer
generates all enabled hint levels and injects the successful trajectory with
the lowest hint level back into that GRPO group.

When every hint level also receives zero reward and
`trainer.with_luffy_expert=true`, the trainer replaces one rollout in that
group with `qwen_expert_trajectory`. As in LUFFY's released `expert.sh`, the
normal GRPO group baseline includes that trajectory and standard-deviation
normalization is disabled. Expert response tokens use LUFFY's released
token-level off-policy objective with `off_policy_reshape=p_div_p_0.1`, while
the other response tokens retain the on-policy term. The loss uses
`sum(masked_loss) / max_response_length`, disables on-policy clipping, and
applies entropy coefficient `0.001`. Saved rollout JSONL records the selected
source as `trajectory_source=rollout|hint|expert`.

Expert injection supports three modes:

- `with_hint=true`, `with_luffy_expert=true`: inject after every hint level fails.
- `with_hint=false`, `with_luffy_expert=true`: inject when the original group has `pass@k=0`.
- `with_hint=false`, `with_luffy_expert=true`, `luffy_expert_every_group=true`:
  replace one rollout in every group. This matches LUFFY's training composition,
  with replacement used instead of prefix-controlled expert injection. The
  every-group option takes precedence and bypasses hint generation if both are enabled.

Set `MODEL_PATH`, `TRAIN_FILE`, and `VAL_FILE`, then run:

```bash
bash examples/scaf_grpo/run_qwen2_5_math_7b.sh
```

The example uses the process-based agent-loop reward manager required by
`math-verify`. Keep `reward_manager.name=naive` for the trainer and set
`reward_model.reward_manager=remote` for the agent loop.
