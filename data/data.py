#!/usr/bin/env python3
"""
Rewrite math SFT/expert trajectories with the Mind the Gap data rewriting recipe.

For each problem:
1. self-alignment: sample from the target model without a reference solution.
2. guided-alignment: if self-alignment fails, sample digest-and-retell answers
   conditioned on the expert solution.
3. fallback: if guided-alignment fails, keep the original expert trajectory.

The default target column is qwen_expert_trajectory so the rewritten parquet can
be used by the existing Scaf/LUFFY config without changing trainer.luffy_expert_key.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SELF_PROMPT = """Please reason step by step, and put your final answer within \\boxed{{}}.

Problem:
{question}
"""


GUIDED_PROMPT = """You are given a math problem.
If you cannot solve it directly, you are also given a teacher's detailed solution and final answer.
Read and learn from it, then try to solve the problem again in your own way, not by copying.

Instructions:
1. First try to understand the problem.
2. Use the teacher's solution only as guidance if you get stuck.
3. Explain the reasoning in your own words, step by step.
4. Conclude with the correct final answer.

Problem:
{question}

Teacher's Solution (for reference):
{solution}

Now, solve the problem in your own way:
"""


def load_model(model_path: str, dtype: str = "bfloat16"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def build_chat_prompt(tokenizer, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return content


@torch.no_grad()
def rollout(
    question: str,
    tokenizer,
    model,
    prompt_template: str,
    solution: str | None = None,
    num_samples: int = 10,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[str]:
    if solution is None:
        prompt = prompt_template.format(question=question)
    else:
        prompt = prompt_template.format(question=question, solution=solution)

    text = build_chat_prompt(tokenizer, prompt)
    inputs = tokenizer([text], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_samples,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[-1]
    return [
        tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
        for output in outputs
    ]


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def get_ground_truth(row, gt_column: str | None = None) -> str:
    if gt_column:
        value = row.get(gt_column)
        if not _is_missing(value):
            return str(value)

    reward_model = _as_dict(row.get("reward_model"))
    if reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])

    extra_info = _as_dict(row.get("extra_info"))
    for key in ("gt_answer", "ground_truth", "answer"):
        if extra_info.get(key) is not None:
            return str(extra_info[key])

    return str(row.get("solution", ""))


def make_verifier(fast: bool = False):
    try:
        from recipe.entropy.reward_score.entropy_math import compute_score

        def verify_with_entropy_math(response: str, ground_truth: str) -> bool:
            return bool(compute_score(response, ground_truth, fast=fast).get("acc", False))

        return verify_with_entropy_math
    except Exception:
        pass

    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

        extraction_config = [
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
        ]

        def verify_with_math_verify(response: str, ground_truth: str) -> bool:
            return bool(
                verify(
                    parse(ground_truth, extraction_config=extraction_config),
                    parse(response, extraction_config=extraction_config),
                )
            )

        return verify_with_math_verify
    except Exception as exc:
        raise RuntimeError(
            "No math verifier is available. Install math-verify/mathruler dependencies "
            "or run inside the verl training environment."
        ) from exc


def choose_correct(
    outputs: list[str],
    ground_truth: str,
    verifier,
    rng: random.Random,
) -> tuple[str | None, list[float]]:
    correct = []
    scores = []
    for output in outputs:
        is_correct = verifier(output, ground_truth)
        scores.append(1.0 if is_correct else 0.0)
        if is_correct:
            correct.append(output)
    return (rng.choice(correct) if correct else None), scores


def get_expert(row, expert_column: str) -> str:
    expert = row.get(expert_column)
    if _is_missing(expert) or str(expert).strip() == "":
        expert = row.get("solution")
    if _is_missing(expert):
        raise KeyError(f"Row has neither {expert_column!r} nor 'solution'.")
    return str(expert)


def rewrite_dataset(
    df,
    tokenizer,
    model,
    *,
    expert_column: str,
    target_column: str,
    gt_column: str | None,
    samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    fast_verify: bool,
    start: int,
    end: int | None,
):
    verifier = make_verifier(fast=fast_verify)
    rng = random.Random(seed)
    statistics = Counter()
    rows = []

    stop = len(df) if end is None else min(end, len(df))
    work_df = df.iloc[start:stop]

    for idx, row in tqdm(work_df.iterrows(), total=len(work_df), desc="Mind the Gap rewriting"):
        question = str(row["question"])
        expert = get_expert(row, expert_column)
        ground_truth = get_ground_truth(row, gt_column)

        self_outputs = rollout(
            question,
            tokenizer,
            model,
            SELF_PROMPT,
            num_samples=samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        rewritten, self_scores = choose_correct(self_outputs, ground_truth, verifier, rng)
        guided_scores = []

        if rewritten is not None:
            stage = "self"
        else:
            guided_outputs = rollout(
                question,
                tokenizer,
                model,
                GUIDED_PROMPT,
                solution=expert,
                num_samples=samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            rewritten, guided_scores = choose_correct(guided_outputs, ground_truth, verifier, rng)
            if rewritten is not None:
                stage = "guided"
            else:
                rewritten = expert
                stage = "fallback"

        new_row = row.copy()
        if f"original_{target_column}" not in new_row:
            new_row[f"original_{target_column}"] = row.get(target_column)
        new_row[target_column] = rewritten
        new_row["mind_the_gap_stage"] = stage
        new_row["mind_the_gap_ground_truth"] = ground_truth
        new_row["mind_the_gap_self_reward_list"] = self_scores
        new_row["mind_the_gap_guided_reward_list"] = guided_scores
        new_row["mind_the_gap_num_self_correct"] = int(sum(self_scores))
        new_row["mind_the_gap_num_guided_correct"] = int(sum(guided_scores))
        if tokenizer is not None:
            new_row[f"{target_column}_token_count"] = len(tokenizer.encode(rewritten, add_special_tokens=False))
        score_column = f"{target_column}_math_verify_score"
        if stage == "fallback":
            existing_score = row.get(score_column)
            new_row[score_column] = existing_score if not _is_missing(existing_score) else float(verifier(rewritten, ground_truth))
        else:
            new_row[score_column] = 1.0
        rows.append(new_row)
        statistics[stage] += 1

    return df.__class__(rows), statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mind the Gap data rewriting for math parquet files.")
    parser.add_argument("--input", default="data/DeepScaler/Qwen2d5_math_7b/train_800.success_rate_k8.right.parquet")
    parser.add_argument("--output", default="data/DeepScaler/Qwen2d5_math_7b/train_800.mind_the_gap.parquet")
    parser.add_argument("--model", default="/workplace/nankai/liting_space/LLM/Qwen2.5-Math-7B")
    parser.add_argument("--expert-column", default="qwen_expert_trajectory")
    parser.add_argument("--target-column", default="qwen_expert_trajectory")
    parser.add_argument("--gt-column", default=None)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--fast-verify", action="store_true", help="Skip slower symbolic fallback in entropy_math verifier.")
    parser.add_argument("--start", type=int, default=0, help="Inclusive row offset for sharding/debugging.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive row offset for sharding/debugging.")
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    tokenizer, model = load_model(args.model, dtype=args.dtype)
    rewritten_df, statistics = rewrite_dataset(
        df,
        tokenizer,
        model,
        expert_column=args.expert_column,
        target_column=args.target_column,
        gt_column=args.gt_column,
        samples=args.samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        fast_verify=args.fast_verify,
        start=args.start,
        end=args.end,
    )
    rewritten_df.to_parquet(output_path, index=False)
    print(f"Wrote {len(rewritten_df)} rows to {output_path}")
    print(f"Stage counts: {dict(statistics)}")


if __name__ == "__main__":
    main()
