# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============== math-verify 0.9.0 in verl 0.7.0 =================
# try:
#     from math_verify.errors import TimeoutException
#     from math_verify.metric import math_metric
#     from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
# except ImportError:
#     print("To use Math-Verify, please install it first by running `pip install math-verify`.")

# def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
#     verify_func = math_metric(
#         gold_extraction_target=(LatexExtractionConfig(),),
#         pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
#     )
#     ret_score = 0.0
#     # Wrap the ground truth in \boxed{} format for verification
#     ground_truth_boxed = "\\boxed{" + ground_truth + "}"
#     try:
#         ret_score, _ = verify_func([ground_truth_boxed], [model_output])
#     except Exception:
#         pass
#     except TimeoutException:
#         ret_score = timeout_score
#     return ret_score
# =============== math-verify 0.9.0 in verl 0.7.0 =================


import re
import unicodedata

try:
    from math_verify import parse, verify
    from sympy import Basic, Integral, Limit, Pow, Product, Sum
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


_EXPENSIVE_POWER_RE = re.compile(r"[\)\]][\s]*(?:\*\*|\^)[\s]*\{?\d{2,}\}?")


def _normalize_unicode_digits(text: str) -> str:
    normalized = []
    for char in text:
        try:
            normalized.append(str(unicodedata.decimal(char)))
        except (TypeError, ValueError):
            normalized.append(char)
    return "".join(normalized)


def _has_expensive_power(answer: Basic) -> bool:
    for power in answer.atoms(Pow):
        exponent = power.exp
        if exponent.is_integer and exponent.is_number and abs(int(exponent)) >= 64 and not power.base.is_Atom:
            return True
    return False


def _is_verifiable_answer(answer) -> bool:
    """Return whether an extracted answer can trigger expensive SymPy evaluation.

    Training prompts require a simplified final answer.  An unevaluated sum,
    product, integral, limit, or large compound-base power is therefore not a
    valid final answer, and passing one to math-verify can make SymPy spend the
    full comparison timeout trying to evaluate it.
    """
    if isinstance(answer, str):
        return _EXPENSIVE_POWER_RE.search(answer) is None
    return isinstance(answer, Basic) and not answer.has(Sum, Product, Integral, Limit) and not _has_expensive_power(answer)


# [ADD] [REWARD] 为了和Luffy的0.6.0以及reward_impl_version=4对齐
def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    try:
        model_output = _normalize_unicode_digits(model_output)
        ground_truth = _normalize_unicode_digits(ground_truth)

        parsed_answers = parse(model_output)
        predicted_answers = [answer for answer in parsed_answers if _is_verifiable_answer(answer)]
        if not predicted_answers:
            return bool(timeout_score)

        golden_answers = parse("$" + ground_truth + "$")
        return bool(verify(golden_answers, predicted_answers, timeout_seconds=1))
    except Exception:
        return bool(timeout_score)
