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


import logging
import os

try:
    from math_verify import parse, verify
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def _configure_math_verify_logging() -> None:
    """Use the same default logging policy as the local LUFFY-source runner."""
    log_level = os.getenv("LUFFY_MATH_VERIFY_LOG_LEVEL", "CRITICAL").upper()
    level = getattr(logging, log_level, logging.CRITICAL)
    logging.getLogger("math_verify").setLevel(level)
    logging.getLogger("math_verify.grader").setLevel(level)


# Match LUFFY-source reward_impl_version=4 exactly: parse the full response,
# parse the dollar-delimited reference answer, then let math-verify compare all
# extracted candidates with its own timeout and fallback behavior.
def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    del timeout_score  # Kept only for compatibility with the verl reward API.
    _configure_math_verify_logging()
    predicted_answers = parse(model_output)
    golden_answers = parse("$" + ground_truth + "$")
    return bool(verify(golden_answers, predicted_answers))
