from unittest.mock import patch

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_verify


def test_scaf_math_verify_data_source_uses_math_verify():
    data_source = "deepscaler-clean-39k_except-still/math-verify"

    with patch("verl.utils.reward_score.math_verify.compute_score", return_value=True) as compute_score:
        result = default_compute_score(
            data_source=data_source,
            solution_str="The answer is \\boxed{2}.",
            ground_truth="2",
        )

    assert result == 1.0
    compute_score.assert_called_once_with("The answer is \\boxed{2}.", "2")


def test_math_verify_uses_luffy_parse_and_verify_flow():
    with (
        patch.object(math_verify, "parse", side_effect=[["prediction"], ["gold"]]) as parse,
        patch.object(math_verify, "verify", return_value=True) as verify,
    ):
        result = math_verify.compute_score("The answer is \\boxed{2}.", "2")

    assert result is True
    assert parse.call_args_list[0].args == ("The answer is \\boxed{2}.",)
    assert parse.call_args_list[1].args == ("$2$",)
    verify.assert_called_once_with(["gold"], ["prediction"])
