from unittest.mock import patch

from sympy import I, Sum, oo, symbols

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
    verify.assert_called_once_with(["gold"], ["prediction"], timeout_seconds=1)


def test_math_verify_rejects_unevaluated_sum_without_verifying():
    n = symbols("n", integer=True, positive=True)

    with (
        patch.object(math_verify, "parse", return_value=[Sum(1 / n, (n, 1, oo))]),
        patch.object(math_verify, "verify") as verify,
    ):
        result = math_verify.compute_score("An unevaluated infinite sum", "1")

    assert result is False
    verify.assert_not_called()


def test_math_verify_keeps_safe_extractions_when_filtering_unevaluated_sum():
    n = symbols("n", integer=True, positive=True)
    unsafe_answer = Sum(1 / n, (n, 1, oo))

    with (
        patch.object(math_verify, "parse", side_effect=[[unsafe_answer, "prediction"], ["gold"]]),
        patch.object(math_verify, "verify", return_value=True) as verify,
    ):
        result = math_verify.compute_score("The answer is \\boxed{2}.", "2")

    assert result is True
    verify.assert_called_once_with(["gold"], ["prediction"], timeout_seconds=1)


def test_math_verify_rejects_large_compound_power_without_verifying():
    unsafe_answer = -((1 - I) ** 2016) + (1 + I) ** 2016

    with (
        patch.object(math_verify, "parse", return_value=[unsafe_answer, "-(1 - I)^{2016} + (1 + I)^{2016}"]),
        patch.object(math_verify, "verify") as verify,
    ):
        result = math_verify.compute_score("An unsimplified high power", "14")

    assert result is False
    verify.assert_not_called()


def test_math_verify_normalizes_unicode_digits_before_parsing():
    with (
        patch.object(math_verify, "parse", side_effect=[["111"], ["111"]]) as parse,
        patch.object(math_verify, "verify", return_value=True) as verify,
    ):
        result = math_verify.compute_score("The answer is \\boxed{๑๑๑}.", "๑๑๑")

    assert result is True
    assert parse.call_args_list[0].args == ("The answer is \\boxed{111}.",)
    assert parse.call_args_list[1].args == ("$111$",)
    verify.assert_called_once_with(["111"], ["111"], timeout_seconds=1)
