"""Synthetic final summaries must not create invalid scientific comparisons."""
import pytest

from experiments.report import comparison_eligibility


def test_final_quality_requires_positive_evidence():
    assert not comparison_eligibility({})["eligible"]
    result = dict(stop_reason="token_target", tokens=100, target_tokens=100, data_epoch=1)
    assert comparison_eligibility(result, {"training": {"target_tokens": 100}})["eligible"]


@pytest.mark.parametrize("change", [
    {"stop_reason": "wallclock_cap"}, {"stop_reason": "pilot_step_cap"},
    {"tokens": 99}, {"tokens": 101}, {"target_tokens": 99}, {"data_epoch": 2},
])
def test_invalid_final_quality_is_excluded(change):
    result = dict(stop_reason="token_target", tokens=100, target_tokens=100, data_epoch=1)
    result.update(change)
    status = comparison_eligibility(result, {"training": {"target_tokens": 100}})
    assert not status["eligible"]
    assert status["reasons"]
