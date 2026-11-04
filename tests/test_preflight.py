"""CPU checks for the preflight's coverage and failure behavior."""
import pytest
import torch

from experiments.config import RUNS, model_for
from experiments.preflight import compare_logits, unique_variants


def test_preflight_covers_every_run_and_all_scientific_variants():
    variants = unique_variants()
    covered = [run_id for item in variants for run_id in item["run_ids"]]
    assert sorted(covered) == sorted(RUNS)
    baseline = next(item for item in variants if "d8-b0-s1" in item["run_ids"])
    assert set(baseline["run_ids"]) == {"d8-b0-s1", "d8-b0-s2", "d8-b0-s3"}
    for run_id in ("d8-gqa2", "d8-gqa2-r75", "d8-gqa2-r25", "d8-gqa2-pm", "d8-gqa4"):
        selected = [item for item in variants if run_id in item["run_ids"]]
        assert len(selected) == 1
        assert selected[0]["run_ids"] == [run_id]
        assert selected[0]["model"] == model_for(run_id)


def test_preflight_accepts_small_bf16_rounding_and_reports_error():
    reference = torch.tensor([[[0.0, 1.0, -2.0]]])
    actual = reference + 0.005
    stats = compare_logits(actual, reference)
    assert stats["max_absolute_error"] == pytest.approx(0.005, abs=1e-6)
    assert stats["rms_error"] == pytest.approx(0.005, abs=1e-6)


@pytest.mark.parametrize("actual", [torch.ones(1, 1, 3), torch.zeros(1, 3),
                                  torch.full((1, 1, 3), float("nan")),
                                  torch.full((1, 1, 3), float("inf"))])
def test_preflight_rejects_wrong_logits_shape_or_nonfinite(actual):
    with pytest.raises(AssertionError):
        compare_logits(actual, torch.zeros(1, 1, 3))
