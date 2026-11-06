"""CPU checks for the preflight's coverage and failure behavior."""
import pytest
import torch

from experiments.config import RUNS, model_for
from experiments.preflight import GRAD_RTOL, compare_backward, compare_logits, unique_variants


def test_preflight_covers_every_run_and_all_scientific_variants():
    variants = unique_variants()
    covered = [run_id for item in variants for run_id in item["run_ids"]]
    assert sorted(covered) == sorted(RUNS)
    baseline = next(item for item in variants if "d8-b0-s1" in item["run_ids"])
    assert set(baseline["run_ids"]) == {"d8-b0-s1", "d8-b0-s2", "d8-b0-s3"}
    gqa2 = next(item for item in variants if "d8-gqa2-s1" in item["run_ids"])
    # Seeds and the matched-token ablation share one architecture check; only token targets differ.
    assert gqa2["run_ids"] == ["d8-gqa2-s1", "d8-gqa2-s2", "d8-gqa2-s3", "d8-gqa2-tok"]
    gqa4 = next(item for item in variants if "d8-gqa4" in item["run_ids"])
    assert gqa4["run_ids"] == ["d8-gqa4", "d8-gqa4-tok"]
    for run_id in ("d8-gqa2-r75", "d8-gqa2-r25", "d8-gqa2-pm", "d8-b0-nove", "d8-gqa2-nove"):
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


def _grads(scale=1.0):
    # "w" is matrix-sized (GRAD_RTOL); "b" is scalar-sized (SCALAR_GRAD_RTOL).
    return torch.tensor(2.0), {"w": torch.linspace(-4.0, 3.0, 65) * scale, "b": torch.tensor([1.0])}


def test_preflight_scalar_gradients_get_looser_tolerance():
    loss, grads = _grads()
    compare_backward((loss, {**grads, "b": grads["b"] * 1.10}), _grads())
    with pytest.raises(AssertionError):
        compare_backward((loss, {**grads, "b": grads["b"] * 1.20}), _grads())


def test_preflight_accepts_near_equal_gradients_and_reports_error():
    stats = compare_backward(_grads(1.01), _grads())
    assert stats["max_grad_relative_error"] == pytest.approx(0.01, rel=1e-4)
    assert stats["worst_parameter"] == "w"
    assert stats["loss_difference"] == 0.0


@pytest.mark.parametrize("actual", [
    _grads(1 + 2 * GRAD_RTOL),
    _grads(float("nan")),
    (torch.tensor(float("inf")), _grads()[1]),
    (torch.tensor(3.0), _grads()[1]),
    (torch.tensor(2.0), {"w": _grads()[1]["w"]}),
])
def test_preflight_rejects_gradient_mismatch_nonfinite_or_parameter_sets(actual):
    with pytest.raises(AssertionError):
        compare_backward(actual, _grads())
