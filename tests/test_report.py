"""Synthetic final summaries must not create invalid scientific comparisons."""
import pytest

from experiments.report import comparison_eligibility


def test_final_quality_requires_positive_evidence():
    assert not comparison_eligibility({})["eligible"]
    result = dict(stop_reason="token_target", tokens=100, target_tokens=100, data_epoch=1, estimated_flops=1000)
    assert comparison_eligibility(result, {"training": {"target_tokens": 100, "flop_budget": 1000}})["eligible"]


@pytest.mark.parametrize("change", [
    {"stop_reason": "wallclock_cap"}, {"stop_reason": "pilot_step_cap"},
    {"tokens": 99}, {"tokens": 101}, {"target_tokens": 99}, {"data_epoch": 2},
    {"estimated_flops": 1001}, {"estimated_flops": None},
])
def test_invalid_final_quality_is_excluded(change):
    result = dict(stop_reason="token_target", tokens=100, target_tokens=100, data_epoch=1, estimated_flops=1000)
    result.update(change)
    status = comparison_eligibility(result, {"training": {"target_tokens": 100, "flop_budget": 1000}})
    assert not status["eligible"]
    assert status["reasons"]


def test_interaction_is_a_difference_of_differences_with_seed_aware_threshold():
    from experiments.report import interaction
    r = lambda name, bpb: dict(run_id=name, val_bpb=bpb, test_bpb=bpb, training_tokens_per_second=1,
                               parameters=1, scaling_parameters=1)
    results = {n: r(n, 1.0) for n in ("d8-b0-s1", "d8-b0-s2", "d8-b0-s3")}
    results |= {n: r(n, 1.03) for n in ("d8-gqa2-s1", "d8-gqa2-s2", "d8-gqa2-s3")}
    results |= {"d8-b0-nove": r("d8-b0-nove", 1.05), "d8-gqa2-nove": r("d8-gqa2-nove", 1.06)}
    out = interaction(results, "d8-gqa2", "d8-b0", "d8-gqa2-nove", "d8-b0-nove", std=0.001)
    assert out["effect_with"] == pytest.approx(0.03) and out["effect_without"] == pytest.approx(0.01)
    assert out["interaction"] == pytest.approx(0.02)
    assert out["threshold"] == pytest.approx(2 * 0.001 * (1 / 3 + 1 / 3 + 1 + 1) ** 0.5)
    assert out["resolved"] is True
    assert interaction({}, "d8-gqa2", "d8-b0", "d8-gqa2-nove", "d8-b0-nove", std=0.001) is None


def test_group_mean_requires_every_seed():
    from experiments.report import pooled
    r = lambda name, bpb: dict(run_id=name, val_bpb=bpb, test_bpb=bpb, training_tokens_per_second=1,
                               parameters=1, scaling_parameters=1)
    assert pooled({"d8-gqa2-s1": r("d8-gqa2-s1", 1.0)}, "d8-gqa2") is None  # incomplete group is not a mean
    three = pooled({n: r(n, v) for n, v in (("d8-gqa2-s1", 1.0), ("d8-gqa2-s2", 1.3), ("d8-gqa2-s3", 1.6))}, "d8-gqa2")
    assert three["members"] == ["d8-gqa2-s1", "d8-gqa2-s2", "d8-gqa2-s3"] and three["test_bpb"] == pytest.approx(1.3)


def test_means_rule_gains_resolution_from_extra_seeds():
    import math
    from experiments.report import SIGNIFICANCE_FACTOR, means_threshold, pair_entry
    s = 0.01
    assert means_threshold(s, 1, 1) == pytest.approx(SIGNIFICANCE_FACTOR * s)  # matches the single-run rule
    assert means_threshold(s, 3, 3) == pytest.approx(2 * s * math.sqrt(2 / 3))
    record = lambda members, bpb: dict(run_id="x", members=members, val_bpb=bpb, test_bpb=bpb, training_tokens_per_second=1)
    ref = record(["b1", "b2", "b3"], 1.0)
    # Gap 0.02: below 2*sqrt(2)*s = 0.0283 and the 1-vs-3 bound 0.0231, above the 3-vs-3 bound 0.0163.
    single = pair_entry(record(["g1"], 1.02), ref, SIGNIFICANCE_FACTOR * s, s)
    triple = pair_entry(record(["g1", "g2", "g3"], 1.02), ref, SIGNIFICANCE_FACTOR * s, s)
    assert (single["n_run"], single["n_ref"], triple["n_run"]) == (1, 3, 3)
    assert not single["resolved"] and not triple["resolved"]  # the pre-registered rule ignores seed counts
    assert not single["resolved_by_means"] and triple["resolved_by_means"]
    assert pair_entry(record(["g1"], 1.02), ref, None, None)["resolved_by_means"] is None


def test_pooled_t_sensitivity_analysis_with_hand_computed_numbers():
    from experiments.report import SIGNIFICANCE_FACTOR, interaction, pair_entry, pooled, pooled_noise, seed_noise
    r = lambda name, bpb: dict(run_id=name, val_bpb=bpb, test_bpb=bpb, training_tokens_per_second=1, parameters=1, scaling_parameters=1)
    results = {f"d8-b0-s{i}": r(f"d8-b0-s{i}", v) for i, v in zip((1, 2, 3), (1.00, 1.02, 0.98))}
    results |= {f"d8-gqa2-s{i}": r(f"d8-gqa2-s{i}", v) for i, v in zip((1, 2, 3), (1.03, 1.05, 1.07))}
    results |= {"d8-b0-nove": r("d8-b0-nove", 1.10), "d8-gqa2-nove": r("d8-gqa2-nove", 1.12)}
    # Each group: mean +- 0.02, sum of squares 0.0008; s_p = sqrt(0.0016 / (6 - 2)) = 0.02 on 4 df.
    pn = pooled_noise(results)
    assert pn["std"] == pytest.approx(0.02) and pn["df"] == 4 and pn["t_critical"] == 2.776 and pn["n"] == 6
    assert pn["members"] == {"d8-b0": ["d8-b0-s1", "d8-b0-s2", "d8-b0-s3"], "d8-gqa2": ["d8-gqa2-s1", "d8-gqa2-s2", "d8-gqa2-s3"]}
    s = seed_noise([1.00, 1.02, 0.98])["std"]
    entry = pair_entry(pooled(results, "d8-gqa2"), pooled(results, "d8-b0"), SIGNIFICANCE_FACTOR * s, s, pn)
    h = 2.776 * 0.02 * (2 / 3) ** 0.5  # 0.04533
    assert entry["test_bpb_delta"] == pytest.approx(0.05)
    assert entry["sensitivity"]["ci95"] == pytest.approx([0.05 - h, 0.05 + h])
    assert entry["sensitivity"]["resolved"] is True and entry["resolved"] is False  # 0.05 < 2*sqrt(2)*0.02 = 0.0566
    out = interaction(results, "d8-gqa2", "d8-b0", "d8-gqa2-nove", "d8-b0-nove", std=s, pn=pn)
    h = 2.776 * 0.02 * (1 / 3 + 1 / 3 + 1 + 1) ** 0.5  # 0.09066
    assert out["interaction"] == pytest.approx(0.03) and out["sensitivity"]["ci95"] == pytest.approx([0.03 - h, 0.03 + h])
    assert out["sensitivity"]["resolved"] is False
    # One seed per group gives df = 0: no pooled estimate, no sensitivity entry.
    assert pooled_noise({"d8-b0-s1": results["d8-b0-s1"], "d8-gqa2-s1": results["d8-gqa2-s1"]}) is None
    assert pair_entry(pooled(results, "d8-gqa2"), pooled(results, "d8-b0"), None, None, None)["sensitivity"] is None


def test_final_validation_rows_are_never_paired_with_interim_rows():
    from experiments.report import matched_observations
    rows = [{"kind": "validation", "tokens": 10, "val_bpb": 1.0}, {"kind": "validation", "final": True, "tokens": 20, "val_bpb": 0.9}]
    ref = [{"kind": "validation", "tokens": 10, "val_bpb": 1.1}, {"kind": "validation", "tokens": 20, "val_bpb": 0.95}]
    assert matched_observations(rows, ref, "tokens") == [{"tokens": 10, "bpb_delta": pytest.approx(-0.1)}]
    ref[1]["final"], rows[1]["final"] = True, False  # the final row is excluded on either side
    assert matched_observations(rows, ref, "tokens") == [{"tokens": 10, "bpb_delta": pytest.approx(-0.1)}]


def test_paired_analysis_uses_same_seed_pairs_with_hand_computed_numbers():
    from experiments.report import paired_analysis, paired_pairs, seed_noise
    seeds, ablation = paired_pairs()
    assert seeds == [("d8-gqa2-s1", "d8-b0-s1"), ("d8-gqa2-s2", "d8-b0-s2"), ("d8-gqa2-s3", "d8-b0-s3")]
    assert ablation == [("d8-gqa2-s1", "d8-gqa2-tok"), ("d8-gqa4", "d8-gqa4-tok")]
    r = lambda bpb: dict(test_bpb=bpb)
    # Baseline seeds spread by +-0.02; GQA-2 tracks the same-seed baseline: differences 0.010, 0.012, 0.014.
    results = {"d8-b0-s1": r(1.00), "d8-b0-s2": r(1.02), "d8-b0-s3": r(0.98),
               "d8-gqa2-s1": r(1.010), "d8-gqa2-s2": r(1.032), "d8-gqa2-s3": r(0.994)}
    assert paired_analysis({k: v for k, v in results.items() if k != "d8-gqa2-s3"}) is None  # every pair required
    noise = seed_noise([1.00, 1.02, 0.98])  # s = 0.02
    out = paired_analysis(results, noise)
    e2 = out["e2_seed_pairs"]
    assert e2["mean"] == pytest.approx(0.012) and e2["std"] == pytest.approx(0.002) and e2["df"] == 2
    h = 4.303 * 0.002 / 3 ** 0.5
    assert e2["ci95"] == pytest.approx([0.012 - h, 0.012 + h]) and e2["resolved"] is True
    assert e2["noise_ratio"] == pytest.approx(0.002 / (2 ** 0.5 * 0.02))
    # The unpaired 3-vs-3 view of the same data cannot resolve 0.012 against s = 0.02.
    assert 0.012 < 2 * 0.02 * (2 / 3) ** 0.5
    assert out["e6_compute_vs_tokens"] == [] and out["n_contrasts"] == 1
    results |= {"d8-gqa2-tok": r(1.015), "d8-gqa4": r(1.02), "d8-gqa4-tok": r(1.021)}
    e6 = {i["run_id"]: i for i in paired_analysis(results, noise)["e6_compute_vs_tokens"]}
    assert e6["d8-gqa2-s1"]["ablation"] == "d8-gqa2-tok"
    assert e6["d8-gqa2-s1"]["test_bpb_delta"] == pytest.approx(-0.005) and e6["d8-gqa2-s1"]["resolved"] is True  # 2*s_d = 0.004
    assert e6["d8-gqa4"]["test_bpb_delta"] == pytest.approx(-0.001) and e6["d8-gqa4"]["resolved"] is False
