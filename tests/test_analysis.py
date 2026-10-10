"""Checks for the step-4 analyses on synthetic data where the answer is known."""

import numpy as np
import pandas as pd

from src.metrics import budget_curves, distance_bins, group_gaps, min_recourse_cost


def _synthetic(n=4000, seed=0, gap_from_start_only=True):
    """Two groups. Group B starts further from approval (lower probability).
    If ``gap_from_start_only``, recourse depends only on starting probability,
    so B's raw gap is entirely explained by where it starts."""
    rng = np.random.default_rng(seed)
    grp = np.where(rng.random(n) < 0.5, "A", "B")
    p0 = np.where(grp == "A", rng.uniform(0.2, 0.5, n), rng.uniform(0.0, 0.3, n))
    chance = p0 * 2 if gap_from_start_only else np.where(grp == "A", 0.6, 0.3)
    found = rng.random(n) < chance
    cost = np.where(found, rng.uniform(0.5, 6, n), np.nan)
    rec = pd.DataFrame({"found": found, "cost": cost, "prob_original": p0,
                        "min_cost_over_budget": np.where(~found & (rng.random(n) < 0.5), 8.0, np.nan)})
    return rec, pd.DataFrame({"grp": grp}, index=rec.index)


def _gap(gaps, metric):
    return gaps.set_index("metric").loc[metric]


def test_gap_explained_by_starting_distance_disappears_when_matched():
    # large n: only the overlap region (~1/5 of B) is compared, so noise matters
    rec, groups = _synthetic(n=40_000, gap_from_start_only=True)
    gaps = group_gaps(rec, groups, {"grp": "A"}, n_bins=10, n_boot=200)
    raw, matched = _gap(gaps, "recourse_rate_gap"), _gap(gaps, "recourse_rate_gap_same_start")
    assert raw["estimate"] < -0.2 and raw["significant"]  # B looks much worse...
    assert abs(matched["estimate"]) < 0.05                # ...but not at the same start
    assert matched["ci_low"] < 0 < matched["ci_high"]


def test_structural_gap_survives_matching():
    rec, groups = _synthetic(gap_from_start_only=False)
    gaps = group_gaps(rec, groups, {"grp": "A"}, n_bins=10, n_boot=200)
    matched = _gap(gaps, "recourse_rate_gap_same_start")
    assert abs(matched["estimate"] - (-0.3)) < 0.06 and matched["significant"]


def test_ci_contains_estimate_and_is_reproducible():
    rec, groups = _synthetic()
    a = group_gaps(rec, groups, {"grp": "A"}, n_boot=200, seed=1)
    b = group_gaps(rec, groups, {"grp": "A"}, n_boot=200, seed=1)
    pd.testing.assert_frame_equal(a, b)
    ok = a.dropna(subset=["ci_low"])
    assert ((ok["ci_low"] <= ok["estimate"]) & (ok["estimate"] <= ok["ci_high"])).all()


def test_budget_curve_is_monotone_and_matches_found_at_budget():
    rec, groups = _synthetic()
    rec.loc[rec["cost"] > 6, "found"] = False  # pretend audit budget is 6
    curves = budget_curves(rec, groups, np.arange(0, 12.25, 0.25))
    for _, c in curves.groupby("group"):
        assert np.all(np.diff(c["recourse_rate"].to_numpy()) >= 0)
    at6 = curves[curves["budget"] == 6].set_index("group")["recourse_rate"]
    expected = rec.join(groups).groupby("grp")["found"].mean()
    assert np.allclose(at6.sort_index(), expected.sort_index())


def test_min_cost_uses_over_budget_cost_and_inf():
    rec = pd.DataFrame({"found": [True, False, False], "cost": [2.0, np.nan, np.nan],
                        "min_cost_over_budget": [np.nan, 9.0, np.nan]})
    assert min_recourse_cost(rec).tolist() == [2.0, 9.0, np.inf]


def test_distance_bins_are_balanced():
    p = pd.Series(np.random.default_rng(0).uniform(0, 0.5, 1000))
    counts = np.bincount(distance_bins(p, 5))
    assert len(counts) == 5 and counts.min() >= 190
