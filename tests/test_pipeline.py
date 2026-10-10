"""Correctness checks for data loading, the model, and recourse search."""

import numpy as np
import pandas as pd
import pytest

from src.model import evaluate
from src.recourse import RecourseSearch

N_APPLICANTS = 10  # denied applicants to run recourse on, per model


# --- data -------------------------------------------------------------------


def test_dataset_shape(data):
    X, y, groups = data
    assert len(X) == len(y) == len(groups) == 48_842


def test_protected_and_dropped_columns_not_model_inputs(data):
    X, _, _ = data
    assert not {"sex", "race", "fnlwgt", "education", "income"} & set(X.columns)


def test_no_missing_values(data):
    X, _, _ = data
    assert X.isna().sum().sum() == 0


def test_positive_rate_matches_published_adult(data):
    _, y, _ = data
    assert 0.23 < y.mean() < 0.25  # Adult's >50K rate is ~0.24


def test_audit_groups_present(data):
    _, _, groups = data
    assert set(groups["sex"]) == {"Male", "Female"}
    assert set(groups["race_group"]) == {"White", "Black", "Other"}
    assert set(groups["age_group"]) == {"<30", "30-49", "50+"}


def test_split_is_stratified(splits):
    _, _, y_train, y_test, *_ = splits
    assert abs(y_train.mean() - y_test.mean()) < 0.005


# --- model ------------------------------------------------------------------


def test_model_performance(model, cfg, splits):
    _, X_test, _, y_test, *_ = splits
    ev = evaluate(model, X_test, y_test, cfg["model"]["threshold"])
    assert ev["accuracy"] > 0.83
    assert ev["roc_auc"] > 0.88


# --- recourse ---------------------------------------------------------------


@pytest.fixture(scope="session")
def recourse_results(model, cfg, splits):
    """Run recourse on denied applicants spread across the probability range."""
    X_train, X_test, *_ = splits
    search = RecourseSearch.from_config(model, cfg, X_train)
    proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
    denied = proba[proba < cfg["model"]["threshold"]].sort_values()
    picks = denied.iloc[np.linspace(0, len(denied) - 1, N_APPLICANTS).astype(int)].index
    return search, [(X_test.loc[i], search.explain(X_test.loc[i])) for i in picks]


def _counterfactual(x, res, features):
    cf = x.copy()
    for f in features:
        if f"cf_{f}" in res:
            cf[f] = res[f"cf_{f}"]
    return cf


def test_some_recourse_is_found(recourse_results):
    _, results = recourse_results
    assert any(res["found"] for _, res in results)


def test_recourse_actually_flips_decision(recourse_results, model, cfg, splits):
    search, results = recourse_results
    dtypes = splits[0].dtypes
    for x, res in results:
        if res["found"]:
            cf = _counterfactual(x, res, search.space.features)
            p = model.predict_proba(cf.to_frame().T.astype(dtypes))[:, 1][0]
            assert p >= cfg["model"]["threshold"], res["changes"]


def test_recourse_respects_constraints(recourse_results, cfg):
    search, results = recourse_results
    num = cfg["actionable"]["numeric"]
    excluded = {
        f: set(s.get("exclude", []))
        for f, s in cfg["actionable"]["categorical"].items()
    }
    for x, res in results:
        if not res["found"]:
            continue
        cf = _counterfactual(x, res, search.space.features)
        for f in cfg["immutable"]:
            assert cf[f] == x[f], f"immutable {f} changed"
        for f, spec in num.items():
            assert (
                spec["min"] <= cf[f] <= spec["max"] or cf[f] == x[f]
            ), f"{f} out of bounds"
            if spec["direction"] == "increase":
                assert cf[f] >= x[f], f"{f} decreased"
        for f, bad in excluded.items():
            assert cf[f] == x[f] or cf[f] not in bad, f"{f} set to excluded {cf[f]}"
        assert res["cost"] <= cfg["recourse"]["max_cost"]
        assert res["n_changed"] <= cfg["recourse"]["max_features_changed"]


def test_recourse_is_deterministic(model, cfg, splits):
    """Same seed -> same recommendation, so figures are reproducible."""
    X_train, X_test, *_ = splits
    proba = model.predict_proba(X_test)[:, 1]
    x = X_test.iloc[
        int(np.argmax(np.where(proba < cfg["model"]["threshold"], proba, -1)))
    ]
    a = RecourseSearch.from_config(model, cfg, X_train).explain(x)
    b = RecourseSearch.from_config(model, cfg, X_train).explain(x)
    assert a["changes"] == b["changes"] and a["cost"] == b["cost"]


# --- unconstrained baseline and violations ----------------------------------

from src.metrics import constraint_violations  # noqa: E402


@pytest.fixture(scope="session")
def both_modes(model, cfg, splits):
    """Constrained and unconstrained recourse for the same denied applicants."""
    X_train, X_test, *_ = splits
    proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
    denied = proba[proba < cfg["model"]["threshold"]].sort_values()
    idx = denied.index[np.linspace(0, len(denied) - 1, N_APPLICANTS).astype(int)]
    out = {}
    for mode in ("constrained", "unconstrained"):
        search = RecourseSearch.from_config(model, cfg, X_train, mode=mode)
        out[mode] = constraint_violations(
            search.run(X_test.loc[idx], verbose=False), X_test, cfg
        )
    return out


def test_constrained_recourse_has_no_violations(both_modes):
    assert (both_modes["constrained"]["n_violations"] == 0).all()


def test_unconstrained_finds_at_least_as_much_recourse(both_modes):
    assert (
        both_modes["unconstrained"]["found"].sum()
        >= both_modes["constrained"]["found"].sum()
    )


def test_unconstrained_recourse_flips_decision(both_modes, model, cfg, splits):
    X_test, dtypes = splits[1], splits[0].dtypes
    rec = both_modes["unconstrained"]
    for idx, r in rec[rec["found"]].iterrows():
        cf = X_test.loc[idx].copy()
        for f in X_test.columns:
            if f"cf_{f}" in r and pd.notna(r[f"cf_{f}"]):
                cf[f] = r[f"cf_{f}"]
        assert (
            model.predict_proba(cf.to_frame().T.astype(dtypes))[:, 1][0]
            >= cfg["model"]["threshold"]
        )


def test_violation_checker_flags_each_rule(cfg, splits):
    """Hand-built counterfactuals: one per rule type, plus a clean one."""
    X_test = splits[1]
    i = X_test.index[
        (X_test["education_num"] > 5) & (X_test["workclass"] == "Private")
    ][0]
    x = X_test.loc[i]
    other_marital = next(
        v for v in X_test["marital_status"].unique() if v != x["marital_status"]
    )
    rec = pd.DataFrame(
        [
            {"found": True, "cf_marital_status": other_marital},  # immutable
            {"found": True, "cf_education_num": x["education_num"] - 1},  # direction
            {"found": True, "cf_hours_per_week": 95},  # bounds
            {"found": True, "cf_workclass": "Without-pay"},  # excluded
            {"found": True, "cf_education_num": x["education_num"] + 1},  # allowed
            {"found": False},
        ],
        index=[i] * 6,
    )
    got = constraint_violations(rec, X_test, cfg)["violations"].tolist()
    assert got == [
        "immutable:marital_status",
        "direction:education_num",
        "bounds:hours_per_week",
        "excluded:workclass",
        "",
        "",
    ]


def test_frozen_features_never_change(model, cfg, splits):
    X_train, X_test, *_ = splits
    proba = model.predict_proba(X_test)[:, 1]
    rows = X_test[proba < cfg["model"]["threshold"]].iloc[:N_APPLICANTS]
    frozen = ["capital_gain", "capital_loss"]
    search = RecourseSearch.from_config(
        model, cfg, X_train, mode="unconstrained", freeze=frozen
    )
    assert not set(frozen) & set(search.space.features)
    rec = search.run(rows, verbose=False)
    for f in frozen:
        assert f"cf_{f}" not in rec or rec[f"cf_{f}"].isna().all()


def test_violation_summaries_are_consistent(both_modes, splits):
    from src.metrics import violation_type_counts, violations_by_group

    rec = both_modes["unconstrained"]
    groups = splits[5].loc[rec.index]
    by_group = violations_by_group(rec, groups)
    assert by_group["violation_rate"].dropna().between(0, 1).all()
    # every attribute's groups partition the same applicants
    assert (by_group.groupby("attribute")["n_denied_audited"].sum() == len(rec)).all()
    types = violation_type_counts(rec)
    n_labels = sum(
        len([t for t in v.split("; ") if t])
        for v in rec.loc[rec["found"], "violations"]
    )
    assert types["count"].sum() == n_labels
