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
    excluded = {f: set(s.get("exclude", [])) for f, s in cfg["actionable"]["categorical"].items()}
    for x, res in results:
        if not res["found"]:
            continue
        cf = _counterfactual(x, res, search.space.features)
        for f in cfg["immutable"]:
            assert cf[f] == x[f], f"immutable {f} changed"
        for f, spec in num.items():
            assert spec["min"] <= cf[f] <= spec["max"] or cf[f] == x[f], f"{f} out of bounds"
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
    x = X_test.iloc[int(np.argmax(np.where(proba < cfg["model"]["threshold"], proba, -1)))]
    a = RecourseSearch.from_config(model, cfg, X_train).explain(x)
    b = RecourseSearch.from_config(model, cfg, X_train).explain(x)
    assert a["changes"] == b["changes"] and a["cost"] == b["cost"]
