"""Checks for the ACS North Carolina loader and that the audit runs on it."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (ACS_EDUCATION, ACS_OCCUPATION, _bin_codes,
                             acs_to_adult_schema, load_config, prepare)
from src.metrics import constraint_violations
from src.model import split, train
from src.recourse import RecourseSearch

ACS_CONFIG = Path(__file__).resolve().parents[1] / "config" / "constraints_acs_nc.yaml"


@pytest.fixture(scope="module")
def acs():
    cfg = load_config(ACS_CONFIG)
    X, y, groups = prepare(cfg)
    return cfg, X, y, groups


# --- recoding (no download needed) ------------------------------------------

def test_education_mapping_is_monotonic_and_on_adult_scale():
    vals = [ACS_EDUCATION[k] for k in sorted(ACS_EDUCATION)]
    assert sorted(ACS_EDUCATION) == list(range(1, 25))
    assert vals == sorted(vals) and vals[0] == 1 and vals[-1] == 16


def test_occupation_binning_edges():
    codes = pd.Series([10, 440, 500, 1005, 3000, 3601, 9830, 9920])
    got = _bin_codes(codes, ACS_OCCUPATION).tolist()
    assert got == ["Management", "Management", "Business-Financial", "Computer-Math",
                   "Healthcare-Practitioner", "Healthcare-Support", "Military", "Unemployed"]


def test_acsincome_filter_and_recoding():
    raw = pd.DataFrame({  # one valid row, then one failing each ACSIncome filter
        "AGEP": [30, 16, 30, 30, 30], "PINCP": [60000, 60000, 50, 60000, 60000],
        "WKHP": [40, 40, 40, 0, 40], "PWGTP": [10, 10, 10, 10, 0],
        "COW": [5] * 5, "SCHL": [21] * 5, "MAR": [5] * 5, "OCCP": [1005] * 5,
        "RELSHIPP": [21] * 5, "POBP": [37] * 5, "SEX": [2] * 5, "RAC1P": [2] * 5,
    })
    df = acs_to_adult_schema(raw)
    assert len(df) == 1
    assert df.iloc[0].to_dict() == {
        "age": 30, "workclass": "Federal-gov", "education_num": 13,
        "marital_status": "Never-married", "occupation": "Computer-Math",
        "relationship": "Spouse", "hours_per_week": 40, "native_country": "United-States",
        "sex": "Female", "race": "Black", "pincp": 60000}


# --- real data (downloads ACS NC once, then cached) -------------------------

def test_acs_dataset(acs):
    cfg, X, y, groups = acs
    assert len(X) == 56_486  # 2023 NC persons passing the ACSIncome filter
    assert X.isna().sum().sum() == 0
    assert not {"sex", "race", "pincp", "income"} & set(X.columns)
    assert X["education_num"].between(1, 16).all()
    assert set(groups["race_group"]) == {"White", "Black", "Other"}
    # label rate follows the configured threshold
    expected = {50000: 0.435}.get(cfg["data"]["income_threshold"])
    if expected is not None:
        assert abs(y.mean() - expected) < 0.005


def test_acs_audit_runs_and_respects_constraints(acs):
    cfg, X, y, groups = acs
    X_tr, X_te, y_tr, y_te, *_ = split(X, y, groups, cfg)
    model = train("gbm", X_tr, y_tr, seed=cfg["data"]["seed"])
    proba = model.predict_proba(X_te)[:, 1]
    denied = X_te[proba < cfg["model"]["threshold"]]
    rows = denied.iloc[np.linspace(0, len(denied) - 1, 8).astype(int)]
    for mode in ("constrained", "unconstrained"):
        rec = constraint_violations(
            RecourseSearch.from_config(model, cfg, X_tr, mode=mode).run(rows, verbose=False), X_te, cfg)
        assert rec["found"].any()
        if mode == "constrained":
            assert (rec["n_violations"] == 0).all()
