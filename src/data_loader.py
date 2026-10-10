"""Load the census data, derive audit groups, and split features from groups."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"


def output_dirs(cfg: dict) -> tuple[Path, Path]:
    """Per-dataset result and figure folders, e.g. data/processed/adult/."""
    name = cfg["data"].get("name", cfg["data"].get("source", "adult"))
    dirs = PROCESSED_DIR / name, FIGURES_DIR / name
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path = ROOT / "config" / "constraints.yaml") -> dict:
    """Load a YAML config. A config may set ``extends: <other.yaml>`` (relative to
    its own folder) to inherit everything and override only what it lists."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    return _merge(load_config(path.parent / parent), cfg) if parent else cfg


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.strip().replace("-", "_").lower())
    if "class" in df.columns:
        df = df.rename(columns={"class": "income"})
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip().fillna("Unknown").replace("?", "Unknown")
            df[c] = df[c].astype(str)
    df["income"] = df["income"].str.replace(".", "", regex=False)
    return df


def load_adult(cache: bool = True) -> pd.DataFrame:
    """UCI Adult (1994 US Census extract) via OpenML, cached as CSV."""
    path = RAW_DIR / "adult.csv"
    if cache and path.exists():
        return pd.read_csv(path)
    from sklearn.datasets import fetch_openml

    df = fetch_openml("adult", version=2, as_frame=True).frame
    df = _clean_columns(df)
    if cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


# --- ACS (folktables) --------------------------------------------------------
# ACSIncome-style extract (Ding et al. 2021, "Retiring Adult") for one state and
# year, recoded to the Adult schema so the same constraints, code, and figures
# apply to both datasets. Code books: https://www.census.gov/programs-surveys/acs/microdata/documentation.html

ACS_WORKCLASS = {1: "Private", 2: "Private-nonprofit", 3: "Local-gov", 4: "State-gov",
                 5: "Federal-gov", 6: "Self-emp-not-inc", 7: "Self-emp-inc",
                 8: "Without-pay", 9: "Unemployed"}
ACS_MARITAL = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never-married"}
# RELSHIPP replaced RELP from 2019 on; collapsed to broad household roles.
ACS_RELATIONSHIP = {20: "Reference-person", 21: "Spouse", 23: "Spouse",
                    22: "Unmarried-partner", 24: "Unmarried-partner",
                    25: "Child", 26: "Child", 27: "Child", 35: "Child",
                    28: "Other-relative", 29: "Other-relative", 30: "Other-relative",
                    31: "Other-relative", 32: "Other-relative", 33: "Other-relative",
                    34: "Non-relative", 36: "Non-relative",
                    37: "Group-quarters", 38: "Group-quarters"}
# SCHL (24 levels) -> Adult's education_num (1-16) so "one step" means the same
# thing in both datasets. Adult splits associate degrees into 11 (vocational) and
# 12 (academic); ACS does not, so they map to 12.
ACS_EDUCATION = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 4, 11: 4,
                 12: 5, 13: 6, 14: 7, 15: 8, 16: 9, 17: 9, 18: 10, 19: 10, 20: 12,
                 21: 13, 22: 14, 23: 15, 24: 16}
# OCCP (~530 codes, 2018 Census occupation list) -> its 23 major groups (lower bound of each range).
ACS_OCCUPATION = [(0, "Management"), (500, "Business-Financial"), (1000, "Computer-Math"),
                  (1300, "Architecture-Engineering"), (1600, "Science"),
                  (2000, "Community-Social-Service"), (2100, "Legal"), (2200, "Education"),
                  (2600, "Arts-Media"), (3000, "Healthcare-Practitioner"),
                  (3600, "Healthcare-Support"), (3700, "Protective-Service"),
                  (4000, "Food-Service"), (4200, "Cleaning-Maintenance"),
                  (4300, "Personal-Care"), (4700, "Sales"), (5000, "Office-Admin"),
                  (6000, "Farming-Fishing"), (6200, "Construction-Extraction"),
                  (7000, "Installation-Repair"), (7700, "Production"),
                  (9000, "Transportation"), (9800, "Military"), (9920, "Unemployed")]
# POBP: 1-56 US states/DC, 60-99 US territories and at sea, then world regions.
ACS_BIRTHPLACE = [(0, "United-States"), (60, "US-territory"), (100, "Europe"),
                  (158, "Asia"), (300, "Americas"), (400, "Africa"), (500, "Oceania")]


def _bin_codes(codes: pd.Series, table: list[tuple[int, str]]) -> pd.Series:
    edges, labels = zip(*table)
    return pd.Series(np.asarray(labels)[np.searchsorted(edges, codes.to_numpy(), side="right") - 1],
                     index=codes.index)


def acs_to_adult_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the ACSIncome filter and recode ACS PUMS columns to Adult's names."""
    d = raw[(raw["AGEP"] > 16) & (raw["PINCP"] > 100) & (raw["WKHP"] > 0) & (raw["PWGTP"] >= 1)]
    return pd.DataFrame({
        "age": d["AGEP"].astype(int),
        "workclass": d["COW"].astype(int).map(ACS_WORKCLASS),
        "education_num": d["SCHL"].astype(int).map(ACS_EDUCATION),
        "marital_status": d["MAR"].astype(int).map(ACS_MARITAL),
        "occupation": _bin_codes(d["OCCP"].astype(int), ACS_OCCUPATION),
        "relationship": d["RELSHIPP"].astype(int).map(ACS_RELATIONSHIP),
        "hours_per_week": d["WKHP"].astype(int),
        "native_country": _bin_codes(d["POBP"].astype(int), ACS_BIRTHPLACE),
        "sex": d["SEX"].map({1: "Male", 2: "Female"}),
        "race": np.where(d["RAC1P"] == 1, "White", np.where(d["RAC1P"] == 2, "Black", "Other")),
        "pincp": d["PINCP"].astype(int),
    }).reset_index(drop=True)


def load_acs(cfg: dict) -> pd.DataFrame:
    """ACSIncome-style data for cfg's state and year via folktables, cached as CSV.
    The label is personal income above ``income_threshold``."""
    d = cfg["data"]
    state, year = d["state"], str(d["year"])
    path = RAW_DIR / f"acs_{state.lower()}_{year}.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        from folktables import ACSDataSource

        raw = ACSDataSource(survey_year=year, horizon="1-Year", survey="person",
                            root_dir=str(RAW_DIR / "folktables")).get_data(states=[state], download=True)
        df = acs_to_adult_schema(raw)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    df["income"] = np.where(df["pincp"] > d["income_threshold"], "above", "below")
    return df


def make_synthetic(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Schema-compatible synthetic stand-in so the pipeline runs offline.

    Group disparities are injected deliberately (via education and occupation
    access) so the audit has something to find; do not draw conclusions from it.
    """
    rng = np.random.default_rng(seed)
    sex = rng.choice(["Male", "Female"], n, p=[0.67, 0.33])
    race = rng.choice(
        ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
        n, p=[0.85, 0.09, 0.03, 0.01, 0.02],
    )
    age = rng.integers(17, 80, n)
    disadvantaged = (race != "White").astype(float) * 0.8 + (sex == "Female") * 0.5
    edu = np.clip(np.round(rng.normal(10.5 - disadvantaged, 2.5)), 1, 16).astype(int)
    hours = np.clip(np.round(rng.normal(40 - 4 * (sex == "Female"), 11)), 1, 99).astype(int)
    occupations = ["Exec-managerial", "Prof-specialty", "Craft-repair", "Sales",
                   "Adm-clerical", "Other-service", "Machine-op-inspct", "Handlers-cleaners"]
    occ_p = np.array([0.14, 0.14, 0.14, 0.12, 0.12, 0.12, 0.11, 0.11])
    occ = np.array([rng.choice(occupations, p=occ_p if d == 0 else occ_p[::-1] / occ_p.sum())
                    for d in (disadvantaged > 0)])
    workclass = rng.choice(["Private", "Self-emp-not-inc", "Local-gov", "State-gov",
                            "Federal-gov", "Self-emp-inc"], n,
                           p=[0.72, 0.08, 0.07, 0.05, 0.04, 0.04])
    marital = rng.choice(["Married-civ-spouse", "Never-married", "Divorced", "Widowed"],
                         n, p=[0.46, 0.33, 0.15, 0.06])
    relationship = np.where(marital == "Married-civ-spouse",
                            np.where(sex == "Male", "Husband", "Wife"),
                            rng.choice(["Not-in-family", "Own-child", "Unmarried"], n))
    cap_gain = np.where(rng.random(n) < 0.08, rng.integers(1000, 20000, n), 0)
    logit = (-9.0 + 0.45 * edu + 0.04 * hours + 0.03 * np.minimum(age, 60)
             + 1.2 * (marital == "Married-civ-spouse")
             + 0.8 * np.isin(occ, ["Exec-managerial", "Prof-specialty"])
             + 0.5 * (workclass == "Self-emp-inc") + cap_gain / 5000)
    income = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), ">50K", "<=50K")
    return pd.DataFrame({
        "age": age, "workclass": workclass, "fnlwgt": rng.integers(20000, 500000, n),
        "education": "n/a", "education_num": edu, "marital_status": marital,
        "occupation": occ, "relationship": relationship, "race": race, "sex": sex,
        "capital_gain": cap_gain, "capital_loss": 0, "hours_per_week": hours,
        "native_country": "United-States", "income": income,
    })


def load_raw(cfg: dict) -> pd.DataFrame:
    source = cfg["data"].get("source", "adult")
    if source == "synthetic":
        return make_synthetic(seed=cfg["data"]["seed"])
    if source == "acs":
        return load_acs(cfg)  # no synthetic fallback: a failed download should be loud
    try:
        return load_adult()
    except Exception as e:  # offline, OpenML down, etc.
        print(f"[data_loader] Could not load Adult ({e!r}); falling back to synthetic data.")
        return make_synthetic(seed=cfg["data"]["seed"])


def add_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["race_group"] = np.where(df["race"].isin(["White", "Black"]), df["race"], "Other")
    df["age_group"] = pd.cut(df["age"], bins=[0, 29, 49, 200],
                             labels=["<30", "30-49", "50+"]).astype(str)
    return df


def prepare(cfg: dict, df: pd.DataFrame | None = None):
    """Return (X, y, groups): model features, binary target, audit-group frame."""
    df = add_group_columns(load_raw(cfg) if df is None else df)
    target = cfg["data"]["target"]
    y = (df[target] == cfg["data"]["positive_label"]).astype(int)
    groups = df[cfg["protected_attributes"]].copy()

    drop = set(cfg["data"].get("drop", [])) | {target, "race_group", "age_group"}
    if not cfg.get("use_protected_in_model", False):
        drop |= {"sex", "race"}
    X = df.drop(columns=[c for c in drop if c in df.columns])
    return X, y, groups
