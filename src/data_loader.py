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


def load_config(path: str | Path = ROOT / "config" / "constraints.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
