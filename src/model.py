"""Train the decision model. Pipelines take raw DataFrames so recourse can
operate in the original (human-readable) feature space."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_model(kind: str, X: pd.DataFrame, seed: int = 42) -> Pipeline:
    numeric = X.select_dtypes("number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    if kind == "logistic":
        clf = LogisticRegression(max_iter=2000)
    elif kind == "gbm":
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=seed)
    else:
        raise ValueError(f"Unknown model kind: {kind!r}")
    return Pipeline([("pre", pre), ("clf", clf)])


def split(X, y, groups, cfg: dict):
    return train_test_split(X, y, groups, test_size=cfg["data"]["test_size"],
                            random_state=cfg["data"]["seed"], stratify=y)


def train(kind: str, X_train, y_train, seed: int = 42) -> Pipeline:
    return build_model(kind, X_train, seed).fit(X_train, y_train)


def evaluate(model: Pipeline, X, y, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, proba >= threshold),
        "roc_auc": roc_auc_score(y, proba),
        "approval_rate": float((proba >= threshold).mean()),
        "n": len(y),
    }
