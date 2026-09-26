"""Group-level audit metrics: who gets denied, and how hard is it to recover?"""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_performance(y_true, proba, groups: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Per-group approval rate and error rates for every protected attribute."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    y = np.asarray(y_true)
    rows = []
    for attr in groups.columns:
        for g in sorted(groups[attr].unique()):
            m = (groups[attr] == g).to_numpy()
            yt, yp = y[m], pred[m]
            pos, neg = yt == 1, yt == 0
            rows.append({
                "attribute": attr, "group": g, "n": int(m.sum()),
                "base_rate": yt.mean(),
                "approval_rate": yp.mean(),
                "accuracy": (yt == yp).mean(),
                "tpr": yp[pos].mean() if pos.any() else np.nan,
                # qualified but denied: the people recourse matters most for
                "fnr": 1 - yp[pos].mean() if pos.any() else np.nan,
                "fpr": yp[neg].mean() if neg.any() else np.nan,
            })
    return pd.DataFrame(rows)


def recourse_by_group(rec: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Recourse availability and burden for denied individuals, per group.

    ``rec`` is the output of RecourseSearch.run, indexed like ``groups``.
    Cost statistics are over individuals for whom recourse was found.
    """
    df = rec.join(groups, how="left")
    rows = []
    for attr in groups.columns:
        for g, d in df.groupby(attr):
            found = d[d["found"]]
            rows.append({
                "attribute": attr, "group": g, "n_denied_audited": len(d),
                "recourse_rate": d["found"].mean(),
                "mean_cost": found["cost"].mean(),
                "median_cost": found["cost"].median(),
                "p90_cost": found["cost"].quantile(0.9),
                "mean_features_changed": found["n_changed"].mean(),
                "mean_prob_original": d["prob_original"].mean(),
            })
    return pd.DataFrame(rows)


def disparity_summary(by_group: pd.DataFrame,
                      metrics=("recourse_rate", "mean_cost", "median_cost")) -> pd.DataFrame:
    """Max-min gap and max/min ratio of each metric within each attribute."""
    rows = []
    for attr, d in by_group.groupby("attribute"):
        for m in metrics:
            s = d.set_index("group")[m].dropna()
            if s.empty:
                continue
            rows.append({
                "attribute": attr, "metric": m,
                "worst_group": s.idxmin() if m == "recourse_rate" else s.idxmax(),
                "best_group": s.idxmax() if m == "recourse_rate" else s.idxmin(),
                "gap": s.max() - s.min(),
                "ratio": s.max() / s.min() if s.min() > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def feature_change_rates(rec: pd.DataFrame, groups: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Share of each group's recourse actions that touch each feature."""
    df = rec[rec["found"]].join(groups, how="left")
    rows = []
    for attr in groups.columns:
        for g, d in df.groupby(attr):
            rows.append({"attribute": attr, "group": g, "n_with_recourse": len(d),
                         **{f: d[f"changed_{f}"].mean() for f in features}})
    return pd.DataFrame(rows)
