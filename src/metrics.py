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


def constraint_violations(rec: pd.DataFrame, X: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Check each recommended counterfactual against the constraints in the config.

    ``rec`` is RecourseSearch.run output (with ``cf_<feature>`` columns) and ``X``
    holds the original rows. Adds ``n_violations`` and ``violations`` (e.g.
    "immutable:marital_status; direction:education_num") for rows with recourse.
    """
    numeric = cfg["actionable"].get("numeric", {})
    categorical = cfg["actionable"].get("categorical", {})
    out = rec.copy()
    labels = []
    for idx, r in rec.iterrows():
        v = []
        if r["found"]:
            x = X.loc[idx]
            for f in X.columns:
                col = f"cf_{f}"
                if col not in r or pd.isna(r[col]) or r[col] == x[f]:
                    continue
                new = r[col]
                if f in numeric:
                    s = numeric[f]
                    if not s["min"] <= float(new) <= s["max"]:
                        v.append(f"bounds:{f}")
                    if (s["direction"] == "increase" and float(new) < float(x[f])) or \
                       (s["direction"] == "decrease" and float(new) > float(x[f])):
                        v.append(f"direction:{f}")
                elif f in categorical:
                    if new in categorical[f].get("exclude", []):
                        v.append(f"excluded:{f}")
                else:  # immutable, or not declared actionable at all
                    v.append(f"immutable:{f}")
        labels.append(v)
    out["n_violations"] = [len(v) for v in labels]
    out["violations"] = ["; ".join(v) for v in labels]
    return out


def violations_by_group(rec: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Per group: how often unconstrained recourse is found, and how often what it
    recommends breaks the constraints. ``rec`` is constraint_violations output."""
    df = rec.join(groups, how="left")
    rows = []
    for attr in groups.columns:
        for g, d in df.groupby(attr):
            found = d[d["found"]]
            types = pd.Series([t.split(":")[1] for v in found["violations"] for t in v.split("; ") if t])
            rows.append({
                "attribute": attr, "group": g, "n_denied_audited": len(d),
                "recourse_rate": d["found"].mean(),
                # share of recommendations that ask for something not allowed
                "violation_rate": (found["n_violations"] > 0).mean() if len(found) else np.nan,
                "top_violated_feature": types.mode()[0] if len(types) else "",
            })
    return pd.DataFrame(rows)


def violation_type_counts(rec: pd.DataFrame) -> pd.DataFrame:
    """How often each (rule, feature) violation appears among found recommendations."""
    found = rec[rec["found"]]
    types = pd.Series([t for v in found["violations"] for t in v.split("; ") if t])
    out = types.value_counts().rename_axis("violation").reset_index(name="count")
    out["share_of_recommendations"] = out["count"] / max(len(found), 1)
    return out
