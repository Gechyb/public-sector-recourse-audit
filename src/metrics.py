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


# --- uncertainty, budget sensitivity, and starting distance -----------------

def min_recourse_cost(rec: pd.DataFrame) -> pd.Series:
    """Cheapest recourse found per individual regardless of budget (inf if none).
    The search is budget-independent; the budget only decides ``found``."""
    over = rec["min_cost_over_budget"] if "min_cost_over_budget" in rec else np.nan
    return rec["cost"].where(rec["found"], over).fillna(np.inf)


def budget_curves(rec: pd.DataFrame, groups: pd.DataFrame, budgets) -> pd.DataFrame:
    """Share of each group's denied applicants with recourse at each budget."""
    cost = min_recourse_cost(rec)
    df = groups.loc[rec.index]
    rows = []
    for attr in groups.columns:
        for g, idx in df.groupby(attr).groups.items():
            c = cost.loc[idx].to_numpy()
            rows += [{"attribute": attr, "group": g, "budget": b, "recourse_rate": (c <= b).mean()}
                     for b in budgets]
    return pd.DataFrame(rows)


def distance_bins(prob: pd.Series, n_bins: int) -> np.ndarray:
    """Quantile band (0..n-1) of each individual's starting approval probability."""
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1))[1:-1])
    return np.searchsorted(edges, prob.to_numpy(), side="right")


def _pair_stats(found, cost, vio, band, in_g, in_r, n_bins) -> dict:
    """Gaps (group minus reference) for one comparison on one (re)sample."""
    out = {"recourse_rate_gap": found[in_g].mean() - found[in_r].mean()}
    # Same-starting-point gap: compare within probability bands both groups
    # occupy, weighted by the group's own band mix (exact matching on bands).
    n_g = np.bincount(band[in_g], minlength=n_bins)
    n_r = np.bincount(band[in_r], minlength=n_bins)
    ok = (n_g > 0) & (n_r > 0)
    if ok.any():
        hit_g = np.bincount(band[in_g], weights=found[in_g], minlength=n_bins)
        hit_r = np.bincount(band[in_r], weights=found[in_r], minlength=n_bins)
        w = n_g[ok] / n_g[ok].sum()
        out["recourse_rate_gap_same_start"] = float((w * (hit_g[ok] / n_g[ok] - hit_r[ok] / n_r[ok])).sum())
    else:
        out["recourse_rate_gap_same_start"] = np.nan
    fg, fr = in_g & found, in_r & found
    out["mean_cost_gap"] = cost[fg].mean() - cost[fr].mean() if fg.any() and fr.any() else np.nan
    for name, (v_found, v_flag) in vio.items():
        vg, vr = in_g & v_found, in_r & v_found
        gap = v_flag[vg].mean() - v_flag[vr].mean() if vg.any() and vr.any() else np.nan
        out[f"violation_rate_gap_{name}"] = gap
    return out


def group_gaps(rec: pd.DataFrame, groups: pd.DataFrame, reference: dict, n_bins: int = 5,
               n_boot: int = 1000, seed: int = 0, unconstrained: dict | None = None) -> pd.DataFrame:
    """Each group vs its reference group, with 95% percentile-bootstrap CIs.

    Metrics (group minus reference): recourse rate, recourse rate at the same
    starting distance from approval, mean cost among those with recourse, and,
    for each unconstrained variant given, the violation rate. A CI that excludes
    0 means the gap is unlikely to be sampling noise.
    """
    g = groups.loc[rec.index]
    found = rec["found"].to_numpy(dtype=bool)
    cost = rec["cost"].to_numpy(dtype=float)
    band = distance_bins(rec["prob_original"], n_bins)
    vio = {name: (u.loc[rec.index, "found"].to_numpy(dtype=bool),
                  (u.loc[rec.index, "n_violations"] > 0).to_numpy())
           for name, u in (unconstrained or {}).items()}
    pairs = [(attr, grp, ref) for attr, ref in reference.items() if attr in g
             for grp in sorted(g[attr].unique()) if grp != ref]
    labels = {(a, x): (g[a] == x).to_numpy() for a, x, r in pairs} | \
             {(a, r): (g[a] == r).to_numpy() for a, x, r in pairs}

    def stats(ix):
        return [_pair_stats(found[ix], cost[ix], {k: (vf[ix], vv[ix]) for k, (vf, vv) in vio.items()},
                            band[ix], labels[(a, x)][ix], labels[(a, r)][ix], n_bins)
                for a, x, r in pairs]

    point = stats(np.arange(len(rec)))
    rng = np.random.default_rng(seed)
    with np.errstate(invalid="ignore"):
        boots = [stats(rng.integers(0, len(rec), len(rec))) for _ in range(n_boot)]
    rows = []
    for k, (a, x, r) in enumerate(pairs):
        for metric, est in point[k].items():
            draws = np.array([b[k][metric] for b in boots], dtype=float)
            lo, hi = np.nanpercentile(draws, [2.5, 97.5]) if np.isfinite(draws).any() else (np.nan, np.nan)
            rows.append({"attribute": a, "group": x, "reference": r, "metric": metric,
                         "estimate": est, "ci_low": lo, "ci_high": hi,
                         "significant": bool(lo > 0 or hi < 0),
                         "n_group": int(labels[(a, x)].sum()), "n_reference": int(labels[(a, r)].sum())})
    return pd.DataFrame(rows)
