"""End-to-end recourse audit.

    python run_audit.py                      # full audit, all models in the config
    python run_audit.py --models logistic    # one model
    python run_audit.py --quick              # small sample, for a smoke test
    python run_audit.py --config config/constraints_acs_nc.yaml   # ACS North Carolina
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from src import metrics, plots
from src.data_loader import load_config, output_dirs, prepare
from src.model import evaluate, split, train
from src.recourse import RecourseSearch


def sample_denied(denied_idx: pd.Index, groups: pd.DataFrame, size: int | None,
                  min_group: int, seed: int) -> pd.Index:
    """Uniform sample of denied individuals, topped up so every protected
    group has at least ``min_group`` members (or all it has)."""
    rng = np.random.default_rng(seed)
    if size is None or size >= len(denied_idx):
        return denied_idx
    chosen = set(rng.choice(denied_idx, size, replace=False))
    g = groups.loc[denied_idx]
    for attr in g.columns:
        for _, members in g.groupby(attr).groups.items():
            have = [i for i in members if i in chosen]
            need = min(min_group, len(members)) - len(have)
            if need > 0:
                pool = np.array([i for i in members if i not in chosen])
                chosen.update(rng.choice(pool, need, replace=False))
    return pd.Index(sorted(chosen))


def audit_model(kind: str, cfg: dict, data) -> pd.DataFrame:
    X_train, X_test, y_train, y_test, g_train, g_test = data
    thr = cfg["model"]["threshold"]
    rc = cfg["recourse"]
    PROCESSED_DIR, FIGURES_DIR = output_dirs(cfg)
    print(f"\n=== {kind} ===")

    model = train(kind, X_train, y_train, seed=cfg["data"]["seed"])
    joblib.dump(model, PROCESSED_DIR / f"model_{kind}.joblib")
    overall = {"model": kind, **evaluate(model, X_test, y_test, thr)}
    print(f"  accuracy={overall['accuracy']:.3f}  auc={overall['roc_auc']:.3f}  "
          f"approval_rate={overall['approval_rate']:.3f}")

    proba = model.predict_proba(X_test)[:, 1]
    perf = metrics.group_performance(y_test, proba, g_test, thr)
    perf.to_csv(PROCESSED_DIR / f"group_performance_{kind}.csv", index=False)

    denied = X_test.index[proba < thr]
    idx = sample_denied(denied, g_test, rc.get("sample_size"), rc.get("min_group_size", 0), rc["seed"])
    print(f"  computing recourse for {len(idx)} of {len(denied)} denied test individuals")
    search = RecourseSearch.from_config(model, cfg, X_train)
    rec = search.run(X_test.loc[idx])
    rec = rec.join(g_test)
    rec.to_csv(PROCESSED_DIR / f"recourse_individual_{kind}.csv")

    g = g_test.loc[idx]
    by_group = metrics.recourse_by_group(rec.drop(columns=g.columns), g)
    by_group.to_csv(PROCESSED_DIR / f"recourse_by_group_{kind}.csv", index=False)
    disp = metrics.disparity_summary(by_group)
    disp.to_csv(PROCESSED_DIR / f"recourse_disparities_{kind}.csv", index=False)
    feats = search.space.features
    changes = metrics.feature_change_rates(rec.drop(columns=g.columns), g, feats)
    changes.to_csv(PROCESSED_DIR / f"feature_changes_{kind}.csv", index=False)

    plots.plot_recourse_rate(by_group, FIGURES_DIR / f"recourse_rate_{kind}.png", kind)
    plots.plot_mean_cost(by_group, FIGURES_DIR / f"recourse_mean_cost_{kind}.png", kind)
    plots.plot_cost_distribution(rec.drop(columns=g.columns), g,
                                 FIGURES_DIR / f"recourse_cost_dist_{kind}.png", kind)
    plots.plot_feature_changes(changes, feats, FIGURES_DIR / f"feature_changes_{kind}.png", kind)
    plots.plot_error_rates(perf, FIGURES_DIR / f"false_negative_rate_{kind}.png", kind)

    print(f"  overall recourse rate={rec['found'].mean():.3f}  "
          f"mean cost={rec.loc[rec['found'], 'cost'].mean():.2f}")
    print(by_group[["attribute", "group", "n_denied_audited", "recourse_rate", "mean_cost"]]
          .to_string(index=False, float_format="%.3f"))
    overall.update(n_denied=len(denied), n_audited=len(idx),
                   recourse_rate=rec["found"].mean(),
                   mean_cost=rec.loc[rec["found"], "cost"].mean())

    # Unconstrained baseline on the same applicants: how often does an
    # off-the-shelf search recommend something the constraints forbid?
    vio_groups, vio_types = [], []
    for variant, freeze in cfg["unconstrained"]["variants"].items():
        print(f"  unconstrained recourse ({variant})")
        uncon = RecourseSearch.from_config(model, cfg, X_train, mode="unconstrained", freeze=freeze)
        u = metrics.constraint_violations(uncon.run(X_test.loc[idx]), X_test, cfg)
        u.join(g_test).to_csv(PROCESSED_DIR / f"recourse_unconstrained_{variant}_{kind}.csv")
        vio_groups.append(metrics.violations_by_group(u, g).assign(variant=variant))
        vio_types.append(metrics.violation_type_counts(u).assign(variant=variant))
        found = u[u["found"]]
        rate = (found["n_violations"] > 0).mean()
        print(f"    recourse rate={u['found'].mean():.3f}  violation rate={rate:.3f}")
        overall.update({f"uncon_recourse_rate_{variant}": u["found"].mean(),
                        f"violation_rate_{variant}": rate})
    vio_groups = pd.concat(vio_groups, ignore_index=True)
    vio_types = pd.concat(vio_types, ignore_index=True)
    vio_groups.to_csv(PROCESSED_DIR / f"violations_by_group_{kind}.csv", index=False)
    vio_types.to_csv(PROCESSED_DIR / f"violation_types_{kind}.csv", index=False)
    plots.plot_violation_rate(vio_groups, FIGURES_DIR / f"violation_rate_{kind}.png", kind)
    plots.plot_violation_types(vio_types, FIGURES_DIR / f"violation_types_{kind}.png", kind)
    return pd.DataFrame([overall])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/constraints.yaml")
    ap.add_argument("--models", nargs="+", help="override model.kinds from the config")
    ap.add_argument("--sample-size", type=int, help="override recourse.sample_size")
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.sample_size is not None:
        cfg["recourse"]["sample_size"] = args.sample_size
    if args.quick:
        cfg["recourse"].update(sample_size=60, min_group_size=10, n_candidates=200)

    PROCESSED_DIR, FIGURES_DIR = output_dirs(cfg)
    X, y, groups = prepare(cfg)
    print(f"[{cfg['data']['name']}] Loaded {len(X)} rows, {X.shape[1]} model features; positive rate {y.mean():.3f}")
    data = split(X, y, groups, cfg)

    summary = pd.concat([audit_model(k, cfg, data) for k in (args.models or cfg["model"]["kinds"])])
    summary.to_csv(PROCESSED_DIR / "model_summary.csv", index=False)
    print(f"\nWrote results to {PROCESSED_DIR} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
