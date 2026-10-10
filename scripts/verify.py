"""Human-readable walkthrough of the pipeline: data -> model -> recourse.

    python scripts/verify.py      (or: make verify)

Prints dataset and group statistics, per-group model performance, and the
recourse found for five denied applicants spread from "just missed" to "far
from approval", re-checking each recommendation independently. Writes nothing.
The same checks run automatically as assertions in tests/ (make test).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_config, prepare  # noqa: E402
from src.model import evaluate, split, train  # noqa: E402
from src.recourse import RecourseSearch  # noqa: E402

pd.set_option("display.width", 140)
cfg = load_config()

print("=" * 70, "\n2a. DATA\n" + "=" * 70)
X, y, groups = prepare(cfg)
print(f"X shape: {X.shape}   (expect 48,842 rows)")
print(f"Model features: {list(X.columns)}")
leaked = {"sex", "race", "fnlwgt", "education", "income"} & set(X.columns)
print(f"CHECK protected/dropped columns absent from X: {'PASS' if not leaked else 'FAIL ' + str(leaked)}")
print(f"Positive rate (income >50K): {y.mean():.3f}   (Adult is known to be ~0.24)")
print(f"Missing values in X: {int(X.isna().sum().sum())}")
for a in groups.columns:
    vc = groups[a].value_counts()
    print(f"  {a}: " + ", ".join(f"{k}={v} ({v/len(groups):.1%}, approved {y[groups[a]==k].mean():.1%})" for k, v in vc.items()))

print("\n" + "=" * 70, "\n2b. MODEL\n" + "=" * 70)
X_tr, X_te, y_tr, y_te, g_tr, g_te = split(X, y, groups, cfg)
print(f"train={len(X_tr)}  test={len(X_te)}  pos rate train={y_tr.mean():.3f} test={y_te.mean():.3f}")
thr = cfg["model"]["threshold"]
models = {}
for kind in cfg["model"]["kinds"]:
    m = train(kind, X_tr, y_tr, seed=cfg["data"]["seed"])
    models[kind] = m
    ev = evaluate(m, X_te, y_te, thr)
    print(f"\n[{kind}] accuracy={ev['accuracy']:.3f}  AUC={ev['roc_auc']:.3f}  approval_rate={ev['approval_rate']:.3f}")
    pred = m.predict_proba(X_te)[:, 1] >= thr
    rows = []
    for a in g_te.columns:
        for k in sorted(g_te[a].unique()):
            mask = (g_te[a] == k).to_numpy()
            pos = mask & (y_te.to_numpy() == 1)
            rows.append({"attr": a, "group": k, "n": mask.sum(),
                         "approved": pred[mask].mean(), "true_rate": y_te[mask].mean(),
                         "FNR": 1 - pred[pos].mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n" + "=" * 70, "\n2c. RECOURSE (gbm, 5 denied applicants)\n" + "=" * 70)
model = models["gbm"]
search = RecourseSearch.from_config(model, cfg, X_tr)
proba = pd.Series(model.predict_proba(X_te)[:, 1], index=X_te.index)
denied = proba[proba < thr]
# spread picks across the probability range: near-miss to far-from-boundary
picks = denied.sort_values().iloc[np.linspace(0, len(denied) - 1, 5).astype(int)[::-1]].index
immutable = cfg["immutable"]
for rid in picks:
    x = X_te.loc[rid]
    res = search.explain(x)
    print(f"\n--- row {rid}  |  {g_te.loc[rid].to_dict()}  |  p(approve)={res['prob_original']:.3f}")
    print("    profile: " + ", ".join(f"{c}={x[c]}" for c in X.columns))
    if res["found"]:
        print(f"    RECOURSE cost={res['cost']:.2f}, {res['n_changed']} change(s): {res['changes']}")
        # independent re-check: apply the changes ourselves and re-score
        cf = x.copy()
        for f in search.space.features:
            if f"cf_{f}" in res:
                cf[f] = res[f"cf_{f}"]
        p_cf = model.predict_proba(cf.to_frame().T.astype(X_tr.dtypes))[:, 1][0]
        imm_ok = all(cf[c] == x[c] for c in immutable)
        bounds_ok = (cf["education_num"] >= x["education_num"]) and (1 <= cf["hours_per_week"] <= 80)
        print(f"    re-scored p={p_cf:.3f} -> {'PASS' if p_cf >= thr else 'FAIL'} approves | "
              f"immutables untouched: {'PASS' if imm_ok else 'FAIL'} | "
              f"bounds/direction: {'PASS' if bounds_ok else 'FAIL'} | "
              f"cost<=budget: {'PASS' if res['cost'] <= cfg['recourse']['max_cost'] else 'FAIL'}")
    elif "min_cost_over_budget" in res:
        print(f"    NO RECOURSE within budget (cheapest found costs {res['min_cost_over_budget']:.2f} > {cfg['recourse']['max_cost']})")
    else:
        print("    NO RECOURSE: no allowed change flips the decision")
