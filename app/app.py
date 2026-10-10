"""Interactive recourse explorer (optional).

    streamlit run app/app.py

Pick a model and a denied applicant from the test set to see the cheapest
feasible path to approval, or browse the group-level audit results.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_config, output_dirs, prepare  # noqa: E402
from src.model import split, train  # noqa: E402
from src.recourse import RecourseSearch  # noqa: E402

st.set_page_config(page_title="Recourse Audit", layout="wide")


@st.cache_data(show_spinner="Loading data...")
def get_data():
    cfg = load_config()
    X, y, groups = prepare(cfg)
    return cfg, split(X, y, groups, cfg)


@st.cache_resource(show_spinner="Training model...")
def get_model(kind: str):
    cfg, (X_train, _, y_train, *_rest) = get_data()
    saved = output_dirs(cfg)[0] / f"model_{kind}.joblib"
    if saved.exists():
        import joblib
        return joblib.load(saved)
    return train(kind, X_train, y_train, seed=cfg["data"]["seed"])


cfg, (X_train, X_test, y_train, y_test, g_train, g_test) = get_data()
st.title("Public-sector recourse audit")
st.caption("If the model denies you, what is the cheapest realistic change that "
           "would get you approved, and is that path equally open to every group?")

kind = st.sidebar.selectbox("Model", cfg["model"]["kinds"])
model = get_model(kind)
thr = cfg["model"]["threshold"]
max_cost = st.sidebar.slider("Recourse budget (max cost)", 1.0, 15.0,
                             float(cfg["recourse"]["max_cost"]), 0.5)

tab_ind, tab_audit = st.tabs(["Individual recourse", "Group audit"])

with tab_ind:
    proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
    denied = proba[proba < thr].index
    c1, c2 = st.columns(2)
    attr = c1.selectbox("Filter by attribute", ["(none)", *g_test.columns])
    pool = denied
    if attr != "(none)":
        grp = c2.selectbox("Group", sorted(g_test[attr].unique()))
        pool = denied[g_test.loc[denied, attr] == grp]
    row_id = st.selectbox(f"Denied applicant ({len(pool)} available)", pool[:500])

    x = X_test.loc[row_id]
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Applicant profile")
        st.dataframe(pd.concat([x, g_test.loc[row_id]]).astype(str).rename("value"),
                     use_container_width=True)
        st.metric("Approval probability", f"{proba[row_id]:.1%}", help=f"threshold {thr:.0%}")

    with right:
        st.subheader("Cheapest path to approval")
        search = RecourseSearch.from_config(model, cfg, X_train)
        search.max_cost = max_cost
        with st.spinner("Searching counterfactuals..."):
            res = search.explain(x)
        if res["found"]:
            st.success(f"Recourse found - cost {res['cost']:.2f}, "
                       f"{res['n_changed']} change(s); new probability "
                       f"{res['prob_counterfactual']:.1%}")
            st.table(pd.DataFrame(
                [c.split(": ", 1)[0:1] + c.split(": ", 1)[1].split(" -> ")
                 for c in res["changes"].split("; ")],
                columns=["feature", "current", "recommended"]))
        elif "min_cost_over_budget" in res:
            st.warning(f"Cheapest recourse costs {res['min_cost_over_budget']:.2f}, "
                       f"above the budget of {max_cost:.1f}.")
        else:
            st.error("No feasible recourse: no allowed change to the actionable "
                     "features gets this applicant approved.")
        st.caption("Immutable: " + ", ".join(cfg["immutable"]) +
                   ". Actionable: " + ", ".join(search.space.features) + ".")

with tab_audit:
    PROCESSED_DIR, FIGURES_DIR = output_dirs(cfg)
    path = PROCESSED_DIR / f"recourse_by_group_{kind}.csv"
    if not path.exists():
        st.info("Run `python run_audit.py` first to generate the group audit.")
    else:
        st.subheader("Recourse by group")
        st.dataframe(pd.read_csv(path).round(3), use_container_width=True)
        disp = PROCESSED_DIR / f"recourse_disparities_{kind}.csv"
        if disp.exists():
            st.subheader("Disparities")
            st.dataframe(pd.read_csv(disp).round(3), use_container_width=True)
        for name in ["recourse_rate", "recourse_mean_cost", "recourse_cost_dist",
                     "feature_changes", "false_negative_rate"]:
            fig = FIGURES_DIR / f"{name}_{kind}.png"
            if fig.exists():
                st.image(str(fig), use_container_width=True)
