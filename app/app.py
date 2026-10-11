"""Public-sector recourse audit: interactive explorer.

    streamlit run app/app.py      (or: make app)

Pick a dataset and model, then explore the audit's findings, try individual
applicants (realistic recourse vs what an off-the-shelf tool would say), and
compare 1994 with 2023.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_config, output_dirs, prepare  # noqa: E402
from src.metrics import constraint_violations  # noqa: E402
from src.model import split, train  # noqa: E402
from src.recourse import RecourseSearch  # noqa: E402

DATASETS = {
    "Adult 1994 (UCI benchmark)": "constraints.yaml",
    "ACS North Carolina 2023, $103K bar": "constraints_acs_nc.yaml",
    "ACS North Carolina 2023, $50K bar": "constraints_acs_nc_50k.yaml",
}
MODELS = {"gbm": "Gradient boosting", "logistic": "Logistic regression"}
ATTR_LABELS = {"sex": "Sex", "race_group": "Race", "age_group": "Age group"}
VARIANT_LABELS = {"all_features": "all features mutable",
                  "no_capital": "capital gain/loss held fixed"}

st.set_page_config(page_title="Recourse Audit", page_icon="⚖️", layout="wide")


# --- cached loading ---------------------------------------------------------

@st.cache_data(show_spinner="Loading data...")
def get_data(config: str):
    cfg = load_config(ROOT / "config" / config)
    X, y, groups = prepare(cfg)
    return cfg, split(X, y, groups, cfg)


@st.cache_resource(show_spinner="Training model (first time only)...")
def get_model(config: str, kind: str):
    cfg, (X_train, _, y_train, *_rest) = get_data(config)
    saved = output_dirs(cfg)[0] / f"model_{kind}.joblib"
    if saved.exists():
        return joblib.load(saved)
    return train(kind, X_train, y_train, seed=cfg["data"]["seed"])


@st.cache_data
def get_results(config: str, kind: str) -> dict[str, pd.DataFrame]:
    """The audit's saved CSVs for this dataset and model (empty if not run yet)."""
    cfg = load_config(ROOT / "config" / config)
    proc = output_dirs(cfg)[0]
    out = {}
    for name in ["recourse_by_group", "group_gaps", "violations_by_group", "violation_types",
                 "budget_curves", "group_performance"]:
        path = proc / f"{name}_{kind}.csv"
        if path.exists():
            out[name] = pd.read_csv(path)
    if (proc / "model_summary.csv").exists():
        summary = pd.read_csv(proc / "model_summary.csv")
        out["summary"] = summary[summary["model"] == kind]
    return out


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    dataset = st.selectbox("Dataset", list(DATASETS), help=(
        "Adult is the standard 1994 benchmark. ACS is 2023 North Carolina census data. "
        "$103K is Adult's $50K in 2023 dollars; $50K is the usual ACSIncome bar."))
    kind = st.selectbox("Model", list(MODELS), format_func=MODELS.get)
    config = DATASETS[dataset]
    cfg, (X_train, X_test, y_train, y_test, g_train, g_test) = get_data(config)
    model = get_model(config, kind)
    results = get_results(config, kind)
    thr = cfg["model"]["threshold"]
    max_cost = st.slider("Recourse budget (max effort)", 1.0, 15.0,
                         float(cfg["recourse"]["max_cost"]), 0.5,
                         help="1 unit = one extra year of school. The audit used "
                              f"{cfg['recourse']['max_cost']:g}.")
    st.caption("Research audit on public survey data. Not a real eligibility system.")

st.title("Is recourse equally open to everyone?")
st.markdown(
    "When an automated eligibility model says **no**, is there a realistic change the "
    "person could make to get a **yes**, and is that path as open and as cheap for every "
    "group? This audit compares **realistic recourse** (only changes a person can actually "
    "make) with what an **off-the-shelf counterfactual tool** would recommend.")

tab_find, tab_try = st.tabs(["Findings", "Try an applicant"])


# --- findings ---------------------------------------------------------------

def _pp(v: float) -> str:
    return f"{v * 100:+.0f} pp"


def finding_sentences(gaps: pd.DataFrame) -> list[str]:
    """One plain-language line per comparison, from the bootstrap results."""
    lines = []
    for (attr, grp, ref), d in gaps.groupby(["attribute", "group", "reference"], sort=False):
        d = d.set_index("metric")
        if "recourse_rate_gap" not in d.index:
            continue
        raw, same = d.loc["recourse_rate_gap"], d.loc["recourse_rate_gap_same_start"]
        if not raw["significant"]:
            verdict = "no clear difference"
        elif same["significant"]:
            verdict = (f"**{_pp(raw['estimate'])}**, and still **{_pp(same['estimate'])}** "
                       f"for people starting equally far from approval "
                       f"(95% CI {_pp(same['ci_low'])} to {_pp(same['ci_high'])}): "
                       "not explained by starting distance")
        else:
            verdict = (f"**{_pp(raw['estimate'])}**, but about the same "
                       f"({_pp(same['estimate'])}, CI includes 0) at the same starting distance: "
                       "explained by where people start")
        lines.append(f"**{grp}** vs {ref}: {verdict}.")
    return lines


with tab_find:
    if "summary" not in results:
        st.info("No audit results for this dataset yet. Run `make audit` first.")
    else:
        s = results["summary"].iloc[0]
        variant = "no_capital" if "violation_rate_no_capital" in s else "all_features"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Approved by the model", f"{s['approval_rate']:.0%}",
                  help="Share of the test set the model approves.")
        c2.metric("Denied applicants with realistic recourse", f"{s['recourse_rate']:.0%}",
                  help=f"Within a budget of {cfg['recourse']['max_cost']:g}, changing only "
                       "what a person can actually change.")
        c3.metric("Off-the-shelf tool finds recourse for",
                  f"{s[f'uncon_recourse_rate_{variant}']:.0%}",
                  help=f"Unconstrained baseline ({VARIANT_LABELS[variant]}).")
        c4.metric("...but its advice breaks a rule", f"{s[f'violation_rate_{variant}']:.0%}",
                  help="Share of its recommendations that change something immutable "
                       "(age, marital status, ...), go out of bounds, or lower education.")

        st.subheader("Who has less recourse?")
        st.caption("Each group vs the largest group (Male, White, 30-49), with 95% bootstrap "
                   "confidence intervals. \"Same starting distance\" compares people who were "
                   "equally far from approval.")
        if "group_gaps" in results:
            for line in finding_sentences(results["group_gaps"]):
                st.markdown(f"- {line}")

        if "violation_types" in results:
            vt = results["violation_types"]
            top = vt[vt["variant"] == variant].sort_values("count", ascending=False).head(3)
            asks = ", ".join(f"{r.violation.split(':')[1].replace('_', ' ')} "
                             f"({r.share_of_recommendations:.0%})" for r in top.itertuples())
            st.subheader("What off-the-shelf recourse asks for")
            st.markdown(f"Most often it tells people to change: **{asks}**, none of which "
                        "a person can (or should be told to) change.")


# --- try an applicant -------------------------------------------------------

def run_search(x: pd.Series, mode: str, freeze=(), budget=None) -> dict:
    search = RecourseSearch.from_config(model, cfg, X_train, mode=mode, freeze=freeze)
    if budget is not None:
        search.max_cost = budget
    return search.explain(x)


def changes_table(x: pd.Series, res: dict) -> pd.DataFrame:
    """Current vs recommended value for each changed feature, with any broken rule."""
    rec = pd.DataFrame([res], index=[x.name])
    flagged = constraint_violations(rec, x.to_frame().T, cfg).iloc[0]["violations"]
    rules = dict(v.split(":")[::-1] for v in flagged.split("; ") if v)
    rows = [{"feature": f.replace("_", " "), "current": str(x[f]), "recommended": str(res[f"cf_{f}"]),
             "allowed?": f"no: {rules[f]}" if f in rules else "yes"}
            for f in X_train.columns if f"cf_{f}" in res]
    return pd.DataFrame(rows)


def show_result(x: pd.Series, res: dict, budget: float | None):
    if res["found"]:
        st.metric("Approval probability after the change", f"{res['prob_counterfactual']:.0%}",
                  delta=f"{(res['prob_counterfactual'] - res['prob_original']) * 100:+.0f} pp")
        st.caption(f"Effort cost {res['cost']:.2f} · {res['n_changed']} change(s)")
        table = changes_table(x, res)
        st.dataframe(table, hide_index=True, width="stretch")
        if (table["allowed?"] != "yes").any():
            st.warning("This advice asks for something the person cannot or should not change.")
    elif "min_cost_over_budget" in res:
        st.warning(f"Cheapest path costs {res['min_cost_over_budget']:.2f}, "
                   f"over the budget of {budget:g}.")
    else:
        st.error("No path found: nothing this person can change gets them approved.")


with tab_try:
    proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
    denied = proba[proba < thr].index
    c1, c2 = st.columns(2)
    attr = c1.selectbox("Show applicants from", ["(everyone)", *g_test.columns],
                        format_func=lambda a: ATTR_LABELS.get(a, a))
    pool = denied
    if attr != "(everyone)":
        grp = c2.selectbox("Group", sorted(g_test[attr].unique()))
        pool = denied[g_test.loc[denied, attr] == grp]
    pool = pool[np.random.default_rng(0).permutation(len(pool))][:300]

    def label(i):
        g = g_test.loc[i]
        return f"#{i} · {g['sex']}, {g['race_group']}, age {X_test.loc[i, 'age']} · p={proba[i]:.2f}"

    row_id = st.selectbox(f"Denied applicant ({len(pool)} shown)", pool, format_func=label)
    x = X_test.loc[row_id].copy()

    with st.expander("What if? Edit this applicant's profile"):
        cols = st.columns(4)
        for k, f in enumerate(X_train.columns):
            box = cols[k % 4]
            if pd.api.types.is_numeric_dtype(X_train[f]):
                x[f] = box.number_input(f.replace("_", " "), value=int(x[f]),
                                        min_value=int(X_train[f].min()),
                                        max_value=int(X_train[f].max()), key=f"edit_{f}")
            else:
                opts = sorted(X_train[f].unique())
                x[f] = box.selectbox(f.replace("_", " "), opts, index=opts.index(x[f]), key=f"edit_{f}")
    p0 = float(model.predict_proba(x.to_frame().T.astype(X_train.dtypes))[:, 1][0])

    left, mid, right = st.columns([1, 1.2, 1.2])
    with left:
        st.subheader("Applicant")
        st.metric("Approval probability", f"{p0:.0%}",
                  help=f"Approved at {thr:.0%} or above.")
        prof = pd.concat([x.astype(str), g_test.loc[row_id].astype(str)])
        st.dataframe(prof.rename("value"), width="stretch", height=420)
    if p0 >= thr:
        mid.success("This profile is already approved. Pick another or edit it.")
    else:
        with mid:
            st.subheader("Realistic recourse")
            st.caption("Only changes a person can make: more schooling, different hours, "
                       "sector or occupation, within the budget.")
            with st.spinner("Searching..."):
                show_result(x, run_search(x, "constrained", budget=max_cost), max_cost)
        with right:
            st.subheader("Off-the-shelf recourse")
            variants = cfg["unconstrained"]["variants"]
            var = st.radio("Variant", list(variants), format_func=VARIANT_LABELS.get,
                           horizontal=True, label_visibility="collapsed") if len(variants) > 1 \
                else next(iter(variants))
            st.caption("What a generic counterfactual tool does: any feature may change, "
                       "cheapest by plain distance.")
            with st.spinner("Searching..."):
                show_result(x, run_search(x, "unconstrained", freeze=variants[var]), None)
