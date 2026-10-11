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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import charts  # noqa: E402

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

tab_find, tab_try, tab_groups, tab_compare, tab_method = st.tabs(
    ["Findings", "Try an applicant", "Group audit", "1994 vs 2023", "Method"])


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
    st.info(
        "**What is recourse?** If a model denies you, recourse is a change *you* could make "
        "to get approved. Example from this data: a 32-year-old denied at 40 hours a week "
        "would be approved at **45 hours**. An explanation says *why* you were denied; recourse "
        "says *what you can do about it*. Advice like \"be older\" or \"get married\" may flip "
        "the model, but it is not recourse, because no one can act on it.")
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


# --- group audit ------------------------------------------------------------

def _table(df: pd.DataFrame, label: str = "Show the numbers"):
    with st.expander(label):
        st.dataframe(df.round(3), hide_index=True, width="stretch")


with tab_groups:
    needed = {"recourse_by_group", "violations_by_group", "group_gaps", "budget_curves", "violation_types"}
    if not needed <= set(results):
        st.info("No audit results for this dataset yet. Run `make audit` first.")
    else:
        variants = list(cfg["unconstrained"]["variants"])
        variant = st.radio("Off-the-shelf variant", variants, format_func=VARIANT_LABELS.get,
                           horizontal=True, key="groups_variant") if len(variants) > 1 else variants[0]
        vio = results["violations_by_group"]
        uncon = vio[vio["variant"] == variant]
        attrs = list(dict.fromkeys(results["recourse_by_group"]["attribute"]))

        st.subheader("Who gets a realistic path to approval?")
        st.caption("Share of denied applicants with recourse. The off-the-shelf tool finds a path "
                   "for almost everyone, but much of its advice is impossible to follow.")
        for col, attr in zip(st.columns(len(attrs)), attrs):
            col.altair_chart(charts.recourse_by_group(results["recourse_by_group"], uncon, attr),
                             width="stretch")
        _table(results["recourse_by_group"])

        st.subheader("Which gaps are real?")
        st.caption("Each group vs the largest group, with 95% bootstrap confidence intervals. A bar "
                   "that crosses 0 is not a clear difference. If a gap shrinks to 0 at the same "
                   "starting distance, it comes from where people start, not from the model's rules.")
        gaps = results["group_gaps"]
        gaps = gaps[gaps["metric"].isin(["recourse_rate_gap", "recourse_rate_gap_same_start"])].assign(
            how=lambda d: d["metric"].map({"recourse_rate_gap": "raw gap",
                                           "recourse_rate_gap_same_start": "same starting distance"}))
        st.altair_chart(charts.gap_forest(gaps, "how", ["raw gap", "same starting distance"], ""),
                        width="stretch")
        _table(gaps.drop(columns=["metric"]))

        st.subheader("Does more effort close the gap?")
        st.caption("Share with recourse as the allowed effort grows. The dashed line is the budget "
                   "in the sidebar. A curve that stays flat means more effort does not help.")
        for col, attr in zip(st.columns(len(attrs)), attrs):
            col.altair_chart(charts.budget_curves(results["budget_curves"], attr, max_cost),
                             width="stretch")
        _table(results["budget_curves"].pivot_table(index="budget", columns=["attribute", "group"],
                                                    values="recourse_rate").reset_index())

        st.subheader("What does off-the-shelf recourse ask people to change?")
        st.caption("Rules broken by unconstrained recommendations: \"immutable\" means something a "
                   "person cannot (or should not be told to) change.")
        st.altair_chart(charts.violation_types(results["violation_types"], VARIANT_LABELS),
                        width="stretch")
        _table(results["violation_types"])


# --- 1994 vs 2023 -----------------------------------------------------------

with tab_compare:
    st.markdown(
        "The same audit on the 1994 benchmark and on 2023 North Carolina census data. "
        "**$103K** is Adult's $50K in 2023 dollars (same real bar); **$50K** is the standard "
        "ACSIncome bar, kept for comparison with published work.")
    every = {name: get_results(cfg_file, kind) for name, cfg_file in DATASETS.items()}
    every = {k: v for k, v in every.items() if "summary" in v and "group_gaps" in v}
    if len(every) < 2:
        st.info("Run `make audit` to produce results for at least two datasets.")
    else:
        rows = []
        for name, r in every.items():
            s = r["summary"].iloc[0]
            var = "no_capital" if "violation_rate_no_capital" in s and pd.notna(s["violation_rate_no_capital"]) \
                else "all_features"
            rows.append({"dataset": name, "approved": s["approval_rate"],
                         "denied with realistic recourse": s["recourse_rate"],
                         "off-the-shelf advice breaking a rule": s[f"violation_rate_{var}"]})
        head = pd.DataFrame(rows)
        st.dataframe(head.style.format({c: "{:.0%}" for c in head.columns if c != "dataset"}),
                     hide_index=True, width="stretch")
        st.caption("Off-the-shelf figure for Adult holds capital gain/loss fixed, "
                   "so it is comparable with ACS, which has no capital columns.")

        which = st.radio("Gap", ["recourse_rate_gap_same_start", "recourse_rate_gap"], horizontal=True,
                         format_func={"recourse_rate_gap": "raw gap",
                                      "recourse_rate_gap_same_start": "at the same starting distance"}.get)
        gaps = pd.concat([r["group_gaps"].assign(dataset=name) for name, r in every.items()])
        gaps = gaps[gaps["metric"] == which]
        st.altair_chart(charts.gap_forest(gaps, "dataset", list(every), ""), width="stretch")
        st.caption("Each group vs the largest group, 95% bootstrap CIs. Compare the colours within a row "
                   "to see how a gap changed between 1994 and 2023.")
        _table(gaps.drop(columns=["metric"]))


# --- method -----------------------------------------------------------------

def rules_table(cfg: dict, X: pd.DataFrame) -> pd.DataFrame:
    """One row per model feature: how recourse may change it, and at what cost."""
    num, cat = cfg["actionable"].get("numeric", {}), cfg["actionable"].get("categorical", {})
    rows = []
    for f in X.columns:
        if f in num:
            a = num[f]
            kind = "monotonic" if a["direction"] in ("increase", "decrease") else "bounded"
            rule = f"{a['direction']} only" if kind == "monotonic" else "up or down"
            rows.append({"feature": f, "type": kind, "allowed change": f"{rule}, {a['min']:g} to {a['max']:g}",
                         "cost": f"{a['cost_per_unit']:g} per unit"})
        elif f in cat:
            a = cat[f]
            excl = f" (never: {', '.join(a.get('exclude', []))})" if a.get("exclude") else ""
            rows.append({"feature": f, "type": "bounded", "allowed change": f"switch category{excl}",
                         "cost": f"{a['cost']:g} per switch"})
        else:
            rows.append({"feature": f, "type": "immutable", "allowed change": "none", "cost": "n/a"})
    return pd.DataFrame(rows)


with tab_method:
    st.subheader("What counts as recourse")
    st.markdown(
        "Every feature the model uses is **immutable** (the person cannot, or should not be told "
        "to, change it), **monotonic** (one direction only), or **bounded** (either direction, "
        "within realistic limits). Sex and race are never model inputs, so recourse can never "
        "recommend changing them, but proxies such as relationship and marital status are.")
    st.dataframe(rules_table(cfg, X_train), hide_index=True, width="stretch")
    st.caption("Costs are effort units: 1 = one extra year of school. They are assumptions made "
               "explicit so they can be argued with; compare them across groups, not as absolutes.")
    cfg_path = ROOT / "config" / config
    with st.expander(f"The constraints file, with a reason for every rule ({cfg_path.name})"):
        st.code(cfg_path.read_text(), language="yaml")
        parent = yaml.safe_load(cfg_path.read_text()).get("extends")
        if parent:  # this config only overrides a few lines of another
            st.caption(f"Everything else is inherited from {parent}:")
            st.code((cfg_path.parent / parent).read_text(), language="yaml")

    st.subheader("How the audit works")
    st.markdown(f"""
1. **Train** logistic regression and gradient boosting on 70% of the data; audit denied
   applicants from the other 30% (about 800 per model, at least 100 per group).
2. **Realistic recourse.** For each denied applicant, search for the cheapest set of up to
   {cfg['recourse']['max_features_changed']} allowed changes that flips the decision: every
   single change, then {cfg['recourse']['n_candidates']:,} random combinations, then shrink the
   best one back toward the original. Over a budget of {cfg['recourse']['max_cost']:g}, the
   applicant has no recourse.
3. **Off-the-shelf baseline.** The same search with every feature changeable and plain
   distance as the cost, like generic counterfactual tools
   (Wachter et al. 2017; DiCE). Each recommendation is checked against the rules above;
   the share that breaks one is the **violation rate**.
4. **Is a gap real?** Each group is compared with the largest group, with 95% bootstrap
   confidence intervals ({cfg['analysis']['n_bootstrap']:,} resamples).
5. **Same starting distance.** Denied applicants are split into
   {cfg['analysis']['distance_bins']} bands by how close they were to approval, and groups are
   compared within bands. A gap that survives is not explained by where people start.
6. **Budget curves** show whether a gap closes as more effort is allowed.
""")

    st.subheader("Limitations")
    st.markdown("""
- **The label is a proxy.** Income above a threshold is not a real benefit decision; in means-tested
  programs a *lower* income qualifies. The contribution is the audit protocol, not a claim about
  any agency.
- **Costs and immutability are judgments.** Treating marital status as immutable, or 20 extra
  hours as equal to a year of school, are explicit choices. Changing them changes the numbers;
  that is why they live in one readable file.
- **The search is heuristic.** "No recourse found" means none was found within the action space
  and budget, not proof that none exists.
- **Starting-distance bands are coarse.** Five bands reduce, but do not remove, differences in
  where groups start; intervals are wide where few people of a group start near approval
  (e.g. under-30s).
- **Recoding ACS to Adult's schema is approximate** (24 school levels to 16, ~530 occupation
  codes to 23 groups, birthplace to world region).
- **Recourse is for the model as trained.** If the model is retrained, a recommended change may
  no longer lead to approval.
- **Groups are coarse.** Race is White / Black / Other, and intersections (e.g. young Black
  women) are not audited separately.
""")

    st.subheader("Intended use")
    st.markdown("""
A **research audit on public survey data**, not an eligibility system and not advice to real people.
It is meant for **auditors and agencies** evaluating a decision model before deployment, and for
**researchers** comparing recourse methods. It should not be used to tell an individual what to
change, or to certify a model as fair: a model with equal recourse can still be unfair in other ways.
""")

    st.subheader("References")
    st.markdown("""
- Wachter, Mittelstadt & Russell (2017). Counterfactual explanations without opening the black box. *Harvard Journal of Law & Technology*.
- Ustun, Spangher & Liu (2019). Actionable recourse in linear classification. *FAT\\* 2019*.
- Gupta, Nokhiz, Roy & Venkatasubramanian (2019). Equalizing recourse across groups. *arXiv:1909.03166*.
- Mothilal, Sharma & Tan (2020). Explaining machine learning classifiers through diverse counterfactual explanations (DiCE). *FAT\\* 2020*.
- Ding, Hardt, Miller & Schmidt (2021). Retiring Adult: New datasets for fair machine learning. *NeurIPS 2021*.
- Karimi, Barthe, Schölkopf & Valera (2022). A survey of algorithmic recourse. *ACM Computing Surveys*.
- von Kügelgen, Karimi, Bhatt, Valera, Weller & Schölkopf (2022). On the fairness of causal algorithmic recourse. *AAAI 2022*.
""")
