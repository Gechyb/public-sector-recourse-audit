"""Interactive Altair charts for the app. Same colours and group order as the
static figures in src/plots.py; every chart has hover tooltips, and the app
shows the underlying table beside each one."""
from __future__ import annotations

import altair as alt
import pandas as pd

from src.plots import AGE_ORDER, SERIES

ATTR_LABELS = {"sex": "Sex", "race_group": "Race", "age_group": "Age group"}
GROUP_ORDER = {"sex": ["Female", "Male"], "race_group": ["Black", "Other", "White"],
               "age_group": AGE_ORDER}
PCT = ".0%"


def _color(domain: list[str]) -> alt.Color:
    return alt.Color("series:N", title=None, scale=alt.Scale(domain=domain, range=SERIES[: len(domain)]),
                     legend=alt.Legend(orient="bottom"))


def recourse_by_group(by_group: pd.DataFrame, uncon: pd.DataFrame, attr: str) -> alt.Chart:
    """Realistic vs off-the-shelf recourse rate for each group of one attribute."""
    a = by_group[by_group["attribute"] == attr].assign(series="realistic recourse")
    b = uncon[uncon["attribute"] == attr].assign(series="off-the-shelf tool")
    d = pd.concat([a, b])[["group", "series", "recourse_rate", "n_denied_audited"]]
    domain = ["realistic recourse", "off-the-shelf tool"]
    base = alt.Chart(d).encode(
        y=alt.Y("group:N", sort=GROUP_ORDER.get(attr), title=None),
        yOffset=alt.YOffset("series:N", sort=domain),
        x=alt.X("recourse_rate:Q", title="denied applicants with recourse",
                axis=alt.Axis(format=PCT), scale=alt.Scale(domain=[0, 1.15])),
        tooltip=[alt.Tooltip("group:N"), alt.Tooltip("series:N", title="method"),
                 alt.Tooltip("recourse_rate:Q", format=PCT, title="recourse rate"),
                 alt.Tooltip("n_denied_audited:Q", title="denied applicants audited")])
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.9}).encode(color=_color(domain))
    labels = base.mark_text(align="left", dx=3, fontSize=11).encode(text=alt.Text("recourse_rate:Q", format=PCT))
    return (bars + labels).properties(title=ATTR_LABELS.get(attr, attr), height=60 * d["group"].nunique())


def gap_forest(gaps: pd.DataFrame, series_col: str, domain: list[str], title: str) -> alt.Chart:
    """Point + 95% CI per comparison; ``series_col`` picks what the colours mean."""
    d = gaps.assign(comparison=gaps["group"] + " vs " + gaps["reference"],
                    series=gaps[series_col],
                    gap_pp=gaps["estimate"] * 100, lo_pp=gaps["ci_low"] * 100, hi_pp=gaps["ci_high"] * 100)
    order = list(dict.fromkeys(d["comparison"]))
    enc = dict(y=alt.Y("comparison:N", sort=order, title=None),
               yOffset=alt.YOffset("series:N", sort=domain),
               color=_color(domain))
    tip = [alt.Tooltip("comparison:N"), alt.Tooltip("series:N", title=" "),
           alt.Tooltip("gap_pp:Q", format="+.1f", title="gap (pp)"),
           alt.Tooltip("lo_pp:Q", format="+.1f", title="95% CI low (pp)"),
           alt.Tooltip("hi_pp:Q", format="+.1f", title="95% CI high (pp)"),
           alt.Tooltip("n_group:Q", title="n group"), alt.Tooltip("n_reference:Q", title="n reference")]
    x = alt.X("gap_pp:Q", title="difference in recourse rate (percentage points, 95% CI)",
              axis=alt.Axis(labelExpr="format(datum.value, '+.0f') + ' pp'"))
    ci = alt.Chart(d).mark_rule(strokeWidth=2.5).encode(x=alt.X("lo_pp:Q"), x2="hi_pp:Q", tooltip=tip, **enc)
    pts = alt.Chart(d).mark_point(filled=True, size=80, opacity=1, stroke="white", strokeWidth=1.5) \
        .encode(x=x, tooltip=tip, **enc)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="gray").encode(x="x:Q")
    return (zero + ci + pts).properties(title=title, height=46 * len(order) * max(1, len(domain) / 2))


def budget_curves(curves: pd.DataFrame, attr: str, budget: float) -> alt.Chart:
    """Recourse rate vs allowed effort for one attribute, with a hover crosshair."""
    d = curves[curves["attribute"] == attr].rename(columns={"group": "series"})
    domain = [g for g in GROUP_ORDER.get(attr, sorted(d["series"].unique())) if g in set(d["series"])]
    hover = alt.selection_point(fields=["budget"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(d).encode(
        x=alt.X("budget:Q", title="allowed effort (1 = one year of school)"),
        y=alt.Y("recourse_rate:Q", title="denied applicants with recourse",
                axis=alt.Axis(format=PCT), scale=alt.Scale(domain=[0, 1])),
        color=_color(domain))
    lines = base.mark_line(strokeWidth=2.5)
    pts = base.mark_point(filled=True, size=70).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("series:N", title="group"), alt.Tooltip("budget:Q", title="effort"),
                 alt.Tooltip("recourse_rate:Q", format=PCT, title="with recourse")]).add_params(hover)
    rule = alt.Chart(pd.DataFrame({"budget": [budget]})).mark_rule(strokeDash=[4, 4], color="gray") \
        .encode(x="budget:Q")
    return (lines + pts + rule).properties(title=ATTR_LABELS.get(attr, attr), height=260)


def violation_types(types: pd.DataFrame, labels: dict[str, str], top: int = 8) -> alt.Chart:
    """What off-the-shelf recourse asks people to change, per robustness variant."""
    keep = (types.groupby("violation")["share_of_recommendations"].max()
            .sort_values(ascending=False).index[:top])
    d = types[types["violation"].isin(keep)].assign(
        series=types["variant"].map(labels).fillna(types["variant"]),
        rule=lambda t: t["violation"].str.replace(":", ": ").str.replace("_", " "))
    domain = list(dict.fromkeys(d["series"]))
    order = [v.replace(":", ": ").replace("_", " ") for v in keep]
    base = alt.Chart(d).encode(
        y=alt.Y("rule:N", sort=order, title=None),
        yOffset=alt.YOffset("series:N", sort=domain),
        x=alt.X("share_of_recommendations:Q", title="share of off-the-shelf recommendations",
                axis=alt.Axis(format=PCT), scale=alt.Scale(domain=[0, 1.1])),
        tooltip=[alt.Tooltip("rule:N", title="broken rule"), alt.Tooltip("series:N", title="variant"),
                 alt.Tooltip("share_of_recommendations:Q", format=PCT, title="share"),
                 alt.Tooltip("count:Q")])
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.9}).encode(color=_color(domain))
    text = base.mark_text(align="left", dx=3, fontSize=11).encode(
        text=alt.Text("share_of_recommendations:Q", format=PCT))
    return (bars + text).properties(height=34 * len(order) * len(domain))
