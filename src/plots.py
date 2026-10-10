"""Audit figures. One panel per protected attribute; groups are labelled on the
axis so colour is never the only identifier."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots in fixed order (validated for CVD separation up to 3 series).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "font.size": 9.5, "figure.dpi": 150,
})

LABELS = {"sex": "Sex", "race_group": "Race", "age_group": "Age group"}
AGE_ORDER = ["<30", "30-49", "50+"]


def _ordered(d: pd.DataFrame, attr: str) -> pd.DataFrame:
    if attr == "age_group":
        return d.set_index("group").reindex([g for g in AGE_ORDER if g in set(d["group"])]).reset_index()
    return d.sort_values("group")


def _bar_panels(df: pd.DataFrame, metric: str, title: str, xlabel: str, path: Path,
                pct: bool = True):
    attrs = list(dict.fromkeys(df["attribute"]))
    fig, axes = plt.subplots(1, len(attrs), figsize=(3.6 * len(attrs), 2.8), sharex=True)
    axes = np.atleast_1d(axes)
    vmax = df[metric].max() * 1.18
    for ax, attr in zip(axes, attrs):
        d = _ordered(df[df["attribute"] == attr], attr)
        y = np.arange(len(d))[::-1]
        ax.barh(y, d[metric], height=0.6, color=SERIES[: len(d)], edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, d[metric]):
            ax.text(v, yi, f" {v:.0%}" if pct else f" {v:.2f}", va="center", color=INK, fontsize=9)
        ax.set_yticks(y, d["group"])
        ax.set_title(LABELS.get(attr, attr))
        ax.set_xlim(0, vmax)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel)
        if pct:
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_recourse_rate(by_group: pd.DataFrame, path: Path, model_name: str):
    _bar_panels(by_group, "recourse_rate",
                f"Share of denied applicants with feasible recourse ({model_name})",
                "recourse rate", path)


def plot_mean_cost(by_group: pd.DataFrame, path: Path, model_name: str):
    _bar_panels(by_group, "mean_cost",
                f"Mean recourse cost among those with recourse ({model_name})",
                "mean cost (effort units)", path, pct=False)


def plot_error_rates(perf: pd.DataFrame, path: Path, model_name: str):
    _bar_panels(perf, "fnr",
                f"False-negative rate: qualified applicants denied ({model_name})",
                "false-negative rate", path)


def plot_cost_distribution(rec: pd.DataFrame, groups: pd.DataFrame, path: Path, model_name: str):
    df = rec[rec["found"]].join(groups, how="left")
    attrs = list(groups.columns)
    fig, axes = plt.subplots(1, len(attrs), figsize=(3.6 * len(attrs), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, attr in zip(axes, attrs):
        order = AGE_ORDER if attr == "age_group" else sorted(df[attr].unique())
        order = [g for g in order if g in set(df[attr])]
        data = [df.loc[df[attr] == g, "cost"].to_numpy() for g in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.5},
                        whiskerprops={"color": INK_2}, capprops={"color": INK_2})
        for patch, c in zip(bp["boxes"], SERIES):
            patch.set(facecolor=c, alpha=0.85, edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(range(1, len(order) + 1), order)
        ax.set_title(LABELS.get(attr, attr))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("recourse cost (effort units)")
    fig.suptitle(f"Distribution of recourse cost by group ({model_name})",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_changes(changes: pd.DataFrame, features: list[str], path: Path, model_name: str):
    labels = [f"{LABELS.get(a, a)}: {g}" for a, g in zip(changes["attribute"], changes["group"])]
    M = changes[features].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.5 * len(features) + 2.2, 0.38 * len(labels) + 1.2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=8.5,
                    color="white" if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(features)), features)
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    ax.tick_params(length=0)
    ax.set_title(f"How often each action appears in recourse ({model_name})")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


VARIANT_LABELS = {"all_features": "all features mutable",
                  "no_capital": "capital gain/loss held fixed"}


def plot_violation_rate(vio: pd.DataFrame, path: Path, model_name: str):
    """Grouped bars: share of unconstrained recommendations that break a constraint,
    per group, one bar per robustness variant."""
    variants = list(dict.fromkeys(vio["variant"]))
    attrs = list(dict.fromkeys(vio["attribute"]))
    fig, axes = plt.subplots(1, len(attrs), figsize=(3.8 * len(attrs), 3.0), sharex=True)
    axes = np.atleast_1d(axes)
    h = 0.8 / len(variants)
    for ax, attr in zip(axes, attrs):
        base = _ordered(vio[(vio["attribute"] == attr) & (vio["variant"] == variants[0])], attr)
        order = list(base["group"])
        y = np.arange(len(order))[::-1]
        for k, var in enumerate(variants):
            d = vio[(vio["attribute"] == attr) & (vio["variant"] == var)].set_index("group").reindex(order)
            yk = y + (len(variants) - 1) * h / 2 - k * h
            ax.barh(yk, d["violation_rate"], height=h, color=SERIES[k], edgecolor=SURFACE,
                    linewidth=2, label=VARIANT_LABELS.get(var, var))
            for yi, v in zip(yk, d["violation_rate"]):
                ax.text(v, yi, f" {v:.0%}", va="center", color=INK, fontsize=8)
        ax.set_yticks(y, order)
        ax.set_title(LABELS.get(attr, attr))
        ax.set_xlim(0, 1.15)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("recommendations breaking a constraint")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.suptitle(f"Unconstrained recourse: how often it asks for something not allowed ({model_name})",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(variants),
               frameon=False, fontsize=8.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_violation_types(types: pd.DataFrame, path: Path, model_name: str, top: int = 8):
    """Which constraints unconstrained recourse breaks, per robustness variant."""
    variants = list(dict.fromkeys(types["variant"]))
    keep = (types.groupby("violation")["share_of_recommendations"].max()
            .sort_values(ascending=False).index[:top])
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(keep) * len(variants) + 1.2))
    y = np.arange(len(keep))[::-1]
    h = 0.8 / len(variants)
    for k, var in enumerate(variants):
        d = types[types["variant"] == var].set_index("violation").reindex(keep)["share_of_recommendations"].fillna(0)
        yk = y + (len(variants) - 1) * h / 2 - k * h
        ax.barh(yk, d, height=h, color=SERIES[k], edgecolor=SURFACE, linewidth=2,
                label=VARIANT_LABELS.get(var, var))
        for yi, v in zip(yk, d):
            ax.text(v, yi, f" {v:.0%}", va="center", color=INK, fontsize=8)
    ax.set_yticks(y, [v.replace(":", ": ").replace("_", " ") for v in keep])
    ax.set_xlim(0, 1.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of unconstrained recommendations")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title(f"What unconstrained recourse asks people to change ({model_name})")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_budget_curves(curves: pd.DataFrame, path: Path, model_name: str, budget: float | None = None):
    """Recourse rate as the allowed cost grows: does the gap close with more effort?"""
    attrs = list(dict.fromkeys(curves["attribute"]))
    fig, axes = plt.subplots(1, len(attrs), figsize=(3.8 * len(attrs), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, attr in zip(axes, attrs):
        d = curves[curves["attribute"] == attr]
        order = AGE_ORDER if attr == "age_group" else sorted(d["group"].unique())
        for k, grp in enumerate([o for o in order if o in set(d["group"])]):
            s = d[d["group"] == grp]
            ax.plot(s["budget"], s["recourse_rate"], color=SERIES[k], linewidth=2, label=grp)
            ax.text(s["budget"].iloc[-1], s["recourse_rate"].iloc[-1], f" {grp}",
                    va="center", color=INK_2, fontsize=8.5)
        if budget is not None:
            ax.axvline(budget, color=INK_2, linewidth=1, linestyle="--")
            ax.text(budget, 0.02, " audit budget", color=INK_2, fontsize=8, va="bottom")
        ax.set_title(LABELS.get(attr, attr))
        ax.set_xlabel("allowed recourse cost (effort units)")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, d["budget"].max() * 1.18)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    axes[0].set_ylabel("denied applicants with recourse")
    fig.suptitle(f"Recourse rate as the allowed effort grows ({model_name})",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


GAP_LABELS = {"recourse_rate_gap": "raw gap",
              "recourse_rate_gap_same_start": "same starting distance"}


def plot_gaps(gaps: pd.DataFrame, path: Path, model_name: str,
              metrics=("recourse_rate_gap", "recourse_rate_gap_same_start")):
    """Forest plot: each group's recourse-rate gap vs its reference group, with
    95% bootstrap CIs, raw and at the same starting distance from approval."""
    d = gaps[gaps["metric"].isin(metrics)]
    rows = list(dict.fromkeys(zip(d["attribute"], d["group"], d["reference"])))
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(rows) + 1.3))
    y = np.arange(len(rows))[::-1]
    off = 0.32 / max(len(metrics) - 1, 1)
    for k, m in enumerate(metrics):
        s = d[d["metric"] == m].set_index(["attribute", "group", "reference"]).reindex(rows)
        yk = y + 0.16 - k * off
        ax.errorbar(s["estimate"], yk, xerr=[s["estimate"] - s["ci_low"], s["ci_high"] - s["estimate"]],
                    fmt="o", color=SERIES[k], markersize=6, elinewidth=2, capsize=0,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=GAP_LABELS.get(m, m))
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(y, [f"{g} vs {r}" for _, g, r in rows])
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:+.0f} pp"))
    ax.set_xlabel("difference in recourse rate (percentage points, 95% CI)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title(f"Who is less likely to have recourse? ({model_name})")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
