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
