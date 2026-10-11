"""Redraw every figure from the result CSVs in data/processed/, without re-running
the (slow) recourse search.

    python scripts/make_figures.py      (or: make figures)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import plots  # noqa: E402
from src.data_loader import load_config, output_dirs  # noqa: E402

CONFIGS = ["constraints.yaml", "constraints_acs_nc.yaml", "constraints_acs_nc_50k.yaml"]
GROUP_COLS = ["sex", "race_group", "age_group"]


def redraw(cfg: dict) -> int:
    proc, figs = output_dirs(cfg)
    n = 0
    for kind in cfg["model"]["kinds"]:
        if not (proc / f"recourse_individual_{kind}.csv").exists():
            print(f"  skip {proc.name}/{kind}: no results (run the audit first)")
            continue
        read = lambda name: pd.read_csv(proc / f"{name}_{kind}.csv")  # noqa: E731
        by_group, perf, changes = read("recourse_by_group"), read("group_performance"), read("feature_changes")
        rec = read("recourse_individual").set_index("row_id")
        groups = rec[[c for c in GROUP_COLS if c in rec]]
        feats = [c for c in changes.columns if c not in ("attribute", "group", "n_with_recourse")]
        plots.plot_recourse_rate(by_group, figs / f"recourse_rate_{kind}.png", kind)
        plots.plot_mean_cost(by_group, figs / f"recourse_mean_cost_{kind}.png", kind)
        plots.plot_cost_distribution(rec.drop(columns=groups.columns), groups,
                                     figs / f"recourse_cost_dist_{kind}.png", kind)
        plots.plot_feature_changes(changes, feats, figs / f"feature_changes_{kind}.png", kind)
        plots.plot_error_rates(perf, figs / f"false_negative_rate_{kind}.png", kind)
        plots.plot_violation_rate(read("violations_by_group"), figs / f"violation_rate_{kind}.png", kind)
        plots.plot_violation_types(read("violation_types"), figs / f"violation_types_{kind}.png", kind)
        plots.plot_budget_curves(read("budget_curves"), figs / f"budget_curves_{kind}.png", kind,
                                 cfg["recourse"]["max_cost"])
        plots.plot_gaps(read("group_gaps"), figs / f"group_gaps_{kind}.png", kind)
        n += 9
    return n


COMPARISON = {"Adult 1994": "constraints.yaml", "ACS NC 2023, $103K": "constraints_acs_nc.yaml",
              "ACS NC 2023, $50K": "constraints_acs_nc_50k.yaml"}


def redraw_comparison() -> int:
    """1994 vs 2023: the same gaps across datasets, one figure per model and metric."""
    root = Path(__file__).resolve().parents[1]
    cfgs = {k: load_config(root / "config" / v) for k, v in COMPARISON.items()}
    n = 0
    for kind in cfgs["Adult 1994"]["model"]["kinds"]:
        paths = {k: output_dirs(c)[0] / f"group_gaps_{kind}.csv" for k, c in cfgs.items()}
        if not all(p.exists() for p in paths.values()):
            continue
        gaps = {k: pd.read_csv(p) for k, p in paths.items()}
        figs = output_dirs(cfgs["Adult 1994"])[1].parent
        for metric, suffix in [("recourse_rate_gap_same_start", "same_start"), ("recourse_rate_gap", "raw")]:
            plots.plot_gaps_across_datasets(gaps, figs / f"comparison_gaps_{suffix}_{kind}.png", kind, metric)
            n += 1
    return n


if __name__ == "__main__":
    for name in CONFIGS:
        cfg = load_config(Path(__file__).resolve().parents[1] / "config" / name)
        print(f"{cfg['data']['name']}: {redraw(cfg)} figures")
    print(f"1994 vs 2023 comparison: {redraw_comparison()} figures")
