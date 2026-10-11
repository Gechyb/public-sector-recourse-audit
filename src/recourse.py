"""Model-agnostic, constraint-aware counterfactual recourse.

For each denied individual we search for the minimum-cost change to the
*actionable* features (per config/constraints.yaml) that makes the model
approve them. The search is:

1. an exhaustive sweep over every single-feature action,
2. random multi-feature actions (up to ``max_features_changed``),
3. a greedy shrink of the best approving action toward the original values,
   which lowers cost and removes unnecessary changes.

It only needs ``model.predict_proba`` on raw DataFrames, so it works for any
sklearn pipeline (linear or tree-based).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NumericAction:
    direction: str
    min: float
    max: float
    step: float
    cost_per_unit: float

    def values(self, current: float) -> np.ndarray:
        # steps are anchored at the current value (e.g. 40 -> 45, 50, ... hours)
        up = np.arange(current + self.step, self.max + self.step / 2, self.step)
        down = np.arange(current - self.step, self.min - self.step / 2, -self.step)
        if self.direction == "increase":
            return up
        if self.direction == "decrease":
            return down
        return np.concatenate([down[::-1], up])


@dataclass
class CategoricalAction:
    cost: float
    categories: list

    def values(self, current) -> np.ndarray:
        return np.array([c for c in self.categories if c != current], dtype=object)


class ActionSpace:
    def __init__(self, numeric: dict[str, NumericAction], categorical: dict[str, CategoricalAction]):
        self.numeric = numeric
        self.categorical = categorical
        self.features = list(numeric) + list(categorical)

    @classmethod
    def from_config(cls, cfg: dict, X: pd.DataFrame) -> "ActionSpace":
        act = cfg["actionable"]
        numeric = {f: NumericAction(**spec) for f, spec in act.get("numeric", {}).items()}
        categorical = {}
        for f, spec in act.get("categorical", {}).items():
            cats = sorted(set(X[f].unique()) - set(spec.get("exclude", [])))
            categorical[f] = CategoricalAction(cost=spec["cost"], categories=cats)
        overlap = set(numeric) | set(categorical)
        bad = overlap & set(cfg.get("immutable", []))
        if bad:
            raise ValueError(f"Features marked both actionable and immutable: {bad}")
        return cls(numeric, categorical)

    @classmethod
    def unconstrained(cls, cfg: dict, X: pd.DataFrame, freeze=()) -> "ActionSpace":
        """Baseline mimicking off-the-shelf counterfactual tools: every model input
        (except those in ``freeze``) is mutable in any direction within its observed
        range; cost is L1 distance with numeric features scaled by their std."""
        spec = cfg["unconstrained"]
        exclude = set(spec.get("exclude", []))
        numeric, categorical = {}, {}
        for f in X.columns:
            if f in freeze:
                continue
            if pd.api.types.is_numeric_dtype(X[f]):
                numeric[f] = NumericAction(direction="any", min=float(X[f].min()),
                                           max=float(X[f].max()), step=spec["numeric_step"][f],
                                           cost_per_unit=1.0 / float(X[f].std()))
            else:
                categorical[f] = CategoricalAction(cost=1.0,
                                                   categories=sorted(set(X[f].unique()) - exclude))
        return cls(numeric, categorical)

    def options(self, x: pd.Series) -> dict[str, np.ndarray]:
        spec = {**self.numeric, **self.categorical}
        return {f: spec[f].values(x[f]) for f in self.features}

    def cost(self, x: pd.Series, cands: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cost = np.zeros(len(cands))
        n_changed = np.zeros(len(cands), dtype=int)
        for f, a in self.numeric.items():
            d = np.abs(cands[f].to_numpy(dtype=float) - float(x[f]))
            cost += d * a.cost_per_unit
            n_changed += d > 0
        for f, a in self.categorical.items():
            ch = cands[f].to_numpy() != x[f]
            cost += ch * a.cost
            n_changed += ch
        return cost, n_changed


def _repeat(x: pd.Series, n: int, dtypes: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({c: np.repeat(x[c], n) for c in x.index}).astype(dtypes)


class RecourseSearch:
    def __init__(self, model, space: ActionSpace, dtypes: pd.Series, threshold: float = 0.5,
                 n_candidates: int = 1000, max_features_changed: int = 3,
                 max_cost: float = np.inf, seed: int = 0):
        self.model = model
        self.space = space
        self.dtypes = dtypes
        self.threshold = threshold
        self.n_candidates = n_candidates
        self.max_features = max_features_changed
        self.max_cost = max_cost if max_cost is not None else np.inf
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_config(cls, model, cfg: dict, X: pd.DataFrame,
                    mode: str = "constrained", freeze=()) -> "RecourseSearch":
        """``mode="constrained"`` uses the action space in the config;
        ``mode="unconstrained"`` uses the off-the-shelf baseline with no budget,
        holding the features in ``freeze`` fixed."""
        r = cfg["recourse"]
        if mode == "constrained":
            space, max_cost = ActionSpace.from_config(cfg, X), r["max_cost"]
        elif mode == "unconstrained":
            space, max_cost = ActionSpace.unconstrained(cfg, X, freeze), None
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
        return cls(model, space, X.dtypes,
                   threshold=cfg["model"]["threshold"], n_candidates=r["n_candidates"],
                   max_features_changed=r["max_features_changed"], max_cost=max_cost,
                   seed=r["seed"])

    def _proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df)[:, 1]

    def _candidates(self, x: pd.Series, options: dict[str, np.ndarray]) -> pd.DataFrame:
        blocks = []
        # 1. every single-feature action
        for f, vals in options.items():
            if len(vals):
                b = _repeat(x, len(vals), self.dtypes)
                b[f] = vals
                blocks.append(b)
        # 2. random multi-feature actions
        feats = [f for f, v in options.items() if len(v)]
        if len(feats) > 1 and self.n_candidates > 0:
            n = self.n_candidates
            b = _repeat(x, n, self.dtypes)
            k = self.rng.integers(2, min(self.max_features, len(feats)) + 1, n)
            order = np.argsort(self.rng.random((n, len(feats))), axis=1)
            for j, f in enumerate(feats):
                rows = (order == j).argmax(axis=1) < k  # f is among the first k picks
                if rows.any():
                    b.loc[rows, f] = self.rng.choice(options[f], rows.sum())
            blocks.append(b)
        return pd.concat(blocks, ignore_index=True).astype(self.dtypes) if blocks else _repeat(x, 0, self.dtypes)

    def _shrink(self, x: pd.Series, best: pd.Series, best_cost: float) -> tuple[pd.Series, float]:
        """Greedily move changed features back toward x while approval holds."""
        while True:
            rows = []
            for f in self.space.features:
                if best[f] == x[f]:
                    continue
                if f in self.space.numeric:
                    lo, hi = sorted([float(x[f]), float(best[f])])
                    grid = self.space.numeric[f].values(x[f])
                    closer = [v for v in grid if lo <= v <= hi and v != best[f]] + [x[f]]
                else:
                    closer = [x[f]]
                for v in closer:
                    r = best.copy()
                    r[f] = v
                    rows.append(r)
            if not rows:
                return best, best_cost
            cands = pd.DataFrame(rows).astype(self.dtypes)
            cost, _ = self.space.cost(x, cands)
            ok = (self._proba(cands) >= self.threshold) & (cost < best_cost - 1e-9)
            if not ok.any():
                return best, best_cost
            i = np.flatnonzero(ok)[np.argmin(cost[ok])]
            best, best_cost = cands.iloc[i], float(cost[i])

    def explain(self, x: pd.Series) -> dict:
        """Find the cheapest approving counterfactual for one individual."""
        x = x[self.dtypes.index]
        p0 = float(self._proba(x.to_frame().T.astype(self.dtypes))[0])
        out = {"prob_original": p0, "found": False, "cost": np.nan, "n_changed": 0,
               "prob_counterfactual": np.nan, "changes": ""}
        if p0 >= self.threshold:
            out.update(found=True, cost=0.0, prob_counterfactual=p0, changes="(already approved)")
            return out

        cands = self._candidates(x, self.space.options(x))
        if cands.empty:
            return out
        cost, _ = self.space.cost(x, cands)
        ok = self._proba(cands) >= self.threshold
        if not ok.any():
            return out
        i = np.flatnonzero(ok)[np.argmin(cost[ok])]
        best, best_cost = self._shrink(x, cands.iloc[i], float(cost[i]))
        if best_cost > self.max_cost:
            out["min_cost_over_budget"] = best_cost
            return out

        best_df = best.to_frame().T.astype(self.dtypes)
        _, n_changed = self.space.cost(x, best_df)
        changed = [f for f in self.space.features if best[f] != x[f]]
        out.update(
            found=True, cost=best_cost, n_changed=int(n_changed[0]),
            prob_counterfactual=float(self._proba(best_df)[0]),
            changes="; ".join(f"{f}: {x[f]} -> {best[f]}" for f in changed),
            **{f"changed_{f}": f in changed for f in self.space.features},
            **{f"cf_{f}": best[f] for f in changed},
        )
        return out

    def run(self, X: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        records = []
        for n, (idx, x) in enumerate(X.iterrows(), 1):
            records.append({"row_id": idx, **self.explain(x)})
            if verbose and (n % 100 == 0 or n == len(X)):
                print(f"    recourse {n}/{len(X)}", flush=True)
        df = pd.DataFrame(records).set_index("row_id")
        for f in self.space.features:
            col = f"changed_{f}"
            df[col] = df[col].eq(True) if col in df else False  # missing -> False
        return df
