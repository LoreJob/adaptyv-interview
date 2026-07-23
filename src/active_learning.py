"""Active-learning backtest for the Round Advisor POC.

Question answered: *given a fixed experimental budget, does an informed
acquisition strategy find the real EGFR binders faster than random testing?*

Setup (pooled design, per the approved plan):

* Pool rounds 1 + 2 into one candidate set (~604 designs); a "hit" is a real
  binder (``binds == True``, 63 of them).
* Start from a small random seed of "already tested" designs.
* Each round, fit a random-forest binder classifier on the tested designs,
  score the untested ones with a UCB acquisition ``mean + kappa * std`` (std =
  disagreement across the forest's trees, i.e. epistemic uncertainty), and
  "test" the top ``batch_size`` — revealing their true labels.
* Baseline: the same budget spent on uniformly random selection.

Output: a discovery curve (hits found vs designs tested) averaged over several
random seeds, plus a headline number at a chosen budget.

This is a retrospective simulation on labels we already know — it estimates the
*sample efficiency* of the strategy, not a production accuracy claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

if __package__ in (None, ""):  # allow `python src/active_learning.py` and IDE Run
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.features import FEATURE_COLUMNS, load_features

RANDOM_STATE = 42


def _forest() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def _ucb_scores(pipe: Pipeline, X: np.ndarray, kappa: float) -> np.ndarray:
    """UCB acquisition = mean binder-probability + kappa * tree-disagreement.

    If the training set held a single class the forest cannot score binder
    probability; fall back to uniform-random exploration for that round.
    """
    imp = pipe.named_steps["impute"]
    rf = pipe.named_steps["rf"]
    if rf.n_classes_ < 2:
        return np.random.default_rng(RANDOM_STATE).random(len(X))
    Xi = imp.transform(X)
    pos = list(rf.classes_).index(1)
    # Per-tree positive-class probability; std across trees = uncertainty.
    per_tree = np.stack([t.predict_proba(Xi)[:, pos] for t in rf.estimators_])
    return per_tree.mean(axis=0) + kappa * per_tree.std(axis=0)


@dataclass
class BacktestResult:
    tested: np.ndarray          # cumulative # designs tested at each step
    informed_hits: np.ndarray   # mean cumulative hits found — informed strategy
    random_hits: np.ndarray     # mean cumulative hits found — random baseline
    total_hits: int
    n_designs: int

    def summary_at(self, budget: int) -> dict[str, float]:
        """Discovery stats at a given budget (nearest recorded step)."""
        i = int(np.argmin(np.abs(self.tested - budget)))
        inf, rnd = self.informed_hits[i], self.random_hits[i]
        return {
            "budget": int(self.tested[i]),
            "informed_found": float(inf),
            "random_found": float(rnd),
            "informed_recall": inf / self.total_hits,
            "random_recall": rnd / self.total_hits,
            "lift": (inf / rnd) if rnd > 0 else float("nan"),
        }


def _stratified_seed(y: np.ndarray, seed_size: int, rng) -> list[int]:
    """Random seed set guaranteed to contain at least one binder.

    A purely random seed of ~20 from a 10%-positive pool is often all-negative,
    which leaves the classifier untrainable. We force one positive in; the rest
    stay random, so the seed still reflects a realistic small labeled start.
    """
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    first = [int(rng.choice(pos))]
    remaining = [i for i in range(len(y)) if i != first[0]]
    rest = rng.choice(remaining, size=seed_size - 1, replace=False).tolist()
    return first + [int(i) for i in rest]


def _one_run(X: np.ndarray, y: np.ndarray, seed: int, seed_size: int,
             batch_size: int, budget: int, kappa: float) -> np.ndarray:
    """One informed simulation; returns cumulative hits at each tested count."""
    rng = np.random.default_rng(seed)
    n = len(y)
    tested = _stratified_seed(y, seed_size, rng)
    untested = [i for i in range(n) if i not in set(tested)]

    cum = [int(y[tested].sum())]
    while len(tested) < min(budget, n) and untested:
        pipe = _forest()
        pipe.fit(X[tested], y[tested])
        scores = _ucb_scores(pipe, X[untested], kappa)
        take = min(batch_size, len(untested))
        pick_local = np.argsort(scores)[::-1][:take]
        picked = [untested[j] for j in pick_local]
        tested.extend(picked)
        untested = [i for i in untested if i not in set(picked)]
        cum.append(int(y[tested].sum()))
    return np.array(cum)


def _random_curve(y: np.ndarray, seed_size: int, batch_size: int,
                  budget: int, n_seeds: int) -> tuple[np.ndarray, np.ndarray]:
    """Expected cumulative hits under random selection, averaged over seeds."""
    n = len(y)
    steps = _step_counts(seed_size, batch_size, budget, n)
    curves = []
    for s in range(n_seeds):
        rng = np.random.default_rng(1000 + s)
        order = rng.permutation(n)
        cum = [int(y[order[:k]].sum()) for k in steps]
        curves.append(cum)
    return steps, np.mean(curves, axis=0)


def _step_counts(seed_size: int, batch_size: int, budget: int, n: int) -> np.ndarray:
    steps = [seed_size]
    while steps[-1] < min(budget, n):
        steps.append(min(steps[-1] + batch_size, min(budget, n)))
    return np.array(steps)


def run_backtest(df: pd.DataFrame | None = None, *, seed_size: int = 20,
                 batch_size: int = 10, budget: int = 200, kappa: float = 1.0,
                 n_seeds: int = 5) -> BacktestResult:
    """Run the pooled active-learning backtest, averaged over ``n_seeds`` runs."""
    if df is None:
        df = load_features()
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["binds"].astype(int).to_numpy()

    steps = _step_counts(seed_size, batch_size, budget, len(y))
    informed = []
    for s in range(n_seeds):
        cum = _one_run(X, y, seed=s, seed_size=seed_size, batch_size=batch_size,
                       budget=budget, kappa=kappa)
        # Align length to steps (last run may stop early if pool exhausts).
        if len(cum) < len(steps):
            cum = np.concatenate([cum, np.full(len(steps) - len(cum), cum[-1])])
        informed.append(cum[:len(steps)])

    _, random_hits = _random_curve(y, seed_size, batch_size, budget, n_seeds)
    return BacktestResult(
        tested=steps,
        informed_hits=np.mean(informed, axis=0),
        random_hits=random_hits,
        total_hits=int(y.sum()),
        n_designs=len(y),
    )


if __name__ == "__main__":
    res = run_backtest()
    print(f"pool={res.n_designs} designs, {res.total_hits} real binders")
    print(f"{'tested':>7} {'informed':>9} {'random':>7}")
    for t, inf, rnd in zip(res.tested, res.informed_hits, res.random_hits):
        print(f"{t:7d} {inf:9.1f} {rnd:7.1f}")
    for b in (50, 100, 150):
        s = res.summary_at(b)
        print(f"\n@budget {s['budget']}: informed found {s['informed_found']:.1f} "
              f"({s['informed_recall']*100:.0f}% of binders) vs random "
              f"{s['random_found']:.1f} ({s['random_recall']*100:.0f}%) "
              f"— {s['lift']:.1f}x")
