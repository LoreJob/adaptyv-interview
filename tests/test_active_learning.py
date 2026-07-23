"""Sanity checks for the Round Advisor data + backtest pipeline.

Fast, deterministic checks — not exhaustive ML validation. They guard the two
claims the POC rests on: the data harmonizes to the expected shape, and the
informed acquisition strategy beats random selection.
"""

from __future__ import annotations

import numpy as np

from src.active_learning import run_backtest
from src.data_loader import UNIFIED_COLUMNS, load_designs
from src.model import precision_at_k


def test_designs_shape_and_schema():
    df = load_designs()
    assert list(df.columns) == UNIFIED_COLUMNS
    assert len(df) == 604
    assert set(df["round"].unique()) == {1, 2}
    # binding labels align with fitted KD in this dataset.
    assert int(df["binds"].sum()) == 63
    assert int(df["kd"].notna().sum()) == 63
    # no binder should carry zero expression.
    assert not ((df["binds"]) & (df["expression"] == "none")).any()


def test_precision_at_k():
    y = np.array([0, 1, 0, 1, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.7])  # top-3 are the 3 positives
    assert precision_at_k(y, scores, 3) == 1.0
    assert precision_at_k(y, scores, 5) == 0.6


def test_informed_beats_random():
    res = run_backtest(seed_size=20, batch_size=10, budget=80, n_seeds=2)
    assert res.total_hits == 63
    # discovery curves are non-decreasing.
    assert np.all(np.diff(res.informed_hits) >= 0)
    assert np.all(np.diff(res.random_hits) >= 0)
    # informed strategy finds at least as many binders as random at the budget.
    s = res.summary_at(80)
    assert s["informed_found"] >= s["random_found"]
    assert s["informed_recall"] > s["random_recall"]
