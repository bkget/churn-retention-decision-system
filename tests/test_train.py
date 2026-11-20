"""Unit tests for split + metric helpers in src/train.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.train import evaluate, f1_optimal_threshold, split_data, top_k_lift


def test_split_is_disjoint_stratified_and_sized():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=1000), "churn": (rng.random(1000) < 0.27).astype(int)})
    train, val, test = split_data(df, test_size=0.2, val_size=0.2, seed=42)
    assert (len(train), len(val), len(test)) == (600, 200, 200)
    assert not set(train.index) & set(val.index)
    assert not set(train.index) & set(test.index)
    assert not set(val.index) & set(test.index)
    for part in (train, val, test):
        assert part["churn"].mean() == pytest.approx(df["churn"].mean(), abs=0.01)


def test_top_k_lift_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    proba = np.linspace(1, 0, 10)
    # top 20% = both churners -> rate 1.0 vs base 0.2 -> lift 5
    assert top_k_lift(y, proba, 0.2) == pytest.approx(5.0)


def test_f1_optimal_threshold_separable():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    thr = f1_optimal_threshold(y, proba)
    assert evaluate(y, proba, thr)["f1"] == pytest.approx(1.0)


def test_evaluate_counts():
    y = np.array([1, 0, 1, 0])
    proba = np.array([0.9, 0.6, 0.4, 0.1])
    m = evaluate(y, proba, 0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["flagged_share"] == pytest.approx(0.5)
