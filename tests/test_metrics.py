"""Smoke tests for evaluation metrics."""
from __future__ import annotations

import numpy as np

from agritime.eval.metrics import (
    brier_score,
    coverage,
    mae,
    mape,
    mase,
    pinball_loss,
    rmse,
    smape,
)


def test_perfect_predictions() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0
    assert smape(y, y) == 0.0


def test_mase_against_naive() -> None:
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = np.array([6.0, 7.0])
    y_pred = np.array([6.0, 7.0])
    assert mase(y_true, y_pred, y_train, season=1) == 0.0


def test_coverage_full_interval() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert coverage(y, y - 1, y + 1) == 1.0
    assert coverage(y, y + 5, y + 10) == 0.0


def test_pinball_loss_nonneg() -> None:
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.5, 1.5, 1.5])
    assert pinball_loss(y, pred, q=0.5) >= 0


def test_brier_score_perfect() -> None:
    y = np.array([0.0, 1.0])
    p = np.array([0.0, 1.0])
    assert brier_score(y, p) == 0.0
