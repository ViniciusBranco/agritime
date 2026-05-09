"""Forecasting and probabilistic metrics."""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(
        np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100
    )


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-8)) * 100)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season: int = 1,
) -> float:
    """Mean Absolute Scaled Error using a seasonal-naive baseline on `y_train`."""
    naive = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    return float(np.mean(np.abs(y_true - y_pred)) / max(naive, 1e-8))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage of a prediction interval."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Pinball / quantile loss at quantile `q`."""
    diff = y_true - y_pred_q
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Brier score for binary probabilistic forecasts."""
    return float(np.mean((p_pred - y_true) ** 2))
