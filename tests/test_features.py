"""Smoke tests for feature engineering helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from agritime.features.lags import add_calendar, add_fourier, add_lags, add_rolling


def _toy_panel() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ts": list(ts) * 2,
            "station_id": ["a"] * 48 + ["b"] * 48,
            "y": np.concatenate([np.arange(48), np.arange(48) * 2]),
        }
    )


def test_add_lags_per_group() -> None:
    df = _toy_panel()
    out = add_lags(df, "y", lags=[1, 2], group="station_id")
    assert "y_lag1" in out.columns
    assert "y_lag2" in out.columns
    a = out[out["station_id"] == "a"].reset_index(drop=True)
    assert pd.isna(a.loc[0, "y_lag1"])
    assert a.loc[1, "y_lag1"] == 0


def test_add_rolling_columns() -> None:
    df = _toy_panel()
    out = add_rolling(df, "y", windows=[3], stats=("mean",), group="station_id")
    assert "y_roll3_mean" in out.columns


def test_add_fourier_terms() -> None:
    df = _toy_panel()
    out = add_fourier(df, "ts", period_hours=24, n_terms=2)
    assert "fourier_sin_24h_k1" in out.columns
    assert "fourier_cos_24h_k2" in out.columns


def test_add_calendar() -> None:
    df = _toy_panel()
    out = add_calendar(df, "ts")
    assert {"hour", "dow", "doy", "month", "is_weekend"} <= set(out.columns)
    assert out["hour"].between(0, 23).all()
