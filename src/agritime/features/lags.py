"""Lag, rolling, Fourier, and calendar features for tabular ML forecasters."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_lags(
    df: pd.DataFrame,
    target: str,
    lags: list[int],
    group: str | None = None,
) -> pd.DataFrame:
    """Append target lag features. Per-group when `group` is given (panel data)."""
    out = df.copy()
    if group is None:
        for k in lags:
            out[f"{target}_lag{k}"] = out[target].shift(k)
    else:
        grouped = out.groupby(group)[target]
        for k in lags:
            out[f"{target}_lag{k}"] = grouped.shift(k).reset_index(level=0, drop=True)
    return out


def add_rolling(
    df: pd.DataFrame,
    target: str,
    windows: list[int],
    stats: tuple[str, ...] = ("mean", "std", "min", "max"),
    group: str | None = None,
) -> pd.DataFrame:
    """Append rolling-window aggregates. Per-group when `group` is given."""
    out = df.copy()
    base = out.groupby(group)[target] if group else out[target]
    for window in windows:
        roll = base.rolling(window, min_periods=max(2, window // 4))
        for stat in stats:
            agg = getattr(roll, stat)()
            col = f"{target}_roll{window}_{stat}"
            out[col] = agg.reset_index(level=0, drop=True) if group else agg
    return out


def add_fourier(
    df: pd.DataFrame,
    ts_col: str,
    period_hours: float,
    n_terms: int = 3,
    prefix: str = "fourier",
) -> pd.DataFrame:
    """Append sin/cos Fourier terms encoding a seasonal period in hours."""
    out = df.copy()
    seconds = pd.to_datetime(out[ts_col]).astype("int64") // 10**9
    hours = seconds / 3600.0
    omega = 2 * np.pi / period_hours
    for k in range(1, n_terms + 1):
        out[f"{prefix}_sin_{int(period_hours)}h_k{k}"] = np.sin(k * omega * hours)
        out[f"{prefix}_cos_{int(period_hours)}h_k{k}"] = np.cos(k * omega * hours)
    return out


def add_calendar(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Append basic calendar features derived from a timestamp column."""
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["doy"] = ts.dt.dayofyear
    out["month"] = ts.dt.month
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return out
