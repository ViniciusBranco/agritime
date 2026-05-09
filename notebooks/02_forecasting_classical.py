# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 02 — Classical Forecasting Baselines
#
# **Goal**: establish honest baselines for hourly temperature and daily rainfall
# using SARIMA, ETS, and Prophet. Every later notebook must beat these on the
# same walk-forward CV scheme — otherwise the added complexity isn't earning
# its keep.
#
# **What you should be able to explain after this notebook**:
# 1. Stationarity diagnostics (ADF, KPSS) and when each disagrees.
# 2. Seasonal decomposition (STL) and what to do with the residual.
# 3. Walk-forward (expanding-window) cross-validation for time series.
# 4. When SARIMA wins, when ETS wins, and when Prophet's prior structure helps.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from agritime.data.storage import read_parquet
from agritime.eval.metrics import mae, mase, rmse, smape

warnings.filterwarnings("ignore")
sns.set_theme(context="notebook", style="whitegrid")

# %% [markdown]
# ## 1. Pick a single station + variable
#
# Start simple: one station, one variable, hourly cadence. Multi-series global
# models come in notebook 03.

# %%
# TODO: load nasa_power_hourly, pick one station, resample to hourly, plot raw series

# %% [markdown]
# ## 2. Stationarity diagnostics
#
# - Augmented Dickey-Fuller (ADF)
# - KPSS
# - Disagreements often mean trend-stationary vs difference-stationary regimes.

# %%
# TODO: ADF + KPSS on raw + first-differenced series

# %% [markdown]
# ## 3. Seasonal decomposition
#
# Hourly weather has strong daily (24h) and weak annual (8766h) seasonality. Use
# `STL` (statsmodels) for a robust decomposition.

# %%
# TODO: STL with period=24, plot trend / seasonal / resid

# %% [markdown]
# ## 4. Walk-forward CV harness
#
# Implement an expanding-window CV. Training history grows; test window slides
# by `horizon`. Track MAE, RMSE, MASE, sMAPE per fold.

# %%
# TODO: walk_forward_cv(model_factory, series, n_splits, horizon)

# %% [markdown]
# ## 5. SARIMA
#
# Use `statsmodels` `SARIMAX` with auto-selection of (p, d, q)(P, D, Q, s) via
# AIC over a small grid, or `pmdarima.auto_arima` if installed.

# %%
# TODO: SARIMA grid search + walk-forward CV

# %% [markdown]
# ## 6. ETS (Error / Trend / Seasonality state-space)
#
# `statsmodels` `ETSModel`. Faster than SARIMA on hourly data and often as good.

# %%
# TODO: ETS fit + walk-forward CV

# %% [markdown]
# ## 7. Prophet
#
# Bayesian additive model with a logistic-prior trend and Fourier seasonality.
# Less elegant on hourly data but a useful comparison point.

# %%
# TODO: Prophet fit + walk-forward CV

# %% [markdown]
# ## 8. Comparison + write-up
#
# Side-by-side metrics, runtime, and residual ACF.
#
# - [ ] Pick the strongest baseline; record its score in `reports/02_forecasting_classical.md`
# - [ ] Note the failure mode of each model (when does it break?)
