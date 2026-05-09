# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 03 — Global ML Forecasting (LightGBM)
#
# **Goal**: build a single global LightGBM model that forecasts every station in
# the panel by treating the panel as a tabular regression problem with engineered
# lag, rolling, Fourier, and calendar features. Compare against the per-series
# classical baselines from notebook 02.
#
# **What you should be able to explain**:
# 1. Why a global tabular model often beats per-series classical models on
#    short, noisy panels — and when it doesn't.
# 2. Feature design: target lags, rolling stats, Fourier seasonality, calendar.
# 3. Optuna-based hyperparameter search with a time-aware CV split.
# 4. Hierarchical reconciliation (município → estado) when forecasts must roll
#    up coherently.

# %%
import logging
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns

from agritime.data.storage import read_parquet
from agritime.eval.metrics import mae, mase, rmse, smape
from agritime.features.lags import add_calendar, add_fourier, add_lags, add_rolling

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
sns.set_theme(context="notebook", style="whitegrid")

# %% [markdown]
# ## 1. Build the panel feature table
#
# One row per (station_id, ts). Target = next-hour value. Features = lags,
# rolling stats, Fourier (24h + annual), calendar.

# %%
# TODO: load nasa_power_hourly, add_lags, add_rolling, add_fourier, add_calendar

# %% [markdown]
# ## 2. Time-aware split
#
# Last 4 weeks → test, prior 4 weeks → validation, rest → train. No shuffling,
# no leakage.

# %%
# TODO: time-aware split helper

# %% [markdown]
# ## 3. Baseline LightGBM
#
# Sane defaults. MLflow autolog so we can compare runs.

# %%
# TODO: lgb.train with regression objective, log to MLflow

# %% [markdown]
# ## 4. Optuna hyperparameter search
#
# Minimize validation MAE over: num_leaves, min_data_in_leaf, learning_rate,
# feature_fraction, bagging_fraction, lambda_l1, lambda_l2.

# %%
# TODO: optuna study with TPE sampler, 50 trials, log best params

# %% [markdown]
# ## 5. Per-station error breakdown
#
# Where does the global model under- vs over-perform per-series classical?
# Likely: stations with strong idiosyncratic patterns (coastal vs interior).

# %%
# TODO: error breakdown by station_id, by hour-of-day

# %% [markdown]
# ## 6. Hierarchical reconciliation
#
# `sktime` / `hierarchicalforecast` to enforce coherent station → município →
# estado aggregations.

# %%
# TODO: hierarchical reconciliation example

# %% [markdown]
# ## Takeaways
#
# - [ ] Beat the notebook 02 baseline (or honestly explain why you didn't)
# - [ ] Record best params + features in `reports/03_forecasting_ml_global.md`
# - [ ] Note feature importance — does it match agronomic intuition?
