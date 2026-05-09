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
# # Notebook 04 — Deep Forecasting: TFT + N-BEATS
#
# **Goal**: train Temporal Fusion Transformer and N-BEATS on the same panel as
# notebook 03 and compare both against the LightGBM global model. End with a
# multi-horizon (1-24h) forecast and TFT attention-weight interpretation.
#
# **What you should be able to explain**:
# 1. Why attention helps with multi-horizon, multi-covariate forecasting.
# 2. The role of static categoricals, known-future, and observed-past inputs.
# 3. When deep models lose to LightGBM — usually when the panel is too short or
#    too noisy to feed enough sequences.
# 4. Quantile loss heads and what they buy you (free prediction intervals).

# %%
import warnings

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from pytorch_forecasting import (
    NBeats,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.metrics import QuantileLoss

from agritime.data.storage import read_parquet

warnings.filterwarnings("ignore")
sns.set_theme(context="notebook", style="whitegrid")

# %% [markdown]
# ## 1. Build the TimeSeriesDataSet
#
# `pytorch_forecasting.TimeSeriesDataSet` wants a long-form panel with:
# - `time_idx` — integer hour offset
# - `group_ids` — `station_id`
# - `target` — the variable being forecast
# - covariates split into time-varying-known / time-varying-unknown / static
# - `max_encoder_length` — context window
# - `max_prediction_length` — forecast horizon

# %%
# TODO: assemble TimeSeriesDataSet for hourly temperature, horizon=24

# %% [markdown]
# ## 2. N-BEATS

# %%
# TODO: NBeats.from_dataset, fit with pytorch-lightning Trainer

# %% [markdown]
# ## 3. Temporal Fusion Transformer
#
# Quantile loss head → free prediction intervals at the same training cost.

# %%
# TODO: TemporalFusionTransformer.from_dataset, QuantileLoss([0.1, 0.5, 0.9])

# %% [markdown]
# ## 4. Compare against LightGBM (notebook 03)
#
# Same walk-forward CV, same metrics. Be honest — deep models often lose on
# small panels.

# %%
# TODO: side-by-side metrics + plot

# %% [markdown]
# ## 5. TFT interpretation
#
# Variable selection weights, encoder/decoder attention by hour-of-day. Pull a
# couple of forecasts where TFT clearly diverges from LightGBM and explain why.

# %%
# TODO: model.interpret_output → variable_importance, attention plots

# %% [markdown]
# ## Takeaways
#
# - [ ] Did TFT/N-BEATS beat LightGBM?
# - [ ] What did the attention weights teach about the data?
# - [ ] Record findings in `reports/04_forecasting_deep.md`
