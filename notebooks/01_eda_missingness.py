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
# # Notebook 01 — EDA & Missingness
#
# **Goal**: profile the joint INMET + NASA POWER hourly weather dataset for a
# target region (default: São Paulo state, 2020-2024) and quantify missingness
# across stations and variables. Finish the notebook with an imputation strategy
# that survives walk-forward forecasting in the rest of the curriculum.
#
# **What you should be able to explain after this notebook**:
# 1. The shape and cadence of each public source.
# 2. How to diagnose MCAR / MAR / MNAR with Little's test + visual gap maps.
# 3. The trade-offs of forward-fill, rolling-mean, and multiple imputation
#    when the downstream task is forecasting.

# %%
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from agritime.data.storage import read_parquet

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
sns.set_theme(context="notebook", style="whitegrid")

# %% [markdown]
# ## 1. Load bootstrapped Parquet snapshots
#
# These come from `python scripts/bootstrap_data.py --years 2020-2024 --uf SP`.

# %%
nasa = read_parquet("nasa_power_hourly")
inmet = read_parquet("inmet_hourly")

print(
    "NASA POWER:",
    nasa.shape,
    "stations:",
    nasa["station_id"].nunique() if not nasa.empty else 0,
)
print(
    "INMET     :",
    inmet.shape,
    "stations:",
    inmet["station_id"].nunique() if not inmet.empty else 0,
)
nasa.head()

# %% [markdown]
# ## 2. Cadence + coverage diagnostic
#
# Confirm both sources resolve to a common hourly UTC grid before going further.

# %%
# TODO: implement an hourly continuity check per station — expected vs observed
#       timestamps over the requested year range, return % completeness.

# %% [markdown]
# ## 3. Missingness profile
#
# - Per-variable null rates
# - Per-station null rates
# - Heatmap of nulls over time (station × month)

# %%
# TODO: missingness_matrix(df) → DataFrame of % nulls per (station, variable, month)

# %% [markdown]
# ## 4. Little's MCAR test
#
# A formal test for whether missingness is independent of observed values. If we
# reject MCAR, naive imputation will bias the forecaster downstream.

# %%
# TODO: implement Little's MCAR test on a sampled subset of variables.
#       Reference: pyampute / missingpy / a from-scratch chi-square implementation.

# %% [markdown]
# ## 5. Imputation strategy comparison
#
# Compare on a held-out window:
# - forward-fill (naive)
# - rolling-mean (k=3, 6, 12)
# - seasonal-naive (lag = 24h)
# - multiple imputation (`IterativeImputer` with `BayesianRidge`)
#
# Metric: MAE on the artificially masked-out values. Downstream forecasting
# impact is deferred to notebook 02 to keep this notebook scoped to the data.

# %%
# TODO: side-by-side imputation comparison + reliability plot

# %% [markdown]
# ## 6. Source-cascade design
#
# When INMET has gaps, fall back to NASA POWER's nearest grid point for the same
# variable. Emit a `source` column tagging which provider supplied each cell so
# downstream code can audit provenance — and so we can trace bias back when it
# bites.

# %%
# TODO: nearest-station map (Haversine) + cascading merge

# %% [markdown]
# ## Takeaways
#
# - [ ] Document the missingness pattern observed
# - [ ] Pick the imputation strategy used by notebooks 02-07
# - [ ] Write `reports/01_eda_missingness.md` with the findings
