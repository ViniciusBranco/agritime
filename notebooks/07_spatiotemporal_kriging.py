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
# # Notebook 07 — Spatio-Temporal Interpolation
#
# **Goal**: estimate weather at an arbitrary point inside the INMET network
# (think: a polygon centroid for a farm with no on-site sensor) using both a
# nearest-station baseline and ordinary kriging / Gaussian Process regression.
# Compare on a held-out station whose readings we hide from the interpolator.
#
# **What you should be able to explain**:
# 1. The semivariogram and how it parametrizes spatial covariance.
# 2. Ordinary kriging vs Gaussian Process regression — same math, different
#    framing and software.
# 3. Why nearest-neighbor often wins on dense networks and loses on sparse ones.
# 4. How to extend to spatio-temporal: separable kernels vs full ST covariance.

# %%
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import Point
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from agritime.data.storage import read_parquet
from agritime.eval.metrics import mae, rmse

warnings.filterwarnings("ignore")
sns.set_theme(context="notebook", style="whitegrid")

# %% [markdown]
# ## 1. Build the spatial frame
#
# Station catalog with lat/lon → GeoDataFrame in EPSG:4326. Pick one hour where
# every station has a reading (no missing temp).

# %%
# TODO: assemble station GeoDataFrame, pick a clean snapshot hour

# %% [markdown]
# ## 2. Hold one station out
#
# Pretend it doesn't exist; interpolate to its location and compare predictions
# to its actual reading. Repeat across all stations to build a leave-one-out
# error distribution.

# %%
# TODO: LOO loop helper

# %% [markdown]
# ## 3. Baseline — nearest neighbor

# %%
# TODO: Haversine nearest-station prediction

# %% [markdown]
# ## 4. Inverse-distance weighting

# %%
# TODO: IDW with k=3, p=2

# %% [markdown]
# ## 5. Gaussian Process regression
#
# Kernel: `ConstantKernel * RBF + WhiteKernel`. Fit on lat/lon → temperature.
# Compare LOO MAE to NN + IDW.

# %%
# TODO: GaussianProcessRegressor + LOO evaluation

# %% [markdown]
# ## 6. Spatio-temporal extension
#
# Separable kernel: spatial RBF × temporal RBF on hour-of-day. Train on a 24h
# window, predict the held-out station's full day.

# %%
# TODO: separable ST kernel, fit on (lat, lon, hour), evaluate

# %% [markdown]
# ## Takeaways
#
# - [ ] How much does GP beat nearest-neighbor — and where (sparse regions)?
# - [ ] Where does GP fail (network edges, regime breaks)?
# - [ ] Discuss kriging vs GP framing: same math, different vocabulary
# - [ ] Record in `reports/07_spatiotemporal_kriging.md`
