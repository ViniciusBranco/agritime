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
# # Notebook 05 — TS Classification + Anomaly Detection
#
# **Two parts in one notebook**:
#
# **Part A — Crop-type classification from NDVI** using ROCKET / MiniRocket /
# InceptionTime on Sentinel-2-derived NDVI series, with MapBiomas annual
# land-use as labels. The point isn't winning a benchmark — it's having a
# defensible take on which method to pick when.
#
# **Part B — Anomaly detection on station telemetry** using Matrix Profile
# (STUMPY) and IsolationForest. We'll detect sensor drift, stuck values, and
# regime shifts in INMET stations.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import stumpy
from aeon.classification.convolution_based import (
    InceptionTimeClassifier,
    MiniRocketClassifier,
    RocketClassifier,
)
from sklearn.ensemble import IsolationForest

from agritime.data.storage import read_parquet

warnings.filterwarnings("ignore")
sns.set_theme(context="notebook", style="whitegrid")

# %% [markdown]
# ## Part A — Crop-type from NDVI
#
# 1. Sample N MapBiomas polygons per crop class.
# 2. Pull Sentinel-2 NDVI series via Microsoft Planetary Computer (STAC).
# 3. Resample to a common cadence + length.
# 4. Train ROCKET / MiniRocket / InceptionTime; compare accuracy + runtime.
# 5. Confusion matrix — which classes are confusable (e.g. soybean vs corn early)?

# %%
# TODO: sentinel2 NDVI extraction via stackstac + pystac-client

# %%
# TODO: ROCKET vs MiniRocket vs InceptionTime training

# %% [markdown]
# ## Part B — Anomaly detection on station telemetry
#
# 1. Pick one INMET station with known coverage gaps from notebook 01.
# 2. Compute Matrix Profile on the temperature series to surface motifs and
#    discords (anomalies).
# 3. Compare against a multivariate IsolationForest over (temp, RH, wind).
# 4. Visualize discords on the raw series.

# %%
# TODO: stumpy.stump on hourly temperature, plot top-5 discords

# %%
# TODO: IsolationForest on multivariate hourly features, score timeline

# %% [markdown]
# ## Takeaways
#
# - [ ] When is ROCKET enough? When does InceptionTime earn its compute?
# - [ ] What do the Matrix Profile discords correspond to in the raw timeline?
# - [ ] Record in `reports/05_ts_classification_anomaly.md`
