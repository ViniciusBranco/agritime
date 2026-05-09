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
# # Notebook 06 — Calibrated Risk + Conformal Prediction
#
# **Goal**: train a probability-of-rain-during-window classifier on the bootstrap
# data, calibrate its outputs, and wrap it in a conformal predictor so the
# prediction interval has a guaranteed coverage rate (e.g. 90%) on exchangeable
# data.
#
# **What you should be able to explain**:
# 1. Why raw classifier probabilities are usually miscalibrated.
# 2. Platt vs isotonic calibration, and how to evaluate calibration (reliability
#    diagram, Brier score, ECE).
# 3. Split-conformal vs CV+ vs Jackknife+ conformal prediction.
# 4. Why conformal coverage is *marginal*, not conditional — and what that
#    means for high-stakes decisions.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mapie.classification import MapieClassifier
from mapie.regression import MapieRegressor
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss

from agritime.data.storage import read_parquet
from agritime.eval.metrics import brier_score, coverage

warnings.filterwarnings("ignore")
sns.set_theme(context="notebook", style="whitegrid")

# %% [markdown]
# ## 1. Build the binary target
#
# `rain_in_next_6h` = (sum of `rain_mm` over t+1..t+6) > 1mm. Features: current
# weather, lags, calendar, source.

# %%
# TODO: build feature/target frame from nasa_power_hourly

# %% [markdown]
# ## 2. Train a baseline classifier
#
# `GradientBoostingClassifier` or `LightGBMClassifier`. Track Brier + log-loss.

# %%
# TODO: baseline classifier, predict_proba on holdout

# %% [markdown]
# ## 3. Calibration diagnostics
#
# - Reliability diagram (binned predicted vs observed)
# - Brier score
# - Expected Calibration Error (ECE)

# %%
# TODO: calibration_curve + reliability plot + Brier + ECE

# %% [markdown]
# ## 4. Calibrate
#
# `CalibratedClassifierCV` with isotonic. Re-plot reliability — ECE should drop.

# %%
# TODO: isotonic calibration, re-evaluate

# %% [markdown]
# ## 5. Conformal prediction sets
#
# `mapie.classification.MapieClassifier` with `method="lac"` for marginal
# coverage. Show that across the holdout, coverage ≈ 1 − α (e.g. 0.9).

# %%
# TODO: MapieClassifier wrapping the calibrated classifier, predict_set

# %% [markdown]
# ## 6. Conformal regression for rainfall amount
#
# Same idea but for the continuous rainfall amount, with `MapieRegressor` and
# CV+ method. Show empirical coverage vs nominal across multiple α.

# %%
# TODO: MapieRegressor, plot empirical coverage curve

# %% [markdown]
# ## Takeaways
#
# - [ ] Quantify the calibration improvement (ECE before/after)
# - [ ] Show that conformal coverage tracks the nominal level
# - [ ] Discuss what fails when exchangeability is violated (e.g. distribution
#       shift between train and deploy)
# - [ ] Record in `reports/06_calibrated_risk_conformal.md`
