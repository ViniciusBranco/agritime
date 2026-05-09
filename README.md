# agritime

A self-contained study lab for **time-series mining and predictive modeling on public agricultural data**, built around a TimescaleDB + Parquet hybrid store, MLflow experiment tracking, and a 7-notebook curriculum that progresses from classical forecasting through deep learning, calibrated uncertainty, and spatio-temporal interpolation.

> **Language convention**: notebook narrative (markdown cells, plot labels, printed messages) is written in **Brazilian Portuguese**. All code identifiers, library names, and the README itself stay in English.

## Stack

- **Python 3.12**, JupyterLab, FastAPI
- **TimescaleDB 2.x** on Postgres 16 — hypertables for hourly station readings
- **Parquet (pyarrow)** — bulk feature tables, immutable raw snapshots, `polars` for fast scans
- **MLflow 2.x** — experiment tracking, model registry, local artifact store
- **Forecasting**: darts, sktime, statsmodels, prophet, pytorch-forecasting (TFT, N-BEATS)
- **TS classification**: aeon (active sktime-classifiers fork)
- **Anomaly**: stumpy (Matrix Profile)
- **Geospatial / raster**: geopandas, xarray, rioxarray, stackstac, pystac-client
- **Calibration / UQ**: mapie (conformal prediction)
- **Tuning / orchestration**: optuna, hydra-core
- **Notebooks**: jupytext-paired `.py` percent-format files (no committed `.ipynb` blobs)

## Quick start

```bash
git clone https://github.com/ViniciusBranco/agritime.git
cd agritime
cp .env.example .env
docker compose up --build -d

# JupyterLab → http://localhost:8888  (token: `agritime`, override via JUPYTER_TOKEN)
# MLflow    → http://localhost:5000
# Postgres  → localhost:5433  (user/pass/db: agritime/agritime/agritime)

# Bootstrap public datasets (NASA POWER + INMET, defaults to São Paulo state, 2020-2024)
docker compose exec jupyter python scripts/bootstrap_data.py --years 2020-2024 --uf SP

# Convert jupytext .py notebooks to .ipynb on first run (paired sync afterwards)
docker compose exec jupyter jupytext --sync notebooks/*.py
```

## Public data sources (no auth required)

| Source | Coverage | Endpoint |
|---|---|---|
| **NASA POWER** | Hourly weather, ~0.5° global grid, 1981+ | https://power.larc.nasa.gov/api |
| **INMET BDMEP** | Brazilian surface stations, hourly | https://portal.inmet.gov.br/dadoshistoricos |
| **Embrapa SATVeg** | NDVI/EVI per polygon, MODIS | https://www.satveg.cnptia.embrapa.br |
| **Sentinel-2 L2A** | 10m optical, 5-day revisit | Microsoft Planetary Computer (STAC) |
| **MapBiomas** | Annual land-use rasters 1985-2024 | https://mapbiomas.org |
| **CONAB** | Crop production by município | https://www.conab.gov.br |
| **ANA HidroWeb** | Streamflow + rainfall stations | https://www.snirh.gov.br/hidroweb |

## Notebook curriculum

Each notebook ships paired with a `reports/NN_*.md` write-up — the artifact a hiring panel would actually read.

| # | Notebook | Theme |
|---|---|---|
| 01 | `01_eda_missingness` | Profile gaps in INMET vs NASA POWER, MCAR/MAR/MNAR diagnosis, multiple imputation |
| 02 | `02_forecasting_classical` | SARIMA / ETS / Prophet baselines, walk-forward CV, residual diagnostics |
| 03 | `03_forecasting_ml_global` | LightGBM with lag/rolling/Fourier features, hierarchical reconciliation |
| 04 | `04_forecasting_deep` | Temporal Fusion Transformer + N-BEATS multi-horizon |
| 05 | `05_ts_classification_anomaly` | ROCKET / InceptionTime crop-type from NDVI; Matrix Profile + IsolationForest |
| 06 | `06_calibrated_risk_conformal` | Probability-of-rain calibration with conformal prediction |
| 07 | `07_spatiotemporal_kriging` | Gaussian Process interpolation across the INMET station network |

## Repo layout

```
agritime/
├── docker/jupyter/Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── Makefile
├── scripts/
│   ├── bootstrap_data.py
│   └── init_timescale.sql
├── src/agritime/
│   ├── data/        # public data clients + parquet/timescale storage
│   ├── features/    # lag, rolling, Fourier, calendar
│   ├── models/      # forecaster wrappers
│   └── eval/        # metrics, calibration, walk-forward CV
├── notebooks/       # jupytext .py percent-format (paired to .ipynb at runtime)
├── reports/         # markdown write-ups per notebook
├── api/             # FastAPI inference for trained forecasters
└── data/            # gitignored — raw/processed parquet
```

## Make targets

```bash
make up            # docker compose up -d
make down          # docker compose down
make rebuild       # docker compose build --no-cache
make logs          # tail container logs
make bootstrap     # run scripts/bootstrap_data.py with sane defaults
make nb-sync       # jupytext --sync notebooks/*.py
make nb-clean      # strip outputs from any committed .ipynb
make lint          # ruff + mypy inside the jupyter container
make test          # pytest inside the jupyter container
```

## License

MIT.
