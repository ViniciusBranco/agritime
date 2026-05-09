"""Minimal FastAPI app exposing trained forecasters from the MLflow registry.

Run inside the jupyter container:

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os

from fastapi import FastAPI

app = FastAPI(title="agritime", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "mlflow": os.environ.get("MLFLOW_TRACKING_URI", "<unset>"),
        "db": os.environ.get("AGRITIME_DB_URL", "<unset>").split("@")[-1],
    }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "agritime",
        "docs": "/docs",
        "notebooks": "see notebooks/ directory",
    }
