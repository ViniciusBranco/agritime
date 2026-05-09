"""Parquet + TimescaleDB storage helpers."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

DEFAULT_DATA_ROOT = Path(os.environ.get("AGRITIME_DATA_ROOT", "/workspace/data"))


def parquet_path(domain: str, *, partitions: dict[str, str] | None = None) -> Path:
    """Canonical Parquet directory for a logical domain (e.g. 'weather_hourly').

    Partitions are folder-style — one directory level per key/value pair. The
    directory is created if missing.
    """
    root = DEFAULT_DATA_ROOT / "raw" / "parquet" / domain
    if partitions:
        for key, value in partitions.items():
            root = root / f"{key}={value}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_parquet(
    df: pd.DataFrame,
    domain: str,
    partitions: dict[str, str] | None = None,
    filename: str = "part.parquet",
) -> Path:
    """Write a DataFrame as a single Parquet file under the partitioned domain path.

    Partition keys are encoded in the directory path (Hive-style) and dropped
    from the DataFrame before writing so a subsequent ``read_parquet`` doesn't
    see a column-vs-partition type clash.
    """
    target = parquet_path(domain, partitions=partitions) / filename
    out = df
    if partitions:
        drop = [k for k in partitions if k in df.columns]
        if drop:
            out = df.drop(columns=drop)
    table = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_table(table, target, compression="zstd")
    return target


def read_parquet(domain: str) -> pd.DataFrame:
    """Read every Parquet file under a domain into a single DataFrame.

    Partition columns are reconstructed from the directory path. Dictionary-
    encoded partition columns are cast back to plain strings so callers can
    treat them as ordinary text.
    """
    root = DEFAULT_DATA_ROOT / "raw" / "parquet" / domain
    if not root.exists():
        return pd.DataFrame()
    df = pd.read_parquet(root)
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].astype(str)
    return df


def get_engine(url: str | None = None) -> Engine:
    """Return a SQLAlchemy engine pointed at the TimescaleDB instance."""
    db_url = url or os.environ["AGRITIME_DB_URL"]
    return create_engine(db_url, future=True, pool_pre_ping=True)


def upsert_weather_hourly(
    df: pd.DataFrame,
    engine: Engine,
    chunk_size: int = 10_000,
) -> int:
    """Append weather rows to the hypertable.

    Expects columns: station_id, ts, temp_c, rh_pct, wind_ms, wind_dir_deg,
    rain_mm, pressure_hpa, solar_wm2. Conflicts on (station_id, ts) are ignored.
    """
    cols = [
        "station_id",
        "ts",
        "temp_c",
        "rh_pct",
        "wind_ms",
        "wind_dir_deg",
        "rain_mm",
        "pressure_hpa",
        "solar_wm2",
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    written = 0
    with engine.begin() as conn:
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size]
            chunk.to_sql("weather_hourly_staging", conn, if_exists="replace", index=False)
            conn.exec_driver_sql(
                """
                INSERT INTO weather_hourly
                SELECT * FROM weather_hourly_staging
                ON CONFLICT (station_id, ts) DO NOTHING
                """
            )
            conn.exec_driver_sql("DROP TABLE IF EXISTS weather_hourly_staging")
            written += len(chunk)
    return written
