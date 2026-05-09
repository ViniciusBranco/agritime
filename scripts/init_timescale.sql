-- TimescaleDB extension + base schema for hourly weather observations.
-- Applied automatically on first container boot via docker-entrypoint-initdb.d.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Station catalog (INMET ground stations + NASA POWER grid points).
CREATE TABLE IF NOT EXISTS station (
    station_id   TEXT PRIMARY KEY,
    source       TEXT NOT NULL,           -- 'inmet' | 'nasa_power'
    name         TEXT,
    uf           TEXT,
    municipio    TEXT,
    lat          DOUBLE PRECISION NOT NULL,
    lon          DOUBLE PRECISION NOT NULL,
    elevation_m  DOUBLE PRECISION,
    metadata     JSONB
);

-- Hypertable for hourly observations.
CREATE TABLE IF NOT EXISTS weather_hourly (
    station_id    TEXT NOT NULL,
    ts            TIMESTAMPTZ NOT NULL,
    temp_c        DOUBLE PRECISION,
    rh_pct        DOUBLE PRECISION,
    wind_ms       DOUBLE PRECISION,
    wind_dir_deg  DOUBLE PRECISION,
    rain_mm       DOUBLE PRECISION,
    pressure_hpa  DOUBLE PRECISION,
    solar_wm2     DOUBLE PRECISION,
    PRIMARY KEY (station_id, ts)
);

SELECT create_hypertable(
    'weather_hourly', 'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS idx_weather_station_ts
    ON weather_hourly (station_id, ts DESC);

-- Compression policy for chunks older than 30 days.
ALTER TABLE weather_hourly SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'station_id'
);
SELECT add_compression_policy('weather_hourly', INTERVAL '30 days', if_not_exists => TRUE);

-- NDVI/EVI hypertable for downstream remote-sensing notebooks.
CREATE TABLE IF NOT EXISTS vegetation_index (
    polygon_id    TEXT NOT NULL,
    ts            TIMESTAMPTZ NOT NULL,
    source        TEXT NOT NULL,           -- 'satveg' | 'sentinel2'
    ndvi          DOUBLE PRECISION,
    evi           DOUBLE PRECISION,
    quality_flag  SMALLINT,
    PRIMARY KEY (polygon_id, ts, source)
);

SELECT create_hypertable(
    'vegetation_index', 'ts',
    chunk_time_interval => INTERVAL '90 days',
    if_not_exists       => TRUE
);
