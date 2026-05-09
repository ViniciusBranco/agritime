"""Bootstrap public datasets into the local Parquet store + TimescaleDB.

Currently materializes:
- NASA POWER hourly weather for a small grid covering São Paulo state (default)
- INMET BDMEP annual archives (UF-filtered)

Usage (inside the jupyter container):

    python scripts/bootstrap_data.py --years 2020-2024 --uf SP
    python scripts/bootstrap_data.py --years 2024 --skip-inmet
    python scripts/bootstrap_data.py --years 2024 --load-timescale
"""
from __future__ import annotations

import argparse
import logging
from datetime import date

from agritime.data.sources import (
    GridPoint,
    fetch_inmet_annual_zip,
    fetch_nasa_power_hourly,
    parse_inmet_zip,
)
from agritime.data.storage import (
    get_engine,
    read_parquet,
    upsert_weather_hourly,
    write_parquet,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("bootstrap")


# Tiny default grid covering São Paulo state. Extend as needed.
DEFAULT_GRID: list[GridPoint] = [
    GridPoint(lat=-22.0, lon=-47.5),
    GridPoint(lat=-22.5, lon=-48.0),
    GridPoint(lat=-23.0, lon=-46.5),
    GridPoint(lat=-23.5, lon=-47.0),
]


def parse_year_range(spec: str) -> list[int]:
    if "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(spec)]


def bootstrap_nasa_power(years: list[int], grid: list[GridPoint]) -> None:
    for point in grid:
        log.info("NASA POWER → %s", point.station_id)
        df = fetch_nasa_power_hourly(
            point,
            start=date(years[0], 1, 1),
            end=date(years[-1], 12, 31),
        )
        if df.empty:
            log.warning("empty response for %s", point.station_id)
            continue
        write_parquet(
            df,
            "nasa_power_hourly",
            partitions={"station_id": point.station_id},
        )
        log.info("wrote %d rows for %s", len(df), point.station_id)


def bootstrap_inmet(years: list[int], uf: str) -> None:
    for year in years:
        log.info("INMET %s/%s → fetching", year, uf)
        try:
            content = fetch_inmet_annual_zip(year)
        except Exception as exc:  # network errors, 404 for very recent years, etc.
            log.warning("INMET %s skipped: %s", year, exc)
            continue
        df = parse_inmet_zip(content, uf=uf)
        if df.empty:
            log.warning("INMET %s/%s parsed empty", year, uf)
            continue
        write_parquet(
            df,
            "inmet_hourly",
            partitions={"year": str(year), "uf": uf},
        )
        log.info("INMET %s/%s wrote %d rows", year, uf, len(df))


def load_into_timescale() -> None:
    df = read_parquet("nasa_power_hourly")
    if df.empty:
        log.warning("no NASA POWER parquet to upsert")
        return
    engine = get_engine()
    n = upsert_weather_hourly(df, engine)
    log.info("upserted %d rows into TimescaleDB", n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", default="2020-2024", help="e.g. 2015-2024 or 2024")
    p.add_argument("--uf", default="SP")
    p.add_argument("--skip-nasa", action="store_true")
    p.add_argument("--skip-inmet", action="store_true")
    p.add_argument(
        "--load-timescale",
        action="store_true",
        help="after parquet write, also upsert NASA POWER into TimescaleDB",
    )
    args = p.parse_args()

    years = parse_year_range(args.years)

    if not args.skip_nasa:
        bootstrap_nasa_power(years, DEFAULT_GRID)

    if not args.skip_inmet:
        bootstrap_inmet(years, uf=args.uf)

    if args.load_timescale:
        load_into_timescale()


if __name__ == "__main__":
    main()
