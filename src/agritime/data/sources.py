"""Public dataset clients.

Currently supported:
- NASA POWER hourly point API (no auth)
- INMET BDMEP annual zip archives (no auth)

Both are intentionally minimal — notebook 01 iterates on parsing and cleanup.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import date

import httpx
import pandas as pd

NASA_POWER_HOURLY_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
INMET_ANNUAL_ZIP_URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"


@dataclass(frozen=True)
class GridPoint:
    """A single (lat, lon) grid point used as a synthetic station id for NASA POWER."""

    lat: float
    lon: float

    @property
    def station_id(self) -> str:
        return (
            f"power_{self.lat:.2f}_{self.lon:.2f}".replace(".", "p").replace("-", "m")
        )


def fetch_nasa_power_hourly(
    point: GridPoint,
    start: date,
    end: date,
    parameters: tuple[str, ...] = (
        "T2M",
        "RH2M",
        "WS2M",
        "WD2M",
        "PRECTOTCORR",
        "PS",
        "ALLSKY_SFC_SW_DWN",
    ),
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Fetch hourly weather from NASA POWER for a single grid point.

    Returns a long-form DataFrame with one row per hour. Columns include the
    requested NASA parameter codes plus station_id, source, and a UTC `ts`.
    """
    params = {
        "parameters": ",".join(parameters),
        "community": "AG",
        "longitude": point.lon,
        "latitude": point.lat,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "UTC",
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.get(NASA_POWER_HOURLY_URL, params=params)
        r.raise_for_status()
        payload = r.json()

    series = payload["properties"]["parameter"]
    df = pd.DataFrame(series)
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H", utc=True)
    df.index.name = "ts"
    df["station_id"] = point.station_id
    df["source"] = "nasa_power"
    df["lat"] = point.lat
    df["lon"] = point.lon
    return df.reset_index()


def fetch_inmet_annual_zip(year: int, timeout: float = 120.0) -> bytes:
    """Download the INMET BDMEP annual archive (returns raw zip bytes)."""
    url = INMET_ANNUAL_ZIP_URL.format(year=year)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.content


def parse_inmet_zip(content: bytes, uf: str | None = None) -> pd.DataFrame:
    """Parse all CSVs in an INMET annual zip into a single long-form DataFrame.

    INMET annual zips contain one CSV per station, semicolon-delimited, latin-1,
    with a small preamble of station metadata. The preamble layout has shifted
    between years — this parser is best-effort and notebook 01 should validate
    coverage on the actual download.
    """
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            if uf is not None and f"_{uf.upper()}_" not in name.upper():
                continue
            with zf.open(name) as fh:
                raw = fh.read().decode("latin-1", errors="replace")
            lines = raw.split("\n")
            preamble = lines[:8]
            body = "\n".join(lines[8:])

            meta: dict[str, str] = {}
            for line in preamble:
                if ":;" in line or ":" in line:
                    key, _, value = line.partition(":")
                    meta[key.strip().upper()] = value.strip().strip(";").strip()

            df = pd.read_csv(
                io.StringIO(body),
                sep=";",
                decimal=",",
                na_values=["-9999", "", " "],
                low_memory=False,
            )
            df["station_id"] = f"inmet_{meta.get('CODIGO (WMO)', name).strip()}"
            df["source"] = "inmet"
            df["uf"] = meta.get("UF")
            df["lat"] = _safe_float(meta.get("LATITUDE"))
            df["lon"] = _safe_float(meta.get("LONGITUDE"))
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None
