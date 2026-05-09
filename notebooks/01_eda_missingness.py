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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 01 — EDA & Análise de Ausência (Missingness)
#
# **Objetivo**: caracterizar o painel horário de clima da NASA POWER (e do
# INMET, quando disponível) na região alvo (default: estado de SP, 2020-2024)
# e quantificar a ausência de dados por estação e variável. Encerramos com
# uma estratégia de imputação que sustente o resto do currículo.
#
# **O que você deve conseguir explicar ao final**:
# 1. A forma e a cadência de cada fonte pública.
# 2. Como diagnosticar MCAR / MAR / MNAR com o teste de Little e mapas
#    visuais de gaps.
# 3. Os trade-offs entre forward-fill, média móvel, sazonal-naive e
#    imputação multivariada quando a tarefa downstream é forecasting.
#
# > **Observação**: NASA POWER é uma reanálise satelital e tem 0% de
# > ausência por construção. Para que as seções de missingness tenham algo
# > para ensinar, **sintetizamos** padrões MCAR/MAR/MNAR no painel antes de
# > aplicar os diagnósticos.

# %%
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from agritime.data.storage import read_parquet

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
sns.set_theme(context="notebook", style="whitegrid")

# %% [markdown]
# ## 1. Carregar snapshots Parquet
#
# Bootstrap via `python scripts/bootstrap_data.py --years 2020-2024 --uf SP`.

# %%
nasa_raw = read_parquet("nasa_power_hourly")
inmet_raw = read_parquet("inmet_hourly")

print(
    "NASA POWER:",
    nasa_raw.shape,
    "estações:",
    nasa_raw["station_id"].nunique() if not nasa_raw.empty else 0,
)
print(
    "INMET     :",
    inmet_raw.shape,
    "estações:",
    inmet_raw["station_id"].nunique() if not inmet_raw.empty else 0,
)
nasa_raw.head()

# %% [markdown]
# ### 1a. Normalizar NASA POWER → schema canônico
#
# A NASA POWER retorna códigos nativos; valores ausentes vêm como `-999`,
# não `NaN`. Renomeamos para o schema do projeto e convertemos pressão de
# kPa para hPa.
#
# | Código NASA | Canônico | Unidade |
# |---|---|---|
# | T2M | temp_c | °C |
# | RH2M | rh_pct | % |
# | WS2M | wind_ms | m/s |
# | WD2M | wind_dir_deg | graus |
# | PRECTOTCORR | rain_mm | mm/h |
# | PS | pressure_hpa | hPa (× 10 a partir de kPa) |
# | ALLSKY_SFC_SW_DWN | solar_wm2 | W/m² |

# %%
NASA_RENAME = {
    "T2M": "temp_c",
    "RH2M": "rh_pct",
    "WS2M": "wind_ms",
    "WD2M": "wind_dir_deg",
    "PRECTOTCORR": "rain_mm",
    "PS": "pressure_hpa",
    "ALLSKY_SFC_SW_DWN": "solar_wm2",
}
WEATHER_VARS = list(NASA_RENAME.values())


def normalize_nasa(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=NASA_RENAME).copy()
    out[WEATHER_VARS] = out[WEATHER_VARS].replace([-999, -999.0], np.nan)
    if "pressure_hpa" in out.columns:
        out["pressure_hpa"] = out["pressure_hpa"] * 10.0
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out.sort_values(["station_id", "ts"]).reset_index(drop=True)


nasa = normalize_nasa(nasa_raw) if not nasa_raw.empty else nasa_raw
nasa.head()

# %% [markdown]
# ## 2. Diagnóstico de cadência + cobertura
#
# Construímos o grid horário esperado por estação, comparamos com os
# timestamps observados, retornamos % de completude. Estações abaixo de
# 95% devem ser examinadas antes de entrar nos modelos de forecasting.

# %%
def hourly_coverage(
    df: pd.DataFrame,
    station_col: str = "station_id",
    ts_col: str = "ts",
) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby(station_col):
        observed = pd.DatetimeIndex(sorted(g[ts_col].unique()))
        if observed.empty:
            continue
        expected = pd.date_range(observed.min(), observed.max(), freq="h", tz="UTC")
        rows.append(
            {
                "station_id": sid,
                "first_ts": observed.min(),
                "last_ts": observed.max(),
                "observed_h": len(observed),
                "expected_h": len(expected),
                "completeness_pct": round(100 * len(observed) / len(expected), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("completeness_pct")


coverage_df = hourly_coverage(nasa) if not nasa.empty else pd.DataFrame()
coverage_df

# %% [markdown]
# Como esperado: NASA POWER retorna **100% de cobertura** nos quatro pontos
# de grid. Comportamento de reanálise — sem gaps.

# %% [markdown]
# ## 3. Sintetizar padrões MCAR / MAR / MNAR
#
# Sem ausências naturais, injetamos três tipos de padrão para que os
# diagnósticos das próximas seções tenham algo substantivo:
#
# - **MCAR** — *Missing Completely At Random*. Descartamos 10% de
#   `rain_mm` uniformemente ao acaso. A probabilidade de ausência
#   independe de qualquer variável, observada ou não. Imputação ingênua é
#   não-viesada nas estimativas de média.
# - **MAR** — *Missing At Random*. Descartamos `rain_mm` quando `temp_c`
#   está abaixo do percentil 10 (proxy para falha de sensor em frio). A
#   probabilidade de ausência depende de uma variável **observada**;
#   imputação que condiciona em `temp_c` (e.g. `IterativeImputer`)
#   recupera a distribuição.
# - **MNAR** — *Missing Not At Random*. Descartamos `temp_c` quando o
#   próprio valor está acima do percentil 95 (proxy para saturação em
#   calor extremo). A probabilidade de ausência depende do valor que se
#   esconderia — nenhum imputador é imparcial sem informação adicional.

# %%
def inject_missingness(
    df: pd.DataFrame,
    *,
    mcar_var: str = "rain_mm",
    mcar_frac: float = 0.10,
    mar_target: str = "rain_mm",
    mar_predictor: str = "temp_c",
    mar_quantile: float = 0.10,
    mnar_var: str = "temp_c",
    mnar_quantile: float = 0.95,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()

    # MCAR — drop uniforme em rain_mm
    obs_mcar = np.where(~out[mcar_var].isna())[0]
    drop_mcar = rng.choice(obs_mcar, int(len(obs_mcar) * mcar_frac), replace=False)
    out.loc[out.index[drop_mcar], mcar_var] = np.nan

    # MAR — drop em rain_mm quando temp_c está no quantil inferior
    cold_threshold = out[mar_predictor].quantile(mar_quantile)
    cold_mask = (out[mar_predictor] <= cold_threshold) & (~out[mar_target].isna())
    out.loc[cold_mask, mar_target] = np.nan

    # MNAR — drop em temp_c na cauda superior
    hot_threshold = out[mnar_var].quantile(mnar_quantile)
    hot_mask = out[mnar_var] >= hot_threshold
    out.loc[hot_mask, mnar_var] = np.nan

    return out


nasa_synth = inject_missingness(nasa) if not nasa.empty else nasa
print("Após injeção de ausência (% nulo por variável):")
print(nasa_synth[WEATHER_VARS].isna().mean().mul(100).round(2).to_string())

# %% [markdown]
# ## 4. Perfil de ausência
#
# - Taxa de nulos por variável no painel completo
# - Taxa de nulos por estação × variável
# - Heatmap de nulos no tempo (estação × mês) para a variável foco

# %%
def per_variable_nulls(df: pd.DataFrame) -> pd.Series:
    return df[WEATHER_VARS].isna().mean().mul(100).round(2).sort_values(ascending=False)


def per_station_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("station_id")[WEATHER_VARS]
        .apply(lambda g: g.isna().mean().mul(100))
        .round(2)
    )


if not nasa_synth.empty:
    print("Nulos por variável (%):")
    print(per_variable_nulls(nasa_synth))
    print()
    print("Nulos por estação (%):")
    display(per_station_nulls(nasa_synth))


# %%
def station_month_nulls(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """% nulo por (station_id, mês calendário)."""
    tmp = df.assign(month=df["ts"].dt.to_period("M").astype(str))
    return (
        tmp.groupby(["station_id", "month"])[var]
        .apply(lambda s: s.isna().mean() * 100)
        .unstack("month")
    )


focal_var = "rain_mm"
if not nasa_synth.empty:
    heat = station_month_nulls(nasa_synth, focal_var)
    fig, ax = plt.subplots(figsize=(14, max(2, 0.45 * len(heat))))
    sns.heatmap(
        heat,
        cmap="rocket_r",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "% nulo"},
        ax=ax,
    )
    ax.set_title(
        f"{focal_var} — taxa de nulos por estação × mês (após injeção MCAR + MAR)"
    )
    ax.set_xlabel("mês")
    ax.set_ylabel("estação")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Teste MCAR de Little
#
# Little (1988) propõe um teste qui-quadrado para *Missing Completely At
# Random*. A ideia: agrupar linhas pelo padrão de ausência, perguntar se
# as médias das variáveis observadas são estatisticamente
# indistinguíveis entre os padrões. Rejeitar MCAR (p baixo) significa que
# imputadores ingênuos viesarão os modelos downstream — precisamos de
# métodos sensíveis a MAR, que condicionem em covariáveis.
#
# A implementação abaixo é uma aproximação EM-style do estatístico de
# Little, escrita do zero. Para uso de produção, prefira `pyampute` ou o
# pacote `naniar` em R.

# %%
def littles_mcar(df: pd.DataFrame) -> dict[str, float | int]:
    X = df.to_numpy(dtype=float)
    n, p = X.shape
    if n == 0 or p == 0:
        return {"chi2": float("nan"), "df": 0, "p_value": float("nan"), "n_patterns": 0}

    M = np.isnan(X).astype(int)
    pattern = ["".join(map(str, row)) for row in M]
    df_pat = pd.DataFrame(X)
    df_pat["__pat__"] = pattern

    mu = np.nanmean(X, axis=0)
    centered = np.where(np.isnan(X), 0.0, X - mu)
    obs_mask = (~np.isnan(X)).astype(float)

    cov = (centered.T @ centered) / np.maximum(obs_mask.T @ obs_mask, 1)
    cov = (cov + cov.T) / 2.0

    chi2 = 0.0
    df_stat = 0
    for pat, sub in df_pat.groupby("__pat__"):
        keep = np.array([c == "0" for c in pat])
        idx = np.where(keep)[0]
        if len(idx) == 0:
            continue
        sub_X = sub.iloc[:, :-1].to_numpy()[:, idx]
        n_g = len(sub_X)
        diff = sub_X.mean(axis=0) - mu[idx]
        sub_cov = cov[np.ix_(idx, idx)]
        try:
            inv = np.linalg.pinv(sub_cov)
        except np.linalg.LinAlgError:
            continue
        chi2 += n_g * float(diff @ inv @ diff)
        df_stat += len(idx)

    df_stat = max(df_stat - p, 1)
    return {
        "chi2": chi2,
        "df": df_stat,
        "p_value": float(1 - stats.chi2.cdf(chi2, df_stat)),
        "n_patterns": int(df_pat["__pat__"].nunique()),
    }


if not nasa_synth.empty:
    sample_synth = nasa_synth[WEATHER_VARS].sample(min(5000, len(nasa_synth)), random_state=42)
    sample_clean = nasa[WEATHER_VARS].sample(min(5000, len(nasa)), random_state=42)
    print("Little MCAR — painel sintetizado (com MAR + MNAR injetados):")
    print(littles_mcar(sample_synth))
    print()
    print("Little MCAR — painel original (sem ausências):")
    print(littles_mcar(sample_clean))

# %% [markdown]
# **Lendo o resultado**: p-valor abaixo de 0.05 rejeita MCAR. Como
# injetamos MAR e MNAR, esperamos rejeição na amostra sintetizada. Na
# amostra original (sem ausências), o teste é trivial — apenas 1 padrão
# único.

# %% [markdown]
# ## 6. Comparação de estratégias de imputação
#
# Pegamos a estação mais completa, mascaramos 15% dos valores de `temp_c`
# uniformemente, imputamos e calculamos o MAE nas células mascaradas.
# Métodos comparados:
#
# - `ffill` — forward-fill (último valor observado)
# - `rolling_mean_6` — média móvel centrada, k = 6 horas
# - `seasonal_naive_24` — valor de 24h atrás (sazonalidade diária)
# - `iterative_bayesian_ridge` — `IterativeImputer(BayesianRidge)` sobre
#   todas as covariáveis simultaneamente
#
# O impacto downstream em forecasting é uma pergunta separada — voltamos
# a ela no notebook 02.

# %%
from sklearn.experimental import enable_iterative_imputer  # noqa: F401, E402
from sklearn.impute import IterativeImputer  # noqa: E402
from sklearn.linear_model import BayesianRidge  # noqa: E402


def evaluate_imputers(
    df: pd.DataFrame,
    target: str,
    mask_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = df[target].to_numpy(dtype=float).copy()
    obs_idx = np.where(~np.isnan(s))[0]
    if len(obs_idx) == 0:
        raise ValueError(f"sem valores observados em {target}")
    n_mask = int(len(obs_idx) * mask_frac)
    mask = rng.choice(obs_idx, size=n_mask, replace=False)
    truth = s[mask].copy()

    masked = s.copy()
    masked[mask] = np.nan
    masked_series = pd.Series(masked, index=df["ts"].to_numpy())

    results: list[tuple[str, float]] = []

    pred = masked_series.ffill().bfill().to_numpy()
    results.append(("ffill", float(np.mean(np.abs(pred[mask] - truth)))))

    pred = (
        masked_series.rolling(6, min_periods=1, center=True)
        .mean()
        .to_numpy()
    )
    pred = np.where(np.isnan(pred), float(np.nanmean(masked)), pred)
    results.append(("rolling_mean_6", float(np.mean(np.abs(pred[mask] - truth)))))

    pred = masked_series.shift(24).to_numpy()
    pred = np.where(np.isnan(pred), float(np.nanmean(masked)), pred)
    results.append(
        ("seasonal_naive_24", float(np.mean(np.abs(pred[mask] - truth))))
    )

    feats = df[WEATHER_VARS].copy()
    feats[target] = masked
    imp = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=5,
        random_state=seed,
    )
    pred_full = imp.fit_transform(feats)
    pred = pred_full[:, list(feats.columns).index(target)]
    results.append(
        ("iterative_bayesian_ridge", float(np.mean(np.abs(pred[mask] - truth))))
    )

    return pd.DataFrame(results, columns=["method", "mae"]).sort_values("mae")


if not coverage_df.empty:
    best_station = coverage_df.iloc[-1]["station_id"]
    focal = (
        nasa[nasa["station_id"] == best_station]
        .sort_values("ts")
        .reset_index(drop=True)
        .head(8000)
    )
    imputation_table = evaluate_imputers(focal, target="temp_c")
    print(
        f"MAE de imputação para temp_c (estação {best_station}, primeiras 8000 horas):"
    )
    display(imputation_table)
else:
    print("Sem dados — execute scripts/bootstrap_data.py primeiro.")

# %% [markdown]
# **Observação**: numa série horária com forte autocorrelação, suavização
# temporal simples (média móvel) costuma vencer imputação multivariada.
# `IterativeImputer` trata as linhas como i.i.d. — ignora completamente o
# índice temporal. Para colher a vantagem multivariada num cenário
# temporal, precisaríamos de uma versão *time-aware* (e.g. com lags como
# features de entrada).

# %% [markdown]
# ## 7. Cascata de fontes
#
# Quando uma estação primária tem gaps, caímos no vizinho mais próximo de
# uma fonte secundária (e.g. NASA POWER → INMET, ou estação a estação
# dentro da mesma fonte). Anotamos cada célula imputada com sua
# proveniência para que o código downstream possa auditar. Distância via
# Haversine em (lat, lon).

# %%
EARTH_RADIUS_KM = 6371.0


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def nearest_station_map(
    primary: pd.DataFrame,
    fallback: pd.DataFrame,
    *,
    self_match_ok: bool = False,
) -> pd.DataFrame:
    p = primary[["station_id", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    f = fallback[["station_id", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    rows = []
    for _, prow in p.iterrows():
        d = haversine_km(prow["lat"], prow["lon"], f["lat"].values, f["lon"].values)
        order = np.argsort(d)
        chosen = order[0]
        if not self_match_ok and f.loc[chosen, "station_id"] == prow["station_id"]:
            if len(order) < 2:
                continue
            chosen = order[1]
        rows.append(
            {
                "primary_id": prow["station_id"],
                "fallback_id": f.loc[chosen, "station_id"],
                "distance_km": round(float(d[chosen]), 2),
            }
        )
    return pd.DataFrame(rows)


if not nasa.empty:
    nn = nearest_station_map(nasa, nasa, self_match_ok=False)
    print("Mapa de vizinho mais próximo (excluindo self-match):")
    display(nn)

# %% [markdown]
# ## Conclusões
#
# - [ ] Documentar o padrão de ausência observado (heatmap da Seção 4)
# - [ ] Registrar o p-valor do teste MCAR de Little em
#       `reports/01_eda_missingness.md`
# - [ ] Eleger a estratégia de imputação usada nos notebooks 02-07
#       (vencedor da Seção 6)
# - [ ] Anotar as distâncias do vizinho mais próximo — qualquer coisa
#       acima de 50 km é suspeita para fallback horário (microclima)
