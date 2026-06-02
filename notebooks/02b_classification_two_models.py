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
# # Notebook 02b — Classificação em Série Temporal: Comparando Dois Modelos
#
# **Cenário operacional**: o gerente agrícola precisa decidir, no instante `t`,
# se a próxima janela de 6 horas tem risco de chuva suficiente para
# **suspender a pulverização** (evitar lavagem do produto). Tratamos a tarefa
# como **classificação binária**:
#
# $$y_t = \mathbb{1}\!\left[\sum_{h=1}^{6}\text{rain}_{t+h} \geq 1\,\text{mm}\right]$$
#
# Comparamos dois modelos clássicos:
#
# - **Regressão Logística** com features de série temporal (lags + rolling
#   + Fourier + calendário). Baseline interpretável.
# - **LightGBM** sobre o mesmo conjunto de features. Modelo flexível,
#   espera-se ganho em PR-AUC quando a relação não é linear.
#
# **Eixos de avaliação** (cada um captura um aspecto diferente):
# 1. Matriz de confusão (em threshold 0.5 e no threshold ótimo de F1)
# 2. Curva ROC + AUC
# 3. Curva Precision-Recall + Average Precision (AP)
# 4. Brier score (combina discriminação + calibração)
# 5. Calibration curve (reliability diagram) antes e depois de calibrar
# 6. Teste de McNemar para diferença estatística entre os dois modelos
#
# **Princípio chave**: nenhuma feature olha para o futuro. Split temporal,
# rolling stats com `closed="left"`, calibração em holdout independente.

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
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)
sns.set_theme(context="notebook", style="whitegrid")

# %% [markdown]
# ## 1. Carregar e normalizar o painel NASA POWER

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


nasa = normalize_nasa(read_parquet("nasa_power_hourly"))
print("Painel:", nasa.shape, "estações:", nasa["station_id"].nunique())

# Focar numa única estação para manter o notebook didático.
# (A versão global multi-série fica em notebook 03 — LightGBM global.)
TARGET_STATION = nasa["station_id"].iloc[0]
focal = (
    nasa[nasa["station_id"] == TARGET_STATION]
    .sort_values("ts")
    .reset_index(drop=True)
    .copy()
)
print(f"Estação foco: {TARGET_STATION} | {len(focal):,} horas")
focal.head()

# %% [markdown]
# ## 2. Construir o target binário (rain ≥ 1 mm nas próximas 6h)
#
# O target em `t` olha exclusivamente para `t+1..t+6`. Para evitar leakage
# em treino, garantimos que nenhuma feature em `t` use `rain` de `t+1` em
# diante.

# %%
def build_target(df: pd.DataFrame, horizon_hours: int = 6, threshold_mm: float = 1.0) -> pd.Series:
    """Soma rain_mm na janela futura (t+1 .. t+horizon) e binariza."""
    future_rain = (
        df["rain_mm"]
        .shift(-1)
        .rolling(window=horizon_hours, min_periods=horizon_hours)
        .sum()
        .shift(-(horizon_hours - 1))
    )
    return (future_rain >= threshold_mm).astype("Int8")


focal["y"] = build_target(focal, horizon_hours=6, threshold_mm=1.0)
class_share = focal["y"].dropna().mean()
print(f"Balanço de classes: {class_share:.1%} positivo (chuva ≥ 1mm em 6h)")
print(f"Total de linhas com target válido: {focal['y'].notna().sum():,}")

# %% [markdown]
# ## 3. Engenharia de features (sem leakage)
#
# - **Lags**: `temp_c`, `rh_pct`, `wind_ms`, `rain_mm` em t-1, t-2, t-3, t-6, t-24
# - **Rolling stats** de `rain_mm` (sum, max, mean) em janelas de 6h, 24h, 168h,
#   sempre `closed="left"` para excluir o instante atual
# - **Fourier**: período 24h (k=2) + período anual 8766h (k=2)
# - **Calendário**: hora do dia, dia do ano

# %%
LAG_VARS = ["temp_c", "rh_pct", "wind_ms", "rain_mm", "pressure_hpa"]
LAGS = [1, 2, 3, 6, 24]
ROLL_WINDOWS = [6, 24, 168]


def add_lag_features(df: pd.DataFrame, vars_: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for v in vars_:
        for k in lags:
            out[f"{v}_lag{k}"] = out[v].shift(k)
    return out


def add_rolling_rain(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        roll = out["rain_mm"].rolling(window=w, min_periods=max(2, w // 4), closed="left")
        out[f"rain_roll{w}_sum"] = roll.sum()
        out[f"rain_roll{w}_max"] = roll.max()
        out[f"rain_roll{w}_mean"] = roll.mean()
    return out


def add_fourier(df: pd.DataFrame, ts_col: str, period_hours: float, k: int, prefix: str) -> pd.DataFrame:
    out = df.copy()
    hours = pd.to_datetime(out[ts_col]).astype("int64") // 10**9 / 3600.0
    omega = 2 * np.pi / period_hours
    for kk in range(1, k + 1):
        out[f"{prefix}_sin_k{kk}"] = np.sin(kk * omega * hours)
        out[f"{prefix}_cos_k{kk}"] = np.cos(kk * omega * hours)
    return out


def add_calendar(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    out["hour"] = ts.dt.hour
    out["doy"] = ts.dt.dayofyear
    return out


feats = add_lag_features(focal, LAG_VARS, LAGS)
feats = add_rolling_rain(feats, ROLL_WINDOWS)
feats = add_fourier(feats, "ts", period_hours=24, k=2, prefix="f24")
feats = add_fourier(feats, "ts", period_hours=8766, k=2, prefix="fyr")
feats = add_calendar(feats, "ts")

FEATURE_COLS = [
    c for c in feats.columns
    if c.startswith(("temp_c_lag", "rh_pct_lag", "wind_ms_lag", "rain_mm_lag",
                     "pressure_hpa_lag", "rain_roll", "f24_", "fyr_"))
    or c in ("hour", "doy")
]
print(f"Features construídas: {len(FEATURE_COLS)}")
print("Exemplos:", FEATURE_COLS[:6], "...")

# Drop linhas com features incompletas (warm-up) e target ausente
data = feats.dropna(subset=FEATURE_COLS + ["y"]).reset_index(drop=True)
print(f"Dataset final: {len(data):,} linhas × {len(FEATURE_COLS)} features")

# %% [markdown]
# ## 4. Split temporal walk-forward
#
# Treino: primeiros 80% do tempo; teste: últimos 20%. Para calibração,
# separamos os últimos 20% do bloco de treino — assim a calibração nunca
# vê o teste.

# %%
def temporal_split(df: pd.DataFrame, train_frac: float = 0.80, calib_frac_of_train: float = 0.20):
    """Divide em treino / calibração / teste preservando ordem temporal."""
    n = len(df)
    n_train_full = int(n * train_frac)
    n_calib = int(n_train_full * calib_frac_of_train)
    n_train = n_train_full - n_calib
    return (
        df.iloc[:n_train].copy(),
        df.iloc[n_train:n_train + n_calib].copy(),
        df.iloc[n_train_full:].copy(),
    )


train, calib, test = temporal_split(data)
for name, part in [("train", train), ("calib", calib), ("test", test)]:
    pos = part["y"].mean()
    print(f"{name:6s}: {len(part):>6,} linhas | {part['ts'].min().date()} → {part['ts'].max().date()} | {pos:.1%} positivo")

# %% [markdown]
# **Por que esse split**: k-fold embaralhado vaza futuro no treino e gera
# métrica otimista. Walk-forward respeita a flecha do tempo. O bloco de
# calibração existe porque calibrar no próprio treino sub-estima a
# incerteza (já vimos esse alvo); calibrar no teste vazaria avaliação.

# %% [markdown]
# ## 5. Modelo A — Regressão Logística

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X_train, y_train = train[FEATURE_COLS].to_numpy(), train["y"].astype(int).to_numpy()
X_calib, y_calib = calib[FEATURE_COLS].to_numpy(), calib["y"].astype(int).to_numpy()
X_test, y_test = test[FEATURE_COLS].to_numpy(), test["y"].astype(int).to_numpy()

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_calib_s = scaler.transform(X_calib)
X_test_s = scaler.transform(X_test)

logreg = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
    n_jobs=-1,
)
logreg.fit(X_train_s, y_train)
p_logreg_test = logreg.predict_proba(X_test_s)[:, 1]
print("Logistic treinada.")

# %% [markdown]
# ## 6. Modelo B — LightGBM

# %%
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_calib, label=y_calib, reference=lgb_train)

lgb_params = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "is_unbalance": True,
    "verbosity": -1,
    "seed": 42,
}

booster = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=600,
    valid_sets=[lgb_valid],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
)
p_lgbm_test = booster.predict(X_test, num_iteration=booster.best_iteration)
print(f"LightGBM treinada. Best iteration: {booster.best_iteration}")

# %% [markdown]
# ## 7. Comparação lado a lado
#
# Cada métrica responde uma pergunta diferente. Reportamos todas e somente
# então afirmamos qual modelo é melhor — afirmar com base em accuracy só
# levaria a um falso vencedor (a classe positiva é minoria).

# %%
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate(name: str, y_true: np.ndarray, p_pred: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (p_pred >= threshold).astype(int)
    return {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, p_pred),
        "pr_auc": average_precision_score(y_true, p_pred),
        "brier": brier_score_loss(y_true, p_pred),
    }


metrics_df = pd.DataFrame(
    [
        evaluate("Logistic Regression", y_test, p_logreg_test, 0.5),
        evaluate("LightGBM", y_test, p_lgbm_test, 0.5),
    ]
).set_index("model")
print("=== Métricas no teste (threshold = 0.5) ===")
display(metrics_df.round(4))

# %% [markdown]
# ### 7a. Matrizes de confusão

# %%
def plot_confusion(ax, y_true, p, threshold, title):
    y_pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Sem chuva (pred)", "Com chuva (pred)"],
        yticklabels=["Sem chuva (real)", "Com chuva (real)"],
    )
    ax.set_title(f"{title}\nthreshold = {threshold:.2f}")


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
plot_confusion(axes[0], y_test, p_logreg_test, 0.5, "Logistic Regression")
plot_confusion(axes[1], y_test, p_lgbm_test, 0.5, "LightGBM")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7b. Curvas ROC e Precision-Recall

# %%
def plot_roc_pr(p_a, p_b, name_a, name_b, y_true):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ROC
    for p, name, color in [(p_a, name_a, "C0"), (p_b, name_b, "C1")]:
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = roc_auc_score(y_true, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="acaso")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Curva ROC")
    axes[0].legend()

    # PR
    base = y_true.mean()
    for p, name, color in [(p_a, name_a, "C0"), (p_b, name_b, "C1")]:
        prec, rec, _ = precision_recall_curve(y_true, p)
        ap = average_precision_score(y_true, p)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, lw=2)
    axes[1].axhline(base, color="k", ls="--", alpha=0.4, label=f"base ({base:.1%})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Curva Precision-Recall")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


plot_roc_pr(p_logreg_test, p_lgbm_test, "Logistic", "LightGBM", y_test)

# %% [markdown]
# ### 7c. Threshold ótimo de F1
#
# Threshold = 0.5 não é especial; é só o ponto onde `argmax(prob) == predição`.
# Quando classes são desbalanceadas, o threshold que maximiza F1 (ou
# F-beta com β > 1 para favorecer recall) costuma ser bem diferente.

# %%
def best_f1_threshold(y_true, p_pred):
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_true, (p_pred >= t).astype(int), zero_division=0) for t in thresholds]
    j = int(np.argmax(f1s))
    return float(thresholds[j]), float(f1s[j])


for name, p in [("Logistic", p_logreg_test), ("LightGBM", p_lgbm_test)]:
    t_opt, f1_opt = best_f1_threshold(y_test, p)
    print(f"{name:10s}: threshold ótimo = {t_opt:.2f}, F1 = {f1_opt:.3f}")

# %% [markdown]
# ## 8. Calibração
#
# Probabilidades bem calibradas significam que "60% de chance de chuva"
# realmente vira chuva em ~60% dos casos com aviso de 60%. Diagrama de
# confiabilidade compara a frequência observada contra a probabilidade
# prevista, em bins.

# %%
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression


def plot_reliability(ax, p, y_true, name, color):
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", color=color, label=name, lw=2)


fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfeitamente calibrado")
plot_reliability(ax, p_logreg_test, y_test, "Logistic", "C0")
plot_reliability(ax, p_lgbm_test, y_test, "LightGBM", "C1")
ax.set_xlabel("probabilidade prevista (bin)")
ax.set_ylabel("frequência observada")
ax.set_title("Reliability diagram — pré-calibração")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8a. Recalibrar via isotônica no bloco de calibração

# %%
p_logreg_calib = logreg.predict_proba(X_calib_s)[:, 1]
p_lgbm_calib = booster.predict(X_calib, num_iteration=booster.best_iteration)

iso_logreg = IsotonicRegression(out_of_bounds="clip").fit(p_logreg_calib, y_calib)
iso_lgbm = IsotonicRegression(out_of_bounds="clip").fit(p_lgbm_calib, y_calib)

p_logreg_test_cal = iso_logreg.transform(p_logreg_test)
p_lgbm_test_cal = iso_lgbm.transform(p_lgbm_test)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfeitamente calibrado")
plot_reliability(ax, p_logreg_test_cal, y_test, "Logistic + isotônica", "C0")
plot_reliability(ax, p_lgbm_test_cal, y_test, "LightGBM + isotônica", "C1")
ax.set_xlabel("probabilidade prevista (bin)")
ax.set_ylabel("frequência observada")
ax.set_title("Reliability diagram — pós-calibração")
ax.legend()
plt.tight_layout()
plt.show()

print()
print("=== Brier score: antes vs depois da calibração ===")
print(f"Logistic   : {brier_score_loss(y_test, p_logreg_test):.4f} → {brier_score_loss(y_test, p_logreg_test_cal):.4f}")
print(f"LightGBM   : {brier_score_loss(y_test, p_lgbm_test):.4f} → {brier_score_loss(y_test, p_lgbm_test_cal):.4f}")

# %% [markdown]
# ## 9. Teste de McNemar
#
# Os dois modelos são estatisticamente diferentes ou estamos vendo ruído?
# McNemar olha para a tabela 2×2 de **discordâncias** entre os dois
# classificadores nas mesmas amostras de teste.
#
# $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{onde:}$$
# - `b` = casos em que A acerta e B erra
# - `c` = casos em que A erra e B acerta
#
# Sob H₀ (modelos equivalentes), `b` e `c` seriam iguais em média.

# %%
def mcnemar(y_true, pred_a, pred_b):
    a_right = pred_a == y_true
    b_right = pred_b == y_true
    n10 = int(np.sum(a_right & ~b_right))  # A acerta, B erra
    n01 = int(np.sum(~a_right & b_right))  # A erra, B acerta
    chi2 = (abs(n10 - n01) - 1) ** 2 / max(n10 + n01, 1)
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return {"a_right_b_wrong": n10, "a_wrong_b_right": n01, "chi2": chi2, "p_value": float(p)}


pred_logreg = (p_logreg_test_cal >= 0.5).astype(int)
pred_lgbm = (p_lgbm_test_cal >= 0.5).astype(int)
print("Teste de McNemar (Logistic vs LightGBM, calibrados, threshold 0.5):")
print(mcnemar(y_test, pred_logreg, pred_lgbm))

# %% [markdown]
# ## 10. Importância de features (LightGBM)
#
# Sanity check: as features que o LightGBM mais usou batem com a
# intuição agronômica?

# %%
imp = pd.DataFrame(
    {"feature": FEATURE_COLS, "gain": booster.feature_importance(importance_type="gain")}
).sort_values("gain", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=imp, x="gain", y="feature", color="C1", ax=ax)
ax.set_title("Top-15 features por ganho (LightGBM)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Tabela final + conclusões
#
# Recapitulando todas as métricas, agora também com versões calibradas.

# %%
final = pd.DataFrame(
    [
        evaluate("Logistic (raw)", y_test, p_logreg_test, 0.5),
        evaluate("Logistic (isotônica)", y_test, p_logreg_test_cal, 0.5),
        evaluate("LightGBM (raw)", y_test, p_lgbm_test, 0.5),
        evaluate("LightGBM (isotônica)", y_test, p_lgbm_test_cal, 0.5),
    ]
).set_index("model").round(4)
display(final)

# %% [markdown]
# ### Pontos a defender na entrevista
#
# 1. **Por que não usei accuracy como critério primário** — a classe
#    positiva é minoria; um modelo trivial que diz "nunca chove" teria
#    accuracy alta e zero utilidade.
# 2. **Por que separei calibração** — calibrar no treino subestima a
#    incerteza; no teste vazaria a avaliação. Holdout temporal específico
#    para calibração é o caminho honesto.
# 3. **Por que walk-forward** — k-fold embaralhado vaza futuro. Em série
#    temporal sazonal o vazamento mascara falhas graves do modelo.
# 4. **Por que rolling com `closed="left"`** — sem isso, a feature em `t`
#    incorpora o valor de `t`, que está embutido implicitamente no
#    target.
# 5. **Por que McNemar e não t-test pareado** — a saída é binária; a
#    distribuição da diferença não é normal.
# 6. **Próximos passos**: adicionar covariáveis exógenas (índices
#    climáticos, ENSO), avaliar com curva de custo (FP e FN com pesos
#    diferentes), comparar com modelo global multi-estação (Notebook 03).
