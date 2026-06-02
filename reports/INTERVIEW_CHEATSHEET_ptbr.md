# Cheat Sheet — Entrevista TT-4A Embrapa Agricultura Digital

> 15 minutos. Time **Mineração de Séries Temporais e Plataforma de Dados**.
> Pontos de fala compactos, em pt-BR, com gatilho técnico para resposta longa
> quando perguntado.

---

## 0. Posicionamento de abertura (30 s)

> "Tenho 4 anos trabalhando com dados em produção, dos últimos 1,5 anos focado
> em agritech — construí uma plataforma de inteligência de aplicações
> agrícolas que cruza telemetria de máquinas com dados de clima de múltiplas
> fontes para classificar a qualidade de cada pulverização. A vaga de
> mineração de séries temporais em uma plataforma de dados na Embrapa cai
> exatamente no centro do que venho fazendo, com a vantagem do contexto
> público e de pesquisa que eu não tenho no privado."

> Em seguida menciono o `agritime` como o repo público que estou usando para
> consolidar a parte de modelagem preditiva — sem nomear a empresa privada.

---

## 1. Por que séries temporais agrícolas são diferentes

| Aspecto | Implicação prática |
|---|---|
| **Forte sazonalidade aninhada** (diária + anual) | STL decomposition; Fourier features; modelos com sazonalidade explícita (SARIMA, Prophet, TFT) |
| **Cobertura desigual entre estações** | Cascata de fontes com proveniência (NASA POWER → INMET → modelo); kriging para interpolar |
| **Eventos raros importam mais** (geada, granizo, chuva forte) | Métricas sensíveis a desbalanceamento (PR-AUC, F1, recall na classe minoritária); reamostragem ou class_weight |
| **Decisões com janela curta** (ex: aplicar ou não em 6h) | Forecasting probabilístico + calibração; conformal prediction para intervalos com cobertura garantida |

---

## 2. MCAR / MAR / MNAR — o que dizer em 30 segundos

- **MCAR**: ausência independe de tudo. Imputação ingênua é não-viesada na média.
- **MAR**: ausência depende de variável **observada** (e.g. sensor falha em frio → temp_c observada explica o gap em rain_mm). `IterativeImputer` condicionando em covariáveis recupera a distribuição.
- **MNAR**: ausência depende do **valor escondido** (e.g. saturação de sensor em calor extremo). Nenhum imputador é imparcial sem informação externa.
- **Teste de Little (1988)**: qui-quadrado que compara médias das observadas entre padrões de ausência. p < 0.05 → rejeitar MCAR.

> Comprovei isso em `notebooks/01_eda_missingness.py` injetando os três tipos sobre dados gap-free da NASA POWER. Little rejeita MCAR com p ≈ 0 na amostra sintetizada e aceita trivialmente na bruta.

---

## 3. Walk-forward CV — por que k-fold mata séries temporais

- K-fold embaralha → vaza futuro no treino → métrica otimista que não se sustenta em produção.
- **Walk-forward / time-series split**: janela de treino cresce (expansiva) ou desliza (sliding), janela de teste avança no tempo.
- Em painéis multi-série globais, garanta que o split é por **tempo**, não por série, senão você vaza padrões entre séries.
- `sklearn.model_selection.TimeSeriesSplit` para CV simples; `darts` e `sktime` têm wrappers para múltiplos horizontes.

---

## 4. Forecasting clássico — quando pesa, quando perde

| Modelo | Bom quando | Falha quando |
|---|---|---|
| **Naive sazonal (lag-24)** | Forte sazonalidade limpa, baseline obrigatório | Tendência forte; ruído curto-prazo |
| **SARIMA** | Série única, estacionária após diferenciação, sazonalidade conhecida | Não-linearidades; múltiplas séries; covariáveis fortes |
| **ETS** | Tendência + sazonalidade aditiva clara | Sazonalidades múltiplas; quebras estruturais |
| **Prophet** | Negócio com calendário pesado (feriados, eventos); usuários não-técnicos | Cadência sub-diária; quando MAE importa mais que interpretabilidade |

**Estacionariedade**: ADF (testa se há raiz unitária — rejeita ⇒ estacionária); KPSS (testa o oposto — rejeita ⇒ não-estacionária). Concordância dos dois é robusta; discordância sugere série trend-stationary.

---

## 5. Modelos globais com features tabulares (LightGBM)

- Trate o painel como tabular: `(series_id, t)` é uma linha; alvo = valor em `t + horizonte`.
- Features chave: **lags** do alvo (1, 2, 3, 24, 168), **rolling stats** (média, std, min, max em janelas), **Fourier** (sin/cos de períodos sazonais), **calendário** (hora, dow, doy, feriado), **covariáveis exógenas** (e.g. temp para prever chuva).
- LightGBM costuma bater SARIMA/Prophet em painéis curtos e ruidosos.
- Cuidado com **leakage**: rolling stats centradas vazam futuro; use `closed="left"`.

> Em `agritime` faço isso em `src/agritime/features/lags.py` (add_lags, add_rolling, add_fourier, add_calendar).

---

## 6. Classificação em série temporal — como comparar dois modelos

### Pipeline de comparação (o pedaço da entrevista)

1. **Target binário operacional**: ex. "vai chover ≥ 1 mm nas próximas 6 h?"
2. **Features**: janela passada (lags + rolling + Fourier) sem peek no futuro.
3. **Modelos**: baseline simples (Logistic Regression) vs. modelo aprendido (LightGBM).
4. **Split temporal**: treino até T0, teste após T0. Nunca shuffle.
5. **Métricas em conjunto** (cada uma capta um eixo):

| Métrica | Captura | Atenção |
|---|---|---|
| **Confusion matrix** (em ≥ 1 threshold) | Tipo de erro (FP vs FN) — crítico em decisão agrícola | Mostre matriz em 0.5 e no threshold ótimo |
| **Accuracy** | Visão geral | Inútil sob desbalanceamento (chuva é rara em SP no inverno) |
| **Precision / Recall / F1** | Trade-off entre alertas falsos e perdidos | Reporte na classe positiva, não macro |
| **ROC-AUC** | Capacidade de ranking | Otimista sob desbalanceamento severo |
| **PR-AUC (AP)** | Ranking quando positivo é raro | Métrica de escolha quando classe + é minoria |
| **Brier score** | Calibração + discriminação juntos | Quanto menor, melhor |
| **Reliability diagram** | Calibração visual | Se distante da diagonal → calibrar (Platt / isotônica) |
| **McNemar's test** | Diferença estatística entre dois classificadores | Apropriado para dados pareados (mesma amostra) |

### Threshold matters

- Argmax/0.5 é só um ponto. Mostre **curva de custo** se FP e FN têm pesos assimétricos (em agro, perder uma chuva = perda de produto; falso alarme = atraso operacional).
- Selecione threshold por **F-beta** (β > 1 favorece recall) ou por **expected cost**.

### Calibração — por que importa em agronomia

- Quando o usuário lê "60% de chance de chuva", isso precisa significar que em 100 dias com aviso de 60%, choveu em ~60. Senão a decisão é pior que random.
- **Platt scaling** (sigmoid) ou **isotonic regression** sobre um holdout independente.
- **Conformal prediction** garante cobertura marginal sem premissas de distribuição.

---

## 7. Geoespacial / multi-estação — vocabulário rápido

- **Haversine** para distância entre estações em (lat, lon).
- **Kriging ordinário** ↔ **Gaussian Process com kernel RBF**: matematicamente equivalentes, vocabulários diferentes (geoestatística vs. ML).
- **Semivariograma** parametriza a covariância espacial: nugget, sill, range.
- **Kernel separável espaço-tempo** = RBF espacial × RBF temporal. Simples e suficiente quando dependências não se cruzam.
- Vizinho mais próximo é baseline difícil de bater em redes densas; perde feio em redes esparsas.

---

## 8. Sensoriamento remoto — falar com naturalidade

- **Sentinel-2 L2A**: 10 m, revisita 5 dias, gratuito via Microsoft Planetary Computer (STAC).
- **NDVI = (NIR - Red) / (NIR + Red)** — proxy de vigor vegetativo. EVI e NDWI são primos.
- **MapBiomas** dá rótulos anuais de uso do solo (mata, agricultura, pastagem...).
- Para classificação de cultura a partir de série de NDVI: **ROCKET** ou **MiniRocket** são baselines fortíssimos (convoluções aleatórias + regressão linear).
- **Embrapa SATVeg** fornece NDVI/EVI MODIS por polígono — está no escopo direto do time.

---

## 9. Plataforma de Dados — como mostro que penso o lado de engenharia

- **TimescaleDB** (hypertable em PostgreSQL) para ingestão e queries com janelas temporais; compressão automática para chunks antigos.
- **Parquet** particionado para snapshots brutos imutáveis (medalhão bronze).
- **MLflow** para tracking de experimentos + registry para promover modelos a prod.
- **Pipeline reprodutível**: Docker compose subindo tudo em um comando. Repo `agritime` faz exatamente isso.
- **Sealed contracts** entre backend e frontend (vivido em projeto privado) — princípio que evita drift entre o que o modelo serve e o que o consumidor espera.

---

## 10. Frases-âncora para o final do papo

- "Onde o time já investe pesquisa, eu chego como executor competente; onde ainda não chegou, eu trago o repertório de produção privada."
- "Não tenho doutorado em estatística — minha vantagem é fechar o loop entre o modelo e a decisão operacional do produtor."
- "Estou nessa carreira por curiosidade pelo dado, e tenho disciplina de operar dentro de processo (clean architecture, traceability, sealed contracts)."

---

## 11. Perguntas que faço se houver tempo

1. Qual é o horizonte de previsão que tem mais valor estratégico hoje — diário, semanal, sazonal?
2. A plataforma serve um endpoint de inferência (modelos em produção) ou ainda é primariamente analítica?
3. Existe um conjunto de dados rotulados maduro para classificação de eventos extremos (geada, granizo, seca) que eu possa atacar nos primeiros 90 dias?
4. Como o time mede sucesso de um modelo preditivo — métrica técnica, adoção pelo produtor, decisão de política pública?

---

## 12. Não-ditos importantes

- **Não cite cliente privado por nome.** Diga "uma fintech de agritech" ou "um cliente do agro" se precisar de contexto.
- **Não venda framework que você não rodou** (TFT, N-BEATS, conformal). Se mencionar, ancore em "li, entendi o papel, ainda não rodei em produção".
- **Não negue o gap**. Se perguntarem por algo que você não fez, diga "ainda não, mas conheço a literatura, faria assim:" e desenhe o approach.
