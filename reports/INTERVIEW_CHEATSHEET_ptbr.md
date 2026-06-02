# Cheat Sheet — Entrevista TT-4A Embrapa Agricultura Digital

> 15 minutos. Time **Mineração de Séries Temporais e Plataforma de Dados**.
> Pontos de fala compactos, em pt-BR, com gatilho técnico para resposta longa
> quando perguntado.
>
> **Convenção**: todo acrônimo é expandido na primeira vez que aparece. Termos
> técnicos novos vêm com uma frase de definição inline.

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

## 1. Fontes de dados públicas que estou usando

Todas gratuitas e sem autenticação obrigatória. O projeto `agritime` materializa cada fonte em **Parquet** (formato colunar de armazenamento) local — snapshots imutáveis — e depois carrega em **TimescaleDB** (extensão do PostgreSQL para séries temporais) para queries com janela.

| Fonte | O que entrega | Resolução / cadência | Status |
|---|---|---|---|
| **NASA POWER** = *Prediction Of Worldwide Energy Resources* | Clima horário (temperatura, umidade relativa, vento, chuva, pressão, radiação) modelado a partir de satélite | Grid global ~0,5°, horário, 1981+ | ✅ Ingerido (4 pontos × 5 anos para SP) |
| **INMET BDMEP** = *Banco de Dados Meteorológicos para Ensino e Pesquisa* | Estações meteorológicas terrestres brasileiras | ~600 estações, horário | ⚠️ Portal está derrubando conexões |
| **Embrapa SATVeg** | Índices de vegetação por polígono, derivados do sensor satelital **MODIS** = *Moderate Resolution Imaging Spectroradiometer* (NASA). Índices NDVI/EVI definidos na Seção 9 | ~250 m, ~16 dias | Planejado para notebook 05 |
| **Sentinel-2 L2A** | Imagens ópticas brutas. L2A = produto com correção atmosférica. Acesso via Microsoft Planetary Computer usando o catálogo **STAC** = *SpatioTemporal Asset Catalog* | 10 m, revisita 5 dias | Planejado para notebook 05 |
| **MapBiomas** | Projeto brasileiro com rótulos anuais de uso e cobertura do solo (mata, agricultura, pastagem…) | 30 m, anual, 1985-2024 | Planejado para notebook 05 |
| **CONAB** = *Companhia Nacional de Abastecimento* | Produção e produtividade de safra por município | Municipal, mensal | Planejado para notebook 03 |
| **ANA HidroWeb** | Vazão de rios e chuva. ANA = *Agência Nacional de Águas* | Estação fluviométrica, diário | Backup para ground-truth de chuva |

**Como falar disso**: "Hoje rodo em cima da NASA POWER porque é a única que está confiável via API pública. INMET tem dado equivalente — só que medido em solo — mas o portal está instável. O plano é triangular as duas (NASA é modelada via satélite, INMET é estação terrestre) e cruzar com NDVI/EVI da SATVeg quando o foco for cultura."

---

## 2. Por que séries temporais agrícolas são diferentes

| Aspecto | Implicação prática |
|---|---|
| **Forte sazonalidade aninhada** (diária + anual) | Decomposição **STL** = *Seasonal-Trend decomposition using Loess* (separa tendência + sazonalidade + resíduo); **features de Fourier** (pares sin/cos com período conhecido como variáveis de entrada do modelo); modelos com sazonalidade explícita como SARIMA / Prophet / TFT (definidos na Seção 5) |
| **Cobertura desigual entre estações** | Cascata de fontes com proveniência (NASA POWER → INMET → modelo); interpolação espacial via **kriging** (técnica geoestatística, definida na Seção 8) |
| **Eventos raros importam mais** (geada, granizo, chuva forte) | Métricas sensíveis a desbalanceamento (PR-AUC e F1 são definidos na Seção 7); reamostragem do conjunto de treino ou parâmetro `class_weight` no modelo |
| **Decisões com janela curta** (ex: aplicar ou não em 6h) | Previsão probabilística + calibração; **conformal prediction** (técnica de calibração com cobertura garantida, definida na Seção 7) |

---

## 3. MCAR / MAR / MNAR — mecanismos de ausência de dados

Três tipos de **por que** um dado está ausente. A diferença importa porque cada um exige uma estratégia de imputação diferente. Framework introduzido por Donald Rubin (1976).

### MCAR — *Missing Completely At Random* (Ausência Completamente ao Acaso)

A probabilidade de um dado estar ausente **não depende de nada** — nem do valor que se perderia, nem de outras variáveis.

> *Exemplo*: data logger pifou por causa de queda de energia. A chance da chuva estar faltando às 14h de quarta independe do quanto choveu, da temperatura, do dia da semana.

**Consequência**: imputação ingênua (média, último valor observado) é **não-viesada na média**. O dado observado é uma amostra aleatória do dado completo.

### MAR — *Missing At Random* (Ausência Condicional ao Acaso)

A probabilidade de ausência depende **apenas de variáveis observadas** (não da variável que está faltando).

> *Exemplo*: sensor de chuva trava quando a temperatura cai abaixo de 5 °C. A chance de `rain_mm` faltar depende de `temp_c` — que está observada. Sabendo a temperatura, a probabilidade de ausência é previsível.

**Consequência**: imputação que **condiciona em covariáveis** recupera a distribuição. `IterativeImputer` (do scikit-learn) com BayesianRidge sobre as outras variáveis funciona. Imputação ingênua é viesada (subestima chuva total porque os dias frios — quando chove menos — são os que faltam).

### MNAR — *Missing Not At Random* (Ausência Não-Aleatória)

A probabilidade de ausência depende do **próprio valor que se esconderia** (ou de outra variável que não está sendo observada).

> *Exemplo*: termômetro satura acima de 45 °C — exatamente os valores mais extremos somem. Sabendo só o que sobrou, você não consegue inferir os ausentes sem informação externa (outra estação, histórico de saturação).

**Consequência**: **nenhum imputador é imparcial** sem informação adicional. Tem que admitir o viés, modelar explicitamente o mecanismo de ausência (*selection model*), ou trazer dado externo.

### Como diagnosticar

- **Visual**: heatmap de nulos por (estação × mês) revela padrão temporal (MAR concentra em estações específicas, MNAR concentra em períodos de extremo).
- **Formal**: **teste de Little (1988)** — qui-quadrado que compara médias das variáveis observadas entre padrões de ausência. H₀ é MCAR. p < 0,05 → rejeitar MCAR (e assumir MAR ou MNAR).
- **Distinguir MAR de MNAR**: estatisticamente impossível só com o dado observado. Tem que vir de conhecimento do mecanismo físico (sabe-se que o sensor satura).

> Em `notebooks/01_eda_missingness.py` injeto os três tipos sobre dados gap-free da NASA POWER. O teste de Little rejeita MCAR com p ≈ 0 na amostra sintetizada e aceita trivialmente na bruta.

---

## 4. Validação cruzada respeitando o tempo (*walk-forward*)

**Cross-validation** (CV, "validação cruzada") é o conjunto de técnicas que estimam o desempenho do modelo em dados que ele nunca viu. A ideia padrão é dividir os dados em pedaços, treinar em alguns e medir em outros, rotacionando.

### Por que k-fold quebra em série temporal

**K-fold** é a CV mais comum: embaralha as linhas, divide em k pedaços (*folds*), treina em k-1 e testa no 1 restante, repete k vezes. Funciona quando as linhas são independentes — o que **não vale** em série temporal: linhas próximas no tempo são correlacionadas (a temperatura de 14h carrega informação sobre 15h).

Quando você embaralha, **vaza futuro** no treino: o modelo é treinado em linhas de Janeiro e testado em Fevereiro do mesmo ano, mas também em Dezembro do ano anterior. Esse Dezembro carrega a sazonalidade que o modelo vai precisar prever em Fevereiro — métrica fica artificialmente boa, e o modelo decepciona em produção. Esse é o **leakage** (vazamento): informação que não estaria disponível no momento da previsão real entra no treino.

### Walk-forward = caminhar pra frente no tempo

A ideia é simples: **treina em passado, testa em futuro, avança a janela**.

- **Variante expansiva** (*expanding window*): o treino sempre começa no início e cresce. Janela de teste avança em saltos do tamanho do horizonte.
- **Variante deslizante** (*sliding window*): o treino tem tamanho fixo e desliza junto com o teste. Melhor quando há quebras estruturais (regime mudou).

Em painel multi-série (muitas estações), o split tem que ser **por tempo**, não por série — senão você treina na estação X durante Janeiro de 2023 e testa em outra estação na mesma data, vazando padrões entre séries simultâneas.

Ferramentas: `sklearn.model_selection.TimeSeriesSplit` para CV simples (expansiva). Bibliotecas como `darts` e `sktime` têm wrappers para múltiplos horizontes e métricas pareadas.

> Em `notebooks/02b_classification_two_models.py` faço split temporal simples (80% treino + 20% teste, mais um bloco de calibração separado dentro do treino). Para a entrevista basta saber explicar **por que** k-fold quebra; o resto é detalhe de implementação.

---

## 5. Forecasting clássico — quando pesa, quando perde

Vocabulário rápido dos modelos:

- **Naive sazonal** (lag-N): previsão = valor de N períodos atrás. Para dados horários com sazonalidade diária, lag-24. Baseline obrigatório.
- **SARIMA** = *Seasonal AutoRegressive Integrated Moving Average*. Combina auto-regressão (AR), diferenciação (I) e média móvel (MA) com componentes sazonais. Clássico para série única estacionária.
- **ETS** = *Error / Trend / Seasonality*. Família de modelos state-space generalizando Holt-Winters. Aditivo ou multiplicativo em cada componente.
- **Prophet**: biblioteca da Meta. Modelo aditivo bayesiano com tendência logística + Fourier para sazonalidade + dummies de feriados.
- **TFT** = *Temporal Fusion Transformer*: arquitetura deep para previsão multi-horizonte com atenção sobre covariáveis. Cobertura conceitual aqui; rodá-lo em produção fica para depois (ver Seção 13).

| Modelo | Bom quando | Falha quando |
|---|---|---|
| Naive sazonal | Forte sazonalidade limpa | Tendência forte; ruído curto-prazo |
| SARIMA | Série única, estacionária após diferenciação, sazonalidade conhecida | Não-linearidades; múltiplas séries; covariáveis fortes |
| ETS | Tendência + sazonalidade aditiva clara | Sazonalidades múltiplas; quebras estruturais |
| Prophet | Negócio com calendário pesado (feriados, eventos); usuários não-técnicos | Cadência sub-diária; quando **MAE** = *Mean Absolute Error* importa mais que interpretabilidade |

**Estacionariedade**: propriedade de a distribuição estatística (média, variância) não mudar no tempo. Dois testes complementares:

- **ADF** = *Augmented Dickey-Fuller*: rejeitar H₀ ⇒ série é estacionária.
- **KPSS** = *Kwiatkowski-Phillips-Schmidt-Shin*: rejeitar H₀ ⇒ série **não** é estacionária.

Concordância dos dois é robusta; discordância sugere série *trend-stationary* (estacionária após remover tendência).

---

## 6. Modelos globais com features tabulares (**LightGBM** = *Light Gradient Boosting Machine*)

**LightGBM** é uma implementação rápida de *gradient boosting* em árvores de decisão. A ideia é treinar um modelo único sobre o painel inteiro (todas as estações × todo o tempo) tratando-o como tabela.

- Trate o painel como tabular: `(series_id, t)` é uma linha; alvo = valor em `t + horizonte`.
- Features chave:
  - **Lags** do alvo (valor em t-1, t-2, t-3, t-24, t-168 horas).
  - **Rolling stats** (média, desvio padrão, mínimo, máximo em janelas).
  - **Fourier** (pares sin/cos com período fixo, e.g. 24 h para diária, 8766 h para anual).
  - **Calendário** (hora, dia da semana, dia do ano, feriado).
  - **Covariáveis exógenas** (e.g. temperatura para prever chuva).
- LightGBM costuma bater SARIMA e Prophet em painéis curtos e ruidosos.
- Cuidado com **vazamento**: rolling stats centradas no instante atual usam dados do futuro. Use `closed="left"` no `.rolling()` do pandas para excluir o ponto atual.

> Em `agritime` faço isso em `src/agritime/features/lags.py` (`add_lags`, `add_rolling`, `add_fourier`, `add_calendar`).

---

## 7. Classificação em série temporal — como comparar dois modelos

### Pipeline de comparação (o pedaço da entrevista)

1. **Target binário operacional**: ex. "vai chover ≥ 1 mm nas próximas 6 h?"
2. **Features**: janela passada (lags + rolling + Fourier) sem peek no futuro.
3. **Modelos**: baseline simples (Regressão Logística) vs. modelo aprendido (LightGBM).
4. **Split temporal**: treino até T0, teste após T0. Nunca shuffle (ver Seção 4).
5. **Métricas em conjunto** (cada uma capta um eixo):

| Métrica | Captura | Atenção |
|---|---|---|
| **Matriz de confusão** (em ≥ 1 threshold) | Tipo de erro (falso positivo vs falso negativo) — crítico em decisão agrícola | Mostre matriz em 0.5 e no threshold ótimo |
| **Accuracy** | Visão geral | Inútil sob desbalanceamento (chuva é rara em SP no inverno) |
| **Precision / Recall / F1** | Trade-off entre alertas falsos e perdidos. F1 = média harmônica de precision e recall | Reporte na classe positiva, não macro |
| **ROC-AUC** = *Receiver Operating Characteristic — Area Under Curve* | Capacidade de ranking do modelo (taxa de verdadeiros positivos vs falsos positivos, variando threshold) | Otimista sob desbalanceamento severo |
| **PR-AUC** = *Precision-Recall — Area Under Curve* (também chamada AP = *Average Precision*) | Mesma ideia da ROC, mas no espaço Precision × Recall; melhor para classe positiva rara | Métrica de escolha quando a classe positiva é minoria |
| **Brier score** | Erro quadrático médio entre probabilidade prevista e label binário. Combina discriminação + calibração | Quanto menor, melhor |
| **Reliability diagram** | Calibração visual: probabilidade prevista vs frequência observada, em bins | Se distante da diagonal → calibrar (Platt ou isotônica, ver abaixo) |
| **Teste de McNemar** | Qui-quadrado em dados pareados que mede se dois classificadores discordam estatisticamente | Apropriado porque os modelos veem as mesmas amostras |

### Threshold importa

- Argmax / 0.5 é só um ponto. Mostre **curva de custo** se falso positivo e falso negativo têm pesos assimétricos (em agro: perder uma chuva = perda de produto; falso alarme = atraso operacional).
- Selecione threshold por **F-beta** (generalização de F1 com peso β > 1 para favorecer recall, β < 1 para favorecer precision) ou por **expected cost**.

### Calibração — por que importa em agronomia

- Quando o usuário lê "60% de chance de chuva", isso precisa significar que em 100 dias com aviso de 60%, choveu em ~60. Senão a decisão é pior que random.
- **Platt scaling**: ajusta uma regressão logística sobre a probabilidade prevista (assume função sigmoide).
- **Regressão isotônica**: ajuste monotônico não-paramétrico — mais flexível, precisa de mais dados de holdout.
- **Conformal prediction**: técnica que garante cobertura marginal (e.g. "o intervalo previsto cobre o verdadeiro em 90% das amostras de teste") sem premissas de distribuição. Mais robusta quando a distribuição muda no tempo.

---

## 8. Geoespacial / multi-estação — vocabulário rápido

- **Fórmula de Haversine**: distância na esfera entre dois pontos (lat, lon). Forma padrão de medir distância entre estações.
- **Kriging ordinário** (geoestatística) ↔ **Gaussian Process** (machine learning) com kernel **RBF** = *Radial Basis Function* (kernel gaussiano): matematicamente equivalentes, vocabulários diferentes.
- **Semivariograma**: função que descreve como a variância entre dois pontos cresce com a distância entre eles. Parâmetros:
  - **Nugget**: variância em distância zero (ruído / micro-escala não resolvida).
  - **Sill**: variância máxima (patamar).
  - **Range** (alcance): distância onde a covariância vira ~zero.
- **Kernel separável espaço-tempo** = kernel espacial × kernel temporal. Simples e suficiente quando dependências não se cruzam.
- Vizinho mais próximo é baseline difícil de bater em redes densas; perde feio em redes esparsas.

---

## 9. Sensoriamento remoto — falar com naturalidade

- **Sentinel-2 L2A**: satélite europeu (programa Copernicus), banda óptica, resolução 10 m, revisita 5 dias. Acesso gratuito via Microsoft Planetary Computer (Seção 1).
- **MODIS** = *Moderate Resolution Imaging Spectroradiometer*: sensor da NASA, resolução ~250 m, revisita diária. Base do produto Embrapa SATVeg.
- **NDVI** = *Normalized Difference Vegetation Index* = (NIR − Red) / (NIR + Red), onde NIR é a banda do infravermelho próximo e Red a do vermelho visível. Proxy de vigor vegetativo entre -1 e 1.
- **EVI** = *Enhanced Vegetation Index*. Refinamento do NDVI menos saturado em vegetação densa.
- **NDWI** = *Normalized Difference Water Index*. Versão para água / umidade.
- Para classificação de cultura a partir de série de NDVI: **ROCKET** = *RandOm Convolutional KErnel Transform* (Dempster et al., 2020) e **MiniRocket** (versão otimizada) são baselines fortíssimos — convoluções aleatórias + regressão linear, treinam em segundos.
- **Embrapa SATVeg** fornece NDVI/EVI MODIS por polígono — está no escopo direto do time onde estou aplicando.

---

## 10. Plataforma de Dados — como mostro que penso o lado de engenharia

- **TimescaleDB**: extensão do PostgreSQL que adiciona *hypertables* (particionamento automático por tempo) para ingestão e queries com janelas temporais; compressão automática para chunks antigos.
- **Parquet**: formato colunar de armazenamento. Usado para snapshots brutos imutáveis (camada "bronze" no jargão de data lake).
- **MLflow**: ferramenta de tracking de experimentos de ML + *model registry* para promover modelos a produção.
- **Pipeline reprodutível**: Docker Compose subindo tudo em um comando. Repo `agritime` faz exatamente isso (Postgres + TimescaleDB + MLflow + JupyterLab).
- **Sealed contracts** entre backend e frontend (vivido em projeto privado) — princípio que evita drift entre o que o modelo serve e o que o consumidor espera.

---

## 11. Frases-âncora para o final do papo

- "Onde o time já investe pesquisa, eu chego como executor competente; onde ainda não chegou, eu trago o repertório de produção privada."
- "Não tenho doutorado em estatística — minha vantagem é fechar o loop entre o modelo e a decisão operacional do produtor."
- "Estou nessa carreira por curiosidade pelo dado, e tenho disciplina de operar dentro de processo (clean architecture, traceability, sealed contracts)."

---

## 12. Perguntas que faço se houver tempo

1. Qual é o horizonte de previsão que tem mais valor estratégico hoje — diário, semanal, sazonal?
2. A plataforma serve um endpoint de inferência (modelos em produção) ou ainda é primariamente analítica?
3. Existe um conjunto de dados rotulados maduro para classificação de eventos extremos (geada, granizo, seca) que eu possa atacar nos primeiros 90 dias?
4. Como o time mede sucesso de um modelo preditivo — métrica técnica, adoção pelo produtor, decisão de política pública?

---

## 13. Não-ditos importantes

- **Não cite cliente privado por nome.** Diga "uma fintech de agritech" ou "um cliente do agro" se precisar de contexto.
- **Não venda framework que você não rodou** (TFT, ROCKET, conformal prediction). Se mencionar, ancore em "li, entendi o papel, ainda não rodei em produção".
- **Não negue o gap**. Se perguntarem por algo que você não fez, diga "ainda não, mas conheço a literatura, faria assim:" e desenhe o approach.
