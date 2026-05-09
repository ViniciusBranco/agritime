# agritime curriculum write-ups

Each notebook ends with one corresponding markdown report capturing:

1. **Findings** — what the data actually looks like
2. **Modeling decisions** — what was picked and why
3. **Open questions and follow-ups**

The reports are the artifact a hiring panel would actually read — keep them tight (1-2 pages each).

| Notebook | Report |
|---|---|
| `01_eda_missingness` | `reports/01_eda_missingness.md` |
| `02_forecasting_classical` | `reports/02_forecasting_classical.md` |
| `03_forecasting_ml_global` | `reports/03_forecasting_ml_global.md` |
| `04_forecasting_deep` | `reports/04_forecasting_deep.md` |
| `05_ts_classification_anomaly` | `reports/05_ts_classification_anomaly.md` |
| `06_calibrated_risk_conformal` | `reports/06_calibrated_risk_conformal.md` |
| `07_spatiotemporal_kriging` | `reports/07_spatiotemporal_kriging.md` |

## Suggested write-up template

```markdown
# Notebook NN — <theme>

## TL;DR
One paragraph. What was the question, what did you do, what did you learn.

## Data
What you used, where it came from, how much was usable.

## Method
What you tried. Bullet list, not prose.

## Results
Table of metrics. Plots inline. Be honest about losses.

## Discussion
What the results imply for an agritech production system. What you'd do next
with another week.

## Open questions
The things you noticed but didn't have time to chase.
```
