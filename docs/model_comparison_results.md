# Consolidated Forecast Model Comparison

## Evaluation setup

- Ticker: RELIANCE.NS
- Development target-date cutoff: 2022-06-14
- Common forecast targets: 3,549
- Common target period: 2008-01-17 to 2022-06-14
- Validation: expanding-window, one-day-ahead forecasts

Only target dates shared by the included models were compared.
Realized prices and returns were checked for consistency.

## Regression results

| Model | Close MAE | Close RMSE | Return MAE (bps) | Return RMSE (bps) | Directional accuracy |
|---|---:|---:|---:|---:|---:|
| Random Walk | 5.321983 | 9.040784 | 148.162015 | 211.029053 | N/A |
| Random Walk with Drift | 5.326888 | 9.040945 | 148.468940 | 211.343515 | 50.38% |
| Ridge — trend only | 5.330085 | 9.047627 | 148.618104 | 211.830023 | 50.41% |
| Ridge — volatility only | 5.332886 | 9.043215 | 148.775039 | 211.798540 | 50.44% |
| Ridge — returns only | 5.343676 | 9.071102 | 148.838349 | 211.750245 | 51.39% |
| Random Forest | 5.336922 | 9.050508 | 148.917898 | 212.040353 | 50.46% |
| Ridge — all features | 5.365147 | 9.094519 | 149.356697 | 212.048669 | 50.72% |
| Ridge baseline | 5.365147 | 9.094519 | 149.356697 | 212.048669 | 50.72% |
| Decision Tree | 5.372325 | 9.099548 | 150.236245 | 214.607755 | 49.90% |

Lower MAE and RMSE indicate smaller forecast errors. Return errors are
measured in basis points of log return, not investment returns.

The Ridge baseline and Ridge all-feature results are identical because
they use the same features and fixed Ridge configuration.

## Logistic Regression classification results

| Model | Accuracy | Balanced accuracy | ROC-AUC | Brier score | Log loss |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 49.99% | 49.85% | 0.5002 | 0.252049 | 0.697324 |
| Historical up-rate probability | 50.38% | 50.00% | 0.4704 | 0.250850 | 0.694855 |
| Constant 50% probability | N/A | N/A | 0.5000 | 0.250000 | 0.693147 |
| Always predict up | 50.38% | 50.00% | N/A | N/A | N/A |

Higher ROC-AUC, accuracy and balanced accuracy are generally better.
Lower Brier score and log loss are better. Classification metrics are
not directly comparable with regression forecast-error metrics.

## Interpretation

1. Random Walk achieved the lowest overall close MAE, close RMSE,
   return MAE and return RMSE.
2. None of the tested Ridge variants, Decision Tree or Random Forest
   improved on Random Walk's overall return errors.
3. Ridge returns-only had the highest directional accuracy among the
   tested regression models, at 51.39%, but did not beat Random Walk
   on overall return MAE or RMSE.
4. Logistic Regression showed no useful overall classification
   advantage over simple benchmarks.
5. Selected annual improvements do not establish a consistent
   predictive advantage over time.

## Research decision

Stop adding generic models for now. Preserve these results as a
completed model-comparison checkpoint.

The next stage is a carefully specified trading-strategy backtest
with explicit position rules, transaction costs, slippage and
comparison against appropriate passive benchmarks.

Any initial backtest using this already-examined development period
will be exploratory, not independent proof of profitability.
The previously evaluated 2022-06-15 to 2026-10-23 period must not
be repeatedly reused to tune strategy rules.

## Limitations

Forecast accuracy does not establish trading profitability. A
strategy must account for turnover, trading costs, slippage, exposure,
drawdowns and risk-adjusted performance. Results from historical
development data may not generalize to future markets.
