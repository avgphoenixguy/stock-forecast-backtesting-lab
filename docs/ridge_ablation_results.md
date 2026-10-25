# Ridge Regression — Feature Ablation Results

## Experiment details

- Ticker: RELIANCE.NS
- Research start: 2005-01-01
- Development cutoff: 2022-06-14
- Forecast origins: 2008-01-16 to 2022-06-13
- Last forecast target: 2022-06-14
- Forecasts per model: 3,549
- Initial training window: 756 observations
- Ridge alpha: 1.0
- Validation: expanding-window, one-day-ahead forecasting

All variants were evaluated on the same forecast dates. Feature and target
scaling were fitted using training data only.

## Overall results

| Model | Close MAE | Close RMSE | Return MAE (bps) | Return RMSE (bps) | Directional accuracy | Mean predicted return (bps) |
|---|---:|---:|---:|---:|---:|---:|
| Random Walk | 5.321983 | 9.040784 | 148.162015 | 211.029053 | N/A | 0.000000 |
| Random Walk with Drift | 5.326888 | 9.040945 | 148.468940 | 211.343515 | 50.38% | 9.877346 |
| Ridge — returns only | 5.343676 | 9.071102 | 148.838349 | 211.750245 | 51.39% | 9.294690 |
| Ridge — trend only | 5.330085 | 9.047627 | 148.618104 | 211.830023 | 50.41% | 9.553353 |
| Ridge — volatility only | 5.332886 | 9.043215 | 148.775039 | 211.798540 | 50.44% | 8.897166 |
| Ridge — all features | 5.365147 | 9.094519 | 149.356697 | 212.048669 | 50.72% | 8.394387 |

Lower MAE and RMSE indicate smaller forecast errors. Return errors are
measured in basis points of log return, not investment returns.

## Interpretation

1. Random Walk produced the lowest overall return MAE and RMSE.
2. Among the Ridge variants, returns-only had the lowest return RMSE.
3. Trend-only had the lowest Ridge return MAE.
4. The all-feature Ridge model had the highest overall errors among the
   Ridge variants.
5. Performance varied across calendar years, so individual strong years
   should not be used alone to select a model.

The results do not demonstrate that any tested Ridge feature group provides
a reliable predictive edge over Random Walk.

## Research decision

Retain these experiments as completed baselines. Stop tuning the current
Ridge feature groups solely to improve historical development-period scores.

The next experiment is a nonlinear tree-based regression model compared
against Random Walk and Random Walk with Drift, using the same development
cutoff and chronological evaluation discipline.

The previously evaluated 2022-06-15 to 2026-10-23 period must not be reused
for model tuning.

## Limitations

These are forecast-error results, not trading returns. No trading strategy
has yet been validated after transaction costs and slippage. Historical
Yahoo Finance data is subject to source and adjustment limitations.
