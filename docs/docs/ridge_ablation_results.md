
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

All models used the same forecast dates. Feature and target scaling were fitted using training data only.

## Overall results

| Model | Return MAE (bps) | Return RMSE (bps) | Directional accuracy |
|---|---:|---:|---:|
| Random Walk | 148.1620 | 211.0291 | Not reported |
| Random Walk with Drift | 148.4689 | 211.3435 | 50.38% |
| Ridge — returns only | 148.8383 | 211.7502 | 51.39% |
| Ridge — trend only | 148.6181 | 211.8300 | 50.41% |
| Ridge — volatility only | 148.7750 | 211.7985 | 50.44% |
| Ridge — all features | 149.3567 | 212.0487 | 50.72% |

Lower return MAE and RMSE indicate smaller forecast errors.

## Interpretation

1. Random Walk produced the lowest overall return MAE and RMSE.
2. Among the Ridge variants, returns-only had the lowest return RMSE.
3. Trend-only had the lowest Ridge return MAE.
4. The all-feature Ridge model had the highest overall errors among the Ridge variants.
5. Performance varied across calendar years, so individual strong years should not be used alone to select a model.

The results do not establish that any Ridge feature group provides a reliable predictive advantage.

## Research decision

Keep these experiments as completed baselines. Do not continue tuning the current Ridge feature groups solely to improve the historical development-period scores.

The next experiment will assess a nonlinear tree-based regression model against Random Walk and Random Walk with Drift, using the same development cutoff and chronological evaluation discipline.

The previously evaluated 2022-06-15 to 2026-10-23 period must not be reused for model tuning.

## Limitations

These are forecast-error results, not trading returns. No trading strategy has yet been validated after transaction costs and slippage. Historical Yahoo Finance data is subject to source and adjustment limitations.
