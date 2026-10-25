# Ridge Regression — Feature Ablation Results

## Experiment details

- Ticker: RELIANCE.NS
- Research start: 2005-01-01
- Development cutoff: 2022-06-14
- Forecast origins: 2008-01-16 to 2022-06-13
- Last forecast target: 2022-06-14
- Forecasts per model: 3,549
- Ridge alpha: 1.0
- Validation: expanding-window, one-day-ahead forecasting

All variants were evaluated on the same forecast dates.

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
2. Among Ridge variants, returns-only had the lowest return RMSE.
3. Trend-only had the lowest Ridge return MAE.
4. The all-feature Ridge model had the highest overall errors among the Ridge variants.
5. Year-to-year variation does not, by itself, establish a reliable predictive advantage.

The results do not demonstrate that any tested Ridge feature group provides a reliable predictive edge over Random Walk.

## Research decision

Retain these experiments as completed baselines. Stop tuning the current Ridge feature groups solely to improve historical development-period scores.

The next experiment will test a nonlinear, tree-based regression model against Random Walk and Random Walk with Drift. It will use the same development cutoff and chronological evaluation discipline.

The previously evaluated period beginning 2022-06-15 must not be reused for model tuning.

## Limitations

These are forecast-error results, not trading returns. No trading strategy has yet been validated after transaction costs and slippage. Historical Yahoo Finance data is subject to source and adjustment limitations.
