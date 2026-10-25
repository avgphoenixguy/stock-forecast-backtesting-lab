# Decision Tree Regression — Experiment Results

## Experiment setup

- Ticker: RELIANCE.NS
- Research start: 2005-01-01
- Development cutoff: 2022-06-14
- Forecast origins: 2008-01-16 to 2022-06-13
- Last forecast target: 2022-06-14
- Number of forecasts: 3,549
- Initial training window: 756 observations
- Model: DecisionTreeRegressor
- Maximum depth: 3
- Minimum samples per leaf: 50
- Random state: 42
- Validation: expanding-window, one-day-ahead forecasting

## Overall results

| Model | Close MAE | Close RMSE | Return MAE (bps) | Return RMSE (bps) | Directional accuracy |
|---|---:|---:|---:|---:|---:|
| Random Walk | 5.321983 | 9.040784 | 148.162015 | 211.029053 | Not reported |
| Random Walk with Drift | 5.326888 | 9.040945 | 148.468940 | 211.343515 | 50.38% |
| Decision Tree | 5.372325 | 9.099548 | 150.236245 | 214.607755 | 49.90% |

Lower forecast errors are better. Return metrics are measured in basis
points of log return, not investment returns.

## Findings

1. Random Walk had the lowest overall close MAE and close RMSE.
2. Random Walk had the lowest overall return MAE and return RMSE.
3. Decision Tree return MAE was approximately 1.4% higher than
   Random Walk, and its return RMSE was approximately 1.7% higher.
4. Decision Tree directional accuracy was 49.90%, below Drift's 50.38%.
5. Some individual years favoured the tree on selected metrics, but
   these improvements were not consistent across the development period.

## Research decision

The fixed shallow Decision Tree did not demonstrate an overall
predictive advantage over the Random Walk benchmark.

Retain this experiment as a negative result. Do not tune this tree
against the previously evaluated 2022-06-15 to 2026-10-23 period.

The next experiment will test a fixed-configuration Random Forest
regressor using the same development cutoff, training window,
features, forecast dates, and benchmarks.

## Limitations

Forecast-error metrics are not trading returns. Transaction costs,
slippage, portfolio construction, and risk-adjusted performance have
not been evaluated. These results do not establish profitability.
