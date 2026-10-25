# Random Forest Regression — Experiment Results

## Experiment setup

- Ticker: RELIANCE.NS
- Research start: 2005-01-01
- Development cutoff: 2022-06-14
- Forecast origins: 2008-01-16 to 2022-06-13
- Last forecast target: 2022-06-14
- Forecasts: 3,549
- Initial training window: 756 observations
- Trees: 50
- Maximum tree depth: 5
- Minimum samples per leaf: 50
- Maximum features per split: 0.7
- Random state: 42
- Validation: Expanding-window, one-day-ahead forecasting

## Overall results

| Model | Close MAE | Close RMSE | Return MAE (bps) | Return RMSE (bps) | Directional accuracy |
|---|---:|---:|---:|---:|---:|
| Random Walk | 5.321983 | 9.040784 | 148.162015 | 211.029053 | N/A |
| Random Walk with Drift | 5.326888 | 9.040945 | 148.468940 | 211.343515 | 50.38% |
| Random Forest | 5.336922 | 9.050508 | 148.917898 | 212.040353 | 50.46% |

Lower MAE and RMSE indicate smaller forecast errors. Return errors
are measured in basis points of log return, not investment returns.

## Findings

1. Random Walk had the lowest overall close MAE and close RMSE.
2. Random Walk had the lowest overall return MAE and return RMSE.
3. Random Forest return MAE was approximately 0.51% higher than
   Random Walk, and return RMSE was approximately 0.48% higher.
4. Random Forest directional accuracy was 50.46%, only about
   0.08 percentage points above Random Walk with Drift.
5. Some years favoured Random Forest on selected metrics, but the
   improvements were not consistent across the development period.

## Research decision

The fixed-configuration Random Forest did not demonstrate an overall
predictive advantage over Random Walk or Random Walk with Drift.

Retain this result as a completed experiment. Do not tune the model
against the previously evaluated 2022-06-15 to 2026-10-23 period.

The next task is to consolidate results from all completed models and
assess forecast-signal stability before deciding whether to proceed
with a trading-strategy backtest. Do not assume that better model
complexity or occasional annual wins imply profitability.

## Limitations

Forecast-error metrics are not trading returns. Transaction costs,
slippage, portfolio construction, and risk-adjusted performance have
not been evaluated. These results do not establish profitability.
