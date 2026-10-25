
# Stock Forecast & Backtesting Lab — Research Log

## Project status

**Stage:** Historical data validation and initial forecasting experiments  
**Instrument:** Reliance Industries (`RELIANCE.NS`)  
**Data source:** Yahoo Finance through `yfinance`  
**Research status:** Exploratory; no profitable strategy or persistent predictive edge has been established.

## 1. Historical data pipeline

The downloader requests maximum available daily history with automatic price adjustment and repair enabled.

The saved Reliance dataset contains:

- Observations: 7,730
- First date: 1996-01-01
- Last date: 2026-10-23
- Fields: date, open, high, low, close, volume, dividends, stock splits

The CSV was verified against its metadata hash, and the OHLC validation was updated to tolerate floating-point-scale differences without altering the original prices.

Data-quality limitations remain: Yahoo Finance is an unofficial third-party data source, and historical coverage, adjustments, and values may change.

## 2. Automated tests

The existing regression suite reported 12 passing tests, covering historical-data validation and expanding-window forecasting behavior.

The suite should be rerun whenever the modelling or data pipeline changes.

## 3. Random Walk and Random Walk with Drift

### Original chronological holdout

Research window: 2005 onward.

- Training period: 2005-01-03 to 2022-06-14
- Test period: 2022-06-15 to 2026-10-23
- Training observations: 4,306
- Test observations: 1,077

| Metric | Random Walk | Random Walk with Drift |
|---|---:|---:|
| Close MAE | 12.5605 | 12.5788 |
| Close RMSE | 17.1853 | 17.2271 |
| Log-return MAE (bps) | 97.1647 | 97.2574 |
| Log-return RMSE (bps) | 131.7682 | 132.0336 |

The Random Walk had slightly lower errors across these metrics. The differences are small and do not establish statistical superiority.

**Research governance:** This later test period has already been examined. It must not be treated as an untouched final holdout for future model selection. Avoid tuning subsequent models against its outcomes.

## 4. Expanding-window Random Walk and Drift

Development cutoff: 2022-06-14.

- Forecasts: 3,550
- Forecast period: 2008-01-16 to 2022-06-14
- Initial training window: 756 observations

| Metric | Random Walk | Drift |
|---|---:|---:|
| Close MAE | 5.3225 | 5.3277 |
| Close RMSE | 9.0403 | 9.0407 |
| Log-return MAE (bps) | 148.1850 | 148.5003 |
| Log-return RMSE (bps) | 211.0346 | 211.3588 |

The Random Walk had marginally lower overall error. Drift performance varied across years.

## 5. Logistic Regression — next-day direction

Target: whether the next adjusted close is higher than the current adjusted close.

Development cutoff: 2022-06-14.

- Forecasts: 3,549
- Target period: 2008-01-17 to 2022-06-14

| Metric | Logistic Regression | Constant 50% probability |
|---|---:|---:|
| ROC-AUC | 0.5002 | 0.5000 |
| Brier score | 0.2520 | 0.2500 |
| Log loss | 0.6973 | 0.6931 |

Logistic Regression accuracy was 49.99%, with balanced accuracy of 49.85%. Its probability scores were worse than the constant 50% benchmark.

Decision: retain the result as a completed experiment; do not continue tuning this model simply to improve the historical score.

## 6. Ridge Regression — next-day log return

Target: next trading day's adjusted-price log return.

Development cutoff: 2022-06-14.

- Forecasts: 3,549
- Forecast origins: 2008-01-16 to 2022-06-13
- Last target date: 2022-06-14
- Initial training window: 756 observations
- Ridge regularization parameter: alpha = 1.0

| Metric | Random Walk | Drift | Ridge |
|---|---:|---:|---:|
| Close MAE | 5.3220 | 5.3269 | 5.3651 |
| Close RMSE | 9.0408 | 9.0409 | 9.0945 |
| Log-return MAE (bps) | 148.1620 | 148.4689 | 149.3567 |
| Log-return RMSE (bps) | 211.0291 | 211.3435 | 212.0487 |

Ridge did not outperform Random Walk or Drift on the overall error metrics. Some annual results were better, but the aggregate result does not establish a reliable predictive edge.

## 7. Current conclusions

1. The historical-data pipeline works for the initial Reliance experiment.
2. Random Walk is a strong baseline for this one-day-ahead forecasting task.
3. Drift has not consistently improved forecast accuracy.
4. Logistic Regression has not demonstrated useful directional discrimination.
5. Ridge Regression has not improved the overall return-forecast errors.

These findings apply to the experiments and periods documented here. They are not proof that every model or feature set will fail.

## 8. Next experiment

Run a controlled Ridge Regression feature-ablation study using the same expanding-window protocol, development cutoff, and forecast dates.

Compare:

- Recent returns only
- Trend features only
- Volatility features only
- All current features

Feature selection and model decisions must use development-period results only. The previously evaluated 2022-06-15 to 2026-10-23 period must not be reused for tuning.

After the feature-ablation study, evaluate any promising configuration under a clearly specified protocol before building trading rules or a dashboard.

## Disclaimer

This is an exploratory quantitative research project, not investment advice. Forecast accuracy is not the same as trading profitability. No strategy has yet been validated after transaction costs and slippage.
