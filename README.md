# Stock Forecast & Backtesting Lab

A Python-based quantitative research project investigating whether machine learning can improve next-day stock return forecasts and whether forecast-driven trading strategies can outperform simple benchmarks after transaction costs.

The project uses historical data for Reliance Industries Limited (`RELIANCE.NS`), with chronological walk-forward validation, baseline comparisons, feature experiments, and an exploratory trading-strategy backtest.

**Research conclusion:** The tested machine-learning models did not outperform the Random Walk benchmark on overall return forecast errors. The initial forecast-driven trading rule also performed poorly after costs. These negative results are documented rather than presented as a successful trading system.

## Project overview

The research pipeline covers:

- Historical daily stock-price acquisition and data validation.
- Random Walk and Random Walk with Drift benchmarks.
- Logistic Regression for next-day price direction.
- Ridge Regression for next-day log-return forecasts.
- Ridge feature ablation across returns, trend, and volatility.
- Decision Tree and Random Forest regression.
- Common-date model comparison.
- An exploratory long-or-cash intraday strategy backtest.
- Transaction-cost sensitivity and buy-and-hold benchmarking.

## Research results

### 1. Regression model comparison

The models below were evaluated on the same 3,549 forecast target dates, from January 17, 2008, through June 14, 2022.

| Model | Return MAE (bps) | Return RMSE (bps) |
|---|---:|---:|
| Random Walk | 148.1620 | 211.0291 |
| Random Walk with Drift | 148.4689 | 211.3435 |
| Ridge — trend only | 148.6181 | 211.8300 |
| Ridge — volatility only | 148.7750 | 211.7985 |
| Ridge — returns only | 148.8383 | 211.7502 |
| Random Forest | 148.9179 | 212.0404 |
| Ridge — all features | 149.3567 | 212.0487 |
| Decision Tree | 150.2362 | 214.6078 |

*Lower MAE and RMSE indicate smaller forecast errors. Return errors are measured in basis points of log return.*

The Random Walk benchmark achieved the lowest overall return MAE and RMSE. None of the tested regression models established an overall forecasting advantage over this simple baseline.

### 2. Logistic Regression

Logistic Regression was evaluated separately as a next-day direction classifier.

| Metric | Logistic Regression | Constant 50% probability baseline |
|---|---:|---:|
| Accuracy | 49.99% | Not applicable |
| ROC-AUC | 0.5002 | 0.5000 |
| Brier score | 0.2520 | 0.2500 |
| Log loss | 0.6973 | 0.6931 |

The classifier did not demonstrate a useful overall predictive advantage. Its accuracy was also below the 50.38% accuracy of the always-predict-up benchmark.

Direction classification and continuous return forecasting measure different objectives; neither metric alone establishes profitability.

### 3. Exploratory trading-strategy backtest

A fixed long-or-cash intraday strategy was evaluated using the existing model forecasts. The strategy entered long at the session open only when forecast-implied upside exceeded the assumed round-trip transaction costs, then exited at the close. Otherwise, it remained in cash.

Results for January 17, 2008, through June 14, 2022:

| Strategy | Gross cumulative return | Net cumulative return |
|---|---:|---:|
| Buy-and-hold | +276.90% | +276.12% |
| Cash, zero interest | 0.00% | 0.00% |
| Random Walk signal | −45.28% | −90.95% |
| Ridge — all features | −29.44% | −92.59% |
| Random Forest signal | −54.86% | −94.66% |
| Decision Tree signal | −73.06% | −96.54% |

The main scenario assumed transaction costs and slippage of 10 basis points per side. This is an illustrative assumption, not a verified brokerage tariff. Cost sensitivity was also examined at 0, 5, 10, and 20 basis points per side.

These strategies have different exposure profiles, so the comparison is not perfectly risk-matched. The simple cost deductions do not model every actual trading fee, tax, liquidity effect, or market impact.

The tested forecast-driven rule was unsuccessful. Its gross cumulative return was negative even before transaction costs, and the strategy showed severe drawdowns. The results do not support a claim of trading profitability.

## Methodology

### Data

- **Instrument:** Reliance Industries (`RELIANCE.NS`)
- **Source:** Yahoo Finance, accessed through `yfinance`
- **Frequency:** Daily
- **Prices:** Adjusted OHLC data produced with automatic price adjustment enabled
- **Research start:** January 1, 2005
- **Development target-date cutoff:** June 14, 2022

The downloader saves a historical CSV alongside metadata containing data-validation information and a SHA-256 file hash. Forecasting scripts validate the saved dataset before running.

### Feature engineering

The regression models use historical features calculated from information available at the forecast origin:

- Log returns over 1, 5, 20, and 60 trading sessions.
- Rolling volatility over 20 and 60 sessions.
- Closing-price distance from the 20-session and 60-session simple moving averages.

Ridge Regression uses feature and target scaling fitted on the available training window. Tree-based models use fixed configurations for their initial comparisons.

### Validation

The core regression and classification experiments use expanding-window, one-day-ahead forecasts. The initial training window is 756 observations.

Both forecast origins and target dates are restricted to the development period. Training targets must already be observable at each forecast origin. This is intended to prevent future target information from leaking into model fitting.

The previously examined June 15, 2022–October 23, 2026 period is **not treated as an untouched final holdout**. It must not be repeatedly reused to tune these models or trading rules.

## Getting started

### Requirements

- Python
- Git
- A network connection for downloading historical data

The project uses NumPy, pandas, yfinance, scikit-learn, Streamlit, Matplotlib, Plotly, and pytest.

### Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/avgphoenixguy/stock-forecast-backtesting-lab.git
cd stock-forecast-backtesting-lab
```

Create and activate a virtual environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Download and validate historical data

Download maximum available daily history for Reliance Industries:

```bash
python scripts/download_data.py --ticker RELIANCE.NS
```

Downloaded data is saved locally under `data/raw/`. Run the downloader again when you deliberately want to refresh the historical dataset.

### Run the main experiments

Run the chronological Random Walk and Drift baseline:

```bash
python scripts/run_walk_forward.py \
  --ticker RELIANCE.NS \
  --start 2005-01-01 \
  --end 2022-06-14
```

Run Ridge Regression:

```bash
python scripts/run_ridge_baseline.py \
  --ticker RELIANCE.NS \
  --start 2005-01-01 \
  --end 2022-06-14
```

Run the Ridge feature-ablation experiment:

```bash
python scripts/run_ridge_ablation.py \
  --ticker RELIANCE.NS \
  --start 2005-01-01 \
  --end 2022-06-14
```

Run the Decision Tree:

```bash
python scripts/run_tree_baseline.py \
  --ticker RELIANCE.NS \
  --start 2005-01-01 \
  --end 2022-06-14
```

Run Random Forest:

```bash
python scripts/run_random_forest.py \
  --ticker RELIANCE.NS \
  --start 2005-01-01 \
  --end 2022-06-14
```

Run Logistic Regression:

```bash
python scripts/run_logistic_baseline.py \
  --ticker RELIANCE.NS \
  --start 2005-01-01 \
  --end 2022-06-14
```

Compare saved model forecasts on common target dates:

```bash
python scripts/compare_model_results.py \
  --ticker RELIANCE.NS \
  --end 2022-06-14
```

Run the exploratory strategy backtest and evaluate its cost sensitivity:

```bash
python scripts/run_strategy_backtest.py \
  --ticker RELIANCE.NS \
  --end 2022-06-14 \
  --cost-bps 10

python scripts/evaluate_strategy_backtest.py \
  --ticker RELIANCE.NS \
  --end 2022-06-14
```

Generated experiment reports are written under `reports/`. The repository's ignore rules keep the main local data files and generated CSV/JSON reports out of Git.

### Run tests

Activate the virtual environment and run:

```bash
python -m pytest
```

The test suite should pass before making code changes available to other users.

## Repository structure

```text
stock-forecast-backtesting-lab/
├── data/
│   ├── raw/                    # Downloaded local historical data
│   └── processed/              # Optional processed datasets
├── docs/
│   ├── experiment_log.md
│   ├── ridge_ablation_results.md
│   ├── tree_baseline_results.md
│   ├── random_forest_results.md
│   ├── model_comparison_results.md
│   └── strategy_backtest_results.md
├── reports/                    # Generated local experiment outputs
├── scripts/
│   ├── download_data.py
│   ├── diagnose_data.py
│   ├── run_baselines.py
│   ├── run_walk_forward.py
│   ├── run_logistic_baseline.py
│   ├── evaluate_logistic_results.py
│   ├── run_ridge_baseline.py
│   ├── run_ridge_ablation.py
│   ├── run_tree_baseline.py
│   ├── run_random_forest.py
│   ├── compare_model_results.py
│   ├── run_strategy_backtest.py
│   └── evaluate_strategy_backtest.py
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Limitations and future work

This is an educational quantitative-research project, not a production trading system.

- Historical forecasts and returns do not establish future performance.
- The development period has already been used to compare multiple models and strategy rules.
- Transaction costs and slippage require more precise market- and broker-specific modelling.
- The strategy comparison does not fully match the risk and exposure of each benchmark.
- The repository does not establish a positive predictive or economic edge.

Future research should begin with a specific economic hypothesis and predeclared evaluation procedure, rather than adding models or changing parameters until an attractive historical result appears.

## Disclaimer

For educational and research purposes only. This repository does not provide investment advice, trading recommendations, or guarantees of performance. The strategies have not been established as profitable and should not be used for live trading without substantially more validation.
