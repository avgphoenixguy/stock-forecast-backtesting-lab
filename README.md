# Stock Forecast & Backtesting Lab

A Python-based quantitative research project investigating whether
machine learning can improve short-horizon equity forecasts and whether
forecast-driven trading strategies can outperform simple benchmarks
after transaction costs.

## Research scope

- Initial stock: Reliance Industries (`RELIANCE.NS`)
- Market: Indian equities
- Forecast horizon: Five trading sessions
- Historical data: Maximum available history, subject to data coverage
- Initial models: Random Walk, Random Walk with Drift, directional
  baselines and logistic regression
- Planned extensions: GRU, LSTM and a reusable multi-stock pipeline
- Interface: Streamlit

## Research experiments

1. Full-history training.
2. Training with predefined event windows excluded.
3. Regime-aware training using past-only market features.

## Evaluation

The project aims to use chronological walk-forward validation,
an untouched final holdout, leakage checks and transaction-cost-aware
backtesting.

Price-level forecasting and directional classification will be
evaluated separately using appropriate metrics.

## Project status

Early development.

The environment and initial dependencies have been installed.
Historical-data validation is still being debugged. No forecasting
models have been trained and no trading performance has been established.

## Technology

- Python
- pandas and NumPy
- yfinance
- scikit-learn
- Streamlit
- Plotly and Matplotlib
- pytest

## Getting started

Setup instructions and the reproducible research workflow will be
expanded as the project develops.

Downloaded market data and generated model outputs are not intended
to be committed to this repository.

## Disclaimer

This project is for educational and research purposes only.
Forecasts are uncertain, historical performance does not guarantee
future results, and the project does not provide financial advice.
