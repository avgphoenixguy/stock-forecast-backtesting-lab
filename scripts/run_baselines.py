
"""Run chronological forecasting baselines for Indian equities.

Models:
1. Random walk: tomorrow's close is predicted to equal today's close.
2. Random walk with drift: estimated average training log return is added.

Reads existing data only. Does not download data or execute trades.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_TICKER = "RELIANCE.NS"
DEFAULT_START_DATE = "2005-01-01"
DEFAULT_TEST_FRACTION = 0.20


def safe_ticker_name(ticker: str) -> str:
    """Create the filename convention used by the downloader."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").upper()


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 hash."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_data(ticker: str) -> tuple[pd.DataFrame, Path, dict]:
    """Load and validate saved historical data and its metadata."""

    name = safe_ticker_name(ticker)
    csv_path = RAW_DATA_DIR / f"{name}_daily.csv"
    metadata_path = RAW_DATA_DIR / f"{name}_metadata.json"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Historical CSV not found: {csv_path}\n"
            "Run scripts/download_data.py first."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("ticker") != ticker:
        raise ValueError("Metadata ticker does not match the requested ticker.")

    if metadata.get("csv_sha256") != sha256_file(csv_path):
        raise ValueError(
            "CSV hash does not match metadata. Verify the input data first."
        )

    if metadata.get("price_adjustment", {}).get("auto_adjust") is not True:
        raise ValueError(
            "This experiment requires the adjusted-price dataset "
            "produced by the project downloader."
        )

    data = pd.read_csv(csv_path)
    data.columns = [str(column).strip().lower() for column in data.columns]

    required = ["date", "open", "high", "low", "close"]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    ).dt.normalize()

    for column in ["open", "high", "low", "close"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data["date"].isna().any():
        raise ValueError("Invalid dates found in CSV.")

    if data["date"].duplicated().any():
        raise ValueError("Duplicate dates found in CSV.")

    if not data["date"].is_monotonic_increasing:
        raise ValueError("CSV dates are not sorted chronologically.")

    if len(data) != metadata.get("row_count"):
        raise ValueError("CSV row count does not match metadata.")

    prices = data[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)

    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("OHLC contains non-finite or non-positive prices.")

    # Recheck OHLC consistency using a small floating-point tolerance.
    open_price = data["open"].to_numpy(dtype=np.float64)
    high_price = data["high"].to_numpy(dtype=np.float64)
    low_price = data["low"].to_numpy(dtype=np.float64)
    close_price = data["close"].to_numpy(dtype=np.float64)

    scale = np.maximum.reduce([
        np.ones(len(data)),
        np.abs(open_price),
        np.abs(high_price),
        np.abs(low_price),
        np.abs(close_price),
    ])
    tolerance = 8 * np.finfo(np.float64).eps * scale

    material_violation = (
        (high_price + tolerance < low_price)
        | (high_price + tolerance < np.maximum(open_price, close_price))
        | (low_price - tolerance > np.minimum(open_price, close_price))
    )

    if material_violation.any():
        raise ValueError(
            f"Found {int(material_violation.sum())} material OHLC violations."
        )

    return data, csv_path, metadata


def calculate_metrics(
    actual_close: np.ndarray,
    actual_log_return: np.ndarray,
    predicted_close: np.ndarray,
    predicted_log_return: np.ndarray,
    model_name: str,
) -> dict:
    """Calculate forecast errors on the held-out test period."""

    close_errors = actual_close - predicted_close
    return_errors = actual_log_return - predicted_log_return

    if np.all(predicted_log_return == 0):
        directional_accuracy = None
    else:
        directional_accuracy = float(
            np.mean(
                np.sign(actual_log_return)
                == np.sign(predicted_log_return)
            ) * 100
        )

    return {
        "model": model_name,
        "test_observations": int(len(actual_close)),
        "close_mae": float(np.mean(np.abs(close_errors))),
        "close_rmse": float(np.sqrt(np.mean(close_errors**2))),
        "log_return_mae_bps": float(
            np.mean(np.abs(return_errors)) * 10_000
        ),
        "log_return_rmse_bps": float(
            np.sqrt(np.mean(return_errors**2)) * 10_000
        ),
        "directional_accuracy_pct": directional_accuracy,
    }


def run_experiment(
    data: pd.DataFrame,
    start_date: str,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit baselines using training history and evaluate chronologically."""

    start = pd.Timestamp(start_date)

    window = (
        data.loc[data["date"] >= start]
        .copy()
        .reset_index(drop=True)
    )

    if len(window) < 100:
        raise ValueError(
            f"Only {len(window)} observations are available from "
            f"{start_date}; at least 100 are required for this experiment."
        )

    # Features used for one-day-ahead forecasts.
    window["previous_close"] = window["close"].shift(1)
    window["actual_log_return"] = np.log(window["close"]).diff()

    split_index = int(np.floor(len(window) * (1 - test_fraction)))

    if split_index < 2 or split_index >= len(window):
        raise ValueError("Invalid chronological train/test split.")

    train = window.iloc[:split_index].copy()
    test = window.iloc[split_index:].copy()

    # Estimate drift on training data only. The test period is not used
    # to fit either baseline.
    training_returns = np.log(train["close"]).diff().dropna()
    mean_daily_log_return = float(training_returns.mean())

    if not np.isfinite(mean_daily_log_return):
        raise ValueError("Training mean log return is not finite.")

    previous_close = test["previous_close"].to_numpy(dtype=float)
    actual_close = test["close"].to_numpy(dtype=float)
    actual_log_return = test["actual_log_return"].to_numpy(dtype=float)

    if not (
        np.isfinite(previous_close).all()
        and np.isfinite(actual_close).all()
        and np.isfinite(actual_log_return).all()
    ):
        raise ValueError("Non-finite values found in the test period.")

    # Baseline 1: random walk.
    rw_predicted_close = previous_close.copy()
    rw_predicted_log_return = np.zeros(len(test))

    # Baseline 2: random walk with drift.
    # Each forecast uses the previous observed close and training-only drift.
    drift_predicted_log_return = np.full(
        len(test), mean_daily_log_return
    )
    drift_predicted_close = (
        previous_close * np.exp(mean_daily_log_return)
    )

    predictions = pd.DataFrame({
        "date": test["date"].to_numpy(),
        "previous_close": previous_close,
        "actual_close": actual_close,
        "actual_log_return": actual_log_return,
        "random_walk_predicted_close": rw_predicted_close,
        "random_walk_predicted_log_return": rw_predicted_log_return,
        "drift_predicted_close": drift_predicted_close,
        "drift_predicted_log_return": drift_predicted_log_return,
    })

    metrics = pd.DataFrame([
        calculate_metrics(
            actual_close,
            actual_log_return,
            rw_predicted_close,
            rw_predicted_log_return,
            "random_walk",
        ),
        calculate_metrics(
            actual_close,
            actual_log_return,
            drift_predicted_close,
            drift_predicted_log_return,
            "random_walk_with_drift",
        ),
    ])

    summary = {
        "project": "Stock Forecast & Backtesting Lab",
        "ticker": str(data.attrs.get("ticker", "")),
        "experiment": "one_day_ahead_baseline_forecasting",
        "price_field": "adjusted close",
        "research_start_date": start_date,
        "test_fraction": test_fraction,
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "training_start": train["date"].min().strftime("%Y-%m-%d"),
        "training_end": train["date"].max().strftime("%Y-%m-%d"),
        "test_start": test["date"].min().strftime("%Y-%m-%d"),
        "test_end": test["date"].max().strftime("%Y-%m-%d"),
        "training_mean_daily_log_return": mean_daily_log_return,
        "training_mean_daily_log_return_bps": (
            mean_daily_log_return * 10_000
        ),
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "One-day-ahead forecast errors on a chronological holdout. "
            "Forecast metrics are not trading returns or proof of "
            "profitability. Drift is estimated using training data only."
        ),
    }

    return predictions, metrics, summary


def main() -> None:
    """Run the baseline forecast experiment."""

    parser = argparse.ArgumentParser(
        description="Compare basic stock-price forecasting baselines."
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help="Ticker symbol; default: RELIANCE.NS",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help="Start date for analysis; default: 2005-01-01",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Final chronological holdout fraction; default: 0.20",
    )

    args = parser.parse_args()
    ticker = args.ticker.strip().upper()

    if not ticker:
        parser.error("Ticker cannot be empty.")

    if not 0.05 <= args.test_fraction <= 0.40:
        parser.error("--test-fraction must be between 0.05 and 0.40.")

    try:
        start_date = pd.Timestamp(args.start).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        parser.error("--start must be a valid date such as 2005-01-01")

    data, csv_path, metadata = load_data(ticker)
    data.attrs["ticker"] = ticker

    predictions, metrics, summary = run_experiment(
        data=data,
        start_date=start_date,
        test_fraction=args.test_fraction,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = safe_ticker_name(ticker)

    predictions_path = REPORTS_DIR / f"baseline_predictions_{name}.csv"
    metrics_path = REPORTS_DIR / f"baseline_metrics_{name}.csv"
    summary_path = REPORTS_DIR / f"baseline_run_{name}.json"

    summary["ticker"] = ticker
    summary["source_csv"] = str(csv_path.relative_to(PROJECT_ROOT))
    summary["source_csv_sha256"] = sha256_file(csv_path)
    summary["source_metadata_row_count"] = metadata["row_count"]

    predictions.to_csv(
        predictions_path,
        index=False,
        date_format="%Y-%m-%d",
    )
    metrics.to_csv(metrics_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n===== BASELINE EXPERIMENT COMPLETE =====")
    print(f"Ticker: {ticker}")
    print(f"Research start: {start_date}")
    print(f"Training rows: {summary['training_rows']:,}")
    print(f"Test rows: {summary['test_rows']:,}")
    print(
        f"Training period: {summary['training_start']} "
        f"to {summary['training_end']}"
    )
    print(
        f"Test period: {summary['test_start']} "
        f"to {summary['test_end']}"
    )
    print(
        "Mean daily training log return: "
        f"{summary['training_mean_daily_log_return_bps']:.4f} bps"
    )

    print("\n===== OUT-OF-SAMPLE METRICS =====")
    print(metrics.to_string(index=False))

    print("\n===== OUTPUT FILES =====")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")

    print(
        "\nThese are forecasting benchmarks, not trading strategies. "
        "No profitability claim is made."
    )


if __name__ == "__main__":
    main()
