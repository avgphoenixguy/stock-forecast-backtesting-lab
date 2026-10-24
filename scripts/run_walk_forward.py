
"""Run expanding-window, one-day-ahead forecasting benchmarks.

Models:
1. Random walk.
2. Random walk with drift estimated from all prior observations.

By default, this script stops at the training end date recorded in the
previous baseline run summary. This keeps the previously evaluated
2022-2026 period out of this development experiment.

No downloads, price edits, or trades are performed.
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
DEFAULT_MIN_TRAINING_OBSERVATIONS = 756


def safe_ticker_name(ticker: str) -> str:
    """Use the project's standard ticker filename convention."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").upper()


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_data(ticker: str) -> tuple[pd.DataFrame, Path, dict]:
    """Load and validate the existing downloaded dataset."""

    name = safe_ticker_name(ticker)
    csv_path = RAW_DATA_DIR / f"{name}_daily.csv"
    metadata_path = RAW_DATA_DIR / f"{name}_metadata.json"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Historical CSV not found: {csv_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Data metadata not found: {metadata_path}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("ticker") != ticker:
        raise ValueError("The metadata ticker does not match the requested ticker.")

    if metadata.get("csv_sha256") != sha256_file(csv_path):
        raise ValueError(
            "CSV SHA-256 does not match metadata. "
            "Verify the input data before modelling."
        )

    if metadata.get("price_adjustment", {}).get("auto_adjust") is not True:
        raise ValueError("This experiment requires adjusted-price history.")

    data = pd.read_csv(csv_path)
    data.columns = [str(col).strip().lower() for col in data.columns]

    required = ["date", "open", "high", "low", "close"]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    ).dt.normalize()

    for column in ["open", "high", "low", "close"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data["date"].isna().any():
        raise ValueError("Invalid dates found in the input CSV.")

    if data["date"].duplicated().any():
        raise ValueError("Duplicate dates found in the input CSV.")

    if not data["date"].is_monotonic_increasing:
        raise ValueError("Input data is not in chronological order.")

    if len(data) != metadata.get("row_count"):
        raise ValueError("CSV row count does not match metadata.")

    o = data["open"].to_numpy(dtype=np.float64)
    h = data["high"].to_numpy(dtype=np.float64)
    low = data["low"].to_numpy(dtype=np.float64)
    c = data["close"].to_numpy(dtype=np.float64)

    prices = np.column_stack([o, h, low, c])

    if not np.isfinite(prices).all():
        raise ValueError("OHLC contains non-finite values.")

    if (prices <= 0).any():
        raise ValueError("OHLC contains non-positive prices.")

    # The same scale-aware floating-point tolerance used by the downloader.
    scale = np.maximum.reduce([
        np.ones(len(data)),
        np.abs(o),
        np.abs(h),
        np.abs(low),
        np.abs(c),
    ])
    tolerance = 8 * np.finfo(np.float64).eps * scale

    material_violation = (
        (h + tolerance < low)
        | (h + tolerance < np.maximum(o, c))
        | (low - tolerance > np.minimum(o, c))
    )

    if material_violation.any():
        raise ValueError(
            f"Found {int(material_violation.sum())} material OHLC violations."
        )

    return data, csv_path, metadata


def get_development_end(
    ticker: str,
    requested_end: str | None,
) -> tuple[pd.Timestamp, str]:
    """Choose an end date without defaulting to the previously tested holdout."""

    if requested_end is not None:
        try:
            end = pd.Timestamp(requested_end).normalize()
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "--end must be a valid date, for example 2022-06-14."
            ) from exc

        if pd.isna(end):
            raise ValueError("--end must be a valid date.")

        return end, "explicit command-line date"

    summary_path = (
        REPORTS_DIR
        / f"baseline_run_{safe_ticker_name(ticker)}.json"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Baseline summary not found: {summary_path}\n"
            "Run scripts/run_baselines.py first, or specify --end explicitly."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if summary.get("ticker") != ticker or not summary.get("training_end"):
        raise ValueError(
            "Baseline summary does not contain a valid training end date "
            "for this ticker."
        )

    end = pd.Timestamp(summary["training_end"]).normalize()

    return end, "training_end from previous baseline summary"


def metric_row(
    frame: pd.DataFrame,
    model_name: str,
    period: str,
    year: int | None,
) -> dict:
    """Calculate metrics for a forecasting model over a given period."""

    actual_close = frame["actual_close"].to_numpy(dtype=float)
    actual_return = frame["actual_log_return"].to_numpy(dtype=float)

    predicted_close = frame[f"{model_name}_predicted_close"].to_numpy(
        dtype=float
    )
    predicted_return = frame[
        f"{model_name}_predicted_log_return"
    ].to_numpy(dtype=float)

    close_errors = actual_close - predicted_close
    return_errors = actual_return - predicted_return

    # Directional accuracy is not informative for a model that always
    # predicts exactly zero return, so leave it undefined for that model.
    if np.all(predicted_return == 0):
        direction_accuracy = None
    else:
        direction_accuracy = float(
            np.mean(np.sign(actual_return) == np.sign(predicted_return))
            * 100
        )

    return {
        "period": period,
        "year": year,
        "model": model_name,
        "forecast_count": int(len(frame)),
        "close_mae": float(np.mean(np.abs(close_errors))),
        "close_rmse": float(np.sqrt(np.mean(close_errors**2))),
        "log_return_mae_bps": float(
            np.mean(np.abs(return_errors)) * 10_000
        ),
        "log_return_rmse_bps": float(
            np.sqrt(np.mean(return_errors**2)) * 10_000
        ),
        "directional_accuracy_pct": direction_accuracy,
    }


def run_walk_forward(
    data: pd.DataFrame,
    start_date: str,
    end_date: pd.Timestamp,
    min_training_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Forecast each date from its past only, using an expanding window."""

    if min_training_observations < 2:
        raise ValueError("Minimum training observations must be at least 2.")

    start = pd.Timestamp(start_date).normalize()

    if end_date < start:
        raise ValueError("End date must be on or after the start date.")

    window = data.loc[
        (data["date"] >= start) & (data["date"] <= end_date)
    ].copy().reset_index(drop=True)

    if len(window) <= min_training_observations:
        raise ValueError(
            f"Only {len(window)} observations fall inside the development "
            f"window. More than {min_training_observations} are required."
        )

    close = window["close"].to_numpy(dtype=np.float64)
    log_close = np.log(close)

    rows = []

    # Each target date is forecast using historical closes only.
    # At index i, close[:i] is available, but close[i] is not.
    for i in range(min_training_observations, len(window)):
        past_closes = close[:i]
        past_log_returns = np.diff(np.log(past_closes))

        if len(past_log_returns) == 0:
            continue

        drift = float(past_log_returns.mean())

        if not np.isfinite(drift):
            raise ValueError(
                f"Non-finite estimated drift before {window.loc[i, 'date']}."
            )

        previous_close = float(close[i - 1])
        actual_close = float(close[i])
        actual_log_return = float(log_close[i] - log_close[i - 1])

        rows.append({
            "date": window.loc[i, "date"],
            "training_observations": i,
            "training_end_date": window.loc[i - 1, "date"],
            "previous_close": previous_close,
            "actual_close": actual_close,
            "actual_log_return": actual_log_return,
            "training_mean_daily_log_return": drift,
            "random_walk_predicted_close": previous_close,
            "random_walk_predicted_log_return": 0.0,
            "drift_predicted_close": previous_close * np.exp(drift),
            "drift_predicted_log_return": drift,
        })

    predictions = pd.DataFrame(rows)

    if predictions.empty:
        raise ValueError("No walk-forward forecasts were produced.")

    if not np.isfinite(
        predictions.select_dtypes(include=[np.number]).to_numpy()
    ).all():
        raise ValueError("Non-finite forecast values were produced.")

    overall_rows = []

    for model_name in ["random_walk", "drift"]:
        overall_rows.append(
            metric_row(
                predictions,
                model_name,
                period="overall",
                year=None,
            )
        )

    overall_metrics = pd.DataFrame(overall_rows)

    annual_rows = []

    predictions["forecast_year"] = predictions["date"].dt.year

    for year, group in predictions.groupby("forecast_year", sort=True):
        for model_name in ["random_walk", "drift"]:
            annual_rows.append(
                metric_row(
                    group,
                    model_name,
                    period="annual",
                    year=int(year),
                )
            )

    annual_metrics = pd.DataFrame(annual_rows)

    summary = {
        "project": "Stock Forecast & Backtesting Lab",
        "experiment": "expanding_window_one_day_ahead_baselines",
        "research_start_date": start.strftime("%Y-%m-%d"),
        "development_end_date": end_date.strftime("%Y-%m-%d"),
        "minimum_training_observations": min_training_observations,
        "observations_in_development_window": int(len(window)),
        "walk_forward_forecasts": int(len(predictions)),
        "first_forecast_date": predictions["date"].min().strftime("%Y-%m-%d"),
        "last_forecast_date": predictions["date"].max().strftime("%Y-%m-%d"),
        "first_forecast_training_end": (
            predictions["training_end_date"].min().strftime("%Y-%m-%d")
        ),
        "last_forecast_training_end": (
            predictions["training_end_date"].max().strftime("%Y-%m-%d")
        ),
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Expanding window. The forecast for each day uses closes and "
            "returns available strictly before that forecast date. Drift "
            "is re-estimated from all historical log returns available "
            "at each forecast origin."
        ),
        "interpretation": (
            "Development-period forecast errors, not trading returns. "
            "This experiment is for model development and should not be "
            "described as an untouched final holdout."
        ),
    }

    return predictions, overall_metrics, annual_metrics, summary


def main() -> None:
    """Run the expanding-window benchmark experiment."""

    parser = argparse.ArgumentParser(
        description="Run expanding-window stock forecasting baselines."
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help="Ticker symbol; default: RELIANCE.NS",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help="Research start date; default: 2005-01-01",
    )
    parser.add_argument(
        "--end",
        default=None,
        help=(
            "Optional development end date. By default, use the training "
            "end recorded in reports/baseline_run_<TICKER>.json."
        ),
    )
    parser.add_argument(
        "--min-training-observations",
        type=int,
        default=DEFAULT_MIN_TRAINING_OBSERVATIONS,
        help="Initial training window; default: 756 observations.",
    )

    args = parser.parse_args()
    ticker = args.ticker.strip().upper()

    if not ticker:
        parser.error("Ticker cannot be empty.")

    try:
        start_date = pd.Timestamp(args.start).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        parser.error("--start must be a valid date.")

    data, csv_path, metadata = load_data(ticker)
    end_date, end_date_source = get_development_end(ticker, args.end)

    predictions, metrics, annual_metrics, summary = run_walk_forward(
        data=data,
        start_date=start_date,
        end_date=end_date,
        min_training_observations=args.min_training_observations,
    )

    summary["ticker"] = ticker
    summary["end_date_source"] = end_date_source
    summary["source_csv"] = str(csv_path.relative_to(PROJECT_ROOT))
    summary["source_csv_sha256"] = sha256_file(csv_path)
    summary["source_metadata_row_count"] = metadata["row_count"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = safe_ticker_name(ticker)

    predictions_path = REPORTS_DIR / f"walk_forward_predictions_{name}.csv"
    metrics_path = REPORTS_DIR / f"walk_forward_metrics_{name}.csv"
    annual_path = REPORTS_DIR / f"walk_forward_annual_metrics_{name}.csv"
    summary_path = REPORTS_DIR / f"walk_forward_run_{name}.json"

    predictions.to_csv(
        predictions_path,
        index=False,
        date_format="%Y-%m-%d",
    )
    metrics.to_csv(metrics_path, index=False)
    annual_metrics.to_csv(annual_path, index=False)

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n===== WALK-FORWARD EXPERIMENT COMPLETE =====")
    print(f"Ticker: {ticker}")
    print(f"Research start: {start_date}")
    print(f"Development end: {end_date.strftime('%Y-%m-%d')}")
    print(f"End-date source: {end_date_source}")
    print(f"Minimum training observations: {args.min_training_observations:,}")
    print(f"Forecasts produced: {len(predictions):,}")
    print(
        f"Forecast period: {summary['first_forecast_date']} "
        f"to {summary['last_forecast_date']}"
    )

    print("\n===== OVERALL WALK-FORWARD METRICS =====")
    print(metrics.to_string(index=False))

    print("\n===== ANNUAL WALK-FORWARD METRICS =====")
    print(annual_metrics.to_string(index=False))

    print("\n===== OUTPUT FILES =====")
    print(f"Predictions: {predictions_path}")
    print(f"Overall metrics: {metrics_path}")
    print(f"Annual metrics: {annual_path}")
    print(f"Run summary: {summary_path}")

    print(
        "\nNo new downloads were performed. Forecasts are not trading "
        "returns or evidence of profitability."
    )


if __name__ == "__main__":
    main()
