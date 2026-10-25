
"""Walk-forward Ridge Regression for next-day log-return forecasting.

Compares:
1. Random Walk: predicts zero log return.
2. Random Walk with Drift: predicts the historical mean log return.
3. Ridge Regression: predicts next-day log return from past price features.

Uses saved historical data only. No downloads or trades are performed.
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
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_TICKER = "RELIANCE.NS"
DEFAULT_START_DATE = "2005-01-01"
DEFAULT_MIN_TRAINING_OBSERVATIONS = 756
RIDGE_ALPHA = 1.0

FEATURE_COLUMNS = [
    "return_1",
    "return_5",
    "return_20",
    "return_60",
    "volatility_20",
    "volatility_60",
    "close_vs_sma_20",
    "close_vs_sma_60",
]


def safe_ticker_name(ticker: str) -> str:
    """Return the filename convention used by the project."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").upper()


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 hash."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_data(ticker: str) -> tuple[pd.DataFrame, Path, dict]:
    """Load saved adjusted-price history and validate its integrity."""

    name = safe_ticker_name(ticker)
    csv_path = RAW_DATA_DIR / f"{name}_daily.csv"
    metadata_path = RAW_DATA_DIR / f"{name}_metadata.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Historical CSV not found: {csv_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("ticker") != ticker:
        raise ValueError("Metadata ticker does not match the requested ticker.")

    if metadata.get("csv_sha256") != sha256_file(csv_path):
        raise ValueError("CSV SHA-256 does not match metadata.")

    if metadata.get("price_adjustment", {}).get("auto_adjust") is not True:
        raise ValueError("This experiment requires adjusted-price history.")

    data = pd.read_csv(csv_path)
    data.columns = [str(col).strip().lower() for col in data.columns]

    required = ["date", "open", "high", "low", "close"]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()

    for column in ["open", "high", "low", "close"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data["date"].isna().any():
        raise ValueError("Invalid dates found in the CSV.")

    if data["date"].duplicated().any():
        raise ValueError("Duplicate dates found in the CSV.")

    if not data["date"].is_monotonic_increasing:
        raise ValueError("CSV dates are not chronologically sorted.")

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
        raise ValueError("OHLC contains non-positive values.")

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


def resolve_end_date(
    ticker: str,
    requested_end: str | None,
) -> tuple[pd.Timestamp, str]:
    """Use the previous baseline's training end unless explicitly supplied."""

    if requested_end is not None:
        try:
            end_date = pd.Timestamp(requested_end).normalize()
        except (ValueError, TypeError) as exc:
            raise ValueError("--end must be a valid date.") from exc

        if pd.isna(end_date):
            raise ValueError("--end must be a valid date.")

        return end_date, "explicit command-line date"

    summary_path = (
        REPORTS_DIR / f"baseline_run_{safe_ticker_name(ticker)}.json"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Baseline summary not found: {summary_path}. "
            "Run scripts/run_baselines.py first or supply --end."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if summary.get("ticker") != ticker or not summary.get("training_end"):
        raise ValueError("Baseline summary does not contain a valid training end.")

    return (
        pd.Timestamp(summary["training_end"]).normalize(),
        "previous baseline training end",
    )


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create predictors at date t and the return from t to the next session."""

    result = data.copy()

    close = result["close"].astype(float)
    log_close = np.log(close)
    daily_log_return = log_close.diff()

    # Predictors use prices available by the current forecast date.
    result["return_1"] = daily_log_return
    result["return_5"] = log_close.diff(5)
    result["return_20"] = log_close.diff(20)
    result["return_60"] = log_close.diff(60)

    result["volatility_20"] = daily_log_return.rolling(20).std()
    result["volatility_60"] = daily_log_return.rolling(60).std()

    result["close_vs_sma_20"] = close / close.rolling(20).mean() - 1.0
    result["close_vs_sma_60"] = close / close.rolling(60).mean() - 1.0

    # Label is the next trading session's log return.
    result["target_date"] = result["date"].shift(-1)
    result["actual_next_close"] = close.shift(-1)
    result["target_log_return"] = log_close.shift(-1) - log_close

    result = result.dropna(
        subset=FEATURE_COLUMNS + [
            "target_date",
            "actual_next_close",
            "target_log_return",
        ]
    ).reset_index(drop=True)

    return result


def create_ridge_model() -> TransformedTargetRegressor:
    """Scale features and target using training data only."""

    return TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            Ridge(alpha=RIDGE_ALPHA),
        ),
        transformer=StandardScaler(),
    )


def metric_row(
    frame: pd.DataFrame,
    model_name: str,
    period: str,
    year: int | None,
) -> dict:
    """Calculate price and return forecast errors."""

    actual_return = frame["actual_log_return"].to_numpy(dtype=float)
    actual_close = frame["actual_next_close"].to_numpy(dtype=float)
    origin_close = frame["origin_close"].to_numpy(dtype=float)

    predicted_return = frame[f"{model_name}_predicted_return"].to_numpy(
        dtype=float
    )
    predicted_close = origin_close * np.exp(predicted_return)

    return_errors = actual_return - predicted_return
    close_errors = actual_close - predicted_close

    # Do not report directional accuracy for a model that always predicts
    # exactly zero return: that is a price-level benchmark, not a direction.
    if np.all(predicted_return == 0):
        direction_accuracy = None
    else:
        direction_accuracy = float(
            np.mean(
                (actual_return > 0) == (predicted_return > 0)
            ) * 100
        )

    return {
        "period": period,
        "year": year,
        "model": model_name,
        "forecast_count": int(len(frame)),
        "close_mae": float(np.mean(np.abs(close_errors))),
        "close_rmse": float(np.sqrt(np.mean(close_errors ** 2))),
        "log_return_mae_bps": float(
            np.mean(np.abs(return_errors)) * 10_000
        ),
        "log_return_rmse_bps": float(
            np.sqrt(np.mean(return_errors ** 2)) * 10_000
        ),
        "directional_accuracy_pct": direction_accuracy,
        "mean_predicted_log_return_bps": float(
            np.mean(predicted_return) * 10_000
        ),
    }


def run_walk_forward(
    data: pd.DataFrame,
    start_date: str,
    end_date: pd.Timestamp,
    min_training_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Generate one-day-ahead forecasts with an expanding training window."""

    if min_training_observations < 100:
        raise ValueError("Minimum training observations must be at least 100.")

    start = pd.Timestamp(start_date).normalize()

    if end_date < start:
        raise ValueError("End date must not precede the research start.")

    prepared = engineer_features(data)

    # Restrict both forecast origins and their target dates to the
    # development window. This prevents targets entering the later holdout.
    prepared = prepared.loc[
        (prepared["date"] >= start)
        & (prepared["date"] < end_date)
        & (prepared["target_date"] <= end_date)
    ].copy().reset_index(drop=True)

    if len(prepared) <= min_training_observations:
        raise ValueError(
            f"Only {len(prepared)} usable rows are available; more than "
            f"{min_training_observations} are required."
        )

    forecasts = []

    for i in range(min_training_observations, len(prepared)):
        training = prepared.iloc[:i]
        current = prepared.iloc[[i]]

        origin_date = current["date"].iloc[0]
        target_date = current["target_date"].iloc[0]

        # A training target must already be observable at this forecast origin.
        if training["target_date"].max() > origin_date:
            raise RuntimeError(
                "Temporal leakage detected: a training target extends "
                "past the forecast origin."
            )

        if target_date > end_date:
            raise RuntimeError("A forecast target crossed the development cutoff.")

        x_train = training[FEATURE_COLUMNS]
        y_train = training["target_log_return"].to_numpy(dtype=float)

        if not np.isfinite(y_train).all():
            raise ValueError("Non-finite training targets were found.")

        model = create_ridge_model()
        model.fit(x_train, y_train)

        ridge_prediction = float(
            np.asarray(model.predict(current[FEATURE_COLUMNS])).reshape(-1)[0]
        )

        # The drift benchmark uses only already observed training returns.
        drift_prediction = float(np.mean(y_train))

        current_close = float(current["close"].iloc[0])
        actual_next_close = float(current["actual_next_close"].iloc[0])
        actual_return = float(current["target_log_return"].iloc[0])

        forecasts.append({
            "forecast_origin_date": origin_date,
            "target_date": target_date,
            "training_observations": int(len(training)),
            "training_end_date": training["target_date"].max(),
            "origin_close": current_close,
            "actual_next_close": actual_next_close,
            "actual_log_return": actual_return,
            "random_walk_predicted_return": 0.0,
            "drift_predicted_return": drift_prediction,
            "ridge_predicted_return": ridge_prediction,
        })

    predictions = pd.DataFrame(forecasts)

    if predictions.empty:
        raise ValueError("No forecasts were generated.")

    numeric_values = predictions.select_dtypes(
        include=[np.number]
    ).to_numpy()

    if not np.isfinite(numeric_values).all():
        raise ValueError("Non-finite values were generated in the forecasts.")

    predictions["forecast_year"] = predictions["target_date"].dt.year

    model_names = [
        "random_walk",
        "drift",
        "ridge",
    ]

    overall_rows = [
        metric_row(
            predictions,
            model_name,
            period="overall",
            year=None,
        )
        for model_name in model_names
    ]

    annual_rows = []

    for year, group in predictions.groupby("forecast_year", sort=True):
        for model_name in model_names:
            annual_rows.append(
                metric_row(
                    group,
                    model_name,
                    period="annual",
                    year=int(year),
                )
            )

    overall_metrics = pd.DataFrame(overall_rows)
    annual_metrics = pd.DataFrame(annual_rows)

    summary = {
        "experiment": "walk_forward_ridge_next_day_log_return",
        "research_start_date": start.strftime("%Y-%m-%d"),
        "development_end_date": end_date.strftime("%Y-%m-%d"),
        "minimum_training_observations": min_training_observations,
        "ridge_alpha": RIDGE_ALPHA,
        "feature_columns": FEATURE_COLUMNS,
        "forecast_count": int(len(predictions)),
        "first_forecast_origin": (
            predictions["forecast_origin_date"].min().strftime("%Y-%m-%d")
        ),
        "last_forecast_origin": (
            predictions["forecast_origin_date"].max().strftime("%Y-%m-%d")
        ),
        "last_forecast_target": (
            predictions["target_date"].max().strftime("%Y-%m-%d")
        ),
        "method": (
            "Expanding-window regression. Each forecast uses historical "
            "features and labels available at its forecast origin. Both "
            "feature and target scaling are fitted on training observations "
            "only. The development cutoff applies to forecast target dates."
        ),
        "interpretation": (
            "Forecast errors are not trading returns or evidence of "
            "profitability. Ridge alpha was fixed at 1.0 for this initial "
            "comparison and was not tuned against the later holdout."
        ),
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return predictions, overall_metrics, annual_metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward Ridge Regression for next-day returns."
    )

    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help="Ticker; default: RELIANCE.NS",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help="Research start date; default: 2005-01-01",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Development cutoff; defaults to the previous baseline training end.",
    )
    parser.add_argument(
        "--min-training-observations",
        type=int,
        default=DEFAULT_MIN_TRAINING_OBSERVATIONS,
        help="Initial training window; default: 756.",
    )

    args = parser.parse_args()
    ticker = args.ticker.strip().upper()

    if not ticker:
        parser.error("Ticker cannot be empty.")

    if args.min_training_observations < 100:
        parser.error("--min-training-observations must be at least 100.")

    try:
        start_date = pd.Timestamp(args.start).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        parser.error("--start must be a valid date.")

    data, csv_path, metadata = load_data(ticker)
    end_date, end_source = resolve_end_date(ticker, args.end)

    predictions, overall_metrics, annual_metrics, summary = run_walk_forward(
        data=data,
        start_date=start_date,
        end_date=end_date,
        min_training_observations=args.min_training_observations,
    )

    summary["ticker"] = ticker
    summary["end_date_source"] = end_source
    summary["source_csv"] = str(csv_path.relative_to(PROJECT_ROOT))
    summary["source_csv_sha256"] = sha256_file(csv_path)
    summary["source_metadata_row_count"] = metadata["row_count"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = safe_ticker_name(ticker)

    predictions_path = REPORTS_DIR / f"ridge_predictions_{name}.csv"
    metrics_path = REPORTS_DIR / f"ridge_metrics_{name}.csv"
    annual_path = REPORTS_DIR / f"ridge_annual_metrics_{name}.csv"
    summary_path = REPORTS_DIR / f"ridge_run_{name}.json"

    predictions.to_csv(
        predictions_path,
        index=False,
        date_format="%Y-%m-%d",
    )
    overall_metrics.to_csv(metrics_path, index=False)
    annual_metrics.to_csv(annual_path, index=False)

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n===== RIDGE REGRESSION EXPERIMENT COMPLETE =====")
    print(f"Ticker: {ticker}")
    print(f"Research start: {start_date}")
    print(f"Development cutoff: {end_date.strftime('%Y-%m-%d')}")
    print(f"Cutoff source: {end_source}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")
    print(f"Initial training observations: {args.min_training_observations:,}")
    print(f"Forecasts produced: {len(predictions):,}")
    print(
        f"Forecast origins: {summary['first_forecast_origin']} "
        f"to {summary['last_forecast_origin']}"
    )
    print(f"Last forecast target: {summary['last_forecast_target']}")

    print("\n===== OVERALL RETURN-FORECAST METRICS =====")
    print(overall_metrics.to_string(index=False))

    print("\n===== ANNUAL RETURN-FORECAST METRICS =====")
    print(annual_metrics.to_string(index=False))

    print("\n===== OUTPUT FILES =====")
    print(f"Predictions: {predictions_path}")
    print(f"Overall metrics: {metrics_path}")
    print(f"Annual metrics: {annual_path}")
    print(f"Summary: {summary_path}")

    print(
        "\nNo new downloads were performed. These are forecast metrics, "
        "not trading returns or proof of profitability."
    )


if __name__ == "__main__":
    main()
