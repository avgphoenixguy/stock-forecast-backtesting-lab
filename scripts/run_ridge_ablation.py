
"""Compare Ridge Regression feature groups using walk-forward validation.

Feature groups:
1. Returns only
2. Trend only
3. Volatility only
4. All current features

Benchmarks:
- Random Walk: predict zero next-day log return.
- Drift: predict mean historical training log return.

Uses saved adjusted prices only. No downloads or trades are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
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

FEATURE_GROUPS = {
    "ridge_returns_only": [
        "return_1",
        "return_5",
        "return_20",
        "return_60",
    ],
    "ridge_trend_only": [
        "close_vs_sma_20",
        "close_vs_sma_60",
    ],
    "ridge_volatility_only": [
        "volatility_20",
        "volatility_60",
    ],
    "ridge_all_features": [
        "return_1",
        "return_5",
        "return_20",
        "return_60",
        "volatility_20",
        "volatility_60",
        "close_vs_sma_20",
        "close_vs_sma_60",
    ],
}


def safe_ticker_name(ticker: str) -> str:
    """Return the project's standard ticker filename."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").upper()


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_data(ticker: str) -> tuple[pd.DataFrame, Path, dict]:
    """Load saved history and verify file integrity."""

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
    data.columns = [str(c).strip().lower() for c in data.columns]

    required = ["date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    ).dt.normalize()

    for column in ["open", "high", "low", "close"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data["date"].isna().any():
        raise ValueError("Invalid dates exist in the CSV.")

    if data["date"].duplicated().any():
        raise ValueError("Duplicate dates exist in the CSV.")

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
        raise ValueError("OHLC contains non-positive prices.")

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
    """Use the previous baseline's training cutoff by default."""

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
        raise ValueError("The baseline summary has no valid training end date.")

    return (
        pd.Timestamp(summary["training_end"]).normalize(),
        "previous baseline training end",
    )


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate trailing predictors and next-session targets."""

    result = data.copy()

    close = result["close"].astype(float)
    log_close = np.log(close)
    daily_log_return = log_close.diff()

    # Trailing price-return predictors.
    result["return_1"] = daily_log_return
    result["return_5"] = log_close.diff(5)
    result["return_20"] = log_close.diff(20)
    result["return_60"] = log_close.diff(60)

    # Trailing volatility predictors.
    result["volatility_20"] = daily_log_return.rolling(20).std()
    result["volatility_60"] = daily_log_return.rolling(60).std()

    # Trend predictors: distance from trailing moving averages.
    result["close_vs_sma_20"] = close / close.rolling(20).mean() - 1.0
    result["close_vs_sma_60"] = close / close.rolling(60).mean() - 1.0

    # Predict the next trading session's log return.
    result["target_date"] = result["date"].shift(-1)
    result["actual_next_close"] = close.shift(-1)
    result["target_log_return"] = log_close.shift(-1) - log_close

    all_features = sorted({
        feature
        for features in FEATURE_GROUPS.values()
        for feature in features
    })

    result = result.dropna(
        subset=all_features + [
            "target_date",
            "actual_next_close",
            "target_log_return",
        ]
    ).reset_index(drop=True)

    return result


def create_ridge_model() -> TransformedTargetRegressor:
    """Scale predictors and target within each training window."""

    return TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            Ridge(alpha=RIDGE_ALPHA),
        ),
        transformer=StandardScaler(),
    )


def calculate_metrics(
    frame: pd.DataFrame,
    model_name: str,
    period: str,
    year: int | None,
) -> dict:
    """Calculate price and log-return forecast errors."""

    actual_close = frame["actual_next_close"].to_numpy(dtype=float)
    origin_close = frame["origin_close"].to_numpy(dtype=float)
    actual_return = frame["actual_log_return"].to_numpy(dtype=float)

    predicted_return = frame[
        f"{model_name}_predicted_return"
    ].to_numpy(dtype=float)

    predicted_close = origin_close * np.exp(predicted_return)

    close_errors = actual_close - predicted_close
    return_errors = actual_return - predicted_return

    if np.all(predicted_return == 0):
        directional_accuracy = None
    else:
        directional_accuracy = float(
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
        "directional_accuracy_pct": directional_accuracy,
        "mean_predicted_log_return_bps": float(
            np.mean(predicted_return) * 10_000
        ),
    }


def run_ablation(
    data: pd.DataFrame,
    start_date: str,
    end_date: pd.Timestamp,
    min_training_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate all feature groups on identical expanding-window dates."""

    if min_training_observations < 100:
        raise ValueError("Minimum training observations must be at least 100.")

    start = pd.Timestamp(start_date).normalize()

    if end_date < start:
        raise ValueError("End date must not precede the start date.")

    # Calculate strictly trailing features. The target is next session's return.
    prepared = engineer_features(data)

    # Restrict both origin dates and target dates to the development window.
    # This ensures that no target crosses into the previously evaluated period.
    prepared = prepared.loc[
        (prepared["date"] >= start)
        & (prepared["date"] < end_date)
        & (prepared["target_date"] <= end_date)
    ].copy().reset_index(drop=True)

    if len(prepared) <= min_training_observations:
        raise ValueError(
            f"Only {len(prepared)} usable rows are available. "
            f"More than {min_training_observations} are required."
        )

    model_names = [
        "random_walk",
        "drift",
        *FEATURE_GROUPS.keys(),
    ]

    forecasts = []
    total_forecasts = len(prepared) - min_training_observations

    print(f"Usable development rows: {len(prepared):,}")
    print(f"Initial training window: {min_training_observations:,}")
    print(f"Forecast origins to evaluate: {total_forecasts:,}")
    print(f"Ridge models per forecast origin: {len(FEATURE_GROUPS)}")
    print(f"Total Ridge model fits planned: {total_forecasts * len(FEATURE_GROUPS):,}")

    started = time.perf_counter()

    for i in range(min_training_observations, len(prepared)):
        training = prepared.iloc[:i]
        current = prepared.iloc[[i]]

        origin_date = current["date"].iloc[0]
        target_date = current["target_date"].iloc[0]

        # A training label must be observable no later than this origin.
        if training["target_date"].max() > origin_date:
            raise RuntimeError(
                f"Temporal leakage detected at origin {origin_date.date()}."
            )

        if target_date > end_date:
            raise RuntimeError(
                f"Target {target_date.date()} crossed the development cutoff."
            )

        y_train = training["target_log_return"].to_numpy(dtype=float)

        if not np.isfinite(y_train).all():
            raise ValueError("Non-finite training targets found.")

        current_close = float(current["close"].iloc[0])

        row = {
            "forecast_origin_date": origin_date,
            "target_date": target_date,
            "training_observations": len(training),
            "training_end_date": training["target_date"].max(),
            "origin_close": current_close,
            "actual_next_close": float(
                current["actual_next_close"].iloc[0]
            ),
            "actual_log_return": float(
                current["target_log_return"].iloc[0]
            ),
            "random_walk_predicted_return": 0.0,
            "drift_predicted_return": float(np.mean(y_train)),
        }

        # Fit a separate pipeline for each feature group. Standardization
        # of predictors and target is estimated from training data only.
        for model_name, feature_columns in FEATURE_GROUPS.items():
            model = create_ridge_model()

            model.fit(
                training[feature_columns],
                y_train,
            )

            prediction = float(
                np.asarray(
                    model.predict(current[feature_columns])
                ).reshape(-1)[0]
            )

            if not np.isfinite(prediction):
                raise ValueError(
                    f"Non-finite prediction from {model_name} "
                    f"at {origin_date.date()}."
                )

            row[f"{model_name}_predicted_return"] = prediction

        forecasts.append(row)

        completed = i - min_training_observations + 1
        if completed % 250 == 0 or completed == total_forecasts:
            elapsed = time.perf_counter() - started
            print(
                f"Progress: {completed:,}/{total_forecasts:,} forecasts | "
                f"elapsed {elapsed:.1f}s",
                flush=True,
            )

    predictions = pd.DataFrame(forecasts)

    if predictions.empty:
        raise ValueError("No forecasts were produced.")

    numeric_values = predictions.select_dtypes(
        include=[np.number]
    ).to_numpy()

    if not np.isfinite(numeric_values).all():
        raise ValueError("Non-finite values found in predictions.")

    if (predictions["target_date"] > end_date).any():
        raise RuntimeError("A forecast target exceeds the development cutoff.")

    predictions["forecast_year"] = predictions["target_date"].dt.year

    overall_rows = [
        calculate_metrics(
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
                calculate_metrics(
                    group,
                    model_name,
                    period="annual",
                    year=int(year),
                )
            )

    summary = {
        "experiment": "ridge_feature_ablation_walk_forward",
        "research_start_date": start.strftime("%Y-%m-%d"),
        "development_end_date": end_date.strftime("%Y-%m-%d"),
        "minimum_training_observations": min_training_observations,
        "ridge_alpha": RIDGE_ALPHA,
        "feature_groups": FEATURE_GROUPS,
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
            "Expanding-window one-day-ahead forecasting. All feature "
            "groups use identical forecast dates, training windows, and "
            "a fixed Ridge alpha of 1.0. Feature and target scalers are "
            "fitted on each training window only."
        ),
        "interpretation": (
            "Feature-ablation forecast errors are not trading returns or "
            "evidence of profitability. The experiment is for model "
            "development; the previously examined 2022-2026 period is "
            "excluded from tuning."
        ),
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return (
        predictions,
        pd.DataFrame(overall_rows),
        pd.DataFrame(annual_rows),
        summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Ridge feature groups using walk-forward validation."
    )

    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help="Ticker; default: RELIANCE.NS",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help="Development start date; default: 2005-01-01",
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
        help="Initial training window; default: 756 observations.",
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

    predictions, overall, annual, summary = run_ablation(
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

    predictions_path = REPORTS_DIR / f"ridge_ablation_predictions_{name}.csv"
    overall_path = REPORTS_DIR / f"ridge_ablation_metrics_{name}.csv"
    annual_path = REPORTS_DIR / f"ridge_ablation_annual_{name}.csv"
    summary_path = REPORTS_DIR / f"ridge_ablation_run_{name}.json"

    predictions.to_csv(
        predictions_path,
        index=False,
        date_format="%Y-%m-%d",
    )
    overall.to_csv(overall_path, index=False)
    annual.to_csv(annual_path, index=False)

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n===== RIDGE FEATURE-ABLATION COMPLETE =====")
    print(f"Ticker: {ticker}")
    print(f"Research start: {start_date}")
    print(f"Development cutoff: {end_date.strftime('%Y-%m-%d')}")
    print(f"Cutoff source: {end_source}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")
    print(f"Forecasts produced: {len(predictions):,}")
    print(
        f"Forecast origins: {summary['first_forecast_origin']} "
        f"to {summary['last_forecast_origin']}"
    )
    print(f"Last forecast target: {summary['last_forecast_target']}")

    print("\n===== OVERALL FEATURE-ABLATION METRICS =====")
    print(overall.to_string(index=False))

    print("\n===== ANNUAL FEATURE-ABLATION METRICS =====")
    print(annual.to_string(index=False))

    print("\n===== OUTPUT FILES =====")
    print(f"Predictions: {predictions_path}")
    print(f"Overall metrics: {overall_path}")
    print(f"Annual metrics: {annual_path}")
    print(f"Run summary: {summary_path}")

    print(
        "\nNo new downloads performed. Compare lower forecast errors, "
        "not model complexity alone. These are forecasts, not trading returns."
    )


if __name__ == "__main__":
    main()
