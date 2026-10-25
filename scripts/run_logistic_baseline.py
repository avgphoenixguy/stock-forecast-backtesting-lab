
"""Walk-forward Logistic Regression for next-day stock direction.

Uses saved adjusted-price data and compares:
1. Logistic Regression.
2. Historical up-day frequency benchmark.

All model fitting uses information available at each forecast origin.
The development evaluation ends before the previously tested 2022-2026
period. No downloads or trades are performed.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_TICKER = "RELIANCE.NS"
DEFAULT_START_DATE = "2005-01-01"
DEFAULT_MIN_TRAINING_OBSERVATIONS = 756

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
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_data(ticker: str) -> tuple[pd.DataFrame, Path, dict]:
    """Load saved historical data and verify its integrity."""

    name = safe_ticker_name(ticker)
    csv_path = RAW_DATA_DIR / f"{name}_daily.csv"
    metadata_path = RAW_DATA_DIR / f"{name}_metadata.json"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Historical CSV not found: {csv_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if metadata.get("ticker") != ticker:
        raise ValueError(
            "Metadata ticker does not match the requested ticker."
        )

    if metadata.get("csv_sha256") != sha256_file(csv_path):
        raise ValueError(
            "CSV SHA-256 does not match metadata. "
            "Verify the data before modelling."
        )

    adjustment = metadata.get("price_adjustment", {})
    if adjustment.get("auto_adjust") is not True:
        raise ValueError(
            "This experiment requires adjusted prices from "
            "the project downloader."
        )

    data = pd.read_csv(csv_path)
    data.columns = [
        str(column).strip().lower()
        for column in data.columns
    ]

    required = ["date", "open", "high", "low", "close"]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    ).dt.normalize()

    for column in ["open", "high", "low", "close"]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

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

    # Use the same scale-aware floating-point tolerance as the downloader.
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
            f"Found {int(material_violation.sum())} "
            "material OHLC violations."
        )

    return data, csv_path, metadata


def resolve_end_date(
    ticker: str,
    requested_end: str | None,
) -> tuple[pd.Timestamp, str]:
    """Resolve the development cutoff from the saved baseline summary."""

    if requested_end is not None:
        try:
            end_date = pd.Timestamp(requested_end).normalize()
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "--end must be a valid date, e.g. 2022-06-14."
            ) from exc

        if pd.isna(end_date):
            raise ValueError("--end must be a valid date.")

        return end_date, "explicit command-line date"

    summary_path = (
        REPORTS_DIR
        / f"baseline_run_{safe_ticker_name(ticker)}.json"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Baseline summary not found: {summary_path}. "
            "Run scripts/run_baselines.py first or supply --end."
        )

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    if summary.get("ticker") != ticker or not summary.get("training_end"):
        raise ValueError(
            "The baseline summary does not contain a valid "
            "training end date for this ticker."
        )

    return (
        pd.Timestamp(summary["training_end"]).normalize(),
        "previous baseline training end",
    )


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build predictors from prices available at each forecast origin."""

    result = data.copy()

    close = result["close"].astype(float)
    log_close = np.log(close)
    daily_log_return = log_close.diff()

    result["return_1"] = daily_log_return
    result["return_5"] = log_close.diff(5)
    result["return_20"] = log_close.diff(20)
    result["return_60"] = log_close.diff(60)

    result["volatility_20"] = daily_log_return.rolling(20).std()
    result["volatility_60"] = daily_log_return.rolling(60).std()

    result["close_vs_sma_20"] = (
        close / close.rolling(20).mean()
    ) - 1.0

    result["close_vs_sma_60"] = (
        close / close.rolling(60).mean()
    ) - 1.0

    # The target is the next available trading day's close.
    result["target_date"] = result["date"].shift(-1)
    result["actual_next_close"] = close.shift(-1)
    result["next_log_return"] = log_close.shift(-1) - log_close

    result = result.dropna(
        subset=FEATURE_COLUMNS + [
            "target_date",
            "actual_next_close",
            "next_log_return",
        ]
    ).copy()

    # A zero return is classified as not-up.
    result["target_up"] = (
        result["next_log_return"] > 0
    ).astype(int)

    return result.reset_index(drop=True)


def calculate_metrics(
    frame: pd.DataFrame,
    model_name: str,
    period: str,
    year: int | None,
) -> dict:
    """Calculate directional and probability-quality metrics."""

    if model_name == "logistic_regression":
        probability_column = "logistic_probability_up"
        prediction_column = "logistic_predicted_up"
    elif model_name == "historical_up_rate":
        probability_column = "historical_probability_up"
        prediction_column = "historical_predicted_up"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    actual = frame["actual_up"].to_numpy(dtype=int)
    probability = frame[probability_column].to_numpy(dtype=float)
    predicted = frame[prediction_column].to_numpy(dtype=int)

    if not np.isfinite(probability).all():
        raise ValueError(
            f"Non-finite probabilities found for {model_name}."
        )

    probability = np.clip(probability, 1e-15, 1 - 1e-15)

    if np.unique(actual).size == 2:
        auc = float(roc_auc_score(actual, probability))
    else:
        auc = None

    return {
        "period": period,
        "year": year,
        "model": model_name,
        "forecast_count": int(len(frame)),
        "actual_up_rate_pct": float(actual.mean() * 100),
        "accuracy_pct": float(
            accuracy_score(actual, predicted) * 100
        ),
        "balanced_accuracy_pct": float(
            balanced_accuracy_score(actual, predicted) * 100
        ),
        "precision_up_pct": float(
            precision_score(actual, predicted, zero_division=0) * 100
        ),
        "recall_up_pct": float(
            recall_score(actual, predicted, zero_division=0) * 100
        ),
        "f1_up": float(
            f1_score(actual, predicted, zero_division=0)
        ),
        "roc_auc": auc,
        "brier_score": float(
            brier_score_loss(actual, probability)
        ),
        "log_loss": float(
            log_loss(actual, probability, labels=[0, 1])
        ),
    }


def run_walk_forward(
    data: pd.DataFrame,
    start_date: str,
    end_date: pd.Timestamp,
    min_training_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Fit Logistic Regression using an expanding training window."""

    if min_training_observations < 100:
        raise ValueError(
            "At least 100 training observations are required."
        )

    start = pd.Timestamp(start_date).normalize()

    if end_date < start:
        raise ValueError(
            "The development end date precedes the start date."
        )

    # Engineer features on the full history first. This allows observations
    # before the research start to supply lookback values, while training
    # labels themselves are restricted to the research window.
    prepared = engineer_features(data)

    # Critical boundary rule: restrict BOTH forecast origins and targets.
    # A forecast whose target date is after the end date is excluded.
    # This prevents an origin on June 14 from using June 15's outcome.
    prepared = prepared.loc[
        (prepared["date"] >= start)
        & (prepared["date"] < end_date)
        & (prepared["target_date"] <= end_date)
    ].copy().reset_index(drop=True)

    if len(prepared) <= min_training_observations:
        raise ValueError(
            f"Only {len(prepared)} usable feature rows are available "
            f"before {end_date.date()}. More than "
            f"{min_training_observations} are required."
        )

    forecasts = []

    for i in range(min_training_observations, len(prepared)):
        training = prepared.iloc[:i]
        current = prepared.iloc[[i]]

        # At the forecast origin, the prior row's next-day label is known.
        if not (
            training["target_date"].max()
            <= current["date"].iloc[0]
        ):
            raise RuntimeError(
                "Temporal leakage detected: a training target extends "
                "beyond the forecast origin."
            )

        x_train = training[FEATURE_COLUMNS]
        y_train = training["target_up"].astype(int)

        if y_train.nunique() < 2:
            raise ValueError(
                "The training window contains only one target class at "
                f"{current['date'].iloc[0].date()}."
            )

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
            ),
        )

        model.fit(x_train, y_train)

        classifier = model.named_steps["logisticregression"]
        up_class_index = list(classifier.classes_).index(1)

        logistic_probability = float(
            model.predict_proba(current[FEATURE_COLUMNS])[0, up_class_index]
        )

        historical_probability = float(y_train.mean())

        origin_date = current["date"].iloc[0]
        target_date = current["target_date"].iloc[0]

        if target_date > end_date:
            raise RuntimeError(
                "Temporal boundary error: a forecast target crossed "
                "the development cutoff."
            )

        if not np.isfinite(logistic_probability):
            raise ValueError(
                f"Non-finite probability for {target_date.date()}."
            )

        forecasts.append({
            "forecast_origin_date": origin_date,
            "target_date": target_date,
            "training_observations": int(len(training)),
            "training_last_known_target_date": training["target_date"].max(),
            "forecast_origin_close": float(current["close"].iloc[0]),
            "actual_next_close": float(current["actual_next_close"].iloc[0]),
            "actual_next_log_return": float(current["next_log_return"].iloc[0]),
            "actual_up": int(current["target_up"].iloc[0]),
            "logistic_probability_up": logistic_probability,
            "logistic_predicted_up": int(logistic_probability >= 0.5),
            "historical_probability_up": historical_probability,
            "historical_predicted_up": int(
                historical_probability > 0.5
            ),
        })

    predictions = pd.DataFrame(forecasts)

    if predictions.empty:
        raise ValueError("No out-of-sample forecasts were produced.")

    numeric_values = predictions.select_dtypes(
        include=[np.number]
    ).to_numpy()

    if not np.isfinite(numeric_values).all():
        raise ValueError("Non-finite prediction values were generated.")

    predictions["forecast_year"] = predictions["target_date"].dt.year

    overall_metrics = pd.DataFrame([
        calculate_metrics(
            predictions,
            model_name,
            period="overall",
            year=None,
        )
        for model_name in [
            "logistic_regression",
            "historical_up_rate",
        ]
    ])

    annual_rows = []

    for year, group in predictions.groupby("forecast_year", sort=True):
        for model_name in [
            "logistic_regression",
            "historical_up_rate",
        ]:
            annual_rows.append(
                calculate_metrics(
                    group,
                    model_name,
                    period="annual",
                    year=int(year),
                )
            )

    annual_metrics = pd.DataFrame(annual_rows)

    summary = {
        "experiment": "walk_forward_logistic_regression_direction",
        "research_start_date": start.strftime("%Y-%m-%d"),
        "development_end_date": end_date.strftime("%Y-%m-%d"),
        "minimum_training_observations": min_training_observations,
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
            "Expanding-window Logistic Regression. Features are calculated "
            "from information available at the forecast origin. The scaler "
            "and classifier are refitted on prior observations only. "
            "The historical up-rate uses prior training labels only."
        ),
        "target_definition": (
            "Up if the next adjusted close is higher than the current "
            "adjusted close; otherwise not-up."
        ),
        "interpretation": (
            "Directional classification and probability scores on the "
            "development period. These metrics are not trading returns "
            "or evidence of profitability."
        ),
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return predictions, overall_metrics, annual_metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward Logistic Regression for next-day stock direction."
        )
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
            "Development end date. Defaults to the baseline training end."
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

    predictions_path = (
        REPORTS_DIR / f"logistic_predictions_{name}.csv"
    )
    metrics_path = (
        REPORTS_DIR / f"logistic_metrics_{name}.csv"
    )
    annual_path = (
        REPORTS_DIR / f"logistic_annual_metrics_{name}.csv"
    )
    summary_path = (
        REPORTS_DIR / f"logistic_run_{name}.json"
    )

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

    print("\n===== LOGISTIC REGRESSION EXPERIMENT COMPLETE =====")
    print(f"Ticker: {ticker}")
    print(f"Research start: {start_date}")
    print(f"Development end: {end_date.strftime('%Y-%m-%d')}")
    print(f"End-date source: {end_source}")
    print(f"Minimum training observations: {args.min_training_observations:,}")
    print(f"Forecasts produced: {len(predictions):,}")
    print(
        f"Forecast origins: {summary['first_forecast_origin']} "
        f"to {summary['last_forecast_origin']}"
    )
    print(f"Last forecast target: {summary['last_forecast_target']}")

    print("\n===== OVERALL DIRECTIONAL METRICS =====")
    print(overall_metrics.to_string(index=False))

    print("\n===== ANNUAL DIRECTIONAL METRICS =====")
    print(annual_metrics.to_string(index=False))

    print("\n===== OUTPUT FILES =====")
    print(f"Predictions: {predictions_path}")
    print(f"Overall metrics: {metrics_path}")
    print(f"Annual metrics: {annual_path}")
    print(f"Summary: {summary_path}")

    print(
        "\nNo new downloads performed. Metrics describe directional "
        "forecasts, not trading returns or profitability."
    )


if __name__ == "__main__":
    main()
