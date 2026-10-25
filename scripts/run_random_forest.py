"""Walk-forward Random Forest regression for next-day log returns.

Compares:
1. Random Walk
2. Random Walk with Drift
3. Random Forest regression

Uses the existing validated historical data, feature engineering,
and metric calculation from run_ridge_baseline.py.

The model uses fixed settings and does not tune against later data.
No new downloads or trades are performed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from run_ridge_baseline import (
    DEFAULT_MIN_TRAINING_OBSERVATIONS,
    DEFAULT_START_DATE,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    REPORTS_DIR,
    engineer_features,
    load_data,
    metric_row,
    resolve_end_date,
    safe_ticker_name,
    sha256_file,
)


# Fixed initial configuration. Do not tune against the later period.
RF_N_ESTIMATORS = 50
RF_MAX_DEPTH = 5
RF_MIN_SAMPLES_LEAF = 50
RF_MAX_FEATURES = 0.7
RF_RANDOM_STATE = 42


def run_walk_forward_random_forest(
    data: pd.DataFrame,
    start_date: str,
    end_date: pd.Timestamp,
    min_training_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Generate one-day-ahead forecasts with expanding-window validation."""

    if min_training_observations < 100:
        raise ValueError(
            "Minimum training observations must be at least 100."
        )

    start = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()

    if end_date < start:
        raise ValueError("End date must not precede the research start.")

    # Reuse the features and target definition already used for Ridge.
    prepared = engineer_features(data)

    # Restrict both forecast origins and their target dates to the
    # development window. No target after the cutoff is permitted.
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

        # Training labels must be available by the forecast origin.
        if training["target_date"].max() > origin_date:
            raise RuntimeError(
                "Temporal leakage detected: a training target extends "
                "past the forecast origin."
            )

        if target_date > end_date:
            raise RuntimeError(
                "A forecast target crossed the development cutoff."
            )

        x_train = training[FEATURE_COLUMNS]
        y_train = training["target_log_return"].to_numpy(dtype=float)
        x_current = current[FEATURE_COLUMNS]

        if not np.isfinite(y_train).all():
            raise ValueError("Non-finite training targets were found.")

        # Fixed Random Forest settings for this first comparison.
        model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            random_state=RF_RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(x_train, y_train)

        random_forest_prediction = float(
            np.asarray(model.predict(x_current)).reshape(-1)[0]
        )

        # Drift benchmark uses only training labels observable so far.
        drift_prediction = float(np.mean(y_train))

        origin_close = float(current["close"].iloc[0])
        actual_next_close = float(current["actual_next_close"].iloc[0])
        actual_return = float(current["target_log_return"].iloc[0])

        forecasts.append(
            {
                "forecast_origin_date": origin_date,
                "target_date": target_date,
                "training_observations": int(len(training)),
                "training_end_date": training["target_date"].max(),
                "origin_close": origin_close,
                "actual_next_close": actual_next_close,
                "actual_log_return": actual_return,
                "random_walk_predicted_return": 0.0,
                "drift_predicted_return": drift_prediction,
                "random_forest_predicted_return": (
                    random_forest_prediction
                ),
            }
        )

        # Occasional progress indicator for a long experiment.
        completed = len(forecasts)
        if completed % 500 == 0:
            print(
                f"Forecasts completed: {completed:,} / "
                f"{len(prepared) - min_training_observations:,}"
            )

    predictions = pd.DataFrame(forecasts)

    if predictions.empty:
        raise ValueError("No forecasts were generated.")

    numeric_values = predictions.select_dtypes(
        include=[np.number]
    ).to_numpy()

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Non-finite values were generated in the forecasts."
        )

    predictions["forecast_year"] = predictions["target_date"].dt.year

    model_names = [
        "random_walk",
        "drift",
        "random_forest",
    ]

    overall_metrics = pd.DataFrame(
        [
            metric_row(
                predictions,
                model_name,
                period="overall",
                year=None,
            )
            for model_name in model_names
        ]
    )

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

    annual_metrics = pd.DataFrame(annual_rows)

    summary = {
        "project": "Stock Forecast & Backtesting Lab",
        "experiment": "walk_forward_random_forest_next_day_log_return",
        "ticker": str(data.attrs.get("ticker", "")),
        "research_start_date": start.strftime("%Y-%m-%d"),
        "development_end_date": end_date.strftime("%Y-%m-%d"),
        "minimum_training_observations": min_training_observations,
        "model": "RandomForestRegressor",
        "n_estimators": RF_N_ESTIMATORS,
        "max_depth": RF_MAX_DEPTH,
        "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
        "max_features": RF_MAX_FEATURES,
        "random_state": RF_RANDOM_STATE,
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
            "Expanding-window, one-day-ahead regression. Each forecast "
            "uses historical features and training labels available by "
            "its forecast origin. Forecast target dates are restricted "
            "to the development cutoff. Model settings are fixed."
        ),
        "interpretation": (
            "Forecast errors are not trading returns or proof of "
            "profitability. Any apparent improvement must be assessed "
            "against the same simple benchmarks and across time."
        ),
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return predictions, overall_metrics, annual_metrics, summary


def main() -> None:
    """Run the Random Forest benchmark experiment."""

    parser = argparse.ArgumentParser(
        description=(
            "Run walk-forward Random Forest regression for next-day returns."
        )
    )

    parser.add_argument(
        "--ticker",
        default="RELIANCE.NS",
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
            "Development cutoff. Defaults to the training end recorded "
            "in the existing baseline summary."
        ),
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
    data.attrs["ticker"] = ticker

    end_date, end_date_source = resolve_end_date(ticker, args.end)

    predictions, overall_metrics, annual_metrics, summary = (
        run_walk_forward_random_forest(
            data=data,
            start_date=start_date,
            end_date=end_date,
            min_training_observations=args.min_training_observations,
        )
    )

    summary["end_date_source"] = end_date_source
    summary["source_csv"] = str(csv_path.relative_to(PROJECT_ROOT))
    summary["source_csv_sha256"] = sha256_file(csv_path)
    summary["source_metadata_row_count"] = metadata["row_count"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = safe_ticker_name(ticker)

    predictions_path = REPORTS_DIR / f"random_forest_predictions_{name}.csv"
    metrics_path = REPORTS_DIR / f"random_forest_metrics_{name}.csv"
    annual_path = (
        REPORTS_DIR / f"random_forest_annual_metrics_{name}.csv"
    )
    summary_path = REPORTS_DIR / f"random_forest_run_{name}.json"

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

    print("\n===== RANDOM FOREST EXPERIMENT COMPLETE =====")
    print(f"Ticker: {ticker}")
    print(f"Research start: {start_date}")
    print(f"Development cutoff: {end_date.strftime('%Y-%m-%d')}")
    print(f"Cutoff source: {end_date_source}")
    print(f"Number of trees: {RF_N_ESTIMATORS}")
    print(f"Maximum tree depth: {RF_MAX_DEPTH}")
    print(f"Minimum samples per leaf: {RF_MIN_SAMPLES_LEAF}")
    print(f"Maximum features per split: {RF_MAX_FEATURES}")
    print(
        f"Initial training observations: "
        f"{args.min_training_observations:,}"
    )
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