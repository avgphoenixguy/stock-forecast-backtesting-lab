
"""Reassess saved Logistic Regression forecasts against simple benchmarks.

Reads existing predictions and reports:
- Logistic Regression classification and probability scores
- Expanding historical up-rate benchmark scores
- Constant 50% probability benchmark scores
- Always-predict-up classification scores

No downloads, model fitting, or changes to historical prices are performed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def safe_ticker_name(ticker: str) -> str:
    """Return the project's ticker filename convention."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").upper()


def probability_scores(
    actual: np.ndarray,
    probability: np.ndarray,
) -> dict:
    """Score predicted probabilities separately from hard decisions."""

    probability = np.clip(probability, 1e-15, 1 - 1e-15)

    auc = (
        float(roc_auc_score(actual, probability))
        if np.unique(actual).size == 2
        else None
    )

    return {
        "roc_auc": auc,
        "brier_score": float(brier_score_loss(actual, probability)),
        "log_loss": float(
            log_loss(actual, probability, labels=[0, 1])
        ),
    }


def classification_scores(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """Score hard class predictions."""

    return {
        "predicted_up_rate_pct": float(predicted.mean() * 100),
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
    }


def evaluate_group(
    frame: pd.DataFrame,
    period: str,
    year: int | None,
) -> list[dict]:
    """Create separate evaluation rows for probability and class baselines."""

    actual = frame["actual_up"].to_numpy(dtype=int)
    rows = []

    definitions = [
        (
            "logistic_regression",
            "logistic_probability_up",
            "logistic_predicted_up",
        ),
        (
            "historical_up_rate",
            "historical_probability_up",
            "historical_predicted_up",
        ),
    ]

    for model_name, probability_column, prediction_column in definitions:
        probability = frame[probability_column].to_numpy(dtype=float)
        predicted = frame[prediction_column].to_numpy(dtype=int)

        scores = probability_scores(actual, probability)
        scores.update(classification_scores(actual, predicted))

        rows.append({
            "period": period,
            "year": year,
            "model": model_name,
            "forecast_count": len(frame),
            "actual_up_rate_pct": float(actual.mean() * 100),
            **scores,
        })

    # Pure probability benchmark: assign 50% to every observation.
    # No hard class decision is implied by this probability forecast.
    constant_probability = np.full(len(frame), 0.5)

    rows.append({
        "period": period,
        "year": year,
        "model": "constant_50pct_probability",
        "forecast_count": len(frame),
        "actual_up_rate_pct": float(actual.mean() * 100),
        "predicted_up_rate_pct": None,
        "accuracy_pct": None,
        "balanced_accuracy_pct": None,
        "precision_up_pct": None,
        "recall_up_pct": None,
        "f1_up": None,
        **probability_scores(actual, constant_probability),
    })

    # Pure majority-class reference: always predict an up day.
    # This is a classification benchmark, not a probability forecast.
    always_up = np.ones(len(frame), dtype=int)

    rows.append({
        "period": period,
        "year": year,
        "model": "always_predict_up",
        "forecast_count": len(frame),
        "actual_up_rate_pct": float(actual.mean() * 100),
        **classification_scores(actual, always_up),
        "roc_auc": None,
        "brier_score": None,
        "log_loss": None,
    })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved Logistic Regression forecasts."
    )
    parser.add_argument(
        "--ticker",
        default="RELIANCE.NS",
        help="Ticker; default RELIANCE.NS",
    )
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()

    if not ticker:
        parser.error("Ticker cannot be empty.")

    name = safe_ticker_name(ticker)

    predictions_path = (
        REPORTS_DIR / f"logistic_predictions_{name}.csv"
    )
    run_summary_path = (
        REPORTS_DIR / f"logistic_run_{name}.json"
    )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {predictions_path}"
        )

    if not run_summary_path.exists():
        raise FileNotFoundError(
            f"Run summary not found: {run_summary_path}"
        )

    run_summary = json.loads(
        run_summary_path.read_text(encoding="utf-8")
    )
    predictions = pd.read_csv(
        predictions_path,
        parse_dates=["forecast_origin_date", "target_date"],
    )

    required = [
        "forecast_origin_date",
        "target_date",
        "actual_up",
        "logistic_probability_up",
        "logistic_predicted_up",
        "historical_probability_up",
        "historical_predicted_up",
    ]

    missing = [column for column in required if column not in predictions]
    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {missing}"
        )

    if predictions.empty:
        raise ValueError("Prediction file is empty.")

    if predictions["target_date"].isna().any():
        raise ValueError("Invalid target dates found.")

    if predictions["target_date"].duplicated().any():
        raise ValueError("Duplicate forecast target dates found.")

    if not predictions["target_date"].is_monotonic_increasing:
        raise ValueError("Forecasts are not sorted by target date.")

    actual = predictions["actual_up"].to_numpy()

    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Actual labels must contain only 0 and 1.")

    cutoff = pd.Timestamp(run_summary["development_end_date"]).normalize()

    if (predictions["target_date"] > cutoff).any():
        raise ValueError(
            "A prediction target extends past the documented "
            f"development cutoff {cutoff.date()}."
        )

    numeric_columns = [
        "logistic_probability_up",
        "historical_probability_up",
    ]

    probability_values = predictions[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(probability_values).all():
        raise ValueError("Non-finite probabilities found.")

    if (
        (probability_values < 0).any()
        or (probability_values > 1).any()
    ):
        raise ValueError("Probabilities must be between 0 and 1.")

    predictions["target_year"] = predictions["target_date"].dt.year

    results = evaluate_group(
        predictions,
        period="overall",
        year=None,
    )

    for year, group in predictions.groupby("target_year", sort=True):
        results.extend(
            evaluate_group(
                group,
                period="annual",
                year=int(year),
            )
        )

    metrics = pd.DataFrame(results)

    overall = metrics.loc[metrics["period"] == "overall"].copy()
    annual = metrics.loc[metrics["period"] == "annual"].copy()

    overall_path = REPORTS_DIR / f"logistic_reassessment_{name}.csv"
    annual_path = REPORTS_DIR / f"logistic_reassessment_annual_{name}.csv"

    overall.to_csv(overall_path, index=False)
    annual.to_csv(annual_path, index=False)

    print("\n===== EVALUATION BOUNDARY CHECK =====")
    print(f"Forecast count: {len(predictions):,}")
    print(
        f"Target period: {predictions['target_date'].min().date()} "
        f"to {predictions['target_date'].max().date()}"
    )
    print(f"Development cutoff: {cutoff.date()}")
    print("No forecast target exceeds the cutoff.")

    print("\n===== OVERALL MODEL COMPARISON =====")
    print(overall.to_string(index=False))

    print("\n===== ANNUAL MODEL COMPARISON =====")
    print(annual.to_string(index=False))

    print("\n===== INTERPRETATION GUIDE =====")
    print("Higher accuracy, balanced accuracy, F1 and ROC-AUC are generally better.")
    print("Lower Brier score and log loss are better.")
    print("The constant 50% benchmark is a probability baseline only.")
    print("The always-up benchmark is a classification baseline only.")
    print("These are forecasts, not trading returns or proof of profitability.")

    print("\n===== SAVED REPORTS =====")
    print(overall_path)
    print(annual_path)


if __name__ == "__main__":
    main()
