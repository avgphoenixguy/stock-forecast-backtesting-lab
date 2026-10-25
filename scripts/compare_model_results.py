
"""Consolidate completed forecasting experiments on common target dates.

Regression models:
- Random Walk
- Random Walk with Drift
- Ridge Regression
- Ridge feature-ablation variants
- Decision Tree
- Random Forest

Logistic Regression is assessed separately as a classification model.

Reads existing reports only. No downloads, model fitting, or trades.
"""

from __future__ import annotations

import argparse
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

DEFAULT_TICKER = "RELIANCE.NS"
DEFAULT_CUTOFF = "2022-06-14"

# Each file and prediction column comes from an existing experiment.
REGRESSION_SPECS = {
    "random_walk": (
        "ridge_predictions",
        "random_walk_predicted_return",
    ),
    "drift": (
        "ridge_predictions",
        "drift_predicted_return",
    ),
    "ridge": (
        "ridge_predictions",
        "ridge_predicted_return",
    ),
    "ridge_returns_only": (
        "ridge_ablation_predictions",
        "ridge_returns_only_predicted_return",
    ),
    "ridge_trend_only": (
        "ridge_ablation_predictions",
        "ridge_trend_only_predicted_return",
    ),
    "ridge_volatility_only": (
        "ridge_ablation_predictions",
        "ridge_volatility_only_predicted_return",
    ),
    "ridge_all_features": (
        "ridge_ablation_predictions",
        "ridge_all_features_predicted_return",
    ),
    "decision_tree": (
        "tree_predictions",
        "tree_predicted_return",
    ),
    "random_forest": (
        "random_forest_predictions",
        "random_forest_predicted_return",
    ),
}


def safe_ticker_name(ticker: str) -> str:
    """Match the project's existing report filename convention."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").upper()


def read_forecasts(
    path: Path,
    cutoff: pd.Timestamp,
    required_columns: list[str],
) -> pd.DataFrame:
    """Read, validate, and standardize a saved prediction file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required prediction file not found: {path}\n"
            "Run the corresponding experiment first."
        )

    frame = pd.read_csv(path)

    missing = [column for column in required_columns if column not in frame]
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {missing}"
        )

    frame["target_date"] = pd.to_datetime(
        frame["target_date"], errors="coerce"
    ).dt.normalize()

    if frame.empty:
        raise ValueError(f"{path.name} is empty.")

    if frame["target_date"].isna().any():
        raise ValueError(f"{path.name} has invalid target dates.")

    if frame["target_date"].duplicated().any():
        raise ValueError(f"{path.name} has duplicate target dates.")

    if not frame["target_date"].is_monotonic_increasing:
        raise ValueError(
            f"{path.name} is not sorted by target date."
        )

    if (frame["target_date"] > cutoff).any():
        raise ValueError(
            f"{path.name} contains a target after the development "
            f"cutoff {cutoff.date()}."
        )

    if "forecast_origin_date" in frame.columns:
        frame["forecast_origin_date"] = pd.to_datetime(
            frame["forecast_origin_date"], errors="coerce"
        ).dt.normalize()

        if frame["forecast_origin_date"].isna().any():
            raise ValueError(
                f"{path.name} has invalid forecast origin dates."
            )

        if (
            frame["forecast_origin_date"] >= frame["target_date"]
        ).any():
            raise ValueError(
                f"{path.name} has an invalid forecast origin/target pair."
            )

    numeric_columns = [
        column
        for column in required_columns
        if column != "target_date"
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column], errors="coerce"
        )

    if not np.isfinite(
        frame[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{path.name} contains missing or non-finite numeric values."
        )

    return frame


def verify_same_actuals(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    name: str,
) -> None:
    """Ensure model files refer to the same realized prices and returns."""

    for column in [
        "actual_next_close",
        "origin_close",
        "actual_log_return",
    ]:
        left = reference[column].to_numpy(dtype=float)
        right = candidate[column].to_numpy(dtype=float)

        if not np.allclose(
            left, right, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                f"Realized values in {name} do not match the reference "
                f"file for column {column}."
            )


def calculate_regression_metrics(
    actual: pd.DataFrame,
    predicted_return: pd.Series,
    model_name: str,
    period: str,
    year: int | None,
) -> dict:
    """Calculate comparable one-day-ahead forecast-error metrics."""

    actual_return = actual["actual_log_return"].to_numpy(dtype=float)
    actual_close = actual["actual_next_close"].to_numpy(dtype=float)
    origin_close = actual["origin_close"].to_numpy(dtype=float)
    predicted = predicted_return.to_numpy(dtype=float)

    predicted_close = origin_close * np.exp(predicted)

    return_errors = actual_return - predicted
    close_errors = actual_close - predicted_close

    if np.all(predicted == 0):
        direction_accuracy = None
    else:
        direction_accuracy = float(
            np.mean(
                (actual_return > 0) == (predicted > 0)
            ) * 100
        )

    return {
        "period": period,
        "year": year,
        "model": model_name,
        "forecast_count": len(actual),
        "close_mae": float(np.mean(np.abs(close_errors))),
        "close_rmse": float(np.sqrt(np.mean(close_errors**2))),
        "log_return_mae_bps": float(
            np.mean(np.abs(return_errors)) * 10_000
        ),
        "log_return_rmse_bps": float(
            np.sqrt(np.mean(return_errors**2)) * 10_000
        ),
        "directional_accuracy_pct": direction_accuracy,
        "mean_predicted_log_return_bps": float(
            np.mean(predicted) * 10_000
        ),
    }


def calculate_classifier_metrics(
    frame: pd.DataFrame,
    period: str,
    year: int | None,
) -> list[dict]:
    """Score Logistic Regression and its simple classification benchmarks."""

    actual = frame["actual_up"].to_numpy(dtype=int)

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

    rows = []

    for model_name, probability_col, prediction_col in definitions:
        probability = np.clip(
            frame[probability_col].to_numpy(dtype=float),
            1e-15,
            1 - 1e-15,
        )
        predicted = frame[prediction_col].to_numpy(dtype=int)

        auc = (
            float(roc_auc_score(actual, probability))
            if np.unique(actual).size == 2
            else None
        )

        rows.append({
            "period": period,
            "year": year,
            "model": model_name,
            "forecast_count": len(frame),
            "actual_up_rate_pct": float(actual.mean() * 100),
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
            "roc_auc": auc,
            "brier_score": float(
                brier_score_loss(actual, probability)
            ),
            "log_loss": float(
                log_loss(actual, probability, labels=[0, 1])
            ),
        })

    # Constant 50% probability baseline; no hard decision is implied.
    constant_probability = np.full(len(actual), 0.5)

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
        "roc_auc": 0.5 if np.unique(actual).size == 2 else None,
        "brier_score": float(
            brier_score_loss(actual, constant_probability)
        ),
        "log_loss": float(
            log_loss(actual, constant_probability, labels=[0, 1])
        ),
    })

    # Majority-class reference; a classification benchmark, not a
    # calibrated probability forecast.
    always_up = np.ones(len(actual), dtype=int)

    rows.append({
        "period": period,
        "year": year,
        "model": "always_predict_up",
        "forecast_count": len(frame),
        "actual_up_rate_pct": float(actual.mean() * 100),
        "predicted_up_rate_pct": 100.0,
        "accuracy_pct": float(
            accuracy_score(actual, always_up) * 100
        ),
        "balanced_accuracy_pct": float(
            balanced_accuracy_score(actual, always_up) * 100
        ),
        "precision_up_pct": float(
            precision_score(actual, always_up, zero_division=0) * 100
        ),
        "recall_up_pct": float(
            recall_score(actual, always_up, zero_division=0) * 100
        ),
        "f1_up": float(
            f1_score(actual, always_up, zero_division=0)
        ),
        "roc_auc": None,
        "brier_score": None,
        "log_loss": None,
    })

    return rows


def markdown_table(frame: pd.DataFrame) -> str:
    """Format a dataframe as Markdown without requiring tabulate."""

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in frame.itertuples(index=False, name=None):
        cells = []

        for value in row:
            if pd.isna(value):
                cells.append("N/A")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare existing forecast reports on common target dates."
        )
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help="Ticker; default RELIANCE.NS",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_CUTOFF,
        help="Development target-date cutoff; default 2022-06-14",
    )
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    if not ticker:
        parser.error("Ticker cannot be empty.")

    try:
        cutoff = pd.Timestamp(args.end).normalize()
    except (ValueError, TypeError):
        parser.error("--end must be a valid date.")

    if pd.isna(cutoff):
        parser.error("--end must be a valid date.")

    name = safe_ticker_name(ticker)

    # Load each distinct regression prediction file once.
    file_prefixes = sorted({
        prefix for prefix, _ in REGRESSION_SPECS.values()
    })

    regression_frames = {}

    for prefix in file_prefixes:
        path = REPORTS_DIR / f"{prefix}_{name}.csv"

        frame = read_forecasts(
            path,
            cutoff,
            [
                "target_date",
                "origin_close",
                "actual_next_close",
                "actual_log_return",
            ],
        )

        regression_frames[prefix] = frame

        print(
            f"Loaded {path.name}: {len(frame):,} forecasts, "
            f"{frame['target_date'].min().date()} to "
            f"{frame['target_date'].max().date()}"
        )

    # Logistic classification is optional here; regression comparison
    # can still run if its prediction file is unavailable.
    logistic_path = REPORTS_DIR / f"logistic_predictions_{name}.csv"
    logistic_frame = None

    if logistic_path.exists():
        logistic_frame = read_forecasts(
            logistic_path,
            cutoff,
            [
                "target_date",
                "actual_up",
                "logistic_probability_up",
                "logistic_predicted_up",
                "historical_probability_up",
                "historical_predicted_up",
            ],
        )

        if not np.isin(
            logistic_frame["actual_up"].to_numpy(), [0, 1]
        ).all():
            raise ValueError("Logistic actual_up must contain only 0 and 1.")

        if (
            (logistic_frame["logistic_probability_up"] < 0).any()
            or (logistic_frame["logistic_probability_up"] > 1).any()
            or (logistic_frame["historical_probability_up"] < 0).any()
            or (logistic_frame["historical_probability_up"] > 1).any()
        ):
            raise ValueError("Classifier probabilities must lie in [0, 1].")

        if not np.isin(
            logistic_frame["logistic_predicted_up"].to_numpy(), [0, 1]
        ).all():
            raise ValueError("logistic_predicted_up must contain only 0 or 1.")

        if not np.isin(
            logistic_frame["historical_predicted_up"].to_numpy(), [0, 1]
        ).all():
            raise ValueError("historical_predicted_up must contain only 0 or 1.")

        print(
            f"Loaded {logistic_path.name}: {len(logistic_frame):,} forecasts, "
            f"{logistic_frame['target_date'].min().date()} to "
            f"{logistic_frame['target_date'].max().date()}"
        )
    else:
        print(
            "WARNING: Logistic Regression prediction file was not found. "
            "The regression comparison will still run."
        )

    # Only compare dates available in every included model file.
    date_sets = [
        set(frame["target_date"])
        for frame in regression_frames.values()
    ]

    if logistic_frame is not None:
        date_sets.append(set(logistic_frame["target_date"]))

    common_dates = sorted(set.intersection(*date_sets))

    if len(common_dates) < 100:
        raise ValueError(
            f"Only {len(common_dates)} common target dates remain. "
            "Check the forecast files before comparing models."
        )

    if common_dates[-1] > cutoff:
        raise RuntimeError("A common target date exceeds the cutoff.")

    aligned = {
        prefix: frame.set_index("target_date").loc[common_dates]
        for prefix, frame in regression_frames.items()
    }

    actual = aligned["ridge_predictions"]

    # Check realized values match across all regression predictions.
    for prefix, frame in aligned.items():
        verify_same_actuals(actual, frame, prefix)

    predicted_returns = {}

    for model_name, (prefix, prediction_column) in REGRESSION_SPECS.items():
        frame = aligned[prefix]

        if prediction_column not in frame.columns:
            raise ValueError(
                f"{prefix} prediction file is missing {prediction_column}."
            )

        values = frame[prediction_column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite predictions found for {model_name}."
            )

        predicted_returns[model_name] = pd.Series(
            values, index=common_dates, name=model_name
        )

    overall_rows = []

    for model_name, prediction in predicted_returns.items():
        overall_rows.append(
            calculate_regression_metrics(
                actual=actual,
                predicted_return=prediction,
                model_name=model_name,
                period="overall",
                year=None,
            )
        )

    overall = pd.DataFrame(overall_rows).sort_values(
        "log_return_mae_bps"
    ).reset_index(drop=True)

    annual_rows = []
    common_years = sorted(set(pd.DatetimeIndex(common_dates).year))

    for year in common_years:
        annual_dates = [
            date for date in common_dates if date.year == year
        ]
        year_actual = actual.loc[annual_dates]

        for model_name, prediction in predicted_returns.items():
            annual_rows.append(
                calculate_regression_metrics(
                    actual=year_actual,
                    predicted_return=prediction.loc[annual_dates],
                    model_name=model_name,
                    period="annual",
                    year=int(year),
                )
            )

    annual = pd.DataFrame(annual_rows)

    # Evaluate logistic forecasts on exactly the same common dates,
    # but keep their classification scores separate from return errors.
    classifier_overall = pd.DataFrame()
    classifier_annual = pd.DataFrame()

    if logistic_frame is not None:
        aligned_logistic = (
            logistic_frame.set_index("target_date").loc[common_dates]
        )

        expected_up = (
            actual["actual_log_return"].to_numpy(dtype=float) > 0
        ).astype(int)

        observed_up = aligned_logistic["actual_up"].to_numpy(dtype=int)

        if not np.array_equal(expected_up, observed_up):
            raise ValueError(
                "Logistic actual_up labels do not agree with realized "
                "returns on the common target dates."
            )

        classifier_overall = pd.DataFrame(
            calculate_classifier_metrics(
                aligned_logistic,
                period="overall",
                year=None,
            )
        )

        annual_classifier_rows = []

        for year in common_years:
            annual_dates = [
                date for date in common_dates if date.year == year
            ]

            annual_classifier_rows.extend(
                calculate_classifier_metrics(
                    aligned_logistic.loc[annual_dates],
                    period="annual",
                    year=int(year),
                )
            )

        classifier_annual = pd.DataFrame(annual_classifier_rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    overall_path = REPORTS_DIR / f"model_comparison_overall_{name}.csv"
    annual_path = REPORTS_DIR / f"model_comparison_annual_{name}.csv"
    report_path = REPORTS_DIR / f"model_comparison_report_{name}.md"

    overall.to_csv(overall_path, index=False)
    annual.to_csv(annual_path, index=False)

    classifier_path = None

    if not classifier_overall.empty:
        classifier_path = (
            REPORTS_DIR / f"model_comparison_classifier_{name}.csv"
        )
        classifier_overall.to_csv(classifier_path, index=False)

        classifier_annual_path = (
            REPORTS_DIR / f"model_comparison_classifier_annual_{name}.csv"
        )
        classifier_annual.to_csv(classifier_annual_path, index=False)

    # Summarize annual winners by return MAE.
    winner_indices = annual.groupby("year")[
        "log_return_mae_bps"
    ].idxmin()
    winner_counts = (
        annual.loc[winner_indices, "model"]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="years_with_lowest_return_mae")
    )

    best_model = overall.iloc[0]["model"]
    best_mae = overall.iloc[0]["log_return_mae_bps"]

    report_parts = [
        f"# Forecast Model Comparison — {ticker}",
        "",
        "## Evaluation boundary",
        "",
        f"- Development target-date cutoff: {cutoff.date()}",
        f"- Common forecast targets: {len(common_dates):,}",
        f"- First common target: {common_dates[0].date()}",
        f"- Last common target: {common_dates[-1].date()}",
        f"- Models evaluated: {len(predicted_returns)} regression forecasts",
        "",
        "Only dates present in every included prediction file are used. "
        "Realized closes and returns are checked for consistency across files. "
        "No target after the development cutoff is permitted.",
        "",
        "## Overall regression comparison",
        "",
        markdown_table(
            overall[
                [
                    "model",
                    "forecast_count",
                    "close_mae",
                    "close_rmse",
                    "log_return_mae_bps",
                    "log_return_rmse_bps",
                    "directional_accuracy_pct",
                ]
            ].round(4)
        ),
        "",
        f"Lowest return MAE in this common-window comparison: **{best_model} "
        f"({best_mae:.4f} bps)**.",
        "",
        "Lower close and log-return MAE/RMSE are better. Directional accuracy "
        "is descriptive and is not evidence of profitability. The "
        "Random Walk benchmark always predicts zero return, so its directional "
        "accuracy is intentionally undefined.",
        "",
        "## Annual return-MAE winners",
        "",
        markdown_table(winner_counts),
        "",
        "Annual wins are descriptive only. A model winning a few years does "
        "not establish a stable edge, particularly after trying several models.",
        "",
    ]

    if not classifier_overall.empty:
        report_parts.extend([
            "## Logistic Regression — classification results",
            "",
            markdown_table(
                classifier_overall[
                    [
                        "model",
                        "forecast_count",
                        "actual_up_rate_pct",
                        "accuracy_pct",
                        "balanced_accuracy_pct",
                        "roc_auc",
                        "brier_score",
                        "log_loss",
                    ]
                ].round(4)
            ),
            "",
            "These direction-classification metrics are not directly "
            "comparable with regression return-error metrics. Lower Brier "
            "score and log loss are better; higher ROC-AUC and balanced "
            "accuracy are generally better.",
            "",
        ])
    else:
        report_parts.extend([
            "## Logistic Regression",
            "",
            "Classification results were not included because the saved "
            "Logistic Regression prediction file was not found.",
            "",
        ])

    report_parts.extend([
        "## Research interpretation",
        "",
        "This report consolidates historical development-period forecasts. "
        "It does not evaluate a trading strategy, transaction costs, slippage, "
        "portfolio risk, or risk-adjusted returns. These results do not "
        "establish profitability.",
        "",
        "The separate chronological 80/20 baseline test is not mixed into "
        "this table because it covers a different evaluation period. The "
        "previously evaluated 2022-06-15 to 2026-10-23 period must not be "
        "used for model tuning.",
        "",
    ])

    report_path.write_text(
        "\n".join(report_parts),
        encoding="utf-8",
    )

    print("\n===== COMMON-DATE COMPARISON COMPLETE =====")
    print(f"Ticker: {ticker}")
    print(f"Development cutoff: {cutoff.date()}")
    print(f"Common forecast targets: {len(common_dates):,}")
    print(
        f"Common target range: {common_dates[0].date()} "
        f"to {common_dates[-1].date()}"
    )

    print("\n===== OVERALL REGRESSION COMPARISON =====")
    print(
        overall[
            [
                "model",
                "forecast_count",
                "close_mae",
                "close_rmse",
                "log_return_mae_bps",
                "log_return_rmse_bps",
                "directional_accuracy_pct",
            ]
        ].to_string(index=False)
    )

    if not classifier_overall.empty:
        print("\n===== LOGISTIC CLASSIFICATION COMPARISON =====")
        print(classifier_overall.to_string(index=False))

    print("\n===== REPORT FILES =====")
    print(f"Overall regression metrics: {overall_path}")
    print(f"Annual regression metrics: {annual_path}")
    if classifier_path is not None:
        print(f"Classifier metrics: {classifier_path}")
    print(f"Consolidated Markdown report: {report_path}")

    print(
        "\nThis is a forecast comparison, not a trading backtest "
        "or proof of profitability."
    )


if __name__ == "__main__":
    main()
