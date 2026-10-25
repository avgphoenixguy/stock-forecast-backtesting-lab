
"""Exploratory long-or-cash intraday strategy backtest.

Uses saved one-day-ahead regression forecasts to decide whether to
enter long at the following session's open and exit at its close.

Compares the forecast-driven strategies with:
- Always-long intraday benchmark
- Cash benchmark

The assumed per-side cost is configurable. It is a scenario assumption,
not a verified brokerage tariff.

No data is downloaded and no real trades are placed.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_ridge_baseline import (
    PROJECT_ROOT,
    REPORTS_DIR,
    load_data,
    safe_ticker_name,
    sha256_file,
)


DEFAULT_TICKER = "RELIANCE.NS"
DEFAULT_CUTOFF = "2022-06-14"
DEFAULT_ONE_WAY_COST_BPS = 10.0
TRADING_SESSIONS_PER_YEAR = 252

# Each tuple contains:
# (prediction report filename prefix, predicted next log-return column)
MODEL_SPECS = {
    "random_walk": (
        "ridge_predictions",
        "random_walk_predicted_return",
    ),
    "drift": (
        "ridge_predictions",
        "drift_predicted_return",
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


def read_prediction_file(
    path: Path,
    prediction_column: str,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Load and validate a saved model's forecasts."""

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}\n"
            "Run the relevant forecasting experiment first."
        )

    frame = pd.read_csv(
        path,
        parse_dates=["forecast_origin_date", "target_date"],
    )

    required = [
        "forecast_origin_date",
        "target_date",
        "origin_close",
        "actual_next_close",
        "actual_log_return",
        prediction_column,
    ]

    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {missing}"
        )

    if frame.empty:
        raise ValueError(f"{path.name} is empty.")

    for column in ["forecast_origin_date", "target_date"]:
        frame[column] = pd.to_datetime(
            frame[column], errors="coerce"
        ).dt.normalize()

        if frame[column].isna().any():
            raise ValueError(
                f"{path.name} contains invalid dates in {column}."
            )

    if frame["target_date"].duplicated().any():
        raise ValueError(
            f"{path.name} contains duplicate target dates."
        )

    if not frame["target_date"].is_monotonic_increasing:
        raise ValueError(
            f"{path.name} is not sorted by target date."
        )

    if (frame["forecast_origin_date"] >= frame["target_date"]).any():
        raise ValueError(
            f"{path.name} contains an invalid forecast-origin/target pair."
        )

    if (frame["target_date"] > cutoff).any():
        raise ValueError(
            f"{path.name} includes targets after the cutoff "
            f"{cutoff.date()}."
        )

    numeric_columns = [
        "origin_close",
        "actual_next_close",
        "actual_log_return",
        prediction_column,
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

    if (frame["origin_close"] <= 0).any():
        raise ValueError(
            f"{path.name} contains non-positive origin prices."
        )

    if (frame["actual_next_close"] <= 0).any():
        raise ValueError(
            f"{path.name} contains non-positive actual closing prices."
        )

    return frame.set_index("target_date", drop=False)


def verify_close_values(
    expected: np.ndarray,
    observed: np.ndarray,
    description: str,
) -> None:
    """Fail when forecast files disagree with the validated price history."""

    if not np.allclose(
        expected,
        observed,
        rtol=1e-9,
        atol=1e-8,
    ):
        raise ValueError(
            f"Price consistency check failed: {description}."
        )


def summarize_returns(
    model_name: str,
    period: str,
    year: int | None,
    dates: pd.Series,
    positions: np.ndarray,
    gross_returns: np.ndarray,
    net_returns: np.ndarray,
    one_way_cost_bps: float,
) -> dict:
    """Calculate exploratory portfolio and risk statistics."""

    count = len(net_returns)

    if count == 0:
        raise ValueError("Cannot summarize an empty return series.")

    if (
        not np.isfinite(gross_returns).all()
        or not np.isfinite(net_returns).all()
    ):
        raise ValueError(
            f"Non-finite portfolio returns found for {model_name}."
        )

    if (net_returns <= -1).any():
        raise ValueError(
            f"A daily net return of -100% or less occurred for {model_name}."
        )

    equity_net = np.cumprod(1.0 + net_returns)
    equity_gross = np.cumprod(1.0 + gross_returns)

    net_total_return = float(equity_net[-1] - 1.0)
    gross_total_return = float(equity_gross[-1] - 1.0)

    if equity_net[-1] > 0:
        annualized_return = float(
            equity_net[-1] ** (
                TRADING_SESSIONS_PER_YEAR / count
            ) - 1.0
        )
    else:
        annualized_return = float("nan")

    annualized_volatility = float(
        np.std(net_returns, ddof=1) * np.sqrt(TRADING_SESSIONS_PER_YEAR)
    ) if count > 1 else 0.0

    standard_deviation = float(np.std(net_returns, ddof=1)) if count > 1 else 0.0

    if standard_deviation > 0:
        sharpe = float(
            np.mean(net_returns)
            / standard_deviation
            * np.sqrt(TRADING_SESSIONS_PER_YEAR)
        )
    else:
        sharpe = None

    # Include initial equity of 1.0 in the high-water mark.
    wealth = np.concatenate(([1.0], equity_net))
    high_water_mark = np.maximum.accumulate(wealth)
    drawdowns = wealth / high_water_mark - 1.0
    max_drawdown = float(np.min(drawdowns))

    active = positions.astype(bool)
    active_count = int(active.sum())

    if active_count:
        win_rate = float(
            np.mean(net_returns[active] > 0) * 100.0
        )
        average_active_gross_return = float(
            np.mean(gross_returns[active]) * 100.0
        )
    else:
        win_rate = None
        average_active_gross_return = None

    return {
        "period": period,
        "year": year,
        "model": model_name,
        "sessions": int(count),
        "active_days": active_count,
        "exposure_pct": float(active.mean() * 100.0),
        "gross_total_return_pct": gross_total_return * 100.0,
        "net_total_return_pct": net_total_return * 100.0,
        "annualized_net_return_pct": annualized_return * 100.0,
        "annualized_net_volatility_pct": annualized_volatility * 100.0,
        "net_sharpe_ratio_rf_zero": sharpe,
        "max_net_drawdown_pct": max_drawdown * 100.0,
        "win_rate_active_days_pct": win_rate,
        "average_active_gross_return_pct": average_active_gross_return,
        "one_way_cost_assumption_bps": float(one_way_cost_bps),
        "round_trip_cost_assumption_bps": float(2 * one_way_cost_bps),
        # Simple sum of assumed round-trip costs per active session.
        "estimated_cost_sum_bps": float(
            active_count * 2 * one_way_cost_bps
        ),
        "first_date": pd.Timestamp(dates.iloc[0]).strftime("%Y-%m-%d"),
        "last_date": pd.Timestamp(dates.iloc[-1]).strftime("%Y-%m-%d"),
    }


def build_strategy_rows(
    model_name: str,
    frame: pd.DataFrame,
    predicted_returns: np.ndarray | None,
    origin_close: np.ndarray,
    target_open: np.ndarray,
    target_close: np.ndarray,
    actual_intraday_returns: np.ndarray,
    cutoff: pd.Timestamp,
    one_way_cost_bps: float,
) -> tuple[pd.DataFrame, dict]:
    """Create daily trades and a summary for one strategy."""

    one_way_cost = one_way_cost_bps / 10_000.0
    round_trip_cost = 2.0 * one_way_cost

    if predicted_returns is None:
        # The always-long benchmark buys at every open and exits at every
        # close. The cash benchmark takes no exposure.
        if model_name == "always_long_intraday":
            positions = np.ones(len(frame), dtype=int)
        elif model_name == "cash":
            positions = np.zeros(len(frame), dtype=int)
        else:
            raise ValueError(f"Unknown benchmark: {model_name}")

        implied_open_close_returns = np.full(len(frame), np.nan)
        predicted_target_close = np.full(len(frame), np.nan)

    else:
        with np.errstate(over="raise", invalid="raise"):
            try:
                predicted_target_close = (
                    origin_close * np.exp(predicted_returns)
                )
            except FloatingPointError as exc:
                raise ValueError(
                    f"Price forecast overflow for {model_name}."
                ) from exc

        if not np.isfinite(predicted_target_close).all():
            raise ValueError(
                f"Non-finite predicted closing prices for {model_name}."
            )

        # At the next session's open, the opening price is observable.
        # Translate the prior-close forecast into an implied open-to-close
        # return before deciding whether its expected upside covers costs.
        implied_open_close_returns = (
            predicted_target_close / target_open
        ) - 1.0

        if not np.isfinite(implied_open_close_returns).all():
            raise ValueError(
                f"Non-finite implied open-to-close returns for {model_name}."
            )

        # Fixed rule: take a long position only if forecast upside exceeds
        # the assumed round-trip costs. No short selling is used.
        positions = (
            implied_open_close_returns > round_trip_cost
        ).astype(int)

    gross_portfolio_returns = (
        positions * actual_intraday_returns
    )

    transaction_costs = positions * round_trip_cost

    # Approximate net return deduction for the entry and exit costs.
    net_portfolio_returns = (
        gross_portfolio_returns - transaction_costs
    )

    daily = pd.DataFrame({
        "forecast_origin_date": frame["forecast_origin_date"].to_numpy(),
        "target_date": frame["target_date"].to_numpy(),
        "model": model_name,
        "predicted_log_return": (
            predicted_returns
            if predicted_returns is not None
            else np.full(len(frame), np.nan)
        ),
        "predicted_target_close": predicted_target_close,
        "target_open": target_open,
        "target_close": target_close,
        "forecast_implied_open_close_return_pct": (
            implied_open_close_returns * 100.0
        ),
        "position_long": positions,
        "actual_open_close_return_pct": actual_intraday_returns * 100.0,
        "gross_strategy_return_pct": gross_portfolio_returns * 100.0,
        "estimated_round_trip_cost_pct": transaction_costs * 100.0,
        "net_strategy_return_pct": net_portfolio_returns * 100.0,
        "one_way_cost_assumption_bps": one_way_cost_bps,
    })

    summary = summarize_returns(
        model_name=model_name,
        period="overall",
        year=None,
        dates=daily["target_date"],
        positions=positions,
        gross_returns=gross_portfolio_returns,
        net_returns=net_portfolio_returns,
        one_way_cost_bps=one_way_cost_bps,
    )

    return daily, summary


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Format a DataFrame as Markdown without optional dependencies."""

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
            elif isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            else:
                cells.append(str(value).replace("|", "\\|"))

        lines.append("| " + " | ".join(cells) + " |")

    return "\\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fixed-rule, exploratory intraday strategy backtest."
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help="Ticker symbol; default RELIANCE.NS",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_CUTOFF,
        help="Last permitted forecast target date; default 2022-06-14",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=DEFAULT_ONE_WAY_COST_BPS,
        help=(
            "Assumed all-in cost plus slippage per side in basis points; "
            "default 10.0. This is a scenario assumption, not a broker quote."
        ),
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

    if not np.isfinite(args.cost_bps) or args.cost_bps < 0:
        parser.error("--cost-bps must be a finite, non-negative number.")

    name = safe_ticker_name(ticker)

    # Reuse the project's integrity-checked historical OHLC loader.
    data, csv_path, metadata = load_data(ticker)
    data.attrs["ticker"] = ticker

    if not {"date", "open", "close"}.issubset(data.columns):
        raise ValueError("Historical data must include date, open and close.")

    price_history = (
        data.set_index("date")[["open", "close"]]
        .sort_index()
    )

    if price_history.index.duplicated().any():
        raise ValueError("Duplicate dates in the validated price history.")

    one_way_cost = args.cost_bps / 10_000.0
    round_trip_hurdle = 2.0 * one_way_cost

    # Read each distinct prediction report once.
    unique_prefixes = sorted({
        prefix for prefix, _ in MODEL_SPECS.values()
    })

    loaded: dict[str, pd.DataFrame] = {}

    for prefix in unique_prefixes:
        path = REPORTS_DIR / f"{prefix}_{name}.csv"

        prediction_columns = [
            column
            for pfx, column in MODEL_SPECS.values()
            if pfx == prefix
        ]

        # Require common fields plus all predictions needed from this file.
        frame = pd.read_csv(
            path,
            parse_dates=["forecast_origin_date", "target_date"],
        )

        required = [
            "forecast_origin_date",
            "target_date",
            "origin_close",
            "actual_next_close",
            "actual_log_return",
            *sorted(set(prediction_columns)),
        ]

        missing = [
            column for column in required if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {missing}"
            )

        frame["forecast_origin_date"] = pd.to_datetime(
            frame["forecast_origin_date"], errors="coerce"
        ).dt.normalize()
        frame["target_date"] = pd.to_datetime(
            frame["target_date"], errors="coerce"
        ).dt.normalize()

        if (
            frame["forecast_origin_date"].isna().any()
            or frame["target_date"].isna().any()
        ):
            raise ValueError(f"Invalid dates found in {path.name}.")

        if frame["target_date"].duplicated().any():
            raise ValueError(f"Duplicate target dates found in {path.name}.")

        if not frame["target_date"].is_monotonic_increasing:
            raise ValueError(f"{path.name} is not sorted by target date.")

        if (frame["forecast_origin_date"] >= frame["target_date"]).any():
            raise ValueError(
                f"Invalid forecast origin/target dates in {path.name}."
            )

        if (frame["target_date"] > cutoff).any():
            raise ValueError(
                f"{path.name} contains targets after cutoff {cutoff.date()}."
            )

        numeric_columns = [
            "origin_close",
            "actual_next_close",
            "actual_log_return",
            *sorted(set(prediction_columns)),
        ]

        for column in numeric_columns:
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce"
            )

        if not np.isfinite(
            frame[numeric_columns].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Missing/non-finite forecast values in {path.name}."
            )

        loaded[prefix] = frame.set_index("target_date", drop=False)

        print(
            f"Loaded {path.name}: {len(frame):,} rows; "
            f"through {frame['target_date'].max().date()}"
        )

    # Require all models to use the same target-date intersection.
    date_sets = [
        set(frame.index)
        for frame in loaded.values()
    ]

    common_dates = sorted(set.intersection(*date_sets))

    if len(common_dates) < 100:
        raise ValueError(
            f"Only {len(common_dates)} common dates are available."
        )

    common_index = pd.DatetimeIndex(common_dates)

    aligned = {
        prefix: frame.reindex(common_index)
        for prefix, frame in loaded.items()
    }

    reference = aligned["ridge_predictions"]

    # Validate that all files refer to the same realized observations.
    for prefix, frame in aligned.items():
        for column in [
            "origin_close",
            "actual_next_close",
            "actual_log_return",
        ]:
            verify_close_values(
                reference[column].to_numpy(dtype=float),
                frame[column].to_numpy(dtype=float),
                f"{prefix} {column}",
            )

    target_prices = price_history.reindex(common_index)

    if target_prices[["open", "close"]].isna().any().any():
        raise ValueError(
            "Historical OHLC prices are missing for a forecast target date."
        )

    target_open = target_prices["open"].to_numpy(dtype=float)
    target_close = target_prices["close"].to_numpy(dtype=float)

    if (
        not np.isfinite(target_open).all()
        or not np.isfinite(target_close).all()
        or (target_open <= 0).any()
        or (target_close <= 0).any()
    ):
        raise ValueError("Invalid target-date opening/closing prices.")

    # Verify the prices/returns in the forecasts against the validated file.
    verify_close_values(
        reference["actual_next_close"].to_numpy(dtype=float),
        target_close,
        "forecast actual_next_close versus price history",
    )

    origin_dates = pd.DatetimeIndex(
        reference["forecast_origin_date"].to_numpy()
    )
    origin_history = price_history.reindex(origin_dates)

    if origin_history["close"].isna().any():
        raise ValueError("A forecast origin is missing from price history.")

    verify_close_values(
        reference["origin_close"].to_numpy(dtype=float),
        origin_history["close"].to_numpy(dtype=float),
        "forecast origin_close versus price history",
    )

    actual_intraday_returns = target_close / target_open - 1.0

    if not np.isfinite(actual_intraday_returns).all():
        raise ValueError("Invalid actual intraday returns.")

    all_daily = []
    overall_rows = []

    for model_name, (prefix, prediction_column) in MODEL_SPECS.items():
        frame = aligned[prefix]
        predicted_returns = frame[prediction_column].to_numpy(dtype=float)

        daily, overall_summary = build_strategy_rows(
            model_name=model_name,
            frame=frame,
            predicted_returns=predicted_returns,
            origin_close=frame["origin_close"].to_numpy(dtype=float),
            target_open=target_open,
            target_close=target_close,
            actual_intraday_returns=actual_intraday_returns,
            cutoff=cutoff,
            one_way_cost_bps=args.cost_bps,
        )

        all_daily.append(daily)
        overall_rows.append(overall_summary)

    # Like-for-like intraday benchmarks on every common target date.
    for benchmark_name in ["always_long_intraday", "cash"]:
        daily, overall_summary = build_strategy_rows(
            model_name=benchmark_name,
            frame=reference,
            predicted_returns=None,
            origin_close=reference["origin_close"].to_numpy(dtype=float),
            target_open=target_open,
            target_close=target_close,
            actual_intraday_returns=actual_intraday_returns,
            cutoff=cutoff,
            one_way_cost_bps=args.cost_bps,
        )

        all_daily.append(daily)
        overall_rows.append(overall_summary)

    daily_results = pd.concat(all_daily, ignore_index=True)
    overall_metrics = pd.DataFrame(overall_rows)

    # Annual metrics use the same fixed rule, cost assumption and dates.
    annual_rows = []

    for year, year_frame in daily_results.groupby(
        daily_results["target_date"].dt.year,
        sort=True,
    ):
        for model_name, model_frame in year_frame.groupby("model", sort=False):
            positions = model_frame["position_long"].to_numpy(dtype=int)
            gross_returns = (
                model_frame["gross_strategy_return_pct"].to_numpy(dtype=float)
                / 100.0
            )
            net_returns = (
                model_frame["net_strategy_return_pct"].to_numpy(dtype=float)
                / 100.0
            )

            annual_rows.append(
                summarize_returns(
                    model_name=model_name,
                    period="annual",
                    year=int(year),
                    dates=model_frame["target_date"],
                    positions=positions,
                    gross_returns=gross_returns,
                    net_returns=net_returns,
                    one_way_cost_bps=args.cost_bps,
                )
            )

    annual_metrics = pd.DataFrame(annual_rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    daily_path = REPORTS_DIR / f"strategy_backtest_daily_{name}.csv"
    overall_path = REPORTS_DIR / f"strategy_backtest_metrics_{name}.csv"
    annual_path = REPORTS_DIR / f"strategy_backtest_annual_metrics_{name}.csv"
    summary_path = REPORTS_DIR / f"strategy_backtest_run_{name}.json"
    report_path = REPORTS_DIR / f"strategy_backtest_report_{name}.md"

    daily_results.to_csv(
        daily_path,
        index=False,
        date_format="%Y-%m-%d",
    )
    overall_metrics.to_csv(overall_path, index=False)
    annual_metrics.to_csv(annual_path, index=False)

    summary = {
        "project": "Stock Forecast & Backtesting Lab",
        "experiment": "exploratory_long_cash_intraday_strategy",
        "ticker": ticker,
        "research_start_date": common_dates[0].strftime("%Y-%m-%d"),
        "development_end_date": cutoff.strftime("%Y-%m-%d"),
        "first_common_target_date": common_dates[0].strftime("%Y-%m-%d"),
        "last_common_target_date": common_dates[-1].strftime("%Y-%m-%d"),
        "common_forecast_targets": int(len(common_dates)),
        "one_way_cost_bps_assumption": float(args.cost_bps),
        "round_trip_cost_bps_assumption": float(2 * args.cost_bps),
        "rule": (
            "At the target session open, use the previous-close model "
            "forecast to estimate the target closing price. Enter long "
            "only when forecast-implied upside from target open exceeds "
            "the assumed round-trip cost. Exit at target close; otherwise "
            "remain in cash."
        ),
        "execution_assumption": (
            "The strategy observes the opening price and enters near that "
            "price, with entry and exit slippage represented by the "
            "configurable per-side cost assumption. This is an approximation."
        ),
        "cost_method": (
            "A simple two-sided cost deduction is applied to each active "
            "intraday session. Costs are assumptions, not a verified "
            "brokerage schedule. More precise turnover and fee modelling "
            "would be required before relying on the results."
        ),
        "limitations": (
            "Exploratory development-period backtest, not independent "
            "evidence of profitability. Many models were tested during "
            "development. Do not tune strategy rules on the previously "
            "evaluated 2022-2026 period."
        ),
        "source_csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "source_csv_sha256": sha256_file(csv_path),
        "source_metadata_row_count": metadata["row_count"],
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    display_columns = [
        "model",
        "sessions",
        "active_days",
        "exposure_pct",
        "gross_total_return_pct",
        "net_total_return_pct",
        "annualized_net_return_pct",
        "annualized_net_volatility_pct",
        "net_sharpe_ratio_rf_zero",
        "max_net_drawdown_pct",
        "win_rate_active_days_pct",
    ]

    ordered_overall = overall_metrics[display_columns]

    markdown_lines = [
        f"# Exploratory Strategy Backtest — {ticker}",
        "",
        f"- Target-date cutoff: {cutoff.date()}",
        f"- Common forecast targets: {len(common_dates):,}",
        f"- Evaluation period: {common_dates[0].date()} to {common_dates[-1].date()}",
        f"- Assumed one-way all-in cost and slippage: {args.cost_bps:.2f} bps",
        f"- Assumed round-trip friction: {2 * args.cost_bps:.2f} bps per active day",
        "",
        "## Overall results",
        "",
        dataframe_to_markdown(ordered_overall),
        "",
        "## Fixed trading rule",
        "",
        "At each target-date open, calculate the model's forecast-implied "
        "close from the previous forecast origin. Go long for that session "
        "only if the implied open-to-close upside exceeds the assumed "
        "round-trip friction. Exit at the close; otherwise stay in cash.",
        "",
        "## Interpretation and limitations",
        "",
        "This is an exploratory in-sample strategy test on a development "
        "period already used to compare multiple forecasting models. Net "
        "returns use a simple assumed two-sided transaction-cost deduction. "
        "Annualization assumes 252 sessions and a zero risk-free rate for "
        "the displayed Sharpe ratio. These metrics do not establish "
        "profitability or account for all real trading frictions.",
        "",
        "The always-long intraday benchmark is exposed from every open to "
        "every close and pays the same assumed round-trip friction each "
        "session. It is not the same as conventional overnight buy-and-hold.",
        "",
        "Do not tune the strategy using the previously evaluated "
        "2022-06-15 to 2026-10-23 period. Any later test must be run only "
        "after fixing the rules and cost assumptions.",
        "",
    ]

    report_path.write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )

    print("\n===== STRATEGY BACKTEST COMPLETE =====")
    print(f"Ticker: {ticker}")
    print(f"Development cutoff: {cutoff.date()}")
    print(f"Common forecast targets: {len(common_dates):,}")
    print(
        f"Target period: {common_dates[0].date()} "
        f"to {common_dates[-1].date()}"
    )
    print(f"Assumed one-way cost: {args.cost_bps:.2f} bps")
    print(f"Assumed round-trip cost: {2 * args.cost_bps:.2f} bps")

    print("\n===== OVERALL STRATEGY COMPARISON =====")
    print(ordered_overall.to_string(index=False))

    print("\n===== OUTPUT FILES =====")
    print(f"Daily strategy details: {daily_path}")
    print(f"Overall metrics: {overall_path}")
    print(f"Annual metrics: {annual_path}")
    print(f"Run summary: {summary_path}")
    print(f"Markdown report: {report_path}")

    print(
        "\nExploratory historical results only. No live orders were placed, "
        "and profitability has not been established."
    )


if __name__ == "__main__":
    main()
