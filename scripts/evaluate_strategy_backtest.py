
"""Validate the exploratory strategy backtest.

Adds a conventional buy-and-hold benchmark and calculates cost sensitivity
for 0, 5, 10, and 20 basis points per side.

Reads saved backtest results only. No model fitting, downloads, or trades.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_TICKER = "RELIANCE.NS"
DEFAULT_CUTOFF = "2022-06-14"
COST_SCENARIOS_BPS = [0.0, 5.0, 10.0, 20.0]
TRADING_SESSIONS_PER_YEAR = 252


def safe_ticker_name(ticker: str) -> str:
    """Return the filename convention used by the project."""
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").upper()


def load_daily_results(
    path: Path,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Load and validate the saved daily strategy results."""

    if not path.exists():
        raise FileNotFoundError(
            f"Backtest results not found: {path}\n"
            "Run scripts/run_strategy_backtest.py first."
        )

    frame = pd.read_csv(path)
    required = [
        "target_date",
        "model",
        "position_long",
        "target_open",
        "target_close",
        "gross_strategy_return_pct",
    ]

    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if frame.empty:
        raise ValueError("The daily strategy report is empty.")

    frame["target_date"] = pd.to_datetime(
        frame["target_date"], errors="coerce"
    ).dt.normalize()

    if frame["target_date"].isna().any():
        raise ValueError("Invalid target dates found.")

    if (frame["target_date"] > cutoff).any():
        raise ValueError(
            f"Targets after the cutoff {cutoff.date()} were found."
        )

    for column in [
        "position_long",
        "target_open",
        "target_close",
        "gross_strategy_return_pct",
    ]:
        frame[column] = pd.to_numeric(
            frame[column], errors="coerce"
        )

    numeric = frame[
        [
            "position_long",
            "target_open",
            "target_close",
            "gross_strategy_return_pct",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError("Missing or non-finite values were found.")

    if not frame["position_long"].isin([0, 1]).all():
        raise ValueError("position_long must contain only 0 or 1.")

    if (frame[["target_open", "target_close"]] <= 0).any().any():
        raise ValueError("Non-positive opening or closing prices found.")

    if frame.duplicated(["target_date", "model"]).any():
        raise ValueError("Duplicate model/date rows were found.")

    frame = frame.sort_values(
        ["target_date", "model"]
    ).reset_index(drop=True)

    dates_by_model = {
        model: tuple(group["target_date"])
        for model, group in frame.groupby("model", sort=True)
    }

    date_sequences = list(dates_by_model.values())
    if not date_sequences or any(
        dates != date_sequences[0] for dates in date_sequences[1:]
    ):
        raise ValueError(
            "Models do not share exactly the same target-date sequence."
        )

    # Each target date must have identical observed OHLC values across models.
    for column in ["target_open", "target_close"]:
        spread = frame.groupby("target_date")[column].agg(
            lambda values: float(values.max() - values.min())
        )

        if (spread.abs() > 1e-8).any():
            raise ValueError(
                f"Inconsistent {column} values across model rows."
            )

    return frame


def build_buy_and_hold(
    dates: pd.DatetimeIndex,
    open_prices: np.ndarray,
    close_prices: np.ndarray,
) -> pd.DataFrame:
    """Buy at the first session open and hold until the final close.

    Daily return 1: first close / first open - 1.
    Later daily returns: current close / previous close - 1.
    """

    if len(dates) < 2:
        raise ValueError("At least two sessions are required.")

    gross_returns = np.empty(len(dates), dtype=float)
    gross_returns[0] = close_prices[0] / open_prices[0] - 1.0
    gross_returns[1:] = close_prices[1:] / close_prices[:-1] - 1.0

    return pd.DataFrame({
        "target_date": dates,
        "model": "buy_and_hold",
        "position_long": np.ones(len(dates), dtype=int),
        "gross_return": gross_returns,
        "round_trips": np.zeros(len(dates), dtype=int),
        "entry_or_exit_cost_events": np.zeros(len(dates), dtype=int),
    })


def metrics_for_series(
    model: str,
    dates: pd.Series | pd.DatetimeIndex,
    gross_returns: np.ndarray,
    net_returns: np.ndarray,
    positions: np.ndarray,
    one_way_cost_bps: float,
    round_trips: int,
) -> dict:
    """Calculate total return, risk, exposure, and cost statistics."""

    # Normalize Series and DatetimeIndex inputs to ordinary positional lists.
    date_values = list(dates)

    if len(net_returns) == 0:
        raise ValueError("Cannot calculate metrics for an empty series.")

    if not (
        np.isfinite(gross_returns).all()
        and np.isfinite(net_returns).all()
    ):
        raise ValueError(f"Non-finite portfolio returns for {model}.")

    if (net_returns <= -1).any():
        raise ValueError(f"Daily loss of 100% or more for {model}.")

    gross_equity = np.cumprod(1.0 + gross_returns)
    net_equity = np.cumprod(1.0 + net_returns)

    gross_total = float(gross_equity[-1] - 1.0)
    net_total = float(net_equity[-1] - 1.0)

    sessions = len(net_returns)
    years = sessions / TRADING_SESSIONS_PER_YEAR

    annualized_net = (
        float(net_equity[-1] ** (1.0 / years) - 1.0)
        if net_equity[-1] > 0
        else float("nan")
    )

    daily_std = (
        float(np.std(net_returns, ddof=1))
        if sessions > 1
        else 0.0
    )

    annualized_vol = daily_std * np.sqrt(TRADING_SESSIONS_PER_YEAR)

    sharpe = (
        float(
            np.mean(net_returns)
            / daily_std
            * np.sqrt(TRADING_SESSIONS_PER_YEAR)
        )
        if daily_std > 0
        else None
    )

    wealth = np.concatenate(([1.0], net_equity))
    high_water_mark = np.maximum.accumulate(wealth)
    drawdowns = wealth / high_water_mark - 1.0

    active = positions.astype(bool)
    active_count = int(active.sum())

    return {
        "model": model,
        "sessions": sessions,
        "active_days": active_count,
        "exposure_pct": float(active.mean() * 100.0),
        "gross_total_return_pct": gross_total * 100.0,
        "net_total_return_pct": net_total * 100.0,
        "annualized_net_return_pct": annualized_net * 100.0,
        "annualized_net_volatility_pct": annualized_vol * 100.0,
        "net_sharpe_ratio_rf_zero": sharpe,
        "max_net_drawdown_pct": float(np.min(drawdowns) * 100.0),
        "win_rate_active_days_pct": (
            float(np.mean(net_returns[active] > 0) * 100.0)
            if active_count
            else None
        ),
        "one_way_cost_assumption_bps": one_way_cost_bps,
        "round_trip_count": round_trips,
        "estimated_total_cost_bps": round_trips * 2.0 * one_way_cost_bps,
        "first_date": pd.Timestamp(date_values[0]).strftime("%Y-%m-%d"),
        "last_date": pd.Timestamp(date_values[-1]).strftime("%Y-%m-%d"),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    """Format a dataframe as Markdown without optional packages."""

    columns = [str(column) for column in frame.columns]
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
                cells.append(str(value).replace("|", "\\|"))

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add buy-and-hold and cost-sensitivity comparisons."
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help="Ticker; default RELIANCE.NS",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_CUTOFF,
        help="Development cutoff; default 2022-06-14",
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
    daily_path = REPORTS_DIR / f"strategy_backtest_daily_{name}.csv"

    frame = load_daily_results(daily_path, cutoff)

    date_prices = (
        frame.groupby("target_date", sort=True)
        .agg(
            target_open=("target_open", "first"),
            target_close=("target_close", "first"),
        )
    )

    dates = pd.DatetimeIndex(date_prices.index)
    open_prices = date_prices["target_open"].to_numpy(dtype=float)
    close_prices = date_prices["target_close"].to_numpy(dtype=float)

    # One conventional buy-and-hold position: buy at the first open,
    # remain invested through all overnight gaps, exit at the last close.
    buy_hold = build_buy_and_hold(dates, open_prices, close_prices)

    result_rows = []

    for cost_bps in COST_SCENARIOS_BPS:
        cost = cost_bps / 10_000.0

        for model, group in frame.groupby("model", sort=True):
            group = group.sort_values("target_date")
            positions = group["position_long"].to_numpy(dtype=int)
            gross_returns = (
                group["gross_strategy_return_pct"].to_numpy(dtype=float)
                / 100.0
            )

            # These strategies enter and exit within every active session.
            round_trips = int(positions.sum())
            net_returns = gross_returns - positions * (2.0 * cost)

            result_rows.append(
                metrics_for_series(
                    model=model,
                    dates=group["target_date"].reset_index(drop=True),
                    gross_returns=gross_returns,
                    net_returns=net_returns,
                    positions=positions,
                    one_way_cost_bps=cost_bps,
                    round_trips=round_trips,
                )
            )

        # Buy-and-hold incurs just one entry and one exit cost over the
        # entire period. Do not charge it a round trip every session.
        buy_hold_gross = buy_hold["gross_return"].to_numpy(dtype=float)
        buy_hold_net = buy_hold_gross.copy()

        buy_hold_net[0] -= cost
        buy_hold_net[-1] -= cost

        result_rows.append(
            metrics_for_series(
                model="buy_and_hold",
                dates=dates,
                gross_returns=buy_hold_gross,
                net_returns=buy_hold_net,
                positions=np.ones(len(dates), dtype=int),
                one_way_cost_bps=cost_bps,
                round_trips=1,
            )
        )

    sensitivity = pd.DataFrame(result_rows)

    overall_path = (
        REPORTS_DIR / f"strategy_cost_sensitivity_{name}.csv"
    )
    comparison_path = (
        REPORTS_DIR / f"strategy_cost_comparison_{name}.csv"
    )
    report_path = (
        REPORTS_DIR / f"strategy_validation_report_{name}.md"
    )

    sensitivity.to_csv(overall_path, index=False)

    # Use 10 bps per side as the primary scenario, matching the original run.
    primary = sensitivity.loc[
        sensitivity["one_way_cost_assumption_bps"] == 10.0
    ].sort_values("net_total_return_pct", ascending=False)

    primary.to_csv(comparison_path, index=False)

    sensitivity_by_model = (
        sensitivity.pivot(
            index="model",
            columns="one_way_cost_assumption_bps",
            values="net_total_return_pct",
        )
        .rename_axis(columns="one_way_cost_bps")
        .reset_index()
    )

    report = [
        f"# Strategy Backtest Validation — {ticker}",
        "",
        f"- Development cutoff: {cutoff.date()}",
        f"- Sessions: {len(dates):,}",
        f"- Evaluation period: {dates[0].date()} to {dates[-1].date()}",
        "- Cost scenarios: 0, 5, 10, and 20 basis points per side.",
        "",
        "## Primary scenario: 10 bps per side",
        "",
        markdown_table(
            primary[
                [
                    "model",
                    "sessions",
                    "active_days",
                    "exposure_pct",
                    "gross_total_return_pct",
                    "net_total_return_pct",
                    "annualized_net_return_pct",
                    "net_sharpe_ratio_rf_zero",
                    "max_net_drawdown_pct",
                    "round_trip_count",
                    "estimated_total_cost_bps",
                ]
            ].round(4)
        ),
        "",
        "## Net total return by cost scenario",
        "",
        markdown_table(sensitivity_by_model.round(4)),
        "",
        "## Buy-and-hold definition",
        "",
        "Buy at the first evaluation session's open, remain invested through "
        "overnight gaps and all subsequent sessions, and exit at the last "
        "session's close. It incurs one entry and one exit cost over the "
        "entire period. Its daily return series includes the first "
        "open-to-close return and subsequent close-to-close returns.",
        "",
        "## Limitations",
        "",
        "This is a development-period comparison after multiple models and "
        "rules have already been examined. Transaction costs are scenario "
        "assumptions, not a verified brokerage schedule. The daily "
        "cost deduction is approximate and does not model every Indian "
        "market fee, tax, liquidity effect, or market impact.",
        "",
        "These results are exploratory, not independent proof of "
        "profitability. The previously evaluated 2022-06-15 to 2026-10-23 "
        "period must not be reused for tuning.",
        "",
    ]

    report_path.write_text("\n".join(report), encoding="utf-8")

    display_columns = [
        "model",
        "active_days",
        "exposure_pct",
        "gross_total_return_pct",
        "net_total_return_pct",
        "annualized_net_return_pct",
        "net_sharpe_ratio_rf_zero",
        "max_net_drawdown_pct",
        "round_trip_count",
    ]

    print("\n===== STRATEGY VALIDATION COMPLETE =====")
    print(f"Ticker: {ticker}")
    print(f"Cutoff: {cutoff.date()}")
    print(f"Common sessions: {len(dates):,}")
    print(
        f"Period: {dates[0].date()} to {dates[-1].date()}"
    )

    print("\n===== RESULTS AT 10 BPS PER SIDE =====")
    print(primary[display_columns].to_string(index=False))

    print("\n===== NET TOTAL RETURN BY COST SCENARIO =====")
    print(sensitivity_by_model.to_string(index=False))

    print("\n===== OUTPUT FILES =====")
    print(f"All cost scenarios: {overall_path}")
    print(f"Primary comparison: {comparison_path}")
    print(f"Validation report: {report_path}")

    print(
        "\nExploratory historical analysis only; no live orders were placed."
    )


if __name__ == "__main__":
    main()
