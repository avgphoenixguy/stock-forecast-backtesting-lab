
"""Generate publication-ready charts from saved forecasting experiments.

Outputs PNG charts into docs/assets/ for use in the public README.

Reads existing reports only. Does not download data, fit models, or trade.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"

DEFAULT_TICKER = "RELIANCE.NS"
DEFAULT_CUTOFF = "2022-06-14"

# Keep the cumulative plots readable by selecting representative strategies.
STRATEGIES_TO_PLOT = [
    "random_walk",
    "ridge_all_features",
    "ridge_returns_only",
    "random_forest",
    "decision_tree",
    "always_long_intraday",
    "cash",
]

STRATEGY_LABELS = {
    "random_walk": "Random Walk signal",
    "ridge_all_features": "Ridge — all features",
    "ridge_returns_only": "Ridge — returns only",
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
    "always_long_intraday": "Always-long intraday",
    "cash": "Cash",
    "buy_and_hold": "Buy-and-hold",
}


def safe_ticker_name(ticker: str) -> str:
    """Match the filename convention used by the project."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").upper()


def require_file(path: Path) -> None:
    """Ensure a required local report exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required report not found: {path}\n"
            "Run the corresponding experiment first."
        )


def load_model_comparison(
    path: Path,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Load and validate regression model comparison metrics."""
    require_file(path)
    frame = pd.read_csv(path)

    required = [
        "model",
        "forecast_count",
        "log_return_mae_bps",
        "log_return_rmse_bps",
    ]
    missing = [column for column in required if column not in frame.columns]

    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    if frame.empty:
        raise ValueError(f"{path.name} is empty.")

    for column in [
        "forecast_count",
        "log_return_mae_bps",
        "log_return_rmse_bps",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if not np.isfinite(
        frame[
            [
                "forecast_count",
                "log_return_mae_bps",
                "log_return_rmse_bps",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{path.name} contains non-finite metrics.")

    if (frame["forecast_count"] <= 0).any():
        raise ValueError("Forecast counts must be positive.")

    if (
        (frame["log_return_mae_bps"] < 0).any()
        or (frame["log_return_rmse_bps"] < 0).any()
    ):
        raise ValueError("Forecast errors cannot be negative.")

    return frame.sort_values("log_return_mae_bps").reset_index(drop=True)


def load_strategy_daily(
    path: Path,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Load the daily backtest output and validate its dates and returns."""
    require_file(path)
    frame = pd.read_csv(path)

    required = [
        "target_date",
        "model",
        "position_long",
        "target_open",
        "target_close",
        "net_strategy_return_pct",
        "gross_strategy_return_pct",
    ]

    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    if frame.empty:
        raise ValueError(f"{path.name} is empty.")

    frame["target_date"] = pd.to_datetime(
        frame["target_date"], errors="coerce"
    ).dt.normalize()

    if frame["target_date"].isna().any():
        raise ValueError("Invalid target dates in the daily strategy report.")

    if (frame["target_date"] > cutoff).any():
        raise ValueError(
            f"Strategy report contains targets past {cutoff.date()}."
        )

    for column in [
        "position_long",
        "target_open",
        "target_close",
        "net_strategy_return_pct",
        "gross_strategy_return_pct",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    numeric = frame[
        [
            "position_long",
            "target_open",
            "target_close",
            "net_strategy_return_pct",
            "gross_strategy_return_pct",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite values in daily strategy results.")

    if not frame["position_long"].isin([0, 1]).all():
        raise ValueError("Expected long-or-cash positions of 0 or 1.")

    if (frame[["target_open", "target_close"]] <= 0).any().any():
        raise ValueError("Non-positive open or close prices found.")

    if frame.duplicated(["target_date", "model"]).any():
        raise ValueError("Duplicate model/date rows in the backtest output.")

    return frame.sort_values(["target_date", "model"]).reset_index(drop=True)


def load_cost_sensitivity(path: Path) -> pd.DataFrame:
    """Load strategy net-return estimates for multiple cost assumptions."""
    require_file(path)
    frame = pd.read_csv(path)

    required = [
        "model",
        "one_way_cost_assumption_bps",
        "net_total_return_pct",
    ]
    missing = [column for column in required if column not in frame.columns]

    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    for column in [
        "one_way_cost_assumption_bps",
        "net_total_return_pct",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if not np.isfinite(
        frame[
            ["one_way_cost_assumption_bps", "net_total_return_pct"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite values in cost-sensitivity report.")

    return frame


def format_model_label(model: str) -> str:
    """Make model identifiers easier to read on plots."""
    labels = {
        "random_walk": "Random Walk",
        "drift": "Random Walk with Drift",
        "ridge": "Ridge — all features",
        "ridge_returns_only": "Ridge — returns only",
        "ridge_trend_only": "Ridge — trend only",
        "ridge_volatility_only": "Ridge — volatility only",
        "ridge_all_features": "Ridge — all features",
        "random_forest": "Random Forest",
        "decision_tree": "Decision Tree",
    }
    return labels.get(model, model.replace("_", " ").title())


def save_return_mae_chart(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot overall next-day log-return MAE across regression models."""
    chart = metrics.sort_values(
        "log_return_mae_bps", ascending=True
    ).copy()

    labels = [
        format_model_label(model)
        for model in chart["model"].astype(str)
    ]
    values = chart["log_return_mae_bps"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_positions = np.arange(len(chart))

    ax.barh(y_positions, values)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute log-return forecast error (bps)")
    ax.set_title("Next-Day Forecast Error: Regression Models")
    ax.grid(axis="x", alpha=0.3)

    for position, value in zip(y_positions, values):
        ax.text(
            value + max(values) * 0.005,
            position,
            f"{value:.2f}",
            va="center",
            fontsize=8,
        )

    ax.text(
        0.0,
        -0.14,
        "Common development targets: 2008-01-17 to 2022-06-14. "
        "Lower is better; this is not a trading-return metric.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compute_buy_and_hold_net_returns(
    unique_days: pd.DataFrame,
    one_way_cost_bps: float,
) -> np.ndarray:
    """Compute buy-at-first-open, hold-through-period, sell-at-last-close."""
    opens = unique_days["target_open"].to_numpy(dtype=float)
    closes = unique_days["target_close"].to_numpy(dtype=float)

    if len(opens) < 2:
        raise ValueError("At least two dates are needed for buy-and-hold.")

    daily_returns = np.empty(len(opens), dtype=float)
    daily_returns[0] = closes[0] / opens[0] - 1.0
    daily_returns[1:] = closes[1:] / closes[:-1] - 1.0

    cost = one_way_cost_bps / 10_000.0
    daily_returns[0] -= cost
    daily_returns[-1] -= cost

    if (daily_returns <= -1).any():
        raise ValueError("Buy-and-hold contains a return of -100% or less.")

    return daily_returns


def save_cumulative_return_chart(
    daily: pd.DataFrame,
    cutoff: pd.Timestamp,
    one_way_cost_bps: float,
    output_path: Path,
) -> None:
    """Plot cumulative net returns for representative intraday strategies."""
    unique_days = (
        daily.drop_duplicates("target_date")
        .sort_values("target_date")
        .reset_index(drop=True)
    )
    dates = pd.DatetimeIndex(unique_days["target_date"])

    fig, ax = plt.subplots(figsize=(11, 6))

    for model in STRATEGIES_TO_PLOT:
        group = (
            daily.loc[daily["model"] == model]
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        if group.empty:
            raise ValueError(
                f"Could not find model '{model}' in daily strategy results."
            )

        if not group["target_date"].reset_index(drop=True).equals(
            pd.Series(dates, name="target_date")
        ):
            # Verify each model has precisely the same dates.
            if list(group["target_date"]) != list(dates):
                raise ValueError(
                    f"{model} does not share the common strategy dates."
                )

        returns = (
            group["net_strategy_return_pct"].to_numpy(dtype=float) / 100.0
        )
        wealth = np.cumprod(1.0 + returns)
        cumulative_pct = (wealth - 1.0) * 100.0

        ax.plot(
            dates,
            cumulative_pct,
            label=STRATEGY_LABELS.get(model, format_model_label(model)),
            linewidth=1.4,
        )

    buy_hold_returns = compute_buy_and_hold_net_returns(
        unique_days, one_way_cost_bps
    )
    buy_hold_wealth = np.cumprod(1.0 + buy_hold_returns)
    buy_hold_pct = (buy_hold_wealth - 1.0) * 100.0

    ax.plot(
        dates,
        buy_hold_pct,
        label="Buy-and-hold",
        linewidth=2.0,
    )

    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Target date")
    ax.set_ylabel("Cumulative net return (%)")
    ax.set_title(
        f"Strategy Cumulative Returns — {ticker_label(daily)}\n"
        f"Assumed {one_way_cost_bps:g} bps per side"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=30)

    ax.text(
        0.0,
        -0.20,
        f"Development target-date cutoff: {cutoff.date()}. "
        "Strategy curves use saved daily net returns; buy-and-hold "
        "uses one entry and one exit cost. Historical results are not "
        "evidence of future profitability.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def ticker_label(daily: pd.DataFrame) -> str:
    """Return a concise chart title from the project's saved experiment."""
    return "RELIANCE.NS" if not daily.empty else "Stock"


def save_drawdown_chart(
    daily: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot drawdowns for selected forecast-driven strategies."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for model in STRATEGIES_TO_PLOT:
        if model in {"buy_and_hold"}:
            continue

        group = (
            daily.loc[daily["model"] == model]
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        if group.empty:
            raise ValueError(f"Missing strategy model: {model}")

        returns = (
            group["net_strategy_return_pct"].to_numpy(dtype=float) / 100.0
        )
        wealth = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
        high_water = np.maximum.accumulate(wealth)
        drawdowns = wealth / high_water - 1.0

        ax.plot(
            pd.to_datetime(group["target_date"]),
            drawdowns[1:] * 100.0,
            label=STRATEGY_LABELS.get(model, format_model_label(model)),
            linewidth=1.2,
        )

    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Target date")
    ax.set_ylabel("Drawdown from prior equity peak (%)")
    ax.set_title("Historical Strategy Drawdowns")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=30)

    ax.text(
        0.0,
        -0.15,
        "Drawdowns are calculated from compounded daily net strategy "
        "returns. These are exploratory development-period results.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_cost_sensitivity_chart(
    costs: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot net cumulative return across assumed cost scenarios."""
    selected = [
        "buy_and_hold",
        "cash",
        "random_walk",
        "ridge_all_features",
        "random_forest",
        "decision_tree",
        "always_long_intraday",
    ]
    available = set(costs["model"].astype(str))

    missing = [model for model in selected if model not in available]
    if missing:
        raise ValueError(
            f"Cost sensitivity report is missing models: {missing}"
        )

    chart = costs.loc[costs["model"].isin(selected)].copy()

    pivot = chart.pivot(
        index="one_way_cost_assumption_bps",
        columns="model",
        values="net_total_return_pct",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    for model in selected:
        ax.plot(
            pivot.index.to_numpy(dtype=float),
            pivot[model].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
            label=STRATEGY_LABELS.get(model, format_model_label(model)),
        )

    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Assumed cost per side (bps)")
    ax.set_ylabel("Net cumulative return (%)")
    ax.set_title("Sensitivity to Assumed Trading Costs")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax.text(
        0.0,
        -0.15,
        "Cost levels are illustrative assumptions, not verified brokerage "
        "charges. Net return is calculated using the project's simplified "
        "cost model.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create charts from existing research reports."
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help="Ticker symbol; default RELIANCE.NS",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_CUTOFF,
        help="Latest permitted target date; default 2022-06-14",
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

    metrics_path = REPORTS_DIR / f"model_comparison_overall_{name}.csv"
    daily_path = REPORTS_DIR / f"strategy_backtest_daily_{name}.csv"
    costs_path = REPORTS_DIR / f"strategy_cost_sensitivity_{name}.csv"
    summary_path = REPORTS_DIR / f"strategy_backtest_run_{name}.json"

    metrics = load_model_comparison(metrics_path, cutoff)
    daily = load_strategy_daily(daily_path, cutoff)
    costs = load_cost_sensitivity(costs_path)

    require_file(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    base_cost = float(summary["one_way_cost_bps_assumption"])

    dates = pd.DatetimeIndex(daily["target_date"].unique())
    if len(dates) < 2:
        raise ValueError("At least two strategy target dates are required.")

    if dates.max() > cutoff:
        raise ValueError("Strategy dates exceed the requested cutoff.")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    output_paths = [
        ASSETS_DIR / "model_return_mae.png",
        ASSETS_DIR / "strategy_cumulative_returns.png",
        ASSETS_DIR / "strategy_drawdowns.png",
        ASSETS_DIR / "strategy_cost_sensitivity.png",
    ]

    save_return_mae_chart(metrics, output_paths[0])
    save_cumulative_return_chart(
        daily, cutoff, base_cost, output_paths[1]
    )
    save_drawdown_chart(daily, output_paths[2])
    save_cost_sensitivity_chart(costs, output_paths[3])

    print("\n===== RESEARCH CHARTS CREATED =====")
    print(f"Ticker: {ticker}")
    print(f"Development cutoff: {cutoff.date()}")
    print(f"Model rows plotted: {len(metrics)}")
    print(f"Strategy target dates: {len(dates):,}")
    print(f"Strategy cost assumption: {base_cost:g} bps per side")

    print("\n===== OUTPUT FILES =====")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
