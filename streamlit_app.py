
"""Interactive dashboard for Stock Forecast & Backtesting Lab.

Reads previously generated local experiment reports.
No models are trained, no data is downloaded, and no trades are placed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "dashboard" / "data"

DEFAULT_TICKER = "RELIANCE.NS"
COST_SCENARIOS = [0.0, 5.0, 10.0, 20.0]

STRATEGY_LABELS = {
    "random_walk": "Random Walk signal",
    "drift": "Random Walk with Drift",
    "ridge": "Ridge — all features",
    "ridge_all_features": "Ridge — all features",
    "ridge_returns_only": "Ridge — returns only",
    "ridge_trend_only": "Ridge — trend only",
    "ridge_volatility_only": "Ridge — volatility only",
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
    "always_long_intraday": "Always-long intraday",
    "cash": "Cash",
    "buy_and_hold": "Buy-and-hold",
}

STRATEGIES_FOR_CURVE = [
    "random_walk",
    "ridge_all_features",
    "ridge_returns_only",
    "random_forest",
    "decision_tree",
    "always_long_intraday",
    "cash",
]

REQUIRED_REPORTS = [
    "model_comparison_overall",
    "model_comparison_classifier",
    "strategy_backtest_daily",
    "strategy_cost_sensitivity",
]


def safe_ticker_name(ticker: str) -> str:
    """Use the project's report filename convention."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").upper()


def report_path(prefix: str, ticker: str) -> Path:
    """Return the local report path for a given ticker."""
    return REPORTS_DIR / f"{prefix}_{safe_ticker_name(ticker)}.csv"


def read_report(prefix: str, ticker: str) -> pd.DataFrame:
    """Load one report from the local reports folder."""
    path = report_path(prefix, ticker)

    if not path.exists():
        raise FileNotFoundError(
            f"Report not found: {path}"
        )

    return pd.read_csv(path)


def read_strategy_daily(ticker: str) -> pd.DataFrame:
    """Load and validate daily backtest observations."""
    frame = read_report("strategy_backtest_daily", ticker)

    required = [
        "target_date",
        "model",
        "position_long",
        "target_open",
        "target_close",
        "gross_strategy_return_pct",
        "net_strategy_return_pct",
    ]

    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Daily backtest report is missing columns: {missing}"
        )

    frame["target_date"] = pd.to_datetime(
        frame["target_date"], errors="coerce"
    )

    if frame["target_date"].isna().any():
        raise ValueError("Invalid target dates in the backtest report.")

    numeric_columns = [
        "position_long",
        "target_open",
        "target_close",
        "gross_strategy_return_pct",
        "net_strategy_return_pct",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column], errors="coerce"
        )

    if not np.isfinite(
        frame[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Missing or non-finite values in backtest report.")

    if not frame["position_long"].isin([0, 1]).all():
        raise ValueError("Expected long-or-cash positions of 0 or 1.")

    if frame.duplicated(["target_date", "model"]).any():
        raise ValueError("Duplicate model/target-date rows found.")

    return frame.sort_values(
        ["target_date", "model"]
    ).reset_index(drop=True)


def net_return_at_cost(
    daily: pd.DataFrame,
    one_way_cost_bps: float,
) -> np.ndarray:
    """Recalculate approximate net returns at a selected per-side cost."""
    gross = (
        daily["gross_strategy_return_pct"].to_numpy(dtype=float) / 100.0
    )
    positions = daily["position_long"].to_numpy(dtype=float)

    round_trip_cost = 2.0 * one_way_cost_bps / 10_000.0

    return gross - positions * round_trip_cost


def buy_and_hold_returns(
    dates: pd.DatetimeIndex,
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    one_way_cost_bps: float,
) -> np.ndarray:
    """Calculate buy-at-first-open and sell-at-last-close returns."""
    if len(dates) < 2:
        raise ValueError("At least two sessions are needed.")

    returns = np.empty(len(dates), dtype=float)

    # First day: buy at the opening price and mark to the closing price.
    returns[0] = close_prices[0] / open_prices[0] - 1.0

    # Subsequent days include overnight and intraday price changes.
    returns[1:] = close_prices[1:] / close_prices[:-1] - 1.0

    # One entry fee and one exit fee over the entire holding period.
    cost = one_way_cost_bps / 10_000.0
    returns[0] -= cost
    returns[-1] -= cost

    return returns


def build_equity_curves(
    daily: pd.DataFrame,
    one_way_cost_bps: float,
    selected_models: list[str],
) -> pd.DataFrame:
    """Construct cumulative net returns at the selected cost assumption."""
    dates = pd.DatetimeIndex(
        sorted(daily["target_date"].drop_duplicates())
    )

    price_rows = (
        daily.drop_duplicates("target_date")
        .set_index("target_date")
        .reindex(dates)
    )

    result_rows = []

    for model in selected_models:
        group = (
            daily.loc[daily["model"] == model]
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        if group.empty:
            continue

        if list(group["target_date"]) != list(dates):
            st.warning(
                f"Skipping {STRATEGY_LABELS.get(model, model)}: "
                "it does not have the same date sequence as other models."
            )
            continue

        returns = net_return_at_cost(group, one_way_cost_bps)

        if (returns <= -1).any():
            st.warning(
                f"Skipping {STRATEGY_LABELS.get(model, model)}: "
                "a daily loss of 100% or more was found."
            )
            continue

        equity = np.cumprod(1.0 + returns)

        result_rows.extend(
            {
                "date": date,
                "strategy": STRATEGY_LABELS.get(model, model),
                "cumulative_return_pct": (value - 1.0) * 100.0,
            }
            for date, value in zip(dates, equity)
        )

    # Conventional buy-and-hold uses one entry and one exit fee.
    if "buy_and_hold" in selected_models:
        opens = price_rows["target_open"].to_numpy(dtype=float)
        closes = price_rows["target_close"].to_numpy(dtype=float)

        hold_returns = buy_and_hold_returns(
            dates, opens, closes, one_way_cost_bps
        )
        hold_equity = np.cumprod(1.0 + hold_returns)

        result_rows.extend(
            {
                "date": date,
                "strategy": "Buy-and-hold",
                "cumulative_return_pct": (value - 1.0) * 100.0,
            }
            for date, value in zip(dates, hold_equity)
        )

    return pd.DataFrame(result_rows)


def build_drawdown_curves(equity_curves: pd.DataFrame) -> pd.DataFrame:
    """Calculate drawdowns from the cumulative equity curve."""
    result = []

    for model, group in equity_curves.groupby("strategy", sort=False):
        group = group.sort_values("date")
        wealth = 1.0 + group["cumulative_return_pct"].to_numpy() / 100.0

        high_water_mark = np.maximum.accumulate(
            np.concatenate(([1.0], wealth))
        )[1:]

        drawdown = wealth / high_water_mark - 1.0

        result.extend(
            {
                "date": date,
                "strategy": model,
                "drawdown_pct": value * 100.0,
            }
            for date, value in zip(group["date"], drawdown)
        )

    return pd.DataFrame(result)


def format_model_label(model: str) -> str:
    """Make experiment identifiers readable."""
    return STRATEGY_LABELS.get(model, model.replace("_", " ").title())


def load_all_reports(ticker: str) -> dict[str, object]:
    """Load the reports needed by the dashboard."""
    data = {
        "regression": read_report("model_comparison_overall", ticker),
        "classifier": read_report("model_comparison_classifier", ticker),
        "daily": read_strategy_daily(ticker),
        "costs": read_report("strategy_cost_sensitivity", ticker),
    }

    sensitivity = data["costs"]
    expected = [
        "model",
        "one_way_cost_assumption_bps",
        "net_total_return_pct",
        "annualized_net_return_pct",
        "max_net_drawdown_pct",
        "net_sharpe_ratio_rf_zero",
    ]

    missing = [column for column in expected if column not in sensitivity.columns]
    if missing:
        raise ValueError(
            f"Cost-sensitivity report is missing columns: {missing}"
        )

    for column in [
        "one_way_cost_assumption_bps",
        "net_total_return_pct",
        "annualized_net_return_pct",
        "max_net_drawdown_pct",
        "net_sharpe_ratio_rf_zero",
    ]:
        sensitivity[column] = pd.to_numeric(
            sensitivity[column], errors="coerce"
        )

    data["costs"] = sensitivity
    return data


def show_overview(data: dict[str, object], cost_bps: float) -> None:
    """Render the main research summary."""
    regression = data["regression"].copy()
    costs = data["costs"].copy()

    best = regression.loc[
        regression["log_return_mae_bps"].idxmin()
    ]

    random_walk = regression.loc[
        regression["model"] == "random_walk"
    ]

    st.title("Stock Forecast & Backtesting Lab")
    st.write(
        "An interactive view of historical forecasting experiments and "
        "an exploratory trading-strategy backtest for Reliance Industries."
    )
    st.info(
        "Use the left sidebar to navigate between sections. The Strategy "
        "backtest page contains interactive cumulative-return and drawdown "
        "charts; the cost selector in the sidebar updates the results."
    )

    st.warning(
        "Research only — the tested machine-learning models did not beat "
        "Random Walk on overall return forecast errors. The tested "
        "forecast-driven trading strategies were unsuccessful. "
        "These results do not establish profitability."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Forecast targets",
        f"{int(regression['forecast_count'].max()):,}",
    )
    c2.metric(
        "Regression models",
        f"{len(regression):,}",
    )
    c3.metric(
        "Lowest return MAE",
        format_model_label(str(best["model"])),
        help=f"{best['log_return_mae_bps']:.2f} basis points",
    )

    st.subheader("Regression forecast errors")

    plot_data = regression.sort_values(
        "log_return_mae_bps", ascending=True
    ).copy()
    plot_data["display_model"] = plot_data["model"].map(format_model_label)

    fig = px.bar(
        plot_data,
        x="log_return_mae_bps",
        y="display_model",
        orientation="h",
        labels={
            "log_return_mae_bps": "Return MAE (basis points)",
            "display_model": "Model",
        },
        title="Next-Day Log-Return Forecast Error",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

    if not random_walk.empty:
        rw_error = float(random_walk.iloc[0]["log_return_mae_bps"])
        st.caption(
            f"Random Walk return MAE: {rw_error:.2f} bps. "
            "Lower forecast error is better."
        )

    st.subheader(f"Strategy results at {cost_bps:g} bps per side")

    primary = costs.loc[
        costs["one_way_cost_assumption_bps"] == cost_bps
    ].copy()

    if not primary.empty:
        primary["display_model"] = primary["model"].map(format_model_label)
        primary = primary.sort_values("net_total_return_pct", ascending=False)

        fig = px.bar(
            primary,
            x="net_total_return_pct",
            y="display_model",
            orientation="h",
            labels={
                "net_total_return_pct": "Net cumulative return (%)",
                "display_model": "Strategy",
            },
            title=f"Historical Net Cumulative Returns at {cost_bps:g} bps per side",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "The buy-and-hold benchmark and the intraday strategies have "
            "different exposures. Transaction costs are scenario assumptions."
        )


def show_forecast_comparison(data: dict[str, object]) -> None:
    """Render regression and classification results separately."""
    st.title("Forecast Comparison")

    st.subheader("Regression models")

    regression = data["regression"].copy()
    regression["display_model"] = regression["model"].map(format_model_label)
    regression = regression.sort_values("log_return_mae_bps")

    columns = [
        "display_model",
        "forecast_count",
        "close_mae",
        "close_rmse",
        "log_return_mae_bps",
        "log_return_rmse_bps",
        "directional_accuracy_pct",
    ]

    st.dataframe(
        regression[columns].rename(
            columns={
                "display_model": "Model",
                "forecast_count": "Forecasts",
                "close_mae": "Close MAE",
                "close_rmse": "Close RMSE",
                "log_return_mae_bps": "Return MAE (bps)",
                "log_return_rmse_bps": "Return RMSE (bps)",
                "directional_accuracy_pct": "Direction accuracy (%)",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "MAE and RMSE measure forecast errors, not investment returns. "
        "The Random Walk benchmark's directional accuracy is undefined "
        "because it always predicts zero return."
    )

    st.subheader("Logistic Regression classification")

    classifier = data["classifier"].copy()

    desired = [
        "model",
        "forecast_count",
        "actual_up_rate_pct",
        "accuracy_pct",
        "balanced_accuracy_pct",
        "roc_auc",
        "brier_score",
        "log_loss",
    ]
    available = [column for column in desired if column in classifier.columns]

    classifier["display_model"] = classifier["model"].map(format_model_label)

    rename = {
        "display_model": "Model",
        "forecast_count": "Forecasts",
        "actual_up_rate_pct": "Actual up rate (%)",
        "accuracy_pct": "Accuracy (%)",
        "balanced_accuracy_pct": "Balanced accuracy (%)",
        "roc_auc": "ROC-AUC",
        "brier_score": "Brier score",
        "log_loss": "Log loss",
    }
    desired_display = ["display_model"] + [
        column for column in available if column != "model"
    ]

    st.dataframe(
        classifier[desired_display].rename(columns=rename),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Classification metrics should be interpreted separately from "
        "regression errors. Higher ROC-AUC and balanced accuracy are "
        "generally preferable; lower Brier score and log loss are better."
    )


def show_strategy_backtest(
    data: dict[str, object],
    cost_bps: float,
    selected_models: list[str],
) -> None:
    """Render cumulative returns, drawdowns and metric summaries."""
    st.title("Strategy Backtest")

    st.write(
        "Explore the historical results of the fixed long-or-cash intraday "
        "rule, benchmark strategies, and selected transaction-cost scenario."
    )

    st.info(
        f"Displayed costs: {cost_bps:g} bps per side "
        f"({2 * cost_bps:g} bps assumed per intraday round trip). "
        "These are illustrative assumptions, not verified broker charges."
    )

    daily = data["daily"]
    sensitivity = data["costs"]

    primary = sensitivity.loc[
        sensitivity["one_way_cost_assumption_bps"] == cost_bps
    ].copy()

    primary["display_model"] = primary["model"].map(format_model_label)
    primary = primary.sort_values("net_total_return_pct", ascending=False)

    c1, c2, c3 = st.columns(3)

    best = primary.iloc[0] if not primary.empty else None

    if best is not None:
        c1.metric(
            "Highest net return in table",
            format_model_label(str(best["model"])),
            help=f"{best['net_total_return_pct']:.2f}% cumulative",
        )
        c2.metric(
            "Sessions",
            f"{int(best['sessions']):,}",
        )
        c3.metric(
            "Assumed one-way cost",
            f"{cost_bps:g} bps",
        )

    equity = build_equity_curves(
        daily=daily,
        one_way_cost_bps=cost_bps,
        selected_models=selected_models,
    )

    if equity.empty:
        st.warning("No equity curves are available for this selection.")
        return

    st.subheader("Cumulative net return")

    fig = px.line(
        equity,
        x="date",
        y="cumulative_return_pct",
        color="strategy",
        labels={
            "date": "Target date",
            "cumulative_return_pct": "Cumulative return (%)",
            "strategy": "Strategy",
        },
        title="Cumulative Net Returns",
    )
    fig.add_hline(y=0.0, line_dash="dash")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Drawdowns")

    drawdown = build_drawdown_curves(equity)
    fig = px.line(
        drawdown,
        x="date",
        y="drawdown_pct",
        color="strategy",
        labels={
            "date": "Target date",
            "drawdown_pct": "Drawdown (%)",
            "strategy": "Strategy",
        },
        title="Drawdown from Historical Equity Peak",
    )
    fig.add_hline(y=0.0, line_dash="dash")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Strategy metrics for the selected cost")

    st.dataframe(
        primary[
            [
                "display_model",
                "active_days",
                "exposure_pct",
                "gross_total_return_pct",
                "net_total_return_pct",
                "annualized_net_return_pct",
                "annualized_net_volatility_pct",
                "net_sharpe_ratio_rf_zero",
                "max_net_drawdown_pct",
            ]
        ].rename(
            columns={
                "display_model": "Strategy",
                "active_days": "Active days",
                "exposure_pct": "Exposure (%)",
                "gross_total_return_pct": "Gross return (%)",
                "net_total_return_pct": "Net return (%)",
                "annualized_net_return_pct": "Annualized net return (%)",
                "annualized_net_volatility_pct": "Annualized volatility (%)",
                "net_sharpe_ratio_rf_zero": "Sharpe (risk-free = 0)",
                "max_net_drawdown_pct": "Maximum drawdown (%)",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Historical development-period results only. The model selection "
        "and strategy rule were evaluated on data already used during research."
    )


def show_daily_positions(
    data: dict[str, object],
    cost_bps: float,
) -> None:
    """Render the saved daily decision and return history."""
    st.title("Daily Strategy Details")

    daily = data["daily"].copy()
    models = sorted(daily["model"].dropna().unique().tolist())

    selected_model = st.selectbox(
        "Choose a strategy",
        options=models,
        format_func=format_model_label,
    )

    subset = daily.loc[
        daily["model"] == selected_model
    ].sort_values("target_date").copy()

    subset["net_return_at_selected_cost_pct"] = (
        subset["gross_strategy_return_pct"]
        - subset["position_long"] * (2.0 * cost_bps / 100.0)
    )

    subset["target_date"] = pd.to_datetime(subset["target_date"]).dt.date
    if "forecast_origin_date" in subset.columns:
        subset["forecast_origin_date"] = pd.to_datetime(
            subset["forecast_origin_date"]
        ).dt.date

    display_columns = [
        column
        for column in [
            "forecast_origin_date",
            "target_date",
            "predicted_log_return",
            "forecast_implied_open_close_return_pct",
            "target_open",
            "target_close",
            "position_long",
            "actual_open_close_return_pct",
            "gross_strategy_return_pct",
            "net_return_at_selected_cost_pct",
        ]
        if column in subset.columns
    ]

    active_days = int(subset["position_long"].sum())
    total_days = len(subset)

    c1, c2, c3 = st.columns(3)
    c1.metric("Sessions", f"{total_days:,}")
    c2.metric("Long days", f"{active_days:,}")
    c3.metric(
        "Exposure",
        f"{active_days / total_days * 100:.2f}%" if total_days else "N/A",
    )

    st.dataframe(
        subset[display_columns].rename(
            columns={
                "forecast_origin_date": "Forecast origin",
                "target_date": "Target date",
                "predicted_log_return": "Predicted log return",
                "forecast_implied_open_close_return_pct": (
                    "Implied open-close return (%)"
                ),
                "target_open": "Open",
                "target_close": "Close",
                "position_long": "Position (1=long, 0=cash)",
                "actual_open_close_return_pct": "Actual open-close return (%)",
                "gross_strategy_return_pct": "Gross strategy return (%)",
                "net_return_at_selected_cost_pct": (
                    "Net return at selected cost (%)"
                ),
            }
        ),
        width="stretch",
        hide_index=True,
    )

    csv_bytes = subset[display_columns].to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download displayed daily rows as CSV",
        data=csv_bytes,
        file_name=f"{selected_model}_daily_results.csv",
        mime="text/csv",
    )



def show_latest_forecast(ticker: str) -> None:
    """Display the most recent automated forecast and scored forecast history."""
    st.title("Latest Daily Forecast")
    st.write(
        "Latest next-session baseline forecasts generated from the most recent "
        "validated daily OHLC data. These are research outputs, not trading advice."
    )

    name = safe_ticker_name(ticker)
    latest_path = REPORTS_DIR / f"latest_forecast_{name}.json"
    history_path = REPORTS_DIR / f"daily_forecast_history_{name}.csv"

    if not latest_path.exists():
        st.warning(
            "No automated daily forecast has been published for this ticker yet."
        )
        st.markdown(
            "[Open the daily forecast workflow on GitHub]"
            "(https://github.com/avgphoenixguy/stock-forecast-backtesting-lab/actions/workflows/daily-forecast.yml)"
        )
        st.write(
            "Run the workflow manually using **Run workflow**. It will refresh "
            "and validate daily OHLC, generate baseline forecasts, and publish "
            "the small forecast ledger if the data is current."
        )
        st.caption(
            "The app will show a new forecast only after a successful workflow run."
        )
        return

    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read the latest forecast file: {exc}")
        return

    if payload.get("ticker") != ticker:
        st.warning(
            f"This forecast file is for {payload.get('ticker')}, not {ticker}. "
            "Choose RELIANCE.NS in the ticker selector to view the published forecast."
        )
        return

    latest_ohlc_date = payload.get("latest_ohlc_date", "Unknown")
    origin_date = payload.get("forecast_origin_date", "Unknown")
    generated_at = payload.get("generated_at_utc", "Unknown")

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest OHLC session", str(latest_ohlc_date))
    c2.metric("Forecast origin", str(origin_date))
    c3.metric("Forecast models", f"{len(payload.get('forecasts', []))}")

    st.caption(
        f"Forecast generated at (UTC): {generated_at} · "
        f"Source: {payload.get('data_source', 'not specified')} · "
        f"Price convention: {payload.get('price_adjustment', 'not specified')}"
    )
    st.info(
        "The current implementation publishes Random Walk and Random Walk with "
        "Drift baselines. The target is the next observed trading session, not "
        "necessarily the next calendar day. A zero-return Random Walk forecast "
        "simply carries forward the last adjusted close."
    )

    forecasts = payload.get("forecasts", [])
    if forecasts:
        forecast_frame = pd.DataFrame(forecasts)
        for column in [
            "input_close",
            "predicted_next_log_return_bps",
            "predicted_next_close",
            "actual_close",
            "absolute_close_error",
            "absolute_return_error_bps",
        ]:
            if column in forecast_frame.columns:
                forecast_frame[column] = pd.to_numeric(
                    forecast_frame[column], errors="coerce"
                )

        model_columns = st.columns(min(3, len(forecast_frame)))
        for idx, (_, forecast) in enumerate(forecast_frame.iterrows()):
            with model_columns[idx % len(model_columns)]:
                st.metric(
                    format_model_label(str(forecast.get("model", "model"))),
                    f"₹{float(forecast['predicted_next_close']):,.2f}",
                    delta=(
                        f"{float(forecast['predicted_next_log_return_bps']):+.2f} bps"
                    ),
                    help=(
                        "Forecasted next-session adjusted close; the delta is the "
                        "predicted log return in basis points."
                    ),
                )

        display_columns = [
            column for column in [
                "model",
                "input_close",
                "predicted_next_log_return_bps",
                "predicted_next_close",
                "target_date",
                "actual_close",
                "absolute_close_error",
            ] if column in forecast_frame.columns
        ]
        st.dataframe(
            forecast_frame[display_columns].rename(columns={
                "model": "Model",
                "input_close": "Last adjusted close (₹)",
                "predicted_next_log_return_bps": "Predicted log return (bps)",
                "predicted_next_close": "Predicted next close (₹)",
                "target_date": "Scored target date",
                "actual_close": "Actual close (₹)",
                "absolute_close_error": "Absolute close error (₹)",
            }),
            width="stretch",
            hide_index=True,
        )

    st.caption(str(payload.get("note", "")))

    st.subheader("Historical forecast log")
    if not history_path.exists():
        st.info("Forecast history has not been created yet.")
        return

    try:
        history = pd.read_csv(history_path)
    except (OSError, pd.errors.ParserError) as exc:
        st.error(f"Could not read the forecast history: {exc}")
        return

    required = {
        "origin_date", "target_date", "model", "input_close",
        "predicted_next_close", "actual_close",
        "absolute_close_error", "absolute_return_error_bps",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        st.error(f"Forecast history is missing columns: {missing}")
        return

    history["origin_date"] = pd.to_datetime(history["origin_date"], errors="coerce")
    history["target_date"] = pd.to_datetime(history["target_date"], errors="coerce")
    for column in [
        "input_close", "predicted_next_close", "actual_close",
        "absolute_close_error", "absolute_return_error_bps",
    ]:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    scored = history.dropna(subset=["target_date", "actual_close"]).copy()
    if scored.empty:
        st.info(
            "The first forecast is waiting for the next completed trading session "
            "to arrive, so there are no scored forecasts yet."
        )
        return

    models = sorted(scored["model"].dropna().unique().tolist())
    selected_model = st.selectbox(
        "Model in historical chart",
        options=models,
        format_func=format_model_label,
        key="daily_forecast_history_model",
    )
    model_history = (
        scored.loc[scored["model"] == selected_model]
        .sort_values("target_date")
        .tail(120)
    )

    history_chart = model_history.melt(
        id_vars=["target_date"],
        value_vars=["predicted_next_close", "actual_close"],
        var_name="series",
        value_name="adjusted_close",
    )
    history_chart["series"] = history_chart["series"].map({
        "predicted_next_close": "Forecast close",
        "actual_close": "Actual close",
    })
    fig = px.line(
        history_chart,
        x="target_date",
        y="adjusted_close",
        color="series",
        markers=True,
        labels={
            "target_date": "Target session",
            "adjusted_close": "Adjusted close (₹)",
            "series": "Series",
        },
        title=f"Last {len(model_history)} scored forecasts — {format_model_label(selected_model)}",
    )
    st.plotly_chart(fig, width="stretch")

    recent_columns = [
        column for column in [
            "origin_date", "target_date", "model", "predicted_next_close",
            "actual_close", "absolute_close_error",
            "absolute_return_error_bps", "direction_correct",
        ] if column in scored.columns
    ]
    recent = scored.sort_values("target_date", ascending=False).head(30)
    st.dataframe(
        recent[recent_columns].rename(columns={
            "origin_date": "Forecast origin",
            "target_date": "Target session",
            "model": "Model",
            "predicted_next_close": "Forecast close (₹)",
            "actual_close": "Actual close (₹)",
            "absolute_close_error": "Absolute close error (₹)",
            "absolute_return_error_bps": "Absolute return error (bps)",
            "direction_correct": "Direction correct",
        }),
        width="stretch",
        hide_index=True,
    )
    csv_bytes = recent[recent_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download recent scored forecasts",
        data=csv_bytes,
        file_name=f"{name.lower()}_forecast_history.csv",
        mime="text/csv",
    )



def main() -> None:
    st.set_page_config(
        page_title="Stock Forecast & Backtesting Lab",
        page_icon="📈",
        layout="wide",
    )

    st.sidebar.title("Research Lab")
    st.sidebar.caption("Reliance Industries · RELIANCE.NS")

    ticker = st.sidebar.text_input(
        "Ticker",
        value=DEFAULT_TICKER,
        help=(
            "Change this only if you have generated matching reports "
            "for another ticker."
        ),
    ).strip().upper()

    if not ticker:
        st.error("Please enter a ticker.")
        st.stop()

    name = safe_ticker_name(ticker)

    st.sidebar.caption("Bundled research results are read from dashboard/data/.")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        [
            "Latest forecast",
            "Overview",
            "Forecast comparison",
            "Strategy backtest",
            "Daily positions",
        ],
    )

    cost_bps = st.sidebar.selectbox(
        "Assumed cost per side (bps)",
        options=COST_SCENARIOS,
        index=2,
        help=(
            "Scenario assumption for cost and slippage per side. "
            "20 bps round trip at the default 10 bps."
        ),
    )

    reports = {
        "model_comparison_overall": (
            REPORTS_DIR / f"model_comparison_overall_{name}.csv"
        ),
        "model_comparison_classifier": (
            REPORTS_DIR / f"model_comparison_classifier_{name}.csv"
        ),
        "strategy_backtest_daily": (
            REPORTS_DIR / f"strategy_backtest_daily_{name}.csv"
        ),
        "strategy_cost_sensitivity": (
            REPORTS_DIR / f"strategy_cost_sensitivity_{name}.csv"
        ),
    }

    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in reports.values()
        if not path.exists()
    ]

    if missing:
        st.title("Stock Forecast & Backtesting Lab")
        st.warning(
            "Some generated experiment reports are missing. This dashboard "
            "reads local output files and does not train or download data."
        )

        st.write("Missing report files:")
        for path in missing:
            st.code(path)

        st.subheader("Generate the reports first")

        st.code(
            "./.venv/bin/python scripts/compare_model_results.py "
            "--ticker RELIANCE.NS --end 2022-06-14\n\n"
            "./.venv/bin/python scripts/run_strategy_backtest.py "
            "--ticker RELIANCE.NS --end 2022-06-14 --cost-bps 10\n\n"
            "./.venv/bin/python scripts/evaluate_strategy_backtest.py "
            "--ticker RELIANCE.NS --end 2022-06-14",
            language="bash",
        )

        st.caption(
            "Generated reports are ignored by Git. A fresh clone or a "
            "hosted deployment will need a deliberate report-generation "
            "or curated public snapshot setup."
        )
        st.stop()

    try:
        data = load_all_reports(ticker)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        st.error(f"Could not load the research reports: {exc}")
        st.stop()

    st.sidebar.caption("Development cutoff: 2022-06-14")

    if page == "Latest forecast":
        show_latest_forecast(ticker)
    elif page == "Overview":
        show_overview(data, cost_bps)
    elif page == "Forecast comparison":
        show_forecast_comparison(data)
    elif page == "Strategy backtest":
        st.title("Strategy Backtest")

        available_models = (
            data["daily"]["model"].dropna().unique().tolist()
        )

        model_defaults = [
            model for model in STRATEGIES_FOR_CURVE
            if model in available_models
        ]
        default_models = model_defaults + (
            ["buy_and_hold"] if "buy_and_hold" not in available_models else []
        )

        selected_models = st.multiselect(
            "Strategies shown in charts",
            options=sorted(set(available_models + ["buy_and_hold"])),
            default=default_models,
            format_func=format_model_label,
        )

        show_strategy_backtest(
            data=data,
            cost_bps=cost_bps,
            selected_models=selected_models,
        )
    elif page == "Daily positions":
        show_daily_positions(data, cost_bps)

    st.sidebar.divider()
    st.sidebar.caption(
        "Educational research only. Historical results are not evidence "
        "of future performance or financial advice."
    )


if __name__ == "__main__":
    main()
