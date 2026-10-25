
"""PriceForecastLab: daily multi-model forecasts plus a historical research lab.

Daily model fitting runs in GitHub Actions, not in the Streamlit request.
The app displays published forecast records and historic experiment reports.
No trades are placed.
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

FORECAST_MODEL_LABELS = {
    "median_ensemble": "Median ensemble",
    "random_walk": "Random Walk baseline",
    "drift": "Random Walk with Drift",
    "arima_111": "ARIMA (1, 1, 1)",
    "holt_damped": "Holt damped trend",
    "ridge": "Ridge regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Gradient boosting",
    "lstm": "LSTM",
    "gru": "GRU",
}

FORECAST_MODEL_ORDER = [
    "median_ensemble", "random_walk", "drift", "arima_111", "holt_damped",
    "ridge", "random_forest", "hist_gradient_boosting", "lstm", "gru",
]

APP_CSS = """
<style>
:root {
  --pfl-ink: #eef4fb;
  --pfl-muted: #95a8bf;
  --pfl-line: rgba(148, 169, 194, 0.20);
  --pfl-panel: rgba(18, 30, 48, 0.88);
  --pfl-teal: #53e0c1;
  --pfl-blue: #7ab5ff;
}
.stApp {
  background:
    radial-gradient(ellipse at 8% 0%, rgba(35, 84, 116, 0.23), transparent 34%),
    radial-gradient(ellipse at 95% 10%, rgba(41, 100, 91, 0.15), transparent 28%),
    #0a101a;
  color: var(--pfl-ink);
}
.main .block-container {
  max-width: 1480px;
  padding-top: 2rem;
  padding-bottom: 3rem;
}
h1, h2, h3 {
  color: #f3f7fc;
  letter-spacing: -0.035em;
}
h1 { font-weight: 780; }
h2 { font-weight: 720; }
p, li, label { line-height: 1.55; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1725 0%, #0b121e 100%);
  border-right: 1px solid var(--pfl-line);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }
.pfl-sidebar-brand {
  display: flex; align-items: center; gap: 12px;
  padding: 8px 4px 18px; border-bottom: 1px solid var(--pfl-line);
  margin-bottom: 18px;
}
.pfl-logo {
  display: flex; align-items: center; justify-content: center;
  width: 44px; height: 44px; border-radius: 14px;
  background: linear-gradient(135deg, #53e0c1, #68a7ff);
  color: #08121c; font-size: 15px; font-weight: 900; letter-spacing: -1px;
  box-shadow: 0 8px 24px rgba(83, 224, 193, 0.14);
}
.pfl-brand-name { color: #f4f8ff; font-size: 19px; font-weight: 800; letter-spacing: -0.8px; }
.pfl-brand-accent { color: var(--pfl-teal); }
.pfl-brand-caption { margin-top: 3px; color: var(--pfl-muted); font-size: 9px; letter-spacing: 1.5px; font-weight: 700; }
.pfl-hero {
  position: relative; overflow: hidden;
  background: linear-gradient(115deg, rgba(19, 39, 59, 0.98), rgba(13, 31, 42, 0.92) 58%, rgba(21, 55, 56, 0.82));
  border: 1px solid rgba(111, 176, 199, 0.28);
  border-radius: 26px; padding: 30px 32px 28px; margin: 2px 0 24px;
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.16);
}
.pfl-hero:after {
  content: ""; position: absolute; width: 300px; height: 300px; right: -110px; top: -170px;
  border-radius: 50%; border: 1px solid rgba(83, 224, 193, 0.18);
  box-shadow: 0 0 0 28px rgba(83, 224, 193, 0.035), 0 0 0 58px rgba(122, 181, 255, 0.025);
}
.pfl-kicker { color: var(--pfl-teal); font-size: 10px; font-weight: 800; letter-spacing: 2.1px; margin-bottom: 12px; }
.pfl-hero-title { color: #f5f8fd; font-size: clamp(2.1rem, 4.8vw, 3.5rem); font-weight: 850; letter-spacing: -2.5px; line-height: 1.04; }
.pfl-hero-title span { color: var(--pfl-teal); }
.pfl-hero-copy { max-width: 780px; margin-top: 14px; color: #b7c9dc; font-size: 1rem; line-height: 1.65; }
.pfl-chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 21px; }
.pfl-chip {
  display: inline-block; border: 1px solid rgba(174, 207, 227, 0.20);
  background: rgba(5, 18, 29, 0.28); color: #d8e7f5;
  padding: 6px 10px; border-radius: 999px; font-size: 11px; font-weight: 650;
}
.pfl-section-label { color: var(--pfl-muted); font-size: 11px; font-weight: 800; letter-spacing: 1.6px; text-transform: uppercase; margin: 22px 0 10px; }
.pfl-small-note { color: var(--pfl-muted); font-size: 12px; }
.pfl-forecast-head {
  display: flex; justify-content: space-between; align-items: flex-end; gap: 22px;
  padding: 22px 24px; margin: 22px 0 14px;
  border-radius: 19px; border: 1px solid rgba(122, 181, 255, 0.22);
  background: linear-gradient(115deg, rgba(18, 34, 53, 0.96), rgba(14, 28, 42, 0.92));
}
.pfl-forecast-title {
  color: #f4f8fd; font-size: clamp(1.7rem, 3vw, 2.25rem);
  line-height: 1.1; font-weight: 820; letter-spacing: -1.4px;
}
.pfl-forecast-copy {
  color: #a8bbd0; font-size: 14px; line-height: 1.55;
  max-width: 760px; margin-top: 9px;
}
.pfl-forecast-status {
  display: inline-flex; align-items: center; gap: 8px; white-space: nowrap;
  padding: 9px 12px; border: 1px solid rgba(83, 224, 193, 0.22);
  background: rgba(83, 224, 193, 0.08); border-radius: 999px;
  color: #a6f3df; font-size: 11px; font-weight: 750; letter-spacing: 0.3px;
}
.pfl-forecast-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: #53e0c1; box-shadow: 0 0 0 4px rgba(83, 224, 193, 0.12);
}
.pfl-forecast-meta {
  display: flex; flex-wrap: wrap; gap: 9px 22px;
  padding: 0 4px 12px; color: #91a6bd; font-size: 12px;
}
.pfl-forecast-meta strong { color: #dce8f5; font-weight: 650; }
.pfl-footer {
  display: flex; flex-wrap: wrap; align-items: center; justify-content: center;
  gap: 8px; margin-top: 42px; padding: 20px 10px 8px;
  border-top: 1px solid var(--pfl-line); color: var(--pfl-muted);
  font-size: 12px; text-align: center;
}
.pfl-footer-heart { color: #ff7f9e; }
.pfl-footer a { color: var(--pfl-teal) !important; text-decoration: none; font-weight: 750; }
.pfl-footer a:hover { text-decoration: underline; }
[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(20, 34, 52, 0.94), rgba(14, 24, 39, 0.94));
  border: 1px solid var(--pfl-line); padding: 16px 18px;
  border-radius: 16px; min-height: 106px;
}
[data-testid="stMetricLabel"] p { color: #9fb2c9; font-weight: 650; font-size: 0.82rem; }
[data-testid="stMetricValue"] { color: #f4f8ff; font-weight: 780; letter-spacing: -0.8px; }
[data-testid="stMetricDelta"] { font-size: 0.78rem; }
[data-testid="stAlert"] { border-radius: 15px; border-width: 1px; }
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--pfl-line); border-radius: 13px; overflow: hidden;
}
.stButton > button, [data-testid="stDownloadButton"] button {
  border-radius: 12px; border: 1px solid rgba(83, 224, 193, 0.35);
  font-weight: 700; min-height: 42px;
}
.stButton > button[kind="primary"] { background: var(--pfl-teal); color: #07131a; }
.stTabs [data-baseweb="tab-list"] { gap: 7px; border-bottom: 1px solid var(--pfl-line); }
.stTabs [data-baseweb="tab"] { background: rgba(19, 31, 48, 0.65); border-radius: 10px 10px 0 0; padding: 10px 14px; }
.stTabs [aria-selected="true"] { border-bottom: 2px solid var(--pfl-teal) !important; color: var(--pfl-teal) !important; }
hr { border-color: var(--pfl-line); }
@media (max-width: 700px) {
  .main .block-container { padding: 1rem 0.8rem 2rem; }
  .pfl-hero { padding: 23px 20px; border-radius: 20px; }
  .pfl-hero-title { letter-spacing: -1.5px; }
  .pfl-forecast-head { flex-direction: column; align-items: flex-start; padding: 19px; gap: 14px; }
  .pfl-forecast-title { letter-spacing: -0.8px; }
  .pfl-forecast-meta { gap: 7px 14px; }
}
</style>
"""


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
    st.markdown(
        """
        <section class="pfl-hero">
          <div class="pfl-kicker">DAILY QUANTITATIVE RESEARCH · NSE EQUITIES</div>
          <div class="pfl-hero-title">PriceForecast<span>Lab</span></div>
          <div class="pfl-hero-copy">
            A transparent forecast board combining econometric baselines, machine-learning
            models and recurrent networks. Track the latest estimates, then measure them
            against the next completed session.
          </div>
          <div class="pfl-chip-row">
            <span class="pfl-chip">End-of-day adjusted OHLC</span>
            <span class="pfl-chip">Multi-model forecasts</span>
            <span class="pfl-chip">Forecasts are scored after the next session</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
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
            "Run the scheduled multi-model workflow manually using **Run workflow**. "
            "It refreshes and validates daily OHLC, attempts the configured model suite, "
            "and publishes the latest successful forecasts. If the run fails, open the "
            "workflow log before trusting any previously displayed forecast."
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

    try:
        generated_display = pd.to_datetime(generated_at, utc=True).strftime("%d %b %Y · %H:%M UTC")
    except (ValueError, TypeError):
        generated_display = str(generated_at)
    try:
        session_display = pd.to_datetime(latest_ohlc_date).strftime("%d %b %Y")
    except (ValueError, TypeError):
        session_display = str(latest_ohlc_date)

    st.markdown(
        f"""
        <section class="pfl-forecast-head">
          <div>
            <div class="pfl-kicker">FORECAST SNAPSHOT · {session_display}</div>
            <div class="pfl-forecast-title">Next-session outlook</div>
            <div class="pfl-forecast-copy">
              Compare one-session price estimates across econometric baselines,
              machine-learning models and recurrent networks. Actual accuracy is
              recorded after the next completed session.
            </div>
          </div>
          <div class="pfl-forecast-status"><span class="pfl-forecast-dot"></span> LATEST PUBLISHED RUN</div>
        </section>
        <div class="pfl-forecast-meta">
          <span><strong>Data session</strong> · {session_display}</span>
          <span><strong>Forecast published</strong> · {generated_display}</span>
          <span><strong>Source</strong> · {payload.get('data_source', 'not specified')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Methodology, data and limitations"):
        st.write(
            "Forecast horizon is the next observed trading session, not necessarily "
            "the next calendar day. Random Walk carries forward the last adjusted close; "
            "other models estimate a one-session log return."
        )
        st.write(
            "The median ensemble is the unweighted median of successful model forecasts. "
            "It is not a trained or validated trading strategy. Predictions are experimental "
            "estimates and are not trading signals."
        )
        st.caption(
            f"Price convention: {payload.get('price_adjustment', 'not specified')} · "
            "Data source availability and adjustment history may change."
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

        forecast_frame["model_label"] = forecast_frame["model"].map(
            lambda model: FORECAST_MODEL_LABELS.get(str(model), format_model_label(str(model)))
        )
        forecast_frame["predicted_return_pct"] = (
            forecast_frame["predicted_next_log_return_bps"] / 100.0
        )
        forecast_frame["_model_order"] = forecast_frame["model"].map(
            {model: idx for idx, model in enumerate(FORECAST_MODEL_ORDER)}
        ).fillna(999)
        forecast_frame = forecast_frame.sort_values(
            ["_model_order", "model"]
        ).reset_index(drop=True)

        first_forecast = forecast_frame.iloc[0]
        latest_close = float(forecast_frame["input_close"].dropna().iloc[0])
        c1, c2, c3 = st.columns(3)
        c1.metric("Last adjusted close", f"₹{latest_close:,.2f}")
        c2.metric("Latest OHLC session", str(latest_ohlc_date))
        c3.metric("Successful forecasts", f"{len(forecast_frame)}")

        st.markdown('<div class="pfl-section-label">Forecast disagreement</div>', unsafe_allow_html=True)
        plot_data = forecast_frame.sort_values("predicted_return_pct", ascending=True)
        fig = px.bar(
            plot_data,
            x="predicted_return_pct",
            y="model_label",
            color="model_label",
            orientation="h",
            labels={
                "predicted_return_pct": "Predicted log return (%)",
                "model_label": "Model",
                "color": "Model",
            },
            title="Predicted return by model",
            hover_data={
                "predicted_next_close": ":.2f",
                "predicted_next_log_return_bps": ":.2f",
                "predicted_return_pct": ":.4f",
            },
        )
        fig.add_vline(x=0.0, line_dash="dash", line_color="#94a8bf")
        fig.update_layout(
            showlegend=False,
            height=max(280, 42 * len(plot_data)),
            margin=dict(l=12, r=20, t=60, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown('<div class="pfl-section-label">Model forecast board</div>', unsafe_allow_html=True)
        for offset in range(0, len(forecast_frame), 3):
            model_columns = st.columns(3)
            batch = forecast_frame.iloc[offset : offset + 3]
            for column, (_, forecast) in zip(model_columns, batch.iterrows()):
                with column:
                    st.metric(
                        str(forecast["model_label"]),
                        f"₹{float(forecast['predicted_next_close']):,.2f}",
                    )
                    st.caption(
                        f"Implied log return: "
                        f"{float(forecast['predicted_next_log_return_bps']):+.2f} bps"
                    )

        display_frame = forecast_frame[
            [
                "model_label",
                "input_close",
                "predicted_next_log_return_bps",
                "predicted_next_close",
                "target_date",
                "actual_close",
                "absolute_close_error",
            ]
        ].rename(columns={
            "model_label": "Model",
            "input_close": "Last adjusted close (₹)",
            "predicted_next_log_return_bps": "Predicted log return (bps)",
            "predicted_next_close": "Predicted next close (₹)",
            "target_date": "Scored target date",
            "actual_close": "Actual close (₹)",
            "absolute_close_error": "Absolute close error (₹)",
        })
        display_frame["Scored target date"] = (
            display_frame["Scored target date"].fillna("Pending next session")
        )
        st.dataframe(display_frame, width="stretch", hide_index=True)
    else:
        st.warning("No model forecast rows were published for this date.")

    failures = payload.get("model_failures", [])
    attempted = payload.get("models_attempted", [])
    with st.expander(f"Model diagnostics · {len(failures)} unavailable in latest run"):
        if attempted:
            labels = [FORECAST_MODEL_LABELS.get(str(model), str(model)) for model in attempted]
            st.caption("Models attempted: " + ", ".join(labels))
        if failures:
            diagnostic_frame = pd.DataFrame(failures).rename(
                columns={"model": "Model", "error": "Reason"}
            )
            diagnostic_frame["Model"] = diagnostic_frame["Model"].map(
                lambda model: FORECAST_MODEL_LABELS.get(str(model), str(model))
            )
            st.dataframe(diagnostic_frame, width="stretch", hide_index=True)
            st.caption(
                "A failure for one model does not necessarily invalidate other forecasts. "
                "Check the GitHub Actions log for full details."
            )
        else:
            st.success("All attempted models completed without a recorded exception.")

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

    st.markdown('<div class="pfl-section-label">Performance since live tracking began</div>', unsafe_allow_html=True)
    performance = (
        scored.groupby("model", dropna=True)
        .agg(
            scored_forecasts=("target_date", "count"),
            close_mae=("absolute_close_error", "mean"),
            return_mae_bps=("absolute_return_error_bps", "mean"),
            direction_accuracy_pct=(
                "direction_correct",
                lambda values: float(values.mean() * 100.0) if values.notna().any() else np.nan,
            ),
        )
        .reset_index()
        .sort_values("return_mae_bps", na_position="last")
    )
    performance["model"] = performance["model"].map(
        lambda model: FORECAST_MODEL_LABELS.get(str(model), format_model_label(str(model)))
    )
    st.dataframe(
        performance.rename(columns={
            "model": "Model",
            "scored_forecasts": "Scored forecasts",
            "close_mae": "Close MAE (₹)",
            "return_mae_bps": "Return MAE (bps)",
            "direction_accuracy_pct": "Direction accuracy (%)",
        }),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "This scorecard uses only forecasts recorded since automated tracking began. "
        "Early samples are small; do not treat the current ranking as a reliable model winner."
    )

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



def render_signature_footer() -> None:
    """Render the author's signature and social link at the end of each workspace."""
    st.markdown(
        """
        <footer class="pfl-footer">
          <span>Created with <span class="pfl-footer-heart">♥</span> by
            <strong>avgphoenixguy · Krishna Lahoti</strong>
          </span>
          <span aria-hidden="true">·</span>
          <a href="https://x.com/avgphoenixguy" target="_blank" rel="noopener noreferrer">
            Follow on X / Twitter ↗
          </a>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the product-facing forecast page and the separate research workspace."""
    st.set_page_config(
        page_title="PriceForecastLab | Daily Stock Forecasts",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.sidebar.markdown(
        """
        <div class="pfl-sidebar-brand">
          <div class="pfl-logo">PF</div>
          <div>
            <div class="pfl-brand-name">PriceForecast<span class="pfl-brand-accent">Lab</span></div>
            <div class="pfl-brand-caption">QUANTITATIVE RESEARCH</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("**TRACKED INSTRUMENT**")
    st.sidebar.markdown("Reliance Industries")
    st.sidebar.caption("NSE · RELIANCE.NS · Daily adjusted OHLC")
    st.sidebar.divider()

    workspace = st.sidebar.radio(
        "Workspace",
        ["Price Forecast", "Research Lab"],
        index=0,
        help=(
            "Price Forecast is the public product view. Research Lab contains "
            "the fixed historical model comparisons and backtest tools."
        ),
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Daily data and forecast updates are produced by GitHub Actions. "
        "This app displays published outputs; it does not place trades."
    )

    if workspace == "Price Forecast":
        show_latest_forecast(DEFAULT_TICKER)
        st.markdown(
            '<div class="pfl-small-note">Educational research only · '
            'Historical results do not establish future performance.</div>',
            unsafe_allow_html=True,
        )
        render_signature_footer()
        return

    st.title("Research Lab")
    st.caption(
        "Historical experiments and diagnostics. The documented comparison "
        "period ends June 14, 2022; these tables are not a live forecast feed."
    )
    cost_bps = st.sidebar.selectbox(
        "Assumed cost per side (bps)",
        options=COST_SCENARIOS,
        index=2,
        help=(
            "Illustrative transaction-cost and slippage assumption. "
            "At 10 bps per side, the intraday round trip is modelled as 20 bps."
        ),
    )

    ticker = DEFAULT_TICKER
    name = safe_ticker_name(ticker)
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
        st.warning(
            "The historical research snapshot is incomplete. The live daily "
            "forecast page can still work independently."
        )
        st.write("Missing report files:")
        for report in missing:
            st.code(report)
        st.stop()

    try:
        data = load_all_reports(ticker)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        st.error(f"Could not load the historical research reports: {exc}")
        st.stop()

    st.caption(
        "Data: RELIANCE.NS · Development cutoff: 2022-06-14 · "
        f"Selected cost scenario: {cost_bps:g} bps per side"
    )
    research_tabs = st.tabs(
        ["Overview", "Model comparison", "Strategy backtest", "Daily decisions"]
    )
    with research_tabs[0]:
        show_overview(data, cost_bps)
    with research_tabs[1]:
        show_forecast_comparison(data)
    with research_tabs[2]:
        st.subheader("Historical strategy backtest")
        st.caption(
            "Exploratory long-or-cash intraday strategy. Benchmark exposures differ, "
            "and the cost model is illustrative rather than broker-verified."
        )
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
            "Strategies shown",
            options=sorted(set(available_models + ["buy_and_hold"])),
            default=default_models,
            format_func=format_model_label,
        )
        show_strategy_backtest(
            data=data,
            cost_bps=cost_bps,
            selected_models=selected_models,
        )
    with research_tabs[3]:
        show_daily_positions(data, cost_bps)

    st.markdown(
        '<div class="pfl-small-note">Research workbench · '
        'Not investment advice, not a live execution system.</div>',
        unsafe_allow_html=True,
    )
    render_signature_footer()


if __name__ == "__main__":
    main()
