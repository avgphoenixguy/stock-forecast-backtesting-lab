
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
import plotly.graph_objects as go
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
  gap: 12px; margin-top: 42px; padding: 22px 10px 8px;
  border-top: 1px solid var(--pfl-line); color: var(--pfl-muted);
  font-size: 12px; text-align: center;
}
.pfl-footer-author { display: inline-flex; align-items: center; gap: 4px; }
.pfl-footer-heart-icon {
  display: inline-block; width: 14px; height: 14px; margin: 0 2px;
  vertical-align: -2px; fill: #ff5b83;
  filter: drop-shadow(0 0 4px rgba(255, 91, 131, 0.28));
  flex: 0 0 auto;
}
.pfl-footer-socials { display: inline-flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 8px; }
.pfl-footer-socials a {
  display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px;
  border: 1px solid rgba(122, 181, 255, 0.20); border-radius: 999px;
  background: rgba(122, 181, 255, 0.045); color: #b6d9ff !important;
  text-decoration: none !important; font-weight: 700;
  transition: border-color 140ms ease, background 140ms ease, transform 140ms ease;
}
.pfl-footer-socials a:hover {
  border-color: rgba(83, 224, 193, 0.55); background: rgba(83, 224, 193, 0.08);
  transform: translateY(-1px); text-decoration: none !important;
}
.pfl-footer-socials svg { width: 13px; height: 13px; fill: currentColor; display: block; }
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



@st.cache_data(ttl=24 * 60 * 60, max_entries=2, show_spinner=False)
def load_nifty500_universe() -> pd.DataFrame:
    """Load the latest published Nifty 500 stock directory."""
    path = REPORTS_DIR / "nifty500_universe.csv"
    if not path.exists():
        return pd.DataFrame([{
            "company_name": "Reliance Industries Limited",
            "symbol": "RELIANCE",
            "ticker": "RELIANCE.NS",
            "industry": "Energy",
            "universe": "Fallback",
            "fetched_at_utc": "",
        }])

    try:
        universe = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read the Nifty 500 directory: {exc}") from exc

    required = {"company_name", "symbol", "ticker"}
    missing = sorted(required.difference(universe.columns))
    if missing:
        raise ValueError(f"The Nifty 500 directory is missing columns: {missing}")

    for column in ("company_name", "symbol", "ticker"):
        universe[column] = universe[column].fillna("").astype(str).str.strip()
    universe = universe.loc[
        universe["company_name"].ne("")
        & universe["symbol"].ne("")
        & universe["ticker"].ne("")
    ].drop_duplicates(subset=["ticker"], keep="first")

    if not universe["ticker"].str.endswith(".NS").all():
        raise ValueError("The published universe contains a non-NSE Yahoo ticker.")
    if len(universe) < 450:
        raise ValueError(
            f"The published directory contains only {len(universe)} rows, not a full Nifty 500 snapshot."
        )
    return universe.sort_values(["company_name", "symbol"]).reset_index(drop=True)


@st.cache_data(ttl=60 * 60, max_entries=200, show_spinner=False)
def fetch_ticker_history(ticker: str) -> pd.DataFrame:
    """Fetch and validate recent auto-adjusted OHLC for one selected ticker."""
    import yfinance as yf

    raw = yf.download(
        ticker,
        period="7y",
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        raise ValueError(
            f"No daily OHLC was returned for {ticker}. The symbol may not map to Yahoo Finance, "
            "or the provider may be temporarily unavailable."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        flattened = False
        for level in range(raw.columns.nlevels):
            labels = [str(value).strip().lower() for value in raw.columns.get_level_values(level)]
            if "close" in labels and "open" in labels:
                raw.columns = raw.columns.get_level_values(level)
                flattened = True
                break
        if not flattened:
            raw.columns = [
                "_".join(str(piece) for piece in column if str(piece) not in {"", "None"})
                for column in raw.columns.to_list()
            ]

    raw = raw.reset_index()
    column_lookup = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in raw.columns
    }
    date_col = next(
        (column_lookup[name] for name in ("date", "datetime") if name in column_lookup),
        None,
    )
    if date_col is None:
        raise ValueError(f"Could not identify the date column returned for {ticker}.")

    required_source = {
        "open": ("open",),
        "high": ("high",),
        "low": ("low",),
        "close": ("close",),
        "volume": ("volume",),
    }
    selected = {"date": raw[date_col]}
    for target_col, aliases in required_source.items():
        source_col = next(
            (column_lookup[alias] for alias in aliases if alias in column_lookup),
            None,
        )
        if source_col is None:
            raise ValueError(
                f"Daily OHLC response for {ticker} is missing {target_col.upper()}."
            )
        selected[target_col] = raw[source_col]

    data = pd.DataFrame(selected)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = (
        data.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if len(data) < 252:
        raise ValueError(
            f"Only {len(data)} valid daily sessions were returned for {ticker}; "
            "at least 252 are required for an on-demand preview. Recently listed stocks "
            "with shorter price history are searchable but cannot be forecast by this version."
        )

    prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
    volumes = data["volume"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or not np.isfinite(volumes).all():
        raise ValueError(f"OHLC data for {ticker} contains non-finite values.")
    if (prices <= 0).any() or (volumes < 0).any():
        raise ValueError(f"OHLC data for {ticker} contains invalid price or volume values.")

    latest_date = pd.Timestamp(data["date"].iloc[-1]).date()
    age_days = (pd.Timestamp.now(tz="UTC").date() - latest_date).days
    if age_days < 0:
        raise ValueError(f"The latest OHLC date for {ticker} is in the future.")
    if age_days > 7:
        raise ValueError(
            f"The latest OHLC date for {ticker} is {age_days} days old. "
            "Refusing to produce a forecast from stale data."
        )
    return data.tail(1800).reset_index(drop=True)


@st.cache_data(ttl=60 * 60, max_entries=100, show_spinner=False)
def generate_on_demand_forecast(ticker: str) -> tuple[dict[str, object], pd.DataFrame]:
    """Generate a non-persistent preview when a selected stock has no tracked forecast."""
    from scripts.run_daily_forecast import (
        build_price_history_snapshot,
        make_daily_forecasts,
    )

    data = fetch_ticker_history(ticker)
    generated_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    forecast_rows = make_daily_forecasts(
        data=data,
        ticker=ticker,
        generated_at_utc=generated_at,
        research_start="2005-01-01",
        include_deep_learning=False,
        min_observations=252,
    )
    failures = forecast_rows.attrs.get("model_failures", [])
    attempted = forecast_rows.attrs.get("models_attempted", [])
    origin_date = pd.Timestamp(data["date"].iloc[-1]).strftime("%Y-%m-%d")
    latest_close = float(data["close"].iloc[-1])

    forecasts: list[dict[str, object]] = []
    for _, row in forecast_rows.iterrows():
        forecasts.append({
            "model": str(row["model"]),
            "input_close": float(row["input_close"]),
            "predicted_next_log_return_bps": float(row["predicted_next_log_return_bps"]),
            "predicted_next_close": float(row["predicted_next_close"]),
            "generated_at_utc": str(row["generated_at_utc"]),
            "target_date": None,
            "actual_close": None,
            "absolute_close_error": None,
            "absolute_return_error_bps": None,
            "direction_correct": None,
        })

    payload: dict[str, object] = {
        "ticker": ticker,
        "latest_ohlc_date": origin_date,
        "forecast_origin_date": origin_date,
        "generated_at_utc": generated_at,
        "forecast_horizon": "next observed trading session",
        "forecast_mode": "on_demand",
        "model_scope": attempted,
        "models_attempted": attempted,
        "model_failures": failures,
        "data_source": "Yahoo Finance via yfinance",
        "price_adjustment": "auto_adjust=True",
        "forecasts": forecasts,
        "note": (
            "On-demand research preview. These estimates are not yet part of the "
            "persistent daily tracking ledger. Deep-learning models are only included "
            "in scheduled forecasts where the dedicated forecast runner is configured for them. "
            "Stocks with shorter histories can still receive baseline forecasts; ML models run "
            "only when at least 500 usable training rows remain after feature construction."
        ),
    }
    snapshot = build_price_history_snapshot(data)
    return payload, snapshot


def show_price_history_chart(
    ticker: str,
    latest_close: float,
    forecast_frame: pd.DataFrame,
    price_history_snapshot: pd.DataFrame | None = None,
) -> None:
    """Show recent adjusted OHLC candles and a price-level comparison to forecasts."""
    name = safe_ticker_name(ticker)
    history_path = REPORTS_DIR / f"price_history_{name}.csv"
    st.markdown(
        '<div class="pfl-section-label">Stock price history</div>',
        unsafe_allow_html=True,
    )
    if price_history_snapshot is not None:
        price_history = price_history_snapshot.copy()
    elif history_path.exists():
        try:
            price_history = pd.read_csv(history_path)
        except (OSError, pd.errors.ParserError) as exc:
            st.warning(f"Could not load the price-history snapshot: {exc}")
            return
    else:
        st.info(
            "No saved OHLC snapshot exists for this ticker. Select a supported Nifty 500 "
            "stock to fetch and validate its recent daily history on demand."
        )
        return

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(price_history.columns))
    if missing:
        st.warning(f"The price-history snapshot is missing columns: {missing}")
        return

    price_history["date"] = pd.to_datetime(price_history["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        price_history[column] = pd.to_numeric(price_history[column], errors="coerce")
    price_history = (
        price_history.dropna(subset=["date", "open", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )
    if price_history.empty:
        st.info("No valid daily OHLC observations are available for the chart.")
        return

    range_label = st.selectbox(
        "Chart period",
        options=["3 months", "6 months", "1 year", "3 years", "5 years"],
        index=2,
        key="price_history_period",
    )
    months_by_label = {
        "3 months": 3,
        "6 months": 6,
        "1 year": 12,
        "3 years": 36,
        "5 years": 60,
    }
    last_date = price_history["date"].max()
    cutoff = last_date - pd.DateOffset(months=months_by_label[range_label])
    visible = price_history.loc[price_history["date"] >= cutoff].copy()

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=visible["date"],
                open=visible["open"],
                high=visible["high"],
                low=visible["low"],
                close=visible["close"],
                name="Adjusted OHLC",
                increasing_line_color="#53E0C1",
                decreasing_line_color="#FF7F9E",
                increasing_fillcolor="rgba(83,224,193,0.50)",
                decreasing_fillcolor="rgba(255,127,158,0.42)",
                whiskerwidth=0.5,
            )
        ]
    )
    fig.add_hline(
        y=latest_close,
        line_dash="dot",
        line_color="#7AB5FF",
        annotation_text=f"Latest adjusted close ₹{latest_close:,.2f}",
        annotation_position="top left",
    )
    fig.update_layout(
        title=f"{ticker} · Adjusted daily price history",
        xaxis_title="Trading session",
        yaxis_title="Adjusted price (₹)",
        height=450,
        margin=dict(l=12, r=18, t=60, b=12),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D8E5F3"),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        f"Showing {len(visible):,} sessions through {last_date.strftime('%d %b %Y')}. "
        "Prices are auto-adjusted OHLC from Yahoo Finance via yfinance—not a live quote."
    )

    if forecast_frame.empty:
        return

    comparison = forecast_frame[
        ["model_label", "predicted_next_close", "predicted_next_log_return_bps"]
    ].copy()
    comparison["predicted_next_close"] = pd.to_numeric(
        comparison["predicted_next_close"], errors="coerce"
    )
    comparison = comparison.dropna(subset=["predicted_next_close"])
    if comparison.empty:
        return

    st.markdown(
        '<div class="pfl-section-label">Last close vs. predicted next close</div>',
        unsafe_allow_html=True,
    )
    low = min(float(latest_close), float(comparison["predicted_next_close"].min()))
    high = max(float(latest_close), float(comparison["predicted_next_close"].max()))
    spread = max(high - low, latest_close * 0.001, 0.5)
    fig = px.scatter(
        comparison.sort_values("predicted_next_close"),
        x="predicted_next_close",
        y="model_label",
        color="model_label",
        labels={
            "predicted_next_close": "Predicted next adjusted close (₹)",
            "model_label": "Model",
            "color": "Model",
        },
        hover_data={
            "predicted_next_log_return_bps": ":.2f",
            "predicted_next_close": ":.2f",
        },
        title="Each point is a model's next-session price estimate",
    )
    fig.add_vline(
        x=latest_close,
        line_dash="dash",
        line_color="#7AB5FF",
        annotation_text=f"Reference close ₹{latest_close:,.2f}",
        annotation_position="top",
    )
    fig.update_xaxes(range=[low - spread * 0.35, high + spread * 0.35])
    fig.update_layout(
        showlegend=False,
        height=max(290, 40 * len(comparison)),
        margin=dict(l=12, r=20, t=70, b=16),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D8E5F3"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def show_latest_forecast(ticker: str) -> None:
    """Display the most recent automated forecast and scored forecast history."""
    st.markdown(
        """
        <section class="pfl-hero">
          <div class="pfl-kicker">DAILY QUANTITATIVE RESEARCH · NSE EQUITIES</div>
          <div class="pfl-hero-title">PriceForecast<span>Lab</span></div>
          <div class="pfl-hero-copy">
            Compare next-session stock-price estimates from econometric and machine-learning
            models. Tracked stocks also include recurrent-network outputs; other Nifty 500
            constituents can be previewed on demand and validated against future closes later.
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

    on_demand_price_history: pd.DataFrame | None = None
    forecast_is_on_demand = not latest_path.exists()

    if forecast_is_on_demand:
        try:
            with st.spinner(f"Fetching validated daily OHLC and fitting models for {ticker}..."):
                payload, on_demand_price_history = generate_on_demand_forecast(ticker)
        except Exception as exc:
            st.error(f"Could not generate an on-demand forecast for {ticker}: {exc}")
            st.info(
                "This stock remains searchable even if the market-data provider does not "
                "support its symbol. Try again later or check the ticker mapping."
            )
            st.markdown(
                "[View the Nifty 500 daily workflow]"
                "(https://github.com/avgphoenixguy/stock-forecast-backtesting-lab/actions/workflows/daily-forecast.yml)"
            )
            return
    else:
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
          <div class="pfl-forecast-status"><span class="pfl-forecast-dot"></span> {("ON-DEMAND PREVIEW" if forecast_is_on_demand else "LATEST PUBLISHED RUN")}</div>
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
        c1.metric(
            "Reference price · latest adjusted close",
            f"₹{latest_close:,.2f}",
            help="Most recent adjusted close in the validated daily data, not an intraday live quote.",
        )
        c2.metric("Latest OHLC session", str(latest_ohlc_date))
        c3.metric("Successful model forecasts", f"{len(forecast_frame)}")

        show_price_history_chart(
            ticker,
            latest_close,
            forecast_frame,
            price_history_snapshot=on_demand_price_history,
        )

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
                    predicted_price = float(forecast["predicted_next_close"])
                    price_delta = predicted_price - latest_close
                    price_delta_pct = (price_delta / latest_close) * 100.0
                    st.metric(
                        str(forecast["model_label"]),
                        f"₹{predicted_price:,.2f}",
                        delta=f"{price_delta:+,.2f} ₹ ({price_delta_pct:+.2f}%)",
                        help="Difference from the latest adjusted close reference; this is not an expected profit.",
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
        if forecast_is_on_demand:
            st.info(
                "This is an on-demand preview, not a persistent daily-tracked forecast. "
                "Its next-session outcome is not yet included in the cross-model leaderboard. "
                "Reliance Industries is the first stock with automated, saved forecast history."
            )
        else:
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
    total_forecasts = int(len(history))
    scored_forecast_rows = int(len(scored))
    pending_forecast_rows = total_forecasts - scored_forecast_rows

    st.markdown(
        '<div class="pfl-section-label">Live forecast tracking</div>',
        unsafe_allow_html=True,
    )
    tracker_columns = st.columns(3)
    tracker_columns[0].metric("Recorded model forecasts", f"{total_forecasts:,}")
    tracker_columns[1].metric("Scored forecast records", f"{scored_forecast_rows:,}")
    tracker_columns[2].metric("Pending outcomes", f"{pending_forecast_rows:,}")

    if scored.empty:
        st.info(
            "The first forecasts are waiting for the next completed trading session. "
            "Once the actual close is available, the workflow will score each model "
            "and the live accuracy leaderboard will populate automatically."
        )
        st.caption(
            "Forecast records are created after the most recent validated OHLC session; "
            "they are scored against the next observed session, not the next calendar day."
        )
        return

    st.markdown('<div class="pfl-section-label">Live model leaderboard</div>', unsafe_allow_html=True)
    st.caption(
        "Scores are based only on forecasts that have reached a subsequent observed session. "
        "The sample begins when live tracking started; the table is descriptive, not a model-selection guarantee."
    )

    scored["signed_return_error_bps"] = (
        pd.to_numeric(scored["predicted_next_log_return_bps"], errors="coerce")
        - pd.to_numeric(scored["actual_log_return_bps"], errors="coerce")
    )
    performance = (
        scored.groupby("model", dropna=True)
        .agg(
            scored_forecasts=("target_date", "count"),
            close_mae=("absolute_close_error", "mean"),
            return_mae_bps=("absolute_return_error_bps", "mean"),
            return_rmse_bps=(
                "signed_return_error_bps",
                lambda values: float(np.sqrt(np.nanmean(np.square(values)))),
            ),
            mean_return_error_bps=("signed_return_error_bps", "mean"),
            direction_accuracy_pct=(
                "direction_correct",
                lambda values: float(values.mean() * 100.0) if values.notna().any() else np.nan,
            ),
        )
        .reset_index()
        .sort_values("return_mae_bps", na_position="last")
    )
    performance["model_label"] = performance["model"].map(
        lambda model: FORECAST_MODEL_LABELS.get(str(model), format_model_label(str(model)))
    )

    accuracy_chart = performance.dropna(subset=["return_mae_bps"]).sort_values("return_mae_bps")
    if not accuracy_chart.empty:
        fig = px.bar(
            accuracy_chart,
            x="return_mae_bps",
            y="model_label",
            orientation="h",
            labels={
                "return_mae_bps": "Mean absolute log-return error (bps)",
                "model_label": "Model",
            },
            title="Live return MAE by model · lower is better",
            hover_data={
                "scored_forecasts": True,
                "return_rmse_bps": ":.2f",
                "mean_return_error_bps": ":.2f",
            },
        )
        fig.update_layout(
            height=max(300, 40 * len(accuracy_chart)),
            margin=dict(l=12, r=18, t=58, b=16),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#D8E5F3"),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.dataframe(
        performance[[
            "model_label", "scored_forecasts", "close_mae", "return_mae_bps",
            "return_rmse_bps", "mean_return_error_bps", "direction_accuracy_pct",
        ]].rename(columns={
            "model_label": "Model",
            "scored_forecasts": "Scored forecasts",
            "close_mae": "Close MAE (₹)",
            "return_mae_bps": "Return MAE (bps)",
            "return_rmse_bps": "Return RMSE (bps)",
            "mean_return_error_bps": "Mean return error (bps)",
            "direction_accuracy_pct": "Direction accuracy (%)",
        }),
        width="stretch",
        hide_index=True,
    )

    rolling = scored.sort_values("target_date").copy()
    rolling["rolling_return_mae_bps"] = (
        rolling.groupby("model")["absolute_return_error_bps"]
        .transform(lambda values: values.rolling(window=20, min_periods=5).mean())
    )
    rolling_visible = rolling.dropna(subset=["rolling_return_mae_bps"]).copy()
    st.markdown(
        '<div class="pfl-section-label">Rolling forecast accuracy</div>',
        unsafe_allow_html=True,
    )
    if rolling_visible.empty:
        st.info(
            "Rolling MAE charts will appear after each model has at least five scored "
            "forecasts. Until then, the live leaderboard above is an early-sample preview."
        )
    else:
        rolling_visible["model_label"] = rolling_visible["model"].map(
            lambda model: FORECAST_MODEL_LABELS.get(str(model), format_model_label(str(model)))
        )
        fig = px.line(
            rolling_visible,
            x="target_date",
            y="rolling_return_mae_bps",
            color="model_label",
            markers=True,
            labels={
                "target_date": "Scored target session",
                "rolling_return_mae_bps": "Rolling 20-forecast MAE (bps)",
                "model_label": "Model",
            },
            title="Rolling return MAE · lower is better",
        )
        fig.update_layout(
            height=400,
            margin=dict(l=12, r=18, t=58, b=16),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#D8E5F3"),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
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
    """Render the author's signature and social links at the end of each workspace."""
    st.markdown(
        """
        <footer class="pfl-footer">
          <span class="pfl-footer-author">
            Created with
            <svg class="pfl-footer-heart-icon" viewBox="0 0 24 24" role="img" aria-label="love">
              <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5
                2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09
                C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5
                c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"></path>
            </svg>
            by <strong>avgphoenixguy · Krishna Lahoti</strong>
          </span>
          <nav class="pfl-footer-socials" aria-label="Social profiles">
            <a href="https://x.com/avgphoenixguy" target="_blank" rel="noopener noreferrer"
               aria-label="Krishna Lahoti on X">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M18.9 2H22l-6.78 7.75L23.2 22h-6.25l-4.9-7.35L5.6 22H2.47l7.25-8.29L1.8 2h6.4l4.43 6.75L18.9 2zm-1.1 17.85h1.73L7.27 4.04H5.41L17.8 19.85z"></path>
              </svg>
              X / Twitter
            </a>
            <a href="https://www.linkedin.com/in/krishnalahoti/" target="_blank" rel="noopener noreferrer"
               aria-label="Krishna Lahoti on LinkedIn">
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.35V9h3.414v1.561h.049c.476-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.369 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 1 1 0-4.124 2.062 2.062 0 0 1 0 4.124zM7.119 20.452H3.554V9h3.565v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0z"></path>
              </svg>
              LinkedIn
            </a>
          </nav>
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
        "The Nifty 500 directory is refreshed by GitHub Actions. Reliance has persistent "
        "daily model tracking; other selected stocks can generate cached on-demand previews. "
        "No trades are placed."
    )

    if workspace == "Price Forecast":
        universe = load_nifty500_universe()
        is_fallback_universe = (
            universe.empty
            or universe.get("universe", pd.Series(dtype=str)).eq("Fallback").all()
        )
        if is_fallback_universe:
            st.sidebar.warning(
                "The Nifty 500 directory is being initialized. Reliance is temporarily available."
            )
        else:
            fetched_at_values = universe.get(
                "fetched_at_utc", pd.Series(dtype=str)
            ).dropna()
            fetched_at = str(fetched_at_values.iloc[0]) if not fetched_at_values.empty else "unknown"
            st.sidebar.caption(
                f"Nifty 500 directory · {len(universe):,} constituents · refreshed {fetched_at}"
            )

        ticker_to_label = {
            str(row["ticker"]): (
                f"{row['company_name']} · {row['symbol']}"
                + (f" · {row['industry']}" if str(row.get("industry", "")).strip() else "")
            )
            for _, row in universe.iterrows()
        }
        ticker_options = list(ticker_to_label)
        default_index = (
            ticker_options.index(DEFAULT_TICKER)
            if DEFAULT_TICKER in ticker_options else 0
        )
        st.sidebar.markdown("**SEARCH NIFTY 500**")
        selected_ticker = st.sidebar.selectbox(
            "Company or ticker",
            options=ticker_options,
            index=default_index,
            format_func=lambda value: ticker_to_label.get(value, value),
            key="nifty500_ticker_selector",
            help="Type a company name or NSE symbol to filter the Nifty 500 directory.",
        )
        selected_row = universe.loc[universe["ticker"] == selected_ticker].iloc[0]
        industry = str(selected_row.get("industry", "")).strip()
        st.sidebar.caption(
            f"Selected: {selected_row['symbol']} · NSE"
            + (f" · {industry}" if industry else "")
        )
        if is_fallback_universe:
            st.caption(
                "The full Nifty 500 list has not been published yet. "
                "Run the daily forecast workflow to refresh the official directory."
            )
        show_latest_forecast(selected_ticker)
        st.markdown(
            '<div class="pfl-small-note">Educational research only · '
            'Historical results do not establish future performance.</div>',
            unsafe_allow_html=True,
        )
        render_signature_footer()
        return

    st.sidebar.markdown("**RESEARCH INSTRUMENT**")
    st.sidebar.markdown("Reliance Industries")
    st.sidebar.caption("NSE · RELIANCE.NS · Fixed historical research reports")
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
