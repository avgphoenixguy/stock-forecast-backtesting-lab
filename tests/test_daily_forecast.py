"""Unit tests for the daily forecast ledger."""
import numpy as np
import pandas as pd

from scripts.run_daily_forecast import (
    HISTORY_COLUMNS,
    build_latest_payload,
    build_price_history_snapshot,
    empty_history,
    make_daily_forecasts,
    score_pending_forecasts,
    update_forecast_history,
)


def sample_market(closes):
    """Build internally consistent synthetic OHLCV input."""
    dates = pd.bdate_range("2026-01-05", periods=len(closes))
    close = np.asarray(closes, dtype=float)
    open_prices = close * 0.998
    high_prices = np.maximum(open_prices, close) * 1.002
    low_prices = np.minimum(open_prices, close) * 0.998
    return pd.DataFrame({
        "date": dates,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close,
        "volume": np.full(len(close), 1_000_000.0),
    })


def test_daily_forecasts_include_random_walk_and_drift():
    dates = pd.bdate_range("2020-01-01", periods=800)
    closes = 100.0 * np.exp(np.linspace(0.0, 0.2, len(dates)))
    data = sample_market(closes)
    data["date"] = dates

    result = make_daily_forecasts(
        data, "RELIANCE.NS", "2026-01-01T00:00:00+00:00", "2020-01-01"
    )
    # Econometric dependencies may be absent from the lightweight test env;
    # the baseline and any successfully fit ML forecasts must still be present.
    assert {"random_walk", "drift"}.issubset(set(result["model"]))
    assert len(result) >= 2
    assert np.isfinite(result["predicted_next_close"].to_numpy(dtype=float)).all()
    latest_close = float(closes[-1])
    random_walk = result.loc[result["model"] == "random_walk"].iloc[0]
    assert random_walk["input_close"] == latest_close
    assert random_walk["predicted_next_close"] == latest_close
    assert random_walk["predicted_next_log_return_bps"] == 0.0


def test_pending_forecast_is_scored_on_first_later_market_session():
    history = pd.DataFrame([{
        "ticker": "RELIANCE.NS",
        "origin_date": "2026-01-05",
        "generated_at_utc": "2026-01-05T12:00:00+00:00",
        "model": "drift",
        "input_close": 100.0,
        "predicted_next_log_return_bps": np.log(1.02) * 10_000,
        "predicted_next_close": 102.0,
        "target_date": "",
        "actual_close": np.nan,
        "actual_log_return_bps": np.nan,
        "absolute_close_error": np.nan,
        "absolute_return_error_bps": np.nan,
        "direction_correct": np.nan,
        "data_source": "Yahoo Finance via yfinance",
        "price_adjustment": "auto_adjust=True",
    }], columns=HISTORY_COLUMNS)
    market = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        "close": [100.0, 103.0, 101.0],
    })

    scored = score_pending_forecasts(history, market)
    row = scored.iloc[0]
    assert row["target_date"] == "2026-01-06"
    assert row["actual_close"] == 103.0
    assert np.isclose(row["absolute_close_error"], 1.0)
    assert np.isfinite(row["absolute_return_error_bps"])


def test_update_history_is_idempotent_for_same_origin_date():
    dates = pd.bdate_range("2020-01-01", periods=800)
    closes = 100.0 * np.exp(np.linspace(0.0, 0.2, len(dates)))
    market = sample_market(closes)
    market["date"] = dates

    first = update_forecast_history(
        market, empty_history(), "RELIANCE.NS",
        "2026-01-01T00:00:00+00:00", "2020-01-01",
    )
    second = update_forecast_history(
        market, first, "RELIANCE.NS",
        "2026-01-01T01:00:00+00:00", "2020-01-01",
    )
    assert len(first) >= 2
    assert len(second) == len(first)
    assert not second.duplicated(["origin_date", "model"]).any()
    assert set(second["origin_date"]) == {pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")}


def test_latest_payload_has_latest_forecasts():
    dates = pd.bdate_range("2020-01-01", periods=800)
    closes = 100.0 * np.exp(np.linspace(0.0, 0.2, len(dates)))
    market = sample_market(closes)
    market["date"] = dates
    history = update_forecast_history(
        market, empty_history(), "RELIANCE.NS",
        "2026-01-01T00:00:00+00:00", "2020-01-01",
    )
    payload = build_latest_payload(
        history, "RELIANCE.NS", pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")
    )
    assert payload["ticker"] == "RELIANCE.NS"
    assert len(payload["forecasts"]) >= 2
    assert payload["latest_ohlc_date"] == pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")



def test_price_history_snapshot_is_recent_sorted_and_complete():
    dates = pd.bdate_range("2020-01-01", periods=1300)
    closes = 100.0 * np.exp(np.linspace(0.0, 0.2, len(dates)))
    market = sample_market(closes)
    market["date"] = dates

    snapshot = build_price_history_snapshot(market, observations=1260)

    assert len(snapshot) == 1260
    assert list(snapshot.columns) == [
        "date", "open", "high", "low", "close", "volume"
    ]
    assert snapshot["date"].is_monotonic_increasing
    assert snapshot["date"].iloc[0] == dates[-1260].strftime("%Y-%m-%d")
    assert snapshot["date"].iloc[-1] == dates[-1].strftime("%Y-%m-%d")
    assert np.isfinite(snapshot[["open", "high", "low", "close", "volume"]].to_numpy()).all()


def test_price_history_snapshot_rejects_duplicate_dates():
    market = sample_market([100.0, 101.0, 102.0])
    market.loc[1, "date"] = market.loc[0, "date"]

    try:
        build_price_history_snapshot(market)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected duplicate OHLC dates to be rejected.")


def test_on_demand_forecast_accepts_one_year_plus_history():
    dates = pd.bdate_range(end="2026-10-23", periods=300)
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.0002, scale=0.012, size=len(dates))
    closes = 100.0 * np.exp(np.cumsum(returns))
    market = sample_market(closes)
    market["date"] = dates

    result = make_daily_forecasts(
        market,
        "NEWLISTING.NS",
        "2026-10-25T17:00:00+00:00",
        research_start="2020-01-01",
        min_observations=252,
    )

    assert {"random_walk", "drift"}.issubset(set(result["model"]))
    assert np.isfinite(result["predicted_next_close"].to_numpy(dtype=float)).all()

