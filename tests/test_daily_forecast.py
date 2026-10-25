"""Unit tests for the daily forecast ledger."""
import numpy as np
import pandas as pd

from scripts.run_daily_forecast import (
    HISTORY_COLUMNS,
    build_latest_payload,
    empty_history,
    make_daily_forecasts,
    score_pending_forecasts,
    update_forecast_history,
)


def sample_market(closes):
    dates = pd.bdate_range("2026-01-05", periods=len(closes))
    return pd.DataFrame({"date": dates, "close": closes})


def test_daily_forecasts_include_random_walk_and_drift():
    data = sample_market([100.0, 101.0, 99.0, 102.0])
    # Lower the training minimum only by using a small research window in a
    # lightweight isolated test? The function uses the production minimum,
    # so extend this sample to the required history length.
    dates = pd.bdate_range("2020-01-01", periods=800)
    closes = 100.0 * np.exp(np.linspace(0.0, 0.2, len(dates)))
    data = pd.DataFrame({"date": dates, "close": closes})

    result = make_daily_forecasts(
        data, "RELIANCE.NS", "2026-01-01T00:00:00+00:00", "2020-01-01"
    )
    assert set(result["model"]) == {"random_walk", "drift"}
    assert len(result) == 2
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
    market = pd.DataFrame({"date": dates, "close": closes})

    first = update_forecast_history(
        market, empty_history(), "RELIANCE.NS",
        "2026-01-01T00:00:00+00:00", "2020-01-01",
    )
    second = update_forecast_history(
        market, first, "RELIANCE.NS",
        "2026-01-01T01:00:00+00:00", "2020-01-01",
    )
    assert len(first) == 2
    assert len(second) == 2
    assert not second.duplicated(["origin_date", "model"]).any()
    assert set(second["origin_date"]) == {pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")}


def test_latest_payload_has_latest_forecasts():
    dates = pd.bdate_range("2020-01-01", periods=800)
    closes = 100.0 * np.exp(np.linspace(0.0, 0.2, len(dates)))
    market = pd.DataFrame({"date": dates, "close": closes})
    history = update_forecast_history(
        market, empty_history(), "RELIANCE.NS",
        "2026-01-01T00:00:00+00:00", "2020-01-01",
    )
    payload = build_latest_payload(
        history, "RELIANCE.NS", pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")
    )
    assert payload["ticker"] == "RELIANCE.NS"
    assert len(payload["forecasts"]) == 2
    assert payload["latest_ohlc_date"] == pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")
