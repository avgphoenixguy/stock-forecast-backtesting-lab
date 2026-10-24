
"""Regression tests for the historical-data downloader.

All tests use synthetic data. They do not access Yahoo Finance.
"""

import numpy as np
import pandas as pd
import pytest

from scripts import download_data as downloader


def make_history(
    rows,
    dates=None,
):
    """Create a synthetic yfinance-style history DataFrame."""

    if dates is None:
        dates = [
            f"2025-01-{day:02d}"
            for day in range(1, len(rows) + 1)
        ]

    index = pd.DatetimeIndex(
        pd.to_datetime(dates).date,
        tz="Asia/Kolkata",
        name="Date",
    )

    return pd.DataFrame(
        {
            "Open": [row["open"] for row in rows],
            "High": [row["high"] for row in rows],
            "Low": [row["low"] for row in rows],
            "Close": [row["close"] for row in rows],
            "Volume": [row.get("volume", 1000) for row in rows],
            "Dividends": [0.0] * len(rows),
            "Stock Splits": [0.0] * len(rows),
        },
        index=index,
    )


def install_fake_history(monkeypatch, history):
    """Replace yfinance with deterministic synthetic history."""

    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, **kwargs):
            assert kwargs["period"] == "max"
            assert kwargs["interval"] == "1d"
            return history.copy()

    monkeypatch.setattr(
        downloader.yf,
        "Ticker",
        FakeTicker,
    )


def test_valid_ohlc_data_is_accepted(monkeypatch):
    history = make_history(
        [
            {
                "open": 100.0,
                "high": 110.0,
                "low": 95.0,
                "close": 105.0,
            }
        ]
    )

    install_fake_history(monkeypatch, history)

    result = downloader.download_history("TEST.NS")

    assert len(result) == 1
    assert result.loc[0, "open"] == 100.0
    assert result.loc[0, "high"] == 110.0
    assert result.attrs["ohlc_rounding_level_rows"] == 0


def test_tiny_floating_point_difference_is_accepted(monkeypatch):
    # Close is one representable float above High.
    close_price = np.nextafter(10.0, np.inf)

    history = make_history(
        [
            {
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": close_price,
            }
        ]
    )

    install_fake_history(monkeypatch, history)

    result = downloader.download_history("TEST.NS")

    # The original downloaded price must remain unchanged.
    assert result.loc[0, "close"] == close_price

    # The tiny discrepancy must be recorded.
    assert result.attrs["ohlc_rounding_level_rows"] == 1


def test_material_ohlc_violation_is_rejected(monkeypatch):
    history = make_history(
        [
            {
                "open": 105.0,
                "high": 100.0,
                "low": 95.0,
                "close": 98.0,
            }
        ]
    )

    install_fake_history(monkeypatch, history)

    with pytest.raises(
        RuntimeError,
        match="Material OHLC inconsistencies",
    ):
        downloader.download_history("TEST.NS")


def test_non_finite_prices_are_rejected(monkeypatch):
    history = make_history(
        [
            {
                "open": 100.0,
                "high": 110.0,
                "low": 95.0,
                "close": np.nan,
            }
        ]
    )

    install_fake_history(monkeypatch, history)

    with pytest.raises(
        RuntimeError,
        match="numeric values",
    ):
        downloader.download_history("TEST.NS")


def test_duplicate_dates_are_rejected(monkeypatch):
    history = make_history(
        [
            {
                "open": 100.0,
                "high": 110.0,
                "low": 95.0,
                "close": 105.0,
            },
            {
                "open": 101.0,
                "high": 112.0,
                "low": 96.0,
                "close": 108.0,
            },
        ],
        dates=["2025-01-02", "2025-01-02"],
    )

    install_fake_history(monkeypatch, history)

    with pytest.raises(
        RuntimeError,
        match="Duplicate trading dates",
    ):
        downloader.download_history("TEST.NS")


def test_negative_volume_is_rejected(monkeypatch):
    history = make_history(
        [
            {
                "open": 100.0,
                "high": 110.0,
                "low": 95.0,
                "close": 105.0,
                "volume": -1,
            }
        ]
    )

    install_fake_history(monkeypatch, history)

    with pytest.raises(
        RuntimeError,
        match="Negative volume",
    ):
        downloader.download_history("TEST.NS")
