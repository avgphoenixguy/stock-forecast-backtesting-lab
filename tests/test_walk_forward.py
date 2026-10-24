
"""Regression tests for expanding-window forecasting."""

import numpy as np
import pandas as pd
import pytest

from scripts import run_walk_forward as wf


def make_synthetic_data(n=320):
    """Create reproducible positive prices with business-day dates."""

    dates = pd.bdate_range("2005-01-03", periods=n)
    steps = np.arange(n, dtype=float)

    log_returns = (
        0.0002
        + 0.008 * np.sin(steps / 17)
        + 0.003 * np.cos(steps / 7)
    )

    close = 100.0 * np.exp(np.cumsum(log_returns))

    return pd.DataFrame({
        "date": dates,
        "close": close,
    })


def run_test_experiment(data, min_train=200):
    """Run the forecasting function without downloading any data."""

    return wf.run_walk_forward(
        data=data,
        start_date="2005-01-01",
        end_date=data["date"].max(),
        min_training_observations=min_train,
    )


def test_forecasts_count_and_order():
    data = make_synthetic_data(n=320)

    predictions, overall, annual, summary = run_test_experiment(data)

    assert len(predictions) == 320 - 200
    assert predictions["date"].is_monotonic_increasing
    assert predictions["training_observations"].iloc[0] == 200
    assert predictions["training_observations"].iloc[-1] == 319

    assert len(overall) == 2
    assert not annual.empty
    assert summary["walk_forward_forecasts"] == 120


def test_first_forecast_does_not_use_its_actual_close():
    """Changing a target close must not change its own prediction."""

    original = make_synthetic_data(n=320)
    changed = original.copy()

    first_test_index = 200
    changed.loc[first_test_index, "close"] *= 1.5

    original_predictions, *_ = run_test_experiment(original)
    changed_predictions, *_ = run_test_experiment(changed)

    original_first = original_predictions.iloc[0]
    changed_first = changed_predictions.iloc[0]

    # Predictions for the first test date are made using prior prices only.
    assert (
        original_first["random_walk_predicted_close"]
        == changed_first["random_walk_predicted_close"]
    )
    assert (
        original_first["drift_predicted_close"]
        == changed_first["drift_predicted_close"]
    )
    assert (
        original_first["drift_predicted_log_return"]
        == changed_first["drift_predicted_log_return"]
    )

    # The changed target should affect the observed outcome.
    assert (
        original_first["actual_close"]
        != changed_first["actual_close"]
    )


def test_training_window_expands_one_observation_at_a_time():
    data = make_synthetic_data(n=320)

    predictions, *_ = run_test_experiment(data)

    expected = np.arange(200, 320)

    np.testing.assert_array_equal(
        predictions["training_observations"].to_numpy(),
        expected,
    )


def test_each_forecast_uses_only_preceding_dates():
    data = make_synthetic_data(n=320)

    predictions, *_ = run_test_experiment(data)

    assert (
        predictions["training_end_date"] < predictions["date"]
    ).all()

    for _, row in predictions.iterrows():
        assert row["training_end_date"] == data.loc[
            data["date"] < row["date"], "date"
        ].max()


def test_insufficient_history_is_rejected():
    data = make_synthetic_data(n=200)

    with pytest.raises(
        ValueError,
        match="observations",
    ):
        run_test_experiment(data, min_train=200)


def test_forecasts_and_metrics_are_finite():
    data = make_synthetic_data(n=320)

    predictions, overall, annual, _ = run_test_experiment(data)

    numeric_predictions = predictions.select_dtypes(
        include=[np.number]
    ).to_numpy()

    numeric_metrics = overall[
        [
            "close_mae",
            "close_rmse",
            "log_return_mae_bps",
            "log_return_rmse_bps",
        ]
    ].to_numpy()

    assert np.isfinite(numeric_predictions).all()
    assert np.isfinite(numeric_metrics).all()

    assert (overall["close_mae"] >= 0).all()
    assert (overall["close_rmse"] >= 0).all()
