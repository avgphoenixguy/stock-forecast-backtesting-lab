"""Generate and score multi-model next-session price forecasts.

This is a research workflow. It does not place trades. It writes a small
forecast ledger and a dashboard payload after validating the saved OHLC data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DASHBOARD_DATA_DIR = PROJECT_ROOT / "dashboard" / "data"
DEFAULT_TICKER = "RELIANCE.NS"
DEFAULT_RESEARCH_START = "2005-01-01"
DEFAULT_MAX_STALENESS_DAYS = 7
MIN_OBSERVATIONS = 756
MAX_SUPERVISED_ROWS = 2520
MAX_ECONOMETRIC_ROWS = 1500
SEQUENCE_LENGTH = 30
MAX_SEQUENCE_ROWS = 1260

HISTORY_COLUMNS = [
    "ticker", "origin_date", "generated_at_utc", "model", "input_close",
    "predicted_next_log_return_bps", "predicted_next_close", "target_date",
    "actual_close", "actual_log_return_bps", "absolute_close_error",
    "absolute_return_error_bps", "direction_correct", "data_source",
    "price_adjustment",
]


def safe_ticker_name(ticker: str) -> str:
    """Convert a ticker into the repository's filename convention."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").upper()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validated_data(ticker: str) -> pd.DataFrame:
    """Load downloaded OHLC data and verify its metadata and structure."""
    name = safe_ticker_name(ticker)
    csv_path = RAW_DATA_DIR / f"{name}_daily.csv"
    metadata_path = RAW_DATA_DIR / f"{name}_metadata.json"
    if not csv_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Daily data is missing. Run python scripts/download_data.py --ticker {ticker} first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("ticker") != ticker:
        raise ValueError("The data metadata ticker does not match the requested ticker.")
    if metadata.get("csv_sha256") != sha256_file(csv_path):
        raise ValueError("OHLC CSV checksum does not match its metadata.")
    if metadata.get("price_adjustment", {}).get("auto_adjust") is not True:
        raise ValueError("Expected the project's auto-adjusted OHLC data.")

    data = pd.read_csv(csv_path)
    data.columns = [str(column).strip().lower() for column in data.columns]
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"OHLC CSV is missing required columns: {missing}")

    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data["date"].isna().any():
        raise ValueError("OHLC contains invalid dates.")
    if data["date"].duplicated().any():
        raise ValueError("OHLC contains duplicate dates.")
    if not data["date"].is_monotonic_increasing:
        raise ValueError("OHLC rows are not sorted by date.")
    if len(data) != metadata.get("row_count"):
        raise ValueError("OHLC row count does not match its metadata.")

    prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
    volume = data["volume"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or not np.isfinite(volume).all():
        raise ValueError("OHLC or volume contains missing/non-finite numbers.")
    if (prices <= 0).any():
        raise ValueError("OHLC contains non-positive prices.")
    if (volume < 0).any():
        raise ValueError("Volume contains negative values.")

    open_prices = data["open"].to_numpy(dtype=float)
    high_prices = data["high"].to_numpy(dtype=float)
    low_prices = data["low"].to_numpy(dtype=float)
    close_prices = data["close"].to_numpy(dtype=float)
    scale = np.maximum.reduce([
        np.ones(len(data)), np.abs(open_prices), np.abs(high_prices),
        np.abs(low_prices), np.abs(close_prices),
    ])
    tolerance = 8 * np.finfo(np.float64).eps * scale
    invalid = (
        (high_prices + tolerance < low_prices)
        | (high_prices + tolerance < np.maximum(open_prices, close_prices))
        | (low_prices - tolerance > np.minimum(open_prices, close_prices))
    )
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} invalid OHLC rows.")
    return data.reset_index(drop=True)


def build_ml_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build lagged features that are observable at each forecast origin."""
    close = pd.to_numeric(data["close"], errors="coerce").astype(float)
    log_close = np.log(close)
    daily_returns = log_close.diff()
    features: dict[str, pd.Series] = {}

    for lag in (1, 2, 3, 5, 10, 20, 60):
        features[f"return_lag_{lag}"] = log_close.diff(lag)

    for window in (5, 10, 20, 60):
        features[f"volatility_{window}"] = daily_returns.rolling(window).std()
        features[f"mean_return_{window}"] = daily_returns.rolling(window).mean()

    for window in (10, 20, 50, 100):
        features[f"close_vs_sma_{window}"] = log_close - np.log(
            close.rolling(window).mean()
        )

    open_prices = pd.to_numeric(data["open"], errors="coerce").astype(float)
    high_prices = pd.to_numeric(data["high"], errors="coerce").astype(float)
    low_prices = pd.to_numeric(data["low"], errors="coerce").astype(float)
    volume = pd.to_numeric(data["volume"], errors="coerce").astype(float)

    features["intraday_return"] = np.log(close / open_prices)
    features["intraday_range"] = np.log(high_prices / low_prices)
    log_volume = np.log1p(volume)
    for lag in (1, 5):
        features[f"log_volume_change_{lag}"] = log_volume.diff(lag)

    frame = pd.DataFrame(features, index=data.index)
    return frame.replace([np.inf, -np.inf], np.nan)


def _record_forecast(
    rows: list[dict[str, Any]],
    model: str,
    predicted_log_return: float,
    ticker: str,
    origin_date: str,
    generated_at_utc: str,
    last_close: float,
) -> None:
    """Validate and append one model's forecast."""
    predicted_log_return = float(predicted_log_return)
    predicted_next_close = last_close * float(np.exp(predicted_log_return))
    if not np.isfinite(predicted_log_return) or not np.isfinite(predicted_next_close):
        raise ValueError("Model returned a non-finite forecast.")
    if predicted_next_close <= 0:
        raise ValueError("Model returned a non-positive predicted close.")
    rows.append({
        "ticker": ticker,
        "origin_date": origin_date,
        "generated_at_utc": generated_at_utc,
        "model": model,
        "input_close": last_close,
        "predicted_next_log_return_bps": predicted_log_return * 10_000.0,
        "predicted_next_close": predicted_next_close,
        "target_date": "",
        "actual_close": np.nan,
        "actual_log_return_bps": np.nan,
        "absolute_close_error": np.nan,
        "absolute_return_error_bps": np.nan,
        "direction_correct": np.nan,
        "data_source": "Yahoo Finance via yfinance",
        "price_adjustment": "auto_adjust=True",
    })


def _forecast_random_walk_and_drift(
    log_closes: np.ndarray,
    trained_returns: np.ndarray,
) -> dict[str, float]:
    del log_closes
    return {
        "random_walk": 0.0,
        "drift": float(np.mean(trained_returns[-MAX_SUPERVISED_ROWS:])),
    }


def _forecast_arima(log_closes: np.ndarray) -> float:
    """Forecast log price with a fixed ARIMA(1,1,1) specification."""
    from statsmodels.tsa.arima.model import ARIMA

    series = pd.Series(log_closes[-MAX_ECONOMETRIC_ROWS:])
    fitted = ARIMA(
        series,
        order=(1, 1, 1),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    forecast_log_close = float(np.asarray(fitted.forecast(steps=1))[0])
    return forecast_log_close - float(log_closes[-1])


def _forecast_holt_damped(log_closes: np.ndarray) -> float:
    """Forecast log price with damped-trend exponential smoothing."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    series = pd.Series(log_closes[-MAX_ECONOMETRIC_ROWS:])
    fitted = ExponentialSmoothing(
        series,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True, use_brute=False)
    forecast_log_close = float(np.asarray(fitted.forecast(1))[0])
    return forecast_log_close - float(log_closes[-1])


def _fit_sklearn_forecasters(
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Train small, fixed ML models on the most recent supervised rows."""
    valid_features = features.dropna()
    aligned_target = target.reindex(valid_features.index)
    train_mask = aligned_target.notna() & np.isfinite(aligned_target)
    x_train = valid_features.loc[train_mask].tail(MAX_SUPERVISED_ROWS)
    y_train = aligned_target.loc[x_train.index].astype(float)

    if len(x_train) < 500:
        raise ValueError(
            f"Only {len(x_train)} valid supervised rows; at least 500 are required."
        )
    x_latest = valid_features.iloc[[-1]]
    if x_latest.empty or x_latest.index[0] != features.index[-1]:
        raise ValueError("Could not construct the latest feature row.")

    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }
    results: dict[str, float] = {}
    for name, estimator in models.items():
        estimator.fit(x_train, y_train)
        value = float(np.asarray(estimator.predict(x_latest))[0])
        if not np.isfinite(value):
            raise ValueError(f"{name} produced a non-finite forecast.")
        results[name] = value
    return results


def _fit_one_sklearn_model(
    name: str, x_train: pd.DataFrame, y_train: pd.Series, x_latest: pd.DataFrame
) -> float:
    """Fit one model independently so a failure does not hide other forecasts."""
    if name == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
    elif name == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        )
    else:
        raise ValueError(f"Unknown scikit-learn model: {name}")
    model.fit(x_train, y_train)
    return float(np.asarray(model.predict(x_latest))[0])


def _fit_recurrent_model(
    model_kind: str,
    features: pd.DataFrame,
    target: pd.Series,
) -> float:
    """Fit a modest LSTM or GRU to recent sequences using TensorFlow/Keras."""
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    if model_kind not in {"lstm", "gru"}:
        raise ValueError(f"Unsupported recurrent model: {model_kind}")

    valid_features = features.dropna().tail(MAX_SEQUENCE_ROWS).copy()
    aligned_target = target.reindex(valid_features.index)
    if len(valid_features) < 500:
        raise ValueError(f"Only {len(valid_features)} recent feature rows for {model_kind}.")

    x_raw = valid_features.to_numpy(dtype=np.float32)
    y_raw = pd.to_numeric(aligned_target, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(x_raw).all():
        raise ValueError("Recurrent-model features contain non-finite values.")

    # The last label is usually unavailable, because it is the next session.
    labelled_endpoints = [
        idx for idx in range(SEQUENCE_LENGTH - 1, len(valid_features) - 1)
        if np.isfinite(y_raw[idx])
    ]
    if len(labelled_endpoints) < 250:
        raise ValueError(f"Only {len(labelled_endpoints)} sequences available for {model_kind}.")

    # Scale with observations available through the latest forecast origin.
    x_scaler = StandardScaler()
    x_scaler.fit(x_raw[:-1])
    x_scaled = x_scaler.transform(x_raw).astype(np.float32)

    y_scaler = StandardScaler()
    y_scaler.fit(y_raw[np.isfinite(y_raw)].reshape(-1, 1))
    y_scaled = np.full(y_raw.shape, np.nan, dtype=np.float32)
    mask = np.isfinite(y_raw)
    y_scaled[mask] = y_scaler.transform(y_raw[mask].reshape(-1, 1)).ravel()

    sequence_x = np.stack([
        x_scaled[end - SEQUENCE_LENGTH + 1 : end + 1]
        for end in labelled_endpoints
    ]).astype(np.float32)
    sequence_y = np.asarray([y_scaled[end] for end in labelled_endpoints], dtype=np.float32)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, x_scaled.shape[1]))
    if model_kind == "lstm":
        hidden = tf.keras.layers.LSTM(24)(inputs)
    else:
        hidden = tf.keras.layers.GRU(24)(inputs)
    hidden = tf.keras.layers.Dense(12, activation="relu")(hidden)
    outputs = tf.keras.layers.Dense(1)(hidden)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    model.fit(
        sequence_x,
        sequence_y,
        validation_split=0.15,
        epochs=20,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0,
    )
    latest_sequence = x_scaled[-SEQUENCE_LENGTH:][None, :, :]
    predicted_scaled = float(np.asarray(model.predict(latest_sequence, verbose=0)).reshape(-1)[0])
    predicted_return = float(y_scaler.inverse_transform([[predicted_scaled]])[0, 0])
    tf.keras.backend.clear_session()
    if not np.isfinite(predicted_return):
        raise ValueError(f"{model_kind} produced a non-finite forecast.")
    return predicted_return


def make_daily_forecasts(
    data: pd.DataFrame,
    ticker: str,
    generated_at_utc: str,
    research_start: str = DEFAULT_RESEARCH_START,
    include_deep_learning: bool = False,
) -> pd.DataFrame:
    """Produce all available daily model forecasts, with per-model failure notes."""
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(data.columns):
        raise ValueError(f"Input data must contain: {sorted(required)}")

    window = data.loc[pd.to_datetime(data["date"]) >= pd.Timestamp(research_start)].copy()
    window = window.sort_values("date").reset_index(drop=True)
    if len(window) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Need at least {MIN_OBSERVATIONS} observations since {research_start}; "
            f"found {len(window)}."
        )

    close = pd.to_numeric(window["close"], errors="coerce").to_numpy(dtype=float)
    log_closes = np.log(close)
    log_returns = np.diff(log_closes)
    if len(log_returns) < 2 or not np.isfinite(log_returns).all():
        raise ValueError("Could not estimate historical log returns.")

    last_close = float(close[-1])
    origin_date = pd.Timestamp(window["date"].iloc[-1]).strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def attempt(model_name: str, forecast_function: Callable[[], float]) -> None:
        try:
            forecast_value = forecast_function()
            _record_forecast(
                rows=rows,
                model=model_name,
                predicted_log_return=forecast_value,
                ticker=ticker,
                origin_date=origin_date,
                generated_at_utc=generated_at_utc,
                last_close=last_close,
            )
        except Exception as exc:
            # Keep the other models and the day's whole forecast cycle alive.
            message = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:240]
            failures.append({"model": model_name, "error": message})

    baseline_values = _forecast_random_walk_and_drift(log_closes, log_returns)
    for name, value in baseline_values.items():
        attempt(name, lambda value=value: value)

    attempt("arima_111", lambda: _forecast_arima(log_closes))
    attempt("holt_damped", lambda: _forecast_holt_damped(log_closes))

    features = build_ml_features(window)
    target = np.log(pd.Series(close, index=window.index).shift(-1) / pd.Series(close, index=window.index))
    valid_features = features.dropna()
    aligned_target = target.reindex(valid_features.index)
    train_mask = aligned_target.notna() & np.isfinite(aligned_target)
    x_train = valid_features.loc[train_mask].tail(MAX_SUPERVISED_ROWS)
    y_train = aligned_target.loc[x_train.index].astype(float)
    x_latest = valid_features.iloc[[-1]]

    if len(x_train) < 500 or x_latest.empty:
        failures.append({
            "model": "ml_models",
            "error": f"Insufficient supervised feature rows: {len(x_train)}.",
        })
    else:
        for model_name in ("ridge", "random_forest", "hist_gradient_boosting"):
            attempt(
                model_name,
                lambda model_name=model_name: _fit_one_sklearn_model(
                    model_name, x_train, y_train, x_latest
                ),
            )

    if include_deep_learning:
        for model_name in ("lstm", "gru"):
            attempt(
                model_name,
                lambda model_name=model_name: _fit_recurrent_model(
                    model_name, features, target
                ),
            )

    # A robust consensus price forecast from all successful individual models.
    successful_returns = [
        float(row["predicted_next_log_return_bps"]) / 10_000.0
        for row in rows
    ]
    if len(successful_returns) >= 3:
        attempt("median_ensemble", lambda: float(np.median(successful_returns)))

    results = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    results.attrs["model_failures"] = failures
    results.attrs["models_attempted"] = [
        "random_walk", "drift", "arima_111", "holt_damped", "ridge",
        "random_forest", "hist_gradient_boosting",
    ] + (["lstm", "gru"] if include_deep_learning else []) + ["median_ensemble"]
    if results.empty:
        raise RuntimeError("No model produced a forecast.")
    return results


def empty_history() -> pd.DataFrame:
    """Return an empty forecast-history frame with a stable schema."""
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def normalise_history_dates(history: pd.DataFrame) -> pd.DataFrame:
    """Normalise history date columns while preserving unscored target dates."""
    result = history.copy()
    for column in ("origin_date", "target_date"):
        parsed = pd.to_datetime(result[column], errors="coerce")
        result[column] = parsed.dt.strftime("%Y-%m-%d").fillna("")
    return result


def score_pending_forecasts(history: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Score pending forecasts against the first later observed market session."""
    if history.empty:
        return history.copy()
    result = normalise_history_dates(history)
    market = data.sort_values("date").reset_index(drop=True).copy()
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    dates = market["date"].to_numpy(dtype="datetime64[ns]")
    closes = pd.to_numeric(market["close"], errors="coerce").to_numpy(dtype=float)

    for idx, row in result.iterrows():
        target_text = str(row.get("target_date", "")).strip()
        if target_text and target_text.lower() not in {"nan", "nat", "none"}:
            continue
        origin = pd.Timestamp(row["origin_date"]).to_datetime64().astype("datetime64[ns]")
        next_index = int(np.searchsorted(dates, origin, side="right"))
        if next_index >= len(market):
            continue

        actual_close = float(closes[next_index])
        input_close = float(row["input_close"])
        predicted_close = float(row["predicted_next_close"])
        predicted_return_bps = float(row["predicted_next_log_return_bps"])
        actual_return_bps = float(np.log(actual_close / input_close) * 10_000.0)
        result.at[idx, "target_date"] = pd.Timestamp(dates[next_index]).strftime("%Y-%m-%d")
        result.at[idx, "actual_close"] = actual_close
        result.at[idx, "actual_log_return_bps"] = actual_return_bps
        result.at[idx, "absolute_close_error"] = abs(predicted_close - actual_close)
        result.at[idx, "absolute_return_error_bps"] = abs(predicted_return_bps - actual_return_bps)
        if predicted_return_bps != 0.0 and actual_return_bps != 0.0:
            result.at[idx, "direction_correct"] = float(
                np.sign(predicted_return_bps) == np.sign(actual_return_bps)
            )
    return result


def update_forecast_history(
    data: pd.DataFrame,
    history: pd.DataFrame,
    ticker: str,
    generated_at_utc: str,
    research_start: str = DEFAULT_RESEARCH_START,
    include_deep_learning: bool = False,
) -> pd.DataFrame:
    """Score old forecasts and append one new row per model and origin date."""
    result = score_pending_forecasts(history, data)
    fresh = make_daily_forecasts(
        data, ticker, generated_at_utc, research_start, include_deep_learning
    )
    failures = fresh.attrs.get("model_failures", [])
    attempted = fresh.attrs.get("models_attempted", [])
    if result.empty:
        result = empty_history()
    result = normalise_history_dates(result)

    existing = set(zip(result["origin_date"].astype(str), result["model"].astype(str)))
    new_rows = fresh.loc[
        ~fresh.apply(
            lambda row: (row["origin_date"], row["model"]) in existing,
            axis=1,
        )
    ]
    if not new_rows.empty:
        result = pd.concat([result, new_rows], ignore_index=True)

    result = normalise_history_dates(result)
    result["_sort_date"] = pd.to_datetime(result["origin_date"])
    result = (
        result.sort_values(["_sort_date", "model"])
        .drop(columns="_sort_date")
        .reset_index(drop=True)
    )
    if result.duplicated(["origin_date", "model"]).any():
        raise ValueError("Duplicate forecast records were detected.")
    result = result[HISTORY_COLUMNS]
    result.attrs["model_failures"] = failures
    result.attrs["models_attempted"] = attempted
    return result


def load_history(path: Path) -> pd.DataFrame:
    """Read the forecast ledger or return an empty ledger."""
    if not path.exists():
        return empty_history()
    history = pd.read_csv(path)
    missing = [column for column in HISTORY_COLUMNS if column not in history.columns]
    if missing:
        raise ValueError(f"Existing forecast history is missing columns: {missing}")
    return normalise_history_dates(history[HISTORY_COLUMNS])


def build_latest_payload(
    history: pd.DataFrame,
    ticker: str,
    latest_ohlc_date: str,
    model_failures: list[dict[str, str]] | None = None,
    models_attempted: list[str] | None = None,
) -> dict[str, Any]:
    """Create a compact JSON payload for the dashboard."""
    if history.empty:
        raise ValueError("Cannot build a latest-forecast file from empty history.")
    latest_origin = max(history["origin_date"].astype(str))
    latest = history.loc[history["origin_date"].astype(str) == latest_origin]
    models: list[dict[str, Any]] = []
    for _, row in latest.iterrows():
        models.append({
            "model": str(row["model"]),
            "input_close": float(row["input_close"]),
            "predicted_next_log_return_bps": float(row["predicted_next_log_return_bps"]),
            "predicted_next_close": float(row["predicted_next_close"]),
            "generated_at_utc": str(row["generated_at_utc"]),
            "target_date": str(row["target_date"]) or None,
            "actual_close": None if pd.isna(row["actual_close"]) else float(row["actual_close"]),
            "absolute_close_error": None if pd.isna(row["absolute_close_error"]) else float(row["absolute_close_error"]),
            "absolute_return_error_bps": None if pd.isna(row["absolute_return_error_bps"]) else float(row["absolute_return_error_bps"]),
            "direction_correct": None if pd.isna(row["direction_correct"]) else bool(float(row["direction_correct"])),
        })
    return {
        "ticker": ticker,
        "latest_ohlc_date": latest_ohlc_date,
        "forecast_origin_date": latest_origin,
        "generated_at_utc": str(latest["generated_at_utc"].iloc[-1]),
        "forecast_horizon": "next observed trading session",
        "model_scope": [
            "Random Walk", "Random Walk with Drift", "ARIMA(1,1,1)",
            "Holt damped trend", "Ridge Regression", "Random Forest",
            "Histogram Gradient Boosting", "LSTM", "GRU", "Median ensemble",
        ],
        "data_source": "Yahoo Finance via yfinance",
        "price_adjustment": "auto_adjust=True",
        "models_attempted": models_attempted or [],
        "model_failures": model_failures or [],
        "forecasts": models,
        "note": (
            "Research output only. Forecasts are model estimates, not trading "
            "recommendations, and do not establish predictive or economic edge. "
            "The ensemble is the median of models that successfully produced forecasts."
        ),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically to avoid a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score previous forecasts and generate next-session multi-model forecasts."
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker, default RELIANCE.NS")
    parser.add_argument("--research-start", default=DEFAULT_RESEARCH_START)
    parser.add_argument("--max-staleness-days", type=int, default=DEFAULT_MAX_STALENESS_DAYS)
    parser.add_argument(
        "--include-deep-learning",
        action="store_true",
        help="Train TensorFlow LSTM and GRU models; install requirements-forecast.txt first.",
    )
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()
    if not ticker:
        parser.error("Ticker symbol cannot be empty.")
    if args.max_staleness_days < 0:
        parser.error("--max-staleness-days cannot be negative.")

    data = load_validated_data(ticker)
    latest_date = pd.Timestamp(data["date"].iloc[-1]).normalize()
    utc_today = datetime.now(timezone.utc).date()
    age_days = (utc_today - latest_date.date()).days
    if age_days < 0:
        raise SystemExit("Latest OHLC date is in the future; refusing to publish.")
    if age_days > args.max_staleness_days:
        raise SystemExit(
            f"Latest OHLC date is {latest_date.date()} ({age_days} calendar days old). "
            "Refusing to publish a forecast from stale data."
        )

    name = safe_ticker_name(ticker)
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    history_path = DASHBOARD_DATA_DIR / f"daily_forecast_history_{name}.csv"
    latest_path = DASHBOARD_DATA_DIR / f"latest_forecast_{name}.json"
    history = load_history(history_path)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    updated_history = update_forecast_history(
        data=data,
        history=history,
        ticker=ticker,
        generated_at_utc=generated_at,
        research_start=args.research_start,
        include_deep_learning=args.include_deep_learning,
    )
    latest_date_text = latest_date.strftime("%Y-%m-%d")
    latest_payload = build_latest_payload(
        updated_history,
        ticker,
        latest_date_text,
        model_failures=updated_history.attrs.get("model_failures", []),
        models_attempted=updated_history.attrs.get("models_attempted", []),
    )

    history_temporary = history_path.with_suffix(history_path.suffix + ".tmp")
    updated_history.to_csv(history_temporary, index=False)
    history_temporary.replace(history_path)
    write_json_atomic(latest_path, latest_payload)

    latest_rows = updated_history.loc[
        updated_history["origin_date"] == latest_payload["forecast_origin_date"]
    ]
    print("===== DAILY FORECAST UPDATED =====")
    print(f"Ticker: {ticker}")
    print(f"Latest OHLC date: {latest_date_text}")
    print(f"Forecast origin date: {latest_payload['forecast_origin_date']}")
    print(f"Successful models: {len(latest_payload['forecasts'])}")
    print(f"Model failures: {len(latest_payload['model_failures'])}")
    if latest_payload["model_failures"]:
        print(json.dumps(latest_payload["model_failures"], indent=2))
    print(f"History records: {len(updated_history):,}")
    print(f"New-origin records: {int((updated_history['origin_date'] == latest_date_text).sum())}")
    print(f"History CSV: {history_path.relative_to(PROJECT_ROOT)}")
    print(f"Latest forecast JSON: {latest_path.relative_to(PROJECT_ROOT)}")
    print(latest_rows[[
        "model", "input_close", "predicted_next_log_return_bps",
        "predicted_next_close", "target_date", "absolute_close_error",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
