"""Generate and score one-session-ahead daily baseline forecasts.

The script expects data/raw/<TICKER>_daily.csv and its validation metadata
to have been refreshed by scripts/download_data.py first. It writes a small,
auditable forecast ledger to dashboard/data and never places trades.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DASHBOARD_DATA_DIR = PROJECT_ROOT / "dashboard" / "data"
DEFAULT_TICKER = "RELIANCE.NS"
DEFAULT_RESEARCH_START = "2005-01-01"
DEFAULT_MAX_STALENESS_DAYS = 7
MIN_OBSERVATIONS = 756

HISTORY_COLUMNS = [
    "ticker",
    "origin_date",
    "generated_at_utc",
    "model",
    "input_close",
    "predicted_next_log_return_bps",
    "predicted_next_close",
    "target_date",
    "actual_close",
    "actual_log_return_bps",
    "absolute_close_error",
    "absolute_return_error_bps",
    "direction_correct",
    "data_source",
    "price_adjustment",
]

MODEL_NAMES = ("random_walk", "drift")


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
    """Load the downloaded OHLC file and verify its metadata and structure."""
    name = safe_ticker_name(ticker)
    csv_path = RAW_DATA_DIR / f"{name}_daily.csv"
    metadata_path = RAW_DATA_DIR / f"{name}_metadata.json"

    if not csv_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Daily data is missing. Run "
            f"python scripts/download_data.py --ticker {ticker} first."
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


def make_daily_forecasts(
    data: pd.DataFrame,
    ticker: str,
    generated_at_utc: str,
    research_start: str = DEFAULT_RESEARCH_START,
) -> pd.DataFrame:
    """Produce simple, reproducible Random Walk and Drift forecasts."""
    required = {"date", "close"}
    if not required.issubset(data.columns):
        raise ValueError(f"Input data must contain: {sorted(required)}")

    window = data.loc[
        pd.to_datetime(data["date"]) >= pd.Timestamp(research_start)
    ].copy()
    window = window.sort_values("date").reset_index(drop=True)
    closes = pd.to_numeric(window["close"], errors="coerce").to_numpy(dtype=float)

    if len(closes) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Need at least {MIN_OBSERVATIONS} closes since {research_start}; "
            f"found {len(closes)}."
        )
    if not np.isfinite(closes).all() or (closes <= 0).any():
        raise ValueError("Closing prices must all be finite and positive.")

    origin_date = pd.Timestamp(window["date"].iloc[-1]).strftime("%Y-%m-%d")
    last_close = float(closes[-1])
    log_returns = np.diff(np.log(closes))
    if len(log_returns) == 0 or not np.isfinite(log_returns).all():
        raise ValueError("Could not estimate historical log returns.")

    # Keep this benchmark definition aligned with the existing expanding-window
    # research: estimate mean daily log return from the specified research start.
    mean_log_return = float(np.mean(log_returns))
    predictions = [
        ("random_walk", 0.0),
        ("drift", mean_log_return),
    ]

    rows: list[dict[str, Any]] = []
    for model, predicted_log_return in predictions:
        rows.append({
            "ticker": ticker,
            "origin_date": origin_date,
            "generated_at_utc": generated_at_utc,
            "model": model,
            "input_close": last_close,
            "predicted_next_log_return_bps": predicted_log_return * 10_000.0,
            "predicted_next_close": last_close * float(np.exp(predicted_log_return)),
            "target_date": "",
            "actual_close": np.nan,
            "actual_log_return_bps": np.nan,
            "absolute_close_error": np.nan,
            "absolute_return_error_bps": np.nan,
            "direction_correct": np.nan,
            "data_source": "Yahoo Finance via yfinance",
            "price_adjustment": "auto_adjust=True",
        })

    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


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


def score_pending_forecasts(
    history: pd.DataFrame,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Score each pending forecast against the first later observed session."""
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
        result.at[idx, "absolute_return_error_bps"] = abs(
            predicted_return_bps - actual_return_bps
        )

        # Direction is undefined for zero-return forecasts or a flat actual move.
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
) -> pd.DataFrame:
    """Score old forecasts and append one new pair only for a new OHLC date."""
    result = score_pending_forecasts(history, data)
    fresh = make_daily_forecasts(data, ticker, generated_at_utc, research_start)

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
    return result[HISTORY_COLUMNS]


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
) -> dict[str, Any]:
    """Create a compact JSON file for the dashboard to display."""
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
        "generated_at_utc": str(latest["generated_at_utc"].iloc[0]),
        "forecast_horizon": "next observed trading session",
        "model_scope": "Random Walk and Random Walk with Drift benchmarks",
        "data_source": "Yahoo Finance via yfinance",
        "price_adjustment": "auto_adjust=True",
        "forecasts": models,
        "note": (
            "Research output only. These simple forecasts are not trading "
            "recommendations and do not establish predictive or economic edge."
        ),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically to avoid leaving a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score prior daily forecasts and generate next-session baseline forecasts."
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker, default RELIANCE.NS")
    parser.add_argument("--research-start", default=DEFAULT_RESEARCH_START)
    parser.add_argument(
        "--max-staleness-days",
        type=int,
        default=DEFAULT_MAX_STALENESS_DAYS,
        help="Fail if latest OHLC date is older than this many calendar days.",
    )
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()
    if not ticker:
        parser.error("Ticker cannot be empty.")
    if args.max_staleness_days < 0:
        parser.error("--max-staleness-days cannot be negative.")

    data = load_validated_data(ticker)
    latest_date = pd.Timestamp(data["date"].iloc[-1]).normalize()
    today_utc = datetime.now(timezone.utc).date()
    age_days = (today_utc - latest_date.date()).days
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
    )
    latest_date_text = latest_date.strftime("%Y-%m-%d")
    latest_payload = build_latest_payload(updated_history, ticker, latest_date_text)

    # Persist the scored ledger and latest view only after all computations succeed.
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
    print(f"History records: {len(updated_history):,}")
    print(f"New forecasts: {int((updated_history['origin_date'] == latest_date_text).sum())}")
    print(f"History CSV: {history_path.relative_to(PROJECT_ROOT)}")
    print(f"Latest forecast JSON: {latest_path.relative_to(PROJECT_ROOT)}")
    print(latest_rows[[
        "model", "input_close", "predicted_next_log_return_bps",
        "predicted_next_close", "target_date", "absolute_close_error",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
