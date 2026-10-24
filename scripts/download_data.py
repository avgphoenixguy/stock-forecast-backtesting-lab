"""Download and validate daily historical data for a stock ticker."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_TICKER = "RELIANCE.NS"

# These are research windows, not download limits.
PRIMARY_RESEARCH_START = "2005-01-01"
SECONDARY_RESEARCH_START = "2015-01-01"

MIN_OBSERVATIONS_FOR_INITIAL_RESEARCH = 1000

# Allow only deviations consistent with floating-point precision.
# This is scale-aware and does not alter the downloaded prices.
FLOAT_TOLERANCE_MULTIPLIER = 8.0


def safe_ticker_name(ticker: str) -> str:
    """Convert a ticker into a filesystem-safe filename component."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")
    return name.upper() or "STOCK"


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 hash for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_history(ticker: str) -> pd.DataFrame:
    """Download maximum daily history and validate its structure."""

    print(f"Ticker: {ticker}")
    print("Requested period: maximum available daily history")
    print("Downloading from Yahoo Finance via yfinance...")

    try:
        history = yf.Ticker(ticker).history(
            period="max",
            interval="1d",
            auto_adjust=True,
            actions=True,
            repair=True,
            keepna=False,
            timeout=30,
            raise_errors=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Download failed for {ticker}: "
            f"{type(exc).__name__}: {exc}\n"
            "Check the ticker symbol, internet connection and data-source "
            "availability. No model or research result has been produced."
        ) from exc

    if history is None or history.empty:
        raise RuntimeError(
            f"No historical observations were returned for {ticker}. "
            "Verify the ticker symbol and data availability."
        )

    data = history.copy()

    # Preserve the exchange-local calendar date when removing the timezone.
    index = pd.DatetimeIndex(data.index)

    if index.tz is not None:
        index = index.tz_localize(None)

    data.index = index
    data.index.name = "date"
    data = data.reset_index()

    data.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in data.columns
    ]

    required = ["date", "open", "high", "low", "close", "volume"]
    missing_columns = [
        column for column in required if column not in data.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"Missing required columns: {missing_columns}. "
            f"Received columns: {list(data.columns)}"
        )

    # Keep a consistent schema if corporate-action columns are absent.
    for column in ("dividends", "stock_splits"):
        if column not in data.columns:
            data[column] = 0.0

        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        ).fillna(0.0)

    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    if data["date"].isna().any():
        raise RuntimeError("Invalid dates were found in the downloaded data.")

    data["date"] = data["date"].dt.normalize()
    data = data.sort_values("date").reset_index(drop=True)

    if data["date"].duplicated().any():
        duplicates = (
            data.loc[data["date"].duplicated(keep=False), "date"]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )
        raise RuntimeError(
            f"Duplicate trading dates found: {duplicates[:10]}"
        )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dividends",
        "stock_splits",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    numeric_values = data[numeric_columns].to_numpy(dtype=np.float64)

    if not np.isfinite(numeric_values).all():
        raise RuntimeError(
            "Missing, invalid or infinite numeric values were found."
        )

    price_columns = ["open", "high", "low", "close"]
    prices = data[price_columns]

    if (prices <= 0).any().any():
        raise RuntimeError("Non-positive OHLC prices were found.")

    if (data["volume"] < 0).any():
        raise RuntimeError("Negative volume was found.")

    # ---------------------------------------------------------------
    # OHLC VALIDATION
    # ---------------------------------------------------------------
    # Strict comparisons can incorrectly reject prices when floating-
    # point arithmetic produces tiny differences, e.g. close is higher
    # than high by 1e-15.
    #
    # Tolerance for each row:
    #   8 * machine epsilon * max(1, maximum absolute OHLC price)
    #
    # This accommodates floating-point-scale discrepancies while
    # retaining strict validation against meaningful OHLC errors.
    # No source prices are rounded or changed.
    # ---------------------------------------------------------------

    open_values = data["open"].to_numpy(dtype=np.float64)
    high_values = data["high"].to_numpy(dtype=np.float64)
    low_values = data["low"].to_numpy(dtype=np.float64)
    close_values = data["close"].to_numpy(dtype=np.float64)

    row_maximum = np.maximum.reduce(
        [
            np.abs(open_values),
            np.abs(high_values),
            np.abs(low_values),
            np.abs(close_values),
        ]
    )

    row_scale = np.maximum(1.0, row_maximum)

    row_tolerance = (
        FLOAT_TOLERANCE_MULTIPLIER
        * np.finfo(np.float64).eps
        * row_scale
    )

    max_open_close = np.maximum(open_values, close_values)
    min_open_close = np.minimum(open_values, close_values)

    # Original strict rule, retained to identify tiny rounding-only cases.
    strict_inconsistent = (
        (high_values < low_values)
        | (high_values < max_open_close)
        | (low_values > min_open_close)
    )

    # Tolerance-aware rule: only violations beyond the row-specific
    # floating-point tolerance are considered material.
    material_inconsistent = (
        (high_values + row_tolerance < low_values)
        | (high_values + row_tolerance < max_open_close)
        | (low_values - row_tolerance > min_open_close)
    )

    rounding_level_rows = (
        strict_inconsistent & ~material_inconsistent
    )

    if material_inconsistent.any():
        bad_positions = np.flatnonzero(material_inconsistent)
        bad_dates = (
            data.loc[material_inconsistent, "date"]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        samples = []

        for position in bad_positions[:10]:
            samples.append(
                {
                    "date": data.loc[position, "date"].strftime("%Y-%m-%d"),
                    "open": float(open_values[position]),
                    "high": float(high_values[position]),
                    "low": float(low_values[position]),
                    "close": float(close_values[position]),
                    "allowed_tolerance": float(row_tolerance[position]),
                }
            )

        raise RuntimeError(
            "Material OHLC inconsistencies found beyond floating-point "
            f"tolerance on: {bad_dates[:10]}\n"
            f"Sample observations: {samples}"
        )

    # Keep diagnostics for the metadata and terminal summary.
    data.attrs["ohlc_rounding_level_rows"] = int(
        rounding_level_rows.sum()
    )
    data.attrs["ohlc_max_row_tolerance"] = float(row_tolerance.max())
    data.attrs["ohlc_tolerance_multiplier"] = FLOAT_TOLERANCE_MULTIPLIER

    output_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dividends",
        "stock_splits",
    ]

    return data[output_columns].copy().assign(
        _temporary_placeholder=lambda frame: np.nan
    ).drop(columns="_temporary_placeholder").pipe(
        lambda result: _attach_validation_metadata(
            result,
            data.attrs,
        )
    )


def _attach_validation_metadata(
    data: pd.DataFrame,
    validation_attributes: dict,
) -> pd.DataFrame:
    """Attach validation diagnostics to the returned DataFrame."""
    data.attrs.update(validation_attributes)
    return data


def main() -> None:
    """Run the downloader, validate the data and write metadata."""

    parser = argparse.ArgumentParser(
        description=(
            "Download and validate maximum available daily stock history."
        )
    )

    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help=(
            "Yahoo Finance ticker symbol. Defaults to RELIANCE.NS. "
            "Example: --ticker TCS.NS"
        ),
    )

    args = parser.parse_args()
    ticker = args.ticker.strip().upper()

    if not ticker:
        parser.error("Ticker symbol cannot be empty.")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    filename_prefix = safe_ticker_name(ticker)
    csv_path = RAW_DATA_DIR / f"{filename_prefix}_daily.csv"
    metadata_path = RAW_DATA_DIR / f"{filename_prefix}_metadata.json"

    data = download_history(ticker)

    first_date = data["date"].iloc[0]
    last_date = data["date"].iloc[-1]
    row_count = int(len(data))

    sufficient_observations = (
        row_count >= MIN_OBSERVATIONS_FOR_INITIAL_RESEARCH
    )

    rounding_level_rows = int(
        data.attrs.get("ohlc_rounding_level_rows", 0)
    )
    max_row_tolerance = float(
        data.attrs.get("ohlc_max_row_tolerance", 0.0)
    )

    data.to_csv(csv_path, index=False, date_format="%Y-%m-%d")

    first_date_text = first_date.strftime("%Y-%m-%d")
    last_date_text = last_date.strftime("%Y-%m-%d")

    metadata = {
        "project": "Stock Forecast & Backtesting Lab",
        "ticker": ticker,
        "requested_history": "maximum_available",
        "interval": "1d",
        "first_observation_date": first_date_text,
        "last_observation_date": last_date_text,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "Yahoo Finance via yfinance",
        "data_source_limitation": (
            "Unofficial third-party data access. Coverage, adjusted prices "
            "and historical values may change. Verify important results "
            "against an appropriate authoritative source when possible."
        ),
        "price_adjustment": {
            "auto_adjust": True,
            "repair": True,
            "description": (
                "OHLC prices use yfinance automatic adjustment and "
                "price-repair options. Volume is source-reported. "
                "Dividend and stock-split action columns are retained."
            ),
        },
        "versions": {
            "python": sys.version.split()[0],
            "yfinance": yf.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "row_count": row_count,
        "columns": list(data.columns),
        "research_windows": {
            "planned_primary_start": PRIMARY_RESEARCH_START,
            "planned_secondary_start": SECONDARY_RESEARCH_START,
            "history_starts_by_primary_start": (
                first_date_text <= PRIMARY_RESEARCH_START
            ),
            "history_starts_by_secondary_start": (
                first_date_text <= SECONDARY_RESEARCH_START
            ),
            "note": (
                "These are planned analysis windows. Actual training "
                "eligibility and split dates will be determined later."
            ),
        },
        "research_readiness": {
            "minimum_observations_reference": (
                MIN_OBSERVATIONS_FOR_INITIAL_RESEARCH
            ),
            "meets_minimum_observation_reference": (
                sufficient_observations
            ),
            "note": (
                "This is only a basic coverage indicator, not proof that "
                "the ticker is suitable for modelling. Formal eligibility "
                "checks will be implemented in the research pipeline."
            ),
        },
        "validation": {
            "required_columns_present": True,
            "dates_valid_and_unique": True,
            "dates_sorted": True,
            "numeric_values_finite": True,
            "ohlc_prices_positive": True,
            "volume_non_negative": True,
            "ohlc_consistent_with_floating_point_tolerance": True,
            "rounding_level_ohlc_rows": rounding_level_rows,
            "tolerance_method": (
                "8 * float64 machine epsilon * "
                "max(1, maximum absolute OHLC price per row)"
            ),
            "maximum_row_tolerance": max_row_tolerance,
            "missing_values_by_column": {
                column: int(data[column].isna().sum())
                for column in data.columns
            },
        },
        "csv_sha256": calculate_sha256(csv_path),
        "csv_file": str(csv_path.relative_to(PROJECT_ROOT)),
        "metadata_file": str(metadata_path.relative_to(PROJECT_ROOT)),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n===== DOWNLOAD SUCCESSFUL =====")
    print(f"Ticker: {ticker}")
    print(f"Observations: {row_count:,}")
    print(f"First trading date: {first_date_text}")
    print(f"Last trading date: {last_date_text}")
    print(f"Rounding-level OHLC rows accepted: {rounding_level_rows}")
    print(f"Maximum row tolerance: {max_row_tolerance:.3e}")
    print(
        f"Meets 1,000-observation reference: {sufficient_observations}"
    )
    print(f"CSV: {csv_path}")
    print(f"Metadata: {metadata_path}")
    print(f"CSV SHA-256: {metadata['csv_sha256']}")

    if not sufficient_observations:
        print(
            "\nWARNING: This stock has fewer than 1,000 observations. "
            "The data has been saved, but we must assess training "
            "and validation eligibility before modelling."
        )

    if first_date_text > PRIMARY_RESEARCH_START:
        print(
            f"\nNOTE: This history starts after {PRIMARY_RESEARCH_START}. "
            "The planned primary window cannot be fully used for this "
            "ticker; its actual coverage will be respected."
        )

    print("\n===== FIRST 3 ROWS =====")
    print(data.head(3).to_string(index=False))

    print("\n===== LAST 3 ROWS =====")
    print(data.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()