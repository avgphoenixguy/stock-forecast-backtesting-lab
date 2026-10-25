"""Refresh the searchable Nifty 500 constituent directory from NSE Indices.

The CSV is a reference directory, not a market-data feed. The hosted app uses
it for company-name/symbol autocomplete. Historical prices are fetched separately.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "dashboard" / "data" / "nifty500_universe.csv"
MIN_CONSTITUENTS = 450
MAX_CONSTITUENTS = 550

SOURCE_URLS = (
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,application/octet-stream,*/*",
    "Referer": "https://www.niftyindices.com/",
}


def _normalise_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def parse_constituents_csv(
    content: bytes | str,
    source_url: str,
    fetched_at_utc: str | None = None,
) -> pd.DataFrame:
    """Parse and validate an official Nifty 500 CSV response."""
    frame = pd.read_csv(io.BytesIO(content) if isinstance(content, bytes) else io.StringIO(content))
    column_map = {_normalise_column(column): column for column in frame.columns}

    symbol_key = next(
        (key for key in ("symbol", "nsetrading symbol", "tradingsymbol") if key in column_map),
        None,
    )
    company_key = next(
        (key for key in ("companyname", "company", "name") if key in column_map),
        None,
    )
    if symbol_key is None:
        raise ValueError(f"Constituent CSV does not contain a Symbol column: {list(frame.columns)}")

    industry_key = next((key for key in ("industry", "sector") if key in column_map), None)
    series_key = next((key for key in ("series",) if key in column_map), None)
    isin_key = next((key for key in ("isincode", "isin") if key in column_map), None)

    result = pd.DataFrame()
    result["company_name"] = (
        frame[column_map[company_key]].astype(str).str.strip()
        if company_key else frame[column_map[symbol_key]].astype(str).str.strip()
    )
    result["symbol"] = frame[column_map[symbol_key]].astype(str).str.strip().str.upper()
    result["ticker"] = result["symbol"] + ".NS"
    result["industry"] = (
        frame[column_map[industry_key]].fillna("").astype(str).str.strip()
        if industry_key else ""
    )
    result["series"] = (
        frame[column_map[series_key]].fillna("").astype(str).str.strip().str.upper()
        if series_key else ""
    )
    result["isin"] = (
        frame[column_map[isin_key]].fillna("").astype(str).str.strip().str.upper()
        if isin_key else ""
    )

    result = result.loc[
        result["symbol"].ne("")
        & result["symbol"].ne("NAN")
        & result["company_name"].ne("")
        & result["company_name"].ne("NAN")
    ].copy()
    result = result.drop_duplicates(subset=["symbol"], keep="first")
    result = result.sort_values(["company_name", "symbol"], kind="stable").reset_index(drop=True)

    if not MIN_CONSTITUENTS <= len(result) <= MAX_CONSTITUENTS:
        raise ValueError(
            f"Expected roughly 500 Nifty 500 constituents; received {len(result)}. "
            "Refusing to replace the last good directory."
        )
    if "RELIANCE" not in set(result["symbol"]):
        raise ValueError("The constituent list is missing RELIANCE; refusing to publish it.")

    timestamp = fetched_at_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["universe"] = "NIFTY 500"
    result["source_url"] = source_url
    result["fetched_at_utc"] = timestamp
    result["search_text"] = (
        result["company_name"].str.lower() + " "
        + result["symbol"].str.lower() + " "
        + result["ticker"].str.lower() + " "
        + result["industry"].str.lower()
    )
    columns = [
        "company_name", "symbol", "ticker", "industry", "series", "isin",
        "universe", "source_url", "fetched_at_utc", "search_text",
    ]
    return result[columns]


def fetch_constituents(
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, str]:
    """Fetch the latest Nifty 500 constituent CSV from official NSE Indices URLs."""
    client = session or requests.Session()
    errors: list[str] = []
    for url in SOURCE_URLS:
        try:
            response = client.get(url, headers=HEADERS, timeout=(10, 35))
            response.raise_for_status()
            parsed = parse_constituents_csv(response.content, source_url=url)
            return parsed, url
        except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Could not refresh the Nifty 500 constituent directory from official sources. "
        "The last saved directory, if present, should be retained. Details: "
        + " | ".join(errors)
    )


def write_directory(frame: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Validate the received frame and atomically replace the current directory."""
    required = {"company_name", "symbol", "ticker", "industry", "universe", "source_url"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Directory is missing required columns: {missing}")
    if frame["symbol"].duplicated().any():
        raise ValueError("Directory contains duplicate symbols.")
    if not MIN_CONSTITUENTS <= len(frame) <= MAX_CONSTITUENTS:
        raise ValueError(f"Unexpected constituent count: {len(frame)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the Nifty 500 search directory.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    frame, source_url = fetch_constituents()
    write_directory(frame, args.output)
    print("===== NIFTY 500 DIRECTORY REFRESHED =====")
    print(f"Constituents: {len(frame):,}")
    print(f"Source: {source_url}")
    print(f"Fetched at UTC: {frame['fetched_at_utc'].iloc[0]}")
    print(f"Output: {args.output}")
    print(frame[["company_name", "symbol", "ticker", "industry"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
