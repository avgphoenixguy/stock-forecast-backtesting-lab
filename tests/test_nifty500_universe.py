"""Tests for the Nifty 500 stock directory parser."""
import pandas as pd
import pytest

from scripts.update_nifty500_universe import parse_constituents_csv, write_directory


def sample_csv():
    rows = []
    for idx in range(500):
        symbol = "RELIANCE" if idx == 0 else f"TEST{idx:03d}"
        name = "Reliance Industries Limited" if idx == 0 else f"Test Company {idx:03d}"
        rows.append(f'"{name}","Industrials","{symbol}","EQ","INE{idx:09d}"')
    return 'Company Name,Industry,Symbol,Series,ISIN Code\n' + "\n".join(rows) + "\n"


def test_parse_constituents_builds_searchable_tickers():
    frame = parse_constituents_csv(
        sample_csv(), "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
        fetched_at_utc="2026-10-25T00:00:00+00:00",
    )
    assert len(frame) == 500
    reliance = frame.loc[frame["symbol"] == "RELIANCE"].iloc[0]
    assert reliance["ticker"] == "RELIANCE.NS"
    assert reliance["company_name"] == "Reliance Industries Limited"
    assert "reliance industries limited" in reliance["search_text"]
    assert reliance["universe"] == "NIFTY 500"


def test_parse_constituents_rejects_too_small_directory():
    small = 'Company Name,Industry,Symbol,Series,ISIN Code\n"Reliance Industries","Energy","RELIANCE","EQ","INE000A01000"\n'
    with pytest.raises(ValueError, match="roughly 500"):
        parse_constituents_csv(small, "test-source")


def test_write_directory_rejects_duplicate_symbols(tmp_path):
    frame = parse_constituents_csv(sample_csv(), "test-source")
    frame.loc[1, "symbol"] = frame.loc[0, "symbol"]
    with pytest.raises(ValueError, match="duplicate"):
        write_directory(frame, tmp_path / "nifty500.csv")
