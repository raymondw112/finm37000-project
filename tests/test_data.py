"""Tests for gcsi.data (Databento ingestion, #4). No network: everything runs on the #14 mock."""
from __future__ import annotations

import pandas as pd
import pytest

from gcsi import data, rolls
from gcsi import synthetic as syn

START, END = "2024-12-01", "2025-07-01"


class _Metadata:
    """Adds the free `metadata` endpoints the #14 mock lacks."""

    def __init__(self, cost: float):
        self.cost = cost
        self.cost_calls: list[dict] = []

    def get_cost(self, **kwargs) -> float:
        self.cost_calls.append(kwargs)
        return self.cost


@pytest.fixture
def client(mock_databento_client):
    """The #14 mock, plus get_cost, an instrument_class column and a calendar spread.

    Real parent symbology (GC.FUT) returns spreads like GCG5-GCJ5 alongside the
    outrights, in both the definition and the bar schemas; the mock doesn't, so
    we add one here.
    """
    c = mock_databento_client
    c.metadata = _Metadata(cost=0.10)
    for root, d in c.data.items():
        first, second = d.definitions["raw_symbol"].iloc[:2]
        spread_id = 999_000 + len(root)
        spread_sym = f"{first}-{second}"
        defs = d.definitions.assign(instrument_class="F")
        spread_def = pd.DataFrame({"instrument_id": [spread_id], "raw_symbol": [spread_sym],
                                   "expiration": [defs["expiration"].iloc[0]],
                                   "instrument_class": ["S"]})
        bars = d.bars
        spread_bars = bars[bars["symbol"] == first].assign(
            instrument_id=spread_id, symbol=spread_sym, close=-1.0, open=-1.0, high=-1.0, low=-1.0)
        c.data[root] = syn.PerExpiry(
            bars=pd.concat([bars, spread_bars], ignore_index=True),
            definitions=pd.concat([defs, spread_def], ignore_index=True),
            roll_dates=d.roll_dates, roll_ratios=d.roll_ratios, annual_carry=d.annual_carry)
    return c


@pytest.fixture
def load(client, tmp_path):
    def _load(root="GC", start=START, end=END, **kw):
        return data.load_bars(root, start, end, client, cache_dir=tmp_path, verbose=False, **kw)
    return _load


def test_columns_and_dtypes(load):
    bars = load()
    assert list(bars.columns) == data.BAR_COLUMNS
    assert str(bars["ts_event"].dt.tz) == "UTC"
    assert str(bars["expiration"].dt.tz) == "UTC"
    assert bars["instrument_id"].dtype == "int64" and bars["volume"].dtype == "int64"
    assert all(bars[c].dtype == "float64" for c in ["open", "high", "low", "close"])
    assert bars["ts_event"].is_monotonic_increasing


def test_spreads_removed(load, client):
    bars = load()
    assert not bars["symbol"].str.contains("-").any()
    assert set(bars["symbol"]) == set(client.data["GC"].definitions.query("instrument_class == 'F'")["raw_symbol"])
    assert (bars["close"] > 0).all()


def test_expiration_joined_from_definitions(load, gc_expiries):
    bars = load()
    expected = gc_expiries.definitions.set_index("raw_symbol")["expiration"]
    assert (bars["expiration"] == bars["symbol"].map(expected)).all()


def test_uses_parent_symbology(load, client):
    load("SI")
    assert {c["symbols"][0] for c in client.calls} == {"SI.FUT"}
    assert all(c["stype_in"] == "parent" and c["dataset"] == data.DATASET for c in client.calls)


def test_second_call_served_from_cache(load, client):
    first = load()
    n_calls = len(client.calls)
    assert n_calls > 0
    n_cost = len(client.metadata.cost_calls)
    second = load()
    assert len(client.calls) == n_calls                    # no new pulls
    assert len(client.metadata.cost_calls) == n_cost       # nothing to price either
    pd.testing.assert_frame_equal(first, second)


def test_one_request_per_calendar_year(load, client):
    load()
    years = sorted((c["schema"], c["start"][:4]) for c in client.calls)
    assert years == [("definition", "2024"), ("definition", "2025"),
                     ("ohlcv-1d", "2024"), ("ohlcv-1d", "2025")]


def test_partial_year_repulled_only_when_end_moves(load, client):
    load(end="2025-03-01")
    n = len(client.calls)
    load(end="2025-03-01")
    assert len(client.calls) == n
    later = load(end=END)
    new = client.calls[n:]
    assert {c["start"][:4] for c in new} == {"2025"}        # 2024 stays cached
    assert later["ts_event"].max() >= pd.Timestamp("2025-03-01", tz="UTC")


def test_start_end_respected(load):
    bars = load(start="2025-01-10", end="2025-02-10")
    assert bars["ts_event"].min() >= pd.Timestamp("2025-01-10", tz="UTC")
    assert bars["ts_event"].max() < pd.Timestamp("2025-02-10", tz="UTC")


def test_cost_guard_raises_above_threshold(load, client):
    client.metadata.cost = 50.0
    with pytest.raises(data.CostLimitExceeded):
        load(max_cost=5.0)
    assert client.calls == []                              # checked before any pull
    assert len(load(max_cost=5.0, yes=True)) > 0


def test_end_before_start_rejected(load):
    with pytest.raises(ValueError):
        load(start="2025-02-01", end="2025-01-01")


def test_feeds_build_continuous(load):
    bars = load()
    cont = rolls.build_continuous(bars, "GC", adjustment="ratio", rule="volume", days_before_fnd=2)
    assert cont["roll"].sum() >= 1
    assert cont["contract"].notna().all()


def test_validate_drops_bad_rows_and_reports_missing_months(load):
    bars = load()
    dirty = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)        # duplicate
    dirty.loc[1, "close"] = 0.0                                          # bad price
    dirty.loc[2, "volume"] = -5                                          # bad volume
    clean, report = data.validate_bars(dirty, "GC", START, END)
    assert len(clean) == len(bars) - 2
    assert report.dropped == {"duplicate (contract, day)": 1,
                              "missing or non-positive price": 1, "negative volume": 1}
    # the fixture lists only Feb/Apr/Jun 2025, so Aug 2025 (expires inside a longer window) is missing
    _, report = data.validate_bars(bars, "GC", START, "2025-09-01")
    assert any(m.startswith("GCQ5") for m in report.missing_months)
    assert "GCQ5" in str(report)


def test_api_key_from_env_is_masked(monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "db-test-not-a-real-key")
    key = data.get_api_key()
    assert str.__str__(key) == "db-test-not-a-real-key"
    assert "db-test" not in str(key) and "db-test" not in f"{key}"


def test_outright_definitions_real_layout():
    # real to_df has both a mapped `symbol` and `raw_symbol`, and re-sends each
    # instrument every session; without instrument_class, spreads go by symbol shape
    defs = pd.DataFrame({
        "instrument_id": [1, 1, 2, 3],
        "symbol": ["GC.FUT"] * 4,
        "raw_symbol": ["GCG5", "GCG5", "GCG5-GCJ5", "GCJ5"],
        "expiration": pd.to_datetime(["2025-02-26"] * 3 + ["2025-04-28"], utc=True),
    })
    out = data._outright_definitions(defs, "GC")
    assert list(out.columns) == data.DEF_COLUMNS
    assert list(out["symbol"]) == ["GCG5", "GCJ5"]
