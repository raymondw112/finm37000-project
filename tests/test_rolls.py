"""Tests for gcsi.rolls (roll schedule and continuous series, #5)."""
import numpy as np
import pandas as pd
import pytest

from gcsi import rolls


def _bars(cross_on: str | None = "2025-01-15") -> pd.DataFrame:
    """Two GC contracts (Feb 25, Apr 25) plus a calendar spread row to be ignored.

    GCG5 closes at 100 + i, GCJ5 at 101 + i (a constant $1 carry gap).
    GCJ5 out-trades GCG5 from `cross_on` onward.
    """
    idx = pd.bdate_range("2024-12-02", "2025-03-20", tz="UTC", name="ts_event")
    i = np.arange(len(idx), dtype=float)
    cross = pd.Timestamp(cross_on, tz="UTC") if cross_on else idx.max() + pd.Timedelta(days=1)
    g_alive = idx <= pd.Timestamp("2025-02-26", tz="UTC")
    frames = [
        pd.DataFrame({"ts_event": idx[g_alive], "instrument_id": 1, "symbol": "GCG5",
                      "close": 100 + i[g_alive], "volume": 1000}),
        pd.DataFrame({"ts_event": idx, "instrument_id": 2, "symbol": "GCJ5",
                      "close": 101 + i, "volume": np.where(idx >= cross, 2000, 100)}),
        pd.DataFrame({"ts_event": idx, "instrument_id": 3, "symbol": "GCG5-GCJ5",
                      "close": -1.0, "volume": 5}),
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("sym, ref, expected", [
    ("GCZ6", "2026-10-13", ("GC", 2026, 12)),
    ("GCG7", "2026-10-13", ("GC", 2027, 2)),
    ("SIH0", "2029-11-01", ("SI", 2030, 3)),
    ("MGCJ5", "2025-01-02", ("MGC", 2025, 4)),
    # one-digit years repeat every decade (review on #22)
    ("GCZ0", "2010-06-07", ("GC", 2010, 12)),
    ("GCZ0", "2020-06-01", ("GC", 2020, 12)),
])
def test_parse_symbol(sym, ref, expected):
    assert rolls.parse_symbol(sym, pd.Timestamp(ref)) == expected


def test_parse_symbol_rejects_spreads():
    with pytest.raises(ValueError):
        rolls.parse_symbol("GCG5-GCJ5", pd.Timestamp("2025-01-01"))


def test_fnd_rule_rolls_before_first_notice():
    sched = rolls.roll_schedule(_bars(), "GC", rule="fnd", days_before_fnd=5)
    assert list(sched["symbol"]) == ["GCG5", "GCJ5"]       # spread row ignored
    # GCG5 FND = Fri 31 Jan 2025; 5 business days earlier = Fri 24 Jan -> switch Mon 27 Jan
    assert sched.loc[0, "fnd"] == pd.Timestamp("2025-01-31")
    assert sched.loc[0, "d1"] == pd.Timestamp("2025-01-27", tz="UTC")
    assert sched.loc[1, "d0"] == sched.loc[0, "d1"]


def test_volume_rule_uses_prior_day_volume():
    sched = rolls.roll_schedule(_bars("2025-01-15"), "GC", rule="volume")
    # crossover visible at the close of Wed 15 Jan -> new contract from Thu 16 Jan
    assert sched.loc[0, "d1"] == pd.Timestamp("2025-01-16", tz="UTC")


def test_volume_rule_never_later_than_fnd_rule():
    sched = rolls.roll_schedule(_bars(cross_on=None), "GC", rule="volume", days_before_fnd=5)
    assert sched.loc[0, "d1"] == pd.Timestamp("2025-01-27", tz="UTC")


def test_ratio_series_has_no_roll_jump():
    bars = _bars()
    cont = rolls.build_continuous(bars, "GC", adjustment="ratio")
    switch = pd.Timestamp("2025-01-27", tz="UTC")
    prev = cont.index[cont.index < switch].max()
    j = bars[bars.symbol == "GCJ5"].set_index("ts_event")["close"]
    # return on the switch day is the new contract's own return, not the carry gap
    assert cont.loc[switch, "close"] / cont.loc[prev, "close"] == pytest.approx(j[switch] / j[prev])
    # the raw series does jump by the gap
    assert cont.loc[switch, "raw_close"] - cont.loc[prev, "raw_close"] == pytest.approx(2.0)
    assert cont["roll"].sum() == 1 and cont.loc[switch, "roll"]
    assert cont.loc[prev, "symbol"] == "GCG5" and cont.loc[switch, "symbol"] == "GCJ5"


def test_back_adjusted_series_preserves_dollar_moves():
    cont = rolls.build_continuous(_bars(), "GC", adjustment="back")
    assert cont["close"].diff().dropna().eq(1.0).all()


def test_roll_log_records_gap():
    bars = _bars()
    sched = rolls.roll_schedule(bars, "GC")
    log = rolls.roll_log(bars, sched)
    assert len(log) == 1
    row = log.iloc[0]
    assert row.exec_date == pd.Timestamp("2025-01-24", tz="UTC")
    assert (row.from_symbol, row.to_symbol) == ("GCG5", "GCJ5")
    assert row.gap == pytest.approx(1.0)


def test_never_holds_after_fnd():
    bars = _bars()
    cont = rolls.build_continuous(bars, "GC")
    held_g = cont.index[cont["symbol"] == "GCG5"]
    assert held_g.max() < pd.Timestamp("2025-01-31", tz="UTC")


def test_align_pair_inner_join():
    a = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]))
    b = pd.DataFrame({"close": [5.0, 6.0]}, index=pd.to_datetime(["2025-01-02", "2025-01-03"]))
    out = rolls.align_pair(a, b)
    assert list(out.columns) == ["GC", "SI"] and len(out) == 2


def _decade_chain(start="2010-06-07", end="2021-06-30") -> pd.DataFrame:
    """Every active GC month over 11 years, each listed ~200 days before expiry.

    Long enough that symbols repeat (GCZ0 = Dec 2010 and Dec 2020). Each
    contract gets its own instrument_id and a price that encodes its year, so
    a merged contract would show up as a wrong price.
    """
    from gcsi import contracts
    idx = pd.bdate_range(start, end, tz="UTC", name="ts_event")
    frames = []
    for k, (y, m) in enumerate(contracts.active_contracts("GC", "2010-01-01", end)):
        expiry = contracts.last_trading_day(y, m).tz_localize("UTC")
        alive = idx[(idx <= expiry) & (idx >= expiry - pd.Timedelta(days=200))]
        if len(alive) == 0:
            continue
        frames.append(pd.DataFrame({
            "ts_event": alive, "instrument_id": 100 + k,
            "symbol": contracts.contract_symbol("GC", y, m),
            "close": float(y) + m / 100, "volume": 1000}))
    return pd.concat(frames, ignore_index=True)


def test_symbols_repeating_a_decade_apart_stay_separate():
    bars = _decade_chain()
    assert (bars["symbol"] == "GCZ0").any()
    sched = rolls.roll_schedule(bars, "GC")
    keys = list(zip(sched["year"], sched["month"]))
    assert keys == sorted(keys) and len(keys) == len(set(keys))
    assert {"GCZ2010", "GCZ2020"} <= set(sched["contract"])
    assert sched.loc[sched.contract == "GCZ2020", "d0"].iloc[0].year == 2020
    # every day, the raw price is the one of the contract we say we hold
    cont = rolls.build_continuous(bars, "GC", adjustment="raw", schedule=sched)
    expected = cont["contract"].str[-4:].astype(int) + cont["contract"].map(
        dict(zip(sched["contract"], sched["month"]))) / 100
    assert (cont["raw_close"] == expected).all()
    assert len(rolls.roll_log(bars, sched)) == len(sched) - 1


# --- against the #14 synthetic fixtures (runs once #20 is merged) -------------

def test_volume_rule_recovers_synthetic_roll_dates():
    syn = pytest.importorskip("gcsi.synthetic")
    for root in ("GC", "SI"):
        pe = syn.per_expiry_bars(root, seed=1)
        sched = rolls.roll_schedule(pe.bars, root, rule="volume", days_before_fnd=2)
        for sym, roll_day in pe.roll_dates.items():
            got = sched.loc[sched.symbol == sym, "d1"].iloc[0]
            # crossover on roll_day is acted on at that close, new contract next day
            assert got == roll_day + pd.offsets.BDay(1)


def test_synthetic_ratio_series_removes_carry_jump():
    syn = pytest.importorskip("gcsi.synthetic")
    pe = syn.per_expiry_bars("GC", seed=1, annual_carry=0.04)
    cont = rolls.build_continuous(pe.bars, "GC", adjustment="ratio", rule="volume", days_before_fnd=2)
    raw_ret = cont["raw_close"].pct_change()[cont["roll"]]
    adj_ret = cont["close"].pct_change()[cont["roll"]]
    # raw return on a roll day = market move x carry gap; the ratio series keeps
    # only the market move, so their quotient is exactly the known roll ratio
    np.testing.assert_allclose(((1 + raw_ret) / (1 + adj_ret)).to_numpy(),
                               pe.roll_ratios.to_numpy(), rtol=1e-12)
