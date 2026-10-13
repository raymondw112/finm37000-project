"""Tests for gcsi.contracts (contract specs and calendar helpers)."""
import pandas as pd
import pytest

from gcsi import contracts as c


@pytest.mark.parametrize(
    "root, multiplier, tick_value",
    [("GC", 100, 10.0), ("MGC", 10, 1.0), ("SI", 5000, 25.0), ("SIL", 1000, 5.0)],
)
def test_spec_terms(root, multiplier, tick_value):
    spec = c.get_spec(root)
    assert spec.multiplier == multiplier
    assert spec.tick_value == pytest.approx(tick_value)


def test_micro_is_fraction_of_standard():
    assert c.GC.multiplier / c.MGC.multiplier == 10
    assert c.SI.multiplier / c.SIL.multiplier == 5


def test_unknown_root():
    with pytest.raises(KeyError):
        c.get_spec("CL")


def test_notional_and_pnl():
    assert c.notional("GC", 2500.0, 2) == 500_000.0
    # short 1 SI, price falls 10c: +$500
    assert c.pnl("SI", 30.00, 29.90, -1) == pytest.approx(500.0)


def test_first_notice_day_weekday_and_weekend():
    # Dec 2026 GC: Nov 30 2026 is a Monday -> FND Nov 30.
    assert c.first_notice_day(2026, 12) == pd.Timestamp("2026-11-30")
    # Mar 2026 SI: Feb 28 2026 is a Saturday -> FND Fri Feb 27.
    assert c.first_notice_day(2026, 3) == pd.Timestamp("2026-02-27")


def test_first_notice_day_with_holiday():
    assert c.first_notice_day(2026, 12, holidays=["2026-11-30"]) == pd.Timestamp("2026-11-27")


def test_last_trading_day():
    # Dec 31 2026 is a Thursday: business days ...28, 29, 30, 31 -> third-last is the 29th.
    assert c.last_trading_day(2026, 12) == pd.Timestamp("2026-12-29")


def test_contract_symbol():
    assert c.contract_symbol("GC", 2026, 12) == "GCZ6"
    assert c.contract_symbol("si", 2027, 3) == "SIH7"


def test_active_contracts_respect_cycle():
    gc = c.active_contracts("GC", "2026-01-01", "2026-06-30")
    assert all(m in c.GC.active_months for _, m in gc)
    assert (2026, 2) in gc and (2027, 2) in gc
    si = c.active_contracts("SI", "2026-01-01", "2026-06-30")
    assert (2026, 3) in si and (2026, 2) not in si


@pytest.mark.parametrize(
    "ounces, standard, micro, expected",
    [
        (12_345, "SI", "SIL", (2, 2)),
        (-250, "GC", "MGC", (-2, -5)),
        (40, "GC", "MGC", (0, 4)),
        (0, "SI", "SIL", (0, 0)),
    ],
)
def test_ounce_neutral_mix(ounces, standard, micro, expected):
    assert c.ounce_neutral_mix(ounces, standard, micro) == expected


def test_ounce_neutral_mix_rejects_mixed_metals():
    with pytest.raises(ValueError):
        c.ounce_neutral_mix(100, "GC", "SIL")
