"""Tests for the PnL ledger and attribution (#25)."""
import numpy as np
import pandas as pd
import pytest

from gcsi import backtest, ledger
from gcsi.costs import CostModel

IDX = pd.bdate_range("2025-01-06", periods=6, tz="UTC", name="ts_event")


def _leg(raw, contract, adjusted=None):
    out = pd.DataFrame({"close": raw if adjusted is None else adjusted, "raw_close": raw,
                        "contract": contract}, index=IDX)
    out.attrs["adjustment"] = "ratio"
    return out


def _flat_legs():
    gc = _leg([2000.0, 2010, 2005, 2020, 2030, 2025], ["GCJ2025"] * 6)
    si = _leg([25.0, 25.2, 25.1, 25.4, 25.5, 25.3], ["SIK2025"] * 6)
    return gc, si


def _pos(gc=0, mgc=0, si=0, sil=0):
    return {"GC": gc, "MGC": mgc, "SI": si, "SIL": sil}


def test_pnl_costs_and_fills_by_hand():
    gc, si = _flat_legs()
    pos = pd.DataFrame([_pos(), _pos(5, 0, -6, 0), _pos(5, 0, -6, 0), _pos(5, 0, -6, 0),
                        _pos(), _pos()], index=IDX)
    led = ledger.build(pos, gc, si, CostModel())
    d = led.daily
    # day 2: 5 GC x (2005 - 2010) x 100 + (-6) SI x (25.1 - 25.2) x 5000
    assert d["gross_pnl"].iloc[2] == pytest.approx(5 * -5 * 100 + -6 * -0.1 * 5000)
    # day 4: last day held, close at 2030 / 25.5
    assert d["gross_pnl"].iloc[4] == pytest.approx(5 * 10 * 100 + -6 * 0.1 * 5000)
    # open on day 1 and close on day 4: 5 GC and 6 SI each time
    assert d["trade_costs"].iloc[1] == pytest.approx(5 * 7.5 + 6 * 15.0)
    assert d["trade_costs"].iloc[4] == pytest.approx(5 * 7.5 + 6 * 15.0)
    assert d["roll_costs"].sum() == 0
    assert d["net_pnl"].sum() == pytest.approx(d["gross_pnl"].sum() - d["costs"].sum())
    f = led.fills
    assert len(f) == 4 and set(f["kind"]) == {"trade"}
    assert f.loc[(f.leg == "GC") & (f.qty > 0), "price"].item() == 2010.0
    assert f["cost"].sum() == pytest.approx(d["costs"].sum())


def test_roll_is_charged_as_roll_and_gap_is_not_pnl():
    # GC rolls from J to M between day 2 and day 3; M trades $20 above J
    j = [2000.0, 2010, 2005, np.nan, np.nan, np.nan]
    m = [2020.0, 2030, 2025, 2040, 2050, 2045]
    raw = pd.Series([j[0], j[1], j[2], m[3], m[4], m[5]], index=IDX)
    held = ["GCJ2025"] * 3 + ["GCM2025"] * 3
    adj = raw.copy()
    adj.iloc[3:] = adj.iloc[2] * np.array(m[3:]) / m[2]  # ratio-adjusted continuation
    gc = _leg(raw.to_numpy(), held, adj.to_numpy())
    si = _leg([25.0] * 6, ["SIK2025"] * 6)
    pos = pd.DataFrame([_pos(2)] * 6, index=IDX)
    led = ledger.build(pos, gc, si, CostModel())
    d = led.daily
    # day 3 PnL is M's own move (2040 - 2025), not the J->M gap
    assert d["pnl_GC"].iloc[3] == pytest.approx(2 * (2040 - 2025) * 100)
    assert d["roll_costs"].iloc[2] == pytest.approx(2 * 2 * 7.5)  # close 2 J, open 2 M
    assert d["trade_costs"].iloc[2] == 0
    kinds = led.fills.set_index("kind")
    assert kinds.loc["roll_close", "contract"] == "GCJ2025" and kinds.loc["roll_close", "qty"] == -2
    assert kinds.loc["roll_open", "contract"] == "GCM2025" and kinds.loc["roll_open", "price"] == pytest.approx(2025)
    att = led.attribution()
    assert att["n_rolls"] == 1 and att["roll_costs"] == pytest.approx(30.0)


def test_attribution_sums_to_net():
    gc, si = _flat_legs()
    pos = pd.DataFrame([_pos(), _pos(5, 1, -6, -2), _pos(5, 1, -6, -2), _pos(), _pos(), _pos()],
                       index=IDX)
    oz = pd.DataFrame({"gc_oz": [0, 507.0, 507.0, 0, 0, 0], "si_oz": [0, -32100.0, -32100.0, 0, 0, 0]},
                      index=IDX)
    led = ledger.build(pos, gc, si, CostModel(), target_oz=oz)
    a = led.attribution()
    assert a["ideal_pnl"] + a["sizing_residual"] - a["trade_costs"] - a["roll_costs"] == \
        pytest.approx(a["net_pnl"])
    # the ideal leg earns exact ounces: day 2 = 507 x (2005-2010) + -32100 x (25.1-25.2)
    assert led.daily["ideal_pnl"].iloc[2] == pytest.approx(507 * -5 + -32100 * -0.1)


def test_simulate_matches_ledger_with_flat_costs(coint_pair, toy_positions):
    prices = coint_pair.prices.loc[toy_positions.index]
    gc = _leg_full(prices["GC"], "GCZ2030")
    si = _leg_full(prices["SI"], "SIZ2030")
    a = backtest.simulate(toy_positions, gc, si, cost_per_contract=2.5)
    b = ledger.build(toy_positions, gc, si, CostModel.flat(2.5)).daily
    pd.testing.assert_series_equal(a["net_pnl"], b["net_pnl"])


def _leg_full(raw, contract):
    out = pd.DataFrame({"close": raw, "raw_close": raw, "contract": contract})
    out.attrs["adjustment"] = "ratio"
    return out
