"""Tests for the walk-forward backtest skeleton (#10)."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from gcsi import backtest, contracts, rolls
from gcsi.backtest import LEGS, BacktestConfig

GC_MULT = contracts.GC.multiplier
SI_MULT = contracts.SI.multiplier


def _leg(raw: pd.Series, contract: str | pd.Series = "X", adjusted: pd.Series | None = None,
         adjustment: str = "ratio") -> pd.DataFrame:
    """A minimal `rolls.build_continuous`-shaped frame."""
    held = contract if isinstance(contract, pd.Series) else pd.Series(contract, index=raw.index)
    out = pd.DataFrame({"close": raw if adjusted is None else adjusted, "raw_close": raw,
                        "contract": held})
    out["roll"] = out["contract"].ne(out["contract"].shift()) & out["contract"].shift().notna()
    out.attrs["adjustment"] = adjustment
    return out


@pytest.fixture(scope="module")
def pair_legs(coint_pair):
    """coint_pair prices as single-contract GC and SI legs (no rolls)."""
    prices = coint_pair.prices
    return _leg(prices["GC"], "GCZ2030"), _leg(prices["SI"], "SIZ2030")


CONFIG = BacktestConfig(window=120, z_lookback=40, entry=1.5, cost_per_contract=2.5)


# --------------------------------------------------------------------------- #
# Flat signal
# --------------------------------------------------------------------------- #
def test_flat_signal_gives_zero_pnl_and_cost(pair_legs):
    gc, si = pair_legs
    flat = BacktestConfig(entry=1e9, stop=2e9, cost_per_contract=2.5, initial_equity=1e6)
    out = backtest.run(gc, si, flat)
    assert (out[list(LEGS)] == 0).all().all()
    assert (out["signal"] == 0).all()
    assert (out["gross_pnl"] == 0).all() and (out["costs"] == 0).all()
    assert (out["equity"] == 1e6).all()


# --------------------------------------------------------------------------- #
# Hand-computed PnL on the toy positions
# --------------------------------------------------------------------------- #
def test_toy_positions_pnl_to_the_cent(pair_legs, toy_positions):
    gc, si = (leg.loc[toy_positions.index] for leg in pair_legs)
    cost = 2.5
    out = backtest.simulate(toy_positions, gc, si, cost_per_contract=cost)
    G, S = gc["raw_close"].to_numpy(), si["raw_close"].to_numpy()

    # long spread: +5 GC / -8 SI filled at the close of row 10, closed at row 30
    long_leg = 5 * GC_MULT * (G[30] - G[10]) - 8 * SI_MULT * (S[30] - S[10])
    # short spread: -5 GC / +8 SI from the close of row 50 to the close of row 70
    short_leg = -5 * GC_MULT * (G[70] - G[50]) + 8 * SI_MULT * (S[70] - S[50])
    assert out["gross_pnl"].sum() == pytest.approx(long_leg + short_leg, abs=0.005)

    # one day's mark, and the day of entry earns nothing
    assert out["gross_pnl"].iloc[11] == pytest.approx(
        5 * GC_MULT * (G[11] - G[10]) - 8 * SI_MULT * (S[11] - S[10]), abs=0.005)
    assert out["gross_pnl"].iloc[10] == 0

    # 4 trades (2 entries, 2 exits) of 5 GC + 8 SI each
    assert out["costs"].sum() == pytest.approx(4 * 13 * cost, abs=0.005)
    assert out["costs"].iloc[10] == pytest.approx(13 * cost)
    assert out["net_pnl"].sum() == pytest.approx(long_leg + short_leg - 52 * cost, abs=0.005)
    assert out["equity"].iloc[-1] == pytest.approx(out["net_pnl"].sum())


# --------------------------------------------------------------------------- #
# Rolls
# --------------------------------------------------------------------------- #
def _rolling_gc(n: int = 20, k: int = 10):
    """Flat old contract at 2000, flat new contract at 2040, switch on row k."""
    idx = pd.bdate_range("2025-01-02", periods=n, tz="UTC")
    held = pd.Series(np.where(np.arange(n) < k, "GCG2025", "GCJ2025"), index=idx)
    raw = pd.Series(np.where(np.arange(n) < k, 2000.0, 2040.0), index=idx)
    ratio_adj = pd.Series(2000.0, index=idx)          # ratio splice: no jump
    back_adj = pd.Series(2000.0, index=idx)           # additive splice: no jump either
    log = pd.DataFrame({"exec_date": [idx[k - 1]], "switch_date": [idx[k]],
                        "from_contract": ["GCG2025"], "to_contract": ["GCJ2025"],
                        "from_close": [2000.0], "to_close": [2040.0],
                        "gap": [40.0], "ratio": [1.02]})
    si = _leg(pd.Series(25.0, index=idx), "SIH2025")
    return idx, held, raw, ratio_adj, back_adj, log, si


@pytest.mark.parametrize("source", ["ratio", "back", "roll_log"])
@pytest.mark.parametrize("lots", [1, -3])
def test_roll_gap_is_not_pnl(source, lots):
    idx, held, raw, ratio_adj, back_adj, log, si = _rolling_gc()
    adj = back_adj if source == "back" else ratio_adj
    gc = _leg(raw, held, adj, adjustment="raw" if source == "roll_log" else source)
    positions = pd.DataFrame({"GC": lots, "SI": 0}, index=idx)
    out = backtest.simulate(positions, gc, si, cost_per_contract=1.0,
                            roll_logs={"GC": log} if source == "roll_log" else None)
    assert (out["gross_pnl"] == 0).all()       # raw jumps 40 on the switch day
    # open on day 0, then close old + open new at the close before the switch
    assert out["costs"].iloc[0] == abs(lots)
    assert out["costs"].iloc[9] == 2 * abs(lots)
    assert out["costs"].sum() == 3 * abs(lots)


def test_raw_series_without_roll_log_is_rejected():
    idx, held, raw, *_, si = _rolling_gc()
    gc = _leg(raw, held, raw, adjustment="raw")
    with pytest.raises(ValueError, match="roll_logs"):
        backtest.simulate(pd.DataFrame({"GC": 1}, index=idx), gc, si)


_HAS_ROLLS = "root" in inspect.signature(rolls.build_continuous).parameters


@pytest.mark.skipif(not _HAS_ROLLS, reason="needs rolls.build_continuous from #22")
def test_roll_day_on_build_continuous_output(gc_expiries, si_expiries):
    gc = rolls.build_continuous(gc_expiries.bars, "GC", "ratio", rule="volume")
    si = rolls.build_continuous(si_expiries.bars, "SI", "ratio", rule="volume")
    log = rolls.roll_log(gc_expiries.bars, rolls.roll_schedule(gc_expiries.bars, "GC", "volume"))
    idx = gc.index.intersection(si.index)
    positions = pd.DataFrame({"GC": 2, "SI": 0}, index=idx)

    implied = backtest.simulate(positions, gc.loc[idx], si.loc[idx])
    logged = backtest.simulate(positions, gc.loc[idx], si.loc[idx], roll_logs={"GC": log})
    assert np.allclose(implied["gross_pnl"], logged["gross_pnl"])

    # on each switch day the raw move includes the gap; the PnL must not
    raw = gc.loc[idx, "raw_close"]
    for r in log.itertuples():
        day = r.switch_date
        expected = 2 * GC_MULT * (raw[day] - raw.shift()[day] - r.gap)
        assert implied.loc[day, "gross_pnl"] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# No look-ahead
# --------------------------------------------------------------------------- #
def _bump(leg: pd.DataFrame, day, factor: float) -> pd.DataFrame:
    out = leg.copy()
    out.loc[day, ["close", "raw_close"]] *= factor
    return out


def test_run_trades(pair_legs):
    out = backtest.run(*pair_legs, CONFIG)
    assert (out["signal"] != 0).sum() > 50
    assert out["costs"].sum() > 0


@pytest.mark.parametrize("where", [-1, 1500])
def test_changing_a_price_does_not_change_earlier_positions(pair_legs, where):
    gc, si = pair_legs
    day = gc.index[where]
    base = backtest.run(gc, si, CONFIG)
    for bumped in (backtest.run(_bump(gc, day, 1.10), si, CONFIG),
                   backtest.run(gc, _bump(si, day, 0.90), CONFIG)):
        before = base.index < day
        cols = list(LEGS) + ["alpha", "beta", "z", "signal", "gross_pnl", "costs"]
        pd.testing.assert_frame_equal(base.loc[before, cols], bumped.loc[before, cols])


def test_hedge_ratio_uses_only_prior_days(pair_legs):
    gc, si = pair_legs
    out = backtest.run(gc, si, CONFIG)
    assert out["beta"].iloc[:CONFIG.window].isna().all()
    t = 500
    window = slice(t - CONFIG.window, t)  # rows t-window .. t-1
    x, y = np.log(si["close"].iloc[window]), np.log(gc["close"].iloc[window])
    beta_ols = np.polyfit(x, y, 1)[0]
    assert out["beta"].iloc[t] == pytest.approx(beta_ols, rel=1e-9)


def test_zscore_uses_only_prior_days_for_mean_and_std(pair_legs):
    out = backtest.run(*pair_legs, CONFIG)
    t = 500
    prior = out["spread"].iloc[t - CONFIG.z_lookback:t]  # rows t-L .. t-1
    expected = (out["spread"].iloc[t] - prior.mean()) / prior.std()
    assert out["z"].iloc[t] == pytest.approx(expected, rel=1e-9)


def test_warm_up_is_window_plus_z_lookback(pair_legs):
    out = backtest.run(*pair_legs, CONFIG)
    warm = CONFIG.window + CONFIG.z_lookback
    assert out["z"].iloc[:warm].isna().all() and out["z"].iloc[warm:].notna().all()
    assert (out[list(LEGS)].iloc[:warm] == 0).all().all()


# --------------------------------------------------------------------------- #
# Signs, sizing, config
# --------------------------------------------------------------------------- #
def test_long_spread_is_long_gc_short_si(pair_legs):
    out = backtest.run(*pair_legs, CONFIG)
    gc_oz = out["GC"] * GC_MULT + out["MGC"] * contracts.MGC.multiplier
    si_oz = out["SI"] * SI_MULT + out["SIL"] * contracts.SIL.multiplier
    long, short = out["signal"] == 1, out["signal"] == -1
    assert long.any() and short.any()
    assert (gc_oz[long] > 0).all() and (si_oz[long] < 0).all()
    assert (gc_oz[short] < 0).all() and (si_oz[short] > 0).all()
    assert (gc_oz[out["signal"] == 0] == 0).all()


def test_beta_mode_sizes_si_notional_to_beta_times_gc():
    target = backtest._size_positions(1, 0.8, 2000.0, 25.0, 1_000_000.0, "beta", True)
    gc_notional = (target["GC"] * GC_MULT + target["MGC"] * contracts.MGC.multiplier) * 2000.0
    si_notional = (target["SI"] * SI_MULT + target["SIL"] * contracts.SIL.multiplier) * 25.0
    assert abs(gc_notional - 1_000_000.0) <= contracts.MGC.multiplier * 2000.0
    assert abs(-si_notional - 0.8 * 1_000_000.0) <= contracts.SIL.multiplier * 25.0


def test_config_as_dict_and_missing_columns(pair_legs):
    gc, si = pair_legs
    a = backtest.run(gc, si, CONFIG)
    b = backtest.run(gc, si, {**CONFIG.__dict__})
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError, match="raw_close"):
        backtest.run(gc.drop(columns="raw_close"), si)
