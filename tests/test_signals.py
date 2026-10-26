"""Tests for gcsi.signals (#9): spread, z-score, signals and sizing."""
import numpy as np
import pandas as pd
import pytest

from gcsi import cointegration, contracts, signals


def _series(values, start="2025-01-02"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# --- spread and z-score -------------------------------------------------------

def test_spread_with_series_beta():
    lg, ls = _series([1.0, 2.0]), _series([0.5, 1.0])
    out = signals.spread(lg, ls, alpha=0.1, beta=_series([2.0, 1.0]))
    assert out.tolist() == pytest.approx([1.0 - 0.1 - 1.0, 2.0 - 0.1 - 1.0])


def test_zscore_uses_only_the_window_ending_yesterday():
    s = _series(np.arange(10.0))
    z = signals.zscore(s, lookback=3)
    assert z.iloc[:3].isna().all()
    # day 3: window = days 0..2 -> mean 1, std 1, value 3 -> z = 2
    assert z.iloc[3] == pytest.approx(2.0)


def test_zscore_has_no_look_ahead(coint_pair):
    s = coint_pair.spread
    z = signals.zscore(s, lookback=60)
    bumped = s.copy()
    bumped.iloc[1000:] += 5.0          # change today and the future
    z2 = signals.zscore(bumped, lookback=60)
    pd.testing.assert_series_equal(z.iloc[:1000], z2.iloc[:1000])


def test_consistent_zscore_matches_brute_force_and_differs_from_mixed_beta(coint_pair):
    p = np.log(coint_pair.prices)
    lg, ls = p["GC"], p["SI"]
    beta = pd.Series(np.linspace(0.6, 1.0, len(lg)), index=lg.index)   # a drifting beta
    alpha = pd.Series(0.3, index=lg.index)
    lb = 40
    out = signals.consistent_zscore(lg, ls, alpha, beta, lb)
    t = 500
    window = slice(t - lb, t)
    past = lg.iloc[window] - alpha.iloc[t] - beta.iloc[t] * ls.iloc[window]   # today's beta on the past
    expected = (out["spread"].iloc[t] - past.mean()) / past.std()
    assert out["z"].iloc[t] == pytest.approx(expected, rel=1e-9)
    mixed = signals.zscore(signals.spread(lg, ls, alpha, beta), lb)
    assert not np.isclose(mixed.iloc[t], out["z"].iloc[t])


def test_zscore_rejects_tiny_lookback():
    with pytest.raises(ValueError):
        signals.zscore(_series([1.0, 2.0, 3.0]), lookback=1)


# --- signals ------------------------------------------------------------------

def test_signal_path_enter_hold_exit_stop_reenter():
    z = _series([np.nan, 0.5, 2.1, 1.0, -0.1, -2.5, -1.0, 0.2, 2.5, 4.5, 3.0, 1.5, 2.2, 0.0])
    sig = signals.generate_signals(z, entry=2.0, exit_=0.0, stop=4.0)
    assert sig.tolist() == [0, 0, -1, -1, 0, 1, 1, 0, -1, 0, 0, 0, -1, 0]
    # 2.1 short, hold at 1.0, exit at -0.1; long at -2.5, exit at 0.2;
    # short at 2.5, stopped at 4.5, stay flat at 3.0 (still >= entry), reset at 1.5,
    # re-enter short at 2.2, exit at 0.0


def test_signal_values_and_validation():
    z = _series(np.random.default_rng(0).normal(0, 2, 500))
    sig = signals.generate_signals(z)
    assert set(sig.unique()) <= {-1, 0, 1}
    with pytest.raises(ValueError):
        signals.generate_signals(z, entry=2.0, exit_=0.0, stop=1.5)


# --- sizing -------------------------------------------------------------------

def test_long_spread_is_long_gold_short_silver():
    c = signals.target_contracts(1, beta=0.8, gc_price=2000.0, si_price=25.0, gc_notional=1_000_000)
    assert c["GC"] > 0 and c["SI"] < 0
    c = signals.target_contracts(-1, beta=0.8, gc_price=2000.0, si_price=25.0, gc_notional=1_000_000)
    assert c["GC"] < 0 and c["SI"] > 0


def test_beta_mode_sizes_by_hand():
    # $1m of gold at 2000 = 500 oz = 5 GC; beta 0.8 -> $800k of silver at 25 = 32,000 oz = 6 SI + 2 SIL
    c = signals.target_contracts(1, 0.8, 2000.0, 25.0, 1_000_000, mode="beta", micros=True)
    assert c == {"GC": 5, "MGC": 0, "SI": -6, "SIL": -2}
    c = signals.target_contracts(1, 0.8, 2000.0, 25.0, 1_000_000, mode="dollar", micros=True)
    assert c == {"GC": 5, "MGC": 0, "SI": -8, "SIL": 0}           # $1m / 25 = 40,000 oz
    c = signals.target_contracts(1, 0.8, 2000.0, 25.0, 1_000_000, mode="beta", micros=False)
    assert c == {"GC": 5, "MGC": 0, "SI": -6, "SIL": 0}           # 6.4 SI rounds to 6


def test_flat_or_unknown_beta_gives_no_position():
    assert signals.target_contracts(0, 0.8, 2000.0, 25.0, 1e6) == dict.fromkeys(signals.LEGS, 0)
    assert signals.target_contracts(1, np.nan, 2000.0, 25.0, 1e6) == dict.fromkeys(signals.LEGS, 0)


def test_size_positions_on_fixture_hits_beta_within_one_micro(coint_pair):
    px = coint_pair.prices
    sig = pd.Series(1, index=px.index)
    beta = pd.Series(coint_pair.params.beta, index=px.index)        # true beta 0.8
    out = signals.size_positions(sig, beta, px["GC"], px["SI"], gc_notional=1_000_000)
    assert (out["GC"] * 100 + out["MGC"] * 10 > 0).all()
    # leftover ounces below one micro on each leg
    assert (out["gc_resid_oz"].abs() <= contracts.MGC.multiplier / 2 + 1e-9).all()
    assert (out["si_resid_oz"].abs() <= contracts.SIL.multiplier / 2 + 1e-9).all()
    # SI dollars ~ 0.8 x GC dollars held, within one SIL / one MGC of slack
    slack = (contracts.SIL.multiplier * px["SI"] + 0.8 * contracts.MGC.multiplier * px["GC"]) \
        / (out["GC"] * 100 * px["GC"] + out["MGC"] * 10 * px["GC"])
    assert ((out["si_dollar_ratio"] - 0.8).abs() <= slack + 1e-12).all()


def test_micros_reduce_residual(coint_pair):
    px = coint_pair.prices.iloc[:200]
    sig, beta = pd.Series(1, index=px.index), pd.Series(0.8, index=px.index)
    with_m = signals.size_positions(sig, beta, px["GC"], px["SI"], 1_000_000, micros=True)
    without = signals.size_positions(sig, beta, px["GC"], px["SI"], 1_000_000, micros=False)
    assert with_m["si_resid_oz"].abs().mean() < without["si_resid_oz"].abs().mean()
    assert (without[["MGC", "SIL"]] == 0).all().all()


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        signals.target_contracts(1, 0.8, 2000.0, 25.0, 1e6, mode="ounce")


# --- the backtest uses these functions (placeholders from #30 were swapped out) --

def test_backtest_signal_is_this_module(coint_pair):
    from gcsi import backtest
    px = coint_pair.prices
    leg = lambda x, c: pd.DataFrame({"close": x, "raw_close": x, "contract": c})
    cfg = backtest.BacktestConfig(zscore="simple", fill_lag=0, cost_per_contract=0.0)
    out = backtest.run(leg(px["GC"], "G"), leg(px["SI"], "S"), cfg)
    p = np.log(px)
    hr = cointegration.rolling_hedge_ratio(p["GC"], p["SI"], 252)
    z = signals.zscore(signals.spread(p["GC"], p["SI"], hr["alpha"], hr["beta"]), 60)
    pd.testing.assert_series_equal(out["signal"], signals.generate_signals(z), check_names=False)
