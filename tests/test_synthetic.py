"""Check that the synthetic fixtures really have the parameters they claim (Issue #14).

These tests also double as usage examples for each fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller, coint

from gcsi import synthetic as syn


# --------------------------------------------------------------------------- #
# Reproducibility: fixed seeds give identical data, different seeds do not
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make", [
    lambda s: syn.simulate_ou(n=300, seed=s),
    lambda s: syn.cointegrated_prices(n=300, seed=s).prices,
    lambda s: syn.per_expiry_bars("GC", seed=s).bars,
])
def test_same_seed_same_data(make):
    pd.testing.assert_frame_equal(pd.DataFrame(make(1)), pd.DataFrame(make(1)))
    assert not pd.DataFrame(make(1)).equals(pd.DataFrame(make(2)))


# --------------------------------------------------------------------------- #
# OU spread (for #8)
# --------------------------------------------------------------------------- #
def test_ou_params_closed_forms(ou_params):
    assert ou_params.half_life == pytest.approx(np.log(2) / 0.05)
    assert ou_params.ar1_b == pytest.approx(np.exp(-0.05))


def test_ou_spread_recovers_true_parameters(ou_spread, ou_params):
    # Example of what #8 will do: AR(1) regression X_t = a + b X_{t-1} + e
    x = ou_spread.to_numpy()
    b, a = np.polyfit(x[:-1], x[1:], 1)
    kappa_hat = -np.log(b)
    theta_hat = a / (1 - b)
    sigma_e_hat = np.std(x[1:] - (a + b * x[:-1]), ddof=2)

    assert kappa_hat == pytest.approx(ou_params.kappa, rel=0.25)
    assert theta_hat == pytest.approx(ou_params.theta, abs=0.5 * ou_params.stationary_std)
    assert sigma_e_hat == pytest.approx(ou_params.step_std, rel=0.05)
    assert np.log(2) / kappa_hat == pytest.approx(ou_params.half_life, rel=0.25)


def test_ou_spread_is_stationary(ou_spread):
    assert adfuller(ou_spread, result_object=False)[1] < 0.01


# --------------------------------------------------------------------------- #
# Cointegrated GC/SI prices (for #7, #9)
# --------------------------------------------------------------------------- #
def test_coint_pair_shape_and_levels(coint_pair):
    px = coint_pair.prices
    assert list(px.columns) == ["GC", "SI"]
    assert px.index.tz is not None and str(px.index.tz) == "UTC"
    assert (px > 0).all().all()
    assert px["GC"].iloc[0] / px["SI"].iloc[0] == pytest.approx(80, rel=0.1)


def test_coint_pair_recovers_hedge_ratio(coint_pair):
    # Example of what #7 will do: static OLS of log GC on log SI
    lg, ls = np.log(coint_pair.prices["GC"]), np.log(coint_pair.prices["SI"])
    beta_hat, alpha_hat = np.polyfit(ls, lg, 1)
    assert beta_hat == pytest.approx(coint_pair.params.beta, abs=0.05)

    # beta is estimated, so test the residual with Engle-Granger (MacKinnon
    # critical values) rather than plain ADF, which would be too optimistic
    assert coint(lg, ls)[1] < 0.01                         # cointegrated ...
    assert adfuller(ls, result_object=False)[1] > 0.05     # ... while the legs are I(1)


def test_coint_pair_true_spread_matches_prices(coint_pair):
    p = coint_pair.params
    implied = (np.log(coint_pair.prices["GC"]) - p.alpha
               - p.beta * np.log(coint_pair.prices["SI"]))
    np.testing.assert_allclose(implied.to_numpy(), coint_pair.spread.to_numpy(), atol=1e-12)


# --------------------------------------------------------------------------- #
# Per-expiry bars (for #4, #5)
# --------------------------------------------------------------------------- #
EXPECTED_COLUMNS = ["ts_event", "rtype", "publisher_id", "instrument_id",
                    "open", "high", "low", "close", "volume", "symbol"]


@pytest.mark.parametrize("fixture", ["gc_expiries", "si_expiries"])
def test_per_expiry_schema(fixture, request):
    data = request.getfixturevalue(fixture)
    bars = data.bars
    assert list(bars.columns) == EXPECTED_COLUMNS
    assert str(bars["ts_event"].dt.tz) == "UTC"
    assert bars["symbol"].nunique() == 3
    assert set(bars["instrument_id"]) == set(data.definitions["instrument_id"])
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()


@pytest.mark.parametrize("fixture,root", [("gc_expiries", "GC"), ("si_expiries", "SI")])
def test_per_expiry_prices_on_tick_grid(fixture, root, request):
    bars = request.getfixturevalue(fixture).bars
    ticks = bars[["open", "high", "low", "close"]].to_numpy() / syn.TICK_SIZE[root]
    np.testing.assert_allclose(ticks, np.round(ticks), atol=1e-6)


@pytest.mark.parametrize("fixture", ["gc_expiries", "si_expiries"])
def test_no_trading_after_expiration(fixture, request):
    data = request.getfixturevalue(fixture)
    merged = data.bars.merge(data.definitions, on="instrument_id")
    assert (merged["ts_event"] <= merged["expiration"]).all()


@pytest.mark.parametrize("fixture", ["gc_expiries", "si_expiries"])
def test_volume_leader_switches_on_true_roll_dates(fixture, request):
    # Example of what #5 will do: the roll date is the volume crossover
    data = request.getfixturevalue(fixture)
    vol = data.bars.pivot(index="ts_event", columns="symbol", values="volume").fillna(0)
    leader = vol.idxmax(axis=1)
    switches = leader[leader != leader.shift()].iloc[1:]
    assert list(switches.index) == list(data.roll_dates)


def test_roll_ratio_reflects_carry(gc_expiries):
    # GCG5 -> GCJ5 is ~2 months apart, so the jump is ~exp(0.04 * 2/12)
    first = gc_expiries.roll_ratios.iloc[0]
    assert first == pytest.approx(np.exp(gc_expiries.annual_carry * 61 / 365.25), rel=2e-3)


def test_gc_and_si_roll_on_different_dates(gc_expiries, si_expiries):
    assert set(gc_expiries.roll_dates).isdisjoint(set(si_expiries.roll_dates))


# --------------------------------------------------------------------------- #
# Top of book and positions (for #10)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture,root", [("gc_top_of_book", "GC"), ("si_top_of_book", "SI")])
def test_top_of_book_one_tick_spread(fixture, root, request):
    tob = request.getfixturevalue(fixture)
    tick = syn.TICK_SIZE[root]
    np.testing.assert_allclose(tob["ask_px"] - tob["bid_px"], tick)
    np.testing.assert_allclose(tob["bid_px"] / tick, np.round(tob["bid_px"] / tick), atol=1e-6)
    assert (tob[["bid_sz", "ask_sz"]] > 0).all().all()


def test_toy_positions_known_trades(toy_positions):
    gc, si = toy_positions["GC"], toy_positions["SI"]
    assert (np.sign(gc) == -np.sign(si)).all()                # legs always opposite
    assert ((gc == 0) == (si == 0)).all()
    assert (gc.diff().fillna(0) != 0).sum() == 4               # 2 entries + 2 exits
    assert set(gc.unique()) == {-5, 0, 5} and set(si.unique()) == {-8, 0, 8}


# --------------------------------------------------------------------------- #
# Mock Databento client (for #4)
# --------------------------------------------------------------------------- #
def test_mock_client_serves_bars_and_records_calls(mock_databento_client, gc_expiries):
    store = mock_databento_client.timeseries.get_range(
        dataset="GLBX.MDP3", symbols="GC.FUT", schema="ohlcv-1d",
        start="2025-01-01", end="2025-03-01", stype_in="parent")
    df = store.to_df()

    assert df.index.name == "ts_event"
    assert df.index.min() >= pd.Timestamp("2025-01-01", tz="UTC")
    assert df.index.max() < pd.Timestamp("2025-03-01", tz="UTC")
    assert set(df["symbol"]) <= set(gc_expiries.definitions["raw_symbol"])
    assert len(mock_databento_client.calls) == 1
    assert mock_databento_client.calls[0]["stype_in"] == "parent"


def test_mock_client_definitions_and_multiple_roots(mock_databento_client):
    defs = mock_databento_client.timeseries.get_range(
        dataset="GLBX.MDP3", symbols=["GC.FUT", "SI.FUT"], schema="definition").to_df()
    assert set(defs["raw_symbol"].str[:2]) == {"GC", "SI"}
    assert {"instrument_id", "raw_symbol", "expiration"} <= set(defs.columns)


def test_mock_client_rejects_unknown_requests(mock_databento_client):
    with pytest.raises(ValueError):
        mock_databento_client.timeseries.get_range(
            dataset="GLBX.MDP3", symbols="GC.FUT", schema="mbo")
    with pytest.raises(ValueError):
        mock_databento_client.timeseries.get_range(
            dataset="XNAS.ITCH", symbols="GC.FUT", schema="ohlcv-1d")
    assert len(mock_databento_client.calls) == 2   # still recorded
