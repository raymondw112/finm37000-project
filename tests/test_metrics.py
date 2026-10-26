"""Tests for performance evaluation (#11)."""
import numpy as np
import pandas as pd
import pytest

from gcsi import metrics

IDX = pd.bdate_range("2022-01-03", periods=10, tz="UTC")


def test_summarize_by_hand():
    pnl = pd.Series([0, 1000, -500, 2000, 0, -1000, 500, 0, 0, 1000.0], index=IDX)
    s = metrics.summarize(pnl, capital=100_000)
    r = pnl / 100_000
    assert s["total_pnl"] == 3000
    assert s["ann_return"] == pytest.approx(r.mean() * 252)
    assert s["sharpe"] == pytest.approx(np.sqrt(252) * r.mean() / r.std())
    # equity peaks at 102,500 on day 3, falls to 101,500 on day 5
    assert s["max_drawdown"] == pytest.approx(101_500 / 102_500 - 1)
    assert s["hit_rate"] == pytest.approx(4 / 6) and s["active_share"] == pytest.approx(0.6)


def test_drawdown_from_initial_capital():
    pnl = pd.Series([-100.0, -100.0, 50.0], index=IDX[:3])
    assert metrics.max_drawdown(pnl, 1000) == pytest.approx(-0.2)


def test_trade_stats_counts_round_trips():
    sig = pd.Series([0, 1, 1, 1, 0, 0, -1, -1, 0, 0], index=IDX)
    pnl = pd.Series([0, 0, 10, 10, 10, -5, 0, 4, -2, 0.0], index=IDX)
    t = metrics.trade_stats(sig, pnl, fill_lag=1)
    # held long on rows 2-4 (PnL rows 3-5), short on rows 7-8 (PnL rows 8-9)
    assert t["n_trades"] == 2 and t["long_trades"] == 1 and t["short_trades"] == 1
    assert t["avg_holding_days"] == pytest.approx(2.5)
    assert t["win_rate"] == pytest.approx(0.5)


def test_stress_periods_and_by_year():
    idx = pd.bdate_range("2011-07-01", "2013-05-31", tz="UTC")
    pnl = pd.Series(100.0, index=idx)
    st = metrics.stress_periods(pnl, capital=1e6, beta=pd.Series(0.8, index=idx))
    assert st.loc["Aug-Dec 2011", "total_pnl"] == pytest.approx(100 * len(pnl["2011-08-01":"2011-12-31"]))
    assert st.loc["Apr 2013", "beta_min"] == 0.8
    assert st.loc["Mar 2020", "days"] == 0
    yr = metrics.by_year(pnl, 1e6)
    assert list(yr.index) == [2011, 2012, 2013]
