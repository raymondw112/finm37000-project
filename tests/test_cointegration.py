"""Tests for gcsi.cointegration (#7), on the #14 fixtures with known beta = 0.8."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gcsi import cointegration as co


@pytest.fixture(scope="module")
def logs(coint_pair):
    px = coint_pair.prices
    return np.log(px["GC"]), np.log(px["SI"])


def test_legs_are_i1_and_spread_is_not(logs, coint_pair):
    log_gc, log_si = logs
    assert co.adf_is_i1(log_si)
    assert not co.adf_is_i1(coint_pair.spread)  # stationary in levels


def test_engle_granger_recovers_beta_and_rejects_no_cointegration(logs, coint_pair):
    log_gc, log_si = logs
    res = co.engle_granger(log_gc, log_si)
    assert res["beta"] == pytest.approx(coint_pair.params.beta, abs=0.05)
    assert res["pvalue"] < 0.01
    assert res["tstat"] < res["crit"]["1%"]
    assert res["nobs"] == len(log_gc)
    # resid is the spread definition #9 uses: log GC - alpha - beta * log SI
    expected = log_gc - res["alpha"] - res["beta"] * log_si
    pd.testing.assert_series_equal(res["resid"], expected, check_names=False)


def test_engle_granger_does_not_find_cointegration_in_independent_walks():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2025-01-02", periods=1000)
    a = pd.Series(np.cumsum(rng.standard_normal(1000)) * 0.01, index=idx)
    b = pd.Series(np.cumsum(rng.standard_normal(1000)) * 0.01, index=idx)
    assert co.engle_granger(a, b)["pvalue"] > 0.05


def test_johansen_finds_a_relation_with_the_same_beta(logs, coint_pair):
    log_gc, log_si = logs
    res = co.johansen(log_gc, log_si)
    # r = 0 is rejected. The r <= 1 test is oversized in samples like this one
    # (rank 2 on ~1/3 of seeds), so only require at least one relation.
    assert res["rank_trace"] >= 1
    assert res["trace_stat"][0] > res["trace_crit_5"][0]
    assert res["beta"] == pytest.approx(coint_pair.params.beta, abs=0.05)
    assert res["vector"][0] == 1.0


def test_rolling_hedge_ratio_matches_trailing_ols_and_has_no_look_ahead(logs):
    log_gc, log_si = logs
    window = 250
    hr = co.rolling_hedge_ratio(log_gc, log_si, window)
    assert list(hr.columns) == ["alpha", "beta"]
    assert hr.iloc[:window].isna().all().all()  # warm-up
    # value on day t = OLS on days t-window .. t-1
    t = 1000
    b, a = np.polyfit(log_si.iloc[t - window:t], log_gc.iloc[t - window:t], 1)
    assert hr["beta"].iloc[t] == pytest.approx(b, rel=1e-9)
    assert hr["alpha"].iloc[t] == pytest.approx(a, rel=1e-9)
    # changing the future must not change the past
    shocked = log_gc.copy()
    shocked.iloc[t:] += 1.0
    hr2 = co.rolling_hedge_ratio(shocked, log_si, window)
    pd.testing.assert_frame_equal(hr.iloc[: t + 1], hr2.iloc[: t + 1])


def test_rolling_hedge_ratio_is_close_to_true_beta_on_average(logs, coint_pair):
    hr = co.rolling_hedge_ratio(*logs, window=500)
    assert hr["beta"].median() == pytest.approx(coint_pair.params.beta, abs=0.1)


def test_kalman_hedge_ratio_converges_and_has_no_look_ahead(logs, coint_pair):
    log_gc, log_si = logs
    kf = co.kalman_hedge_ratio(log_gc, log_si)
    assert kf.iloc[:60].isna().all().all()
    assert kf["beta"].iloc[-500:].median() == pytest.approx(coint_pair.params.beta, abs=0.1)
    t = 1500
    shocked = log_gc.copy()
    shocked.iloc[t:] += 1.0
    kf2 = co.kalman_hedge_ratio(shocked, log_si)
    pd.testing.assert_frame_equal(kf.iloc[: t + 1], kf2.iloc[: t + 1])


def test_subperiod_engle_granger_reports_each_year(logs):
    table = co.subperiod_engle_granger(*logs, freq="YE")
    assert {"beta", "pvalue", "nobs", "cointegrated_5pct"} <= set(table.columns)
    assert len(table) >= 8  # ~10 years of fixture data, short edge years skipped
    assert (table["nobs"] >= 120).all()


def test_inputs_are_aligned_on_dates(logs):
    log_gc, log_si = logs
    res_full = co.engle_granger(log_gc, log_si)
    res_shifted = co.engle_granger(log_gc.iloc[5:], log_si.iloc[:-5])  # partial overlap
    assert res_shifted["nobs"] == len(log_gc) - 10
    assert res_shifted["beta"] == pytest.approx(res_full["beta"], abs=0.02)
