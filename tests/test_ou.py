"""Tests for gcsi.ou (#8), on the #14 OU fixture (kappa = 0.05/day, half-life ~13.86 days)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gcsi import cointegration as co
from gcsi import ou


def test_half_life_formula_and_edge_cases():
    assert ou.half_life(0.05) == pytest.approx(np.log(2) / 0.05)
    assert ou.half_life(0.0) == float("inf")
    assert ou.half_life(-0.1) == float("inf")
    assert ou.half_life(float("nan")) == float("inf")


def test_fit_ou_recovers_true_parameters(ou_spread, ou_params):
    fit = ou.fit_ou(ou_spread)
    assert fit["kappa"] == pytest.approx(ou_params.kappa, rel=0.25)
    assert fit["theta"] == pytest.approx(ou_params.theta, abs=0.5 * ou_params.stationary_std)
    assert fit["sigma"] == pytest.approx(ou_params.sigma, rel=0.05)
    assert fit["sigma_eq"] == pytest.approx(ou_params.stationary_std, rel=0.15)
    assert fit["half_life"] == pytest.approx(ou_params.half_life, rel=0.25)
    assert fit["nobs"] == len(ou_spread)


def test_fit_ou_on_a_random_walk_reports_no_mean_reversion():
    rng = np.random.default_rng(3)
    walk = pd.Series(np.cumsum(rng.standard_normal(3000)) * 0.01)
    fit = ou.fit_ou(walk)
    if fit["ar1_b"] >= 1:  # the usual case for a pure random walk
        assert fit["half_life"] == float("inf")
        assert np.isnan(fit["kappa"])
    else:  # b just below 1: mean reversion so slow it is irrelevant
        assert fit["half_life"] > 250


def test_fit_ou_on_engle_granger_spread_matches_the_true_spread(coint_pair):
    """End to end: #7 spread -> #8 half-life loses nothing vs. the true spread.

    On this 2,520-day sample the true spread itself fits a ~20-day half-life
    (sampling noise around 13.9; the median over 30 seeds is 13.8), so compare
    with the true spread's own fit rather than the population value.
    """
    px = coint_pair.prices
    spread = co.engle_granger(np.log(px["GC"]), np.log(px["SI"]))["resid"]
    fit_eg = ou.fit_ou(spread)
    fit_true = ou.fit_ou(coint_pair.spread)
    assert fit_eg["half_life"] == pytest.approx(fit_true["half_life"], rel=0.05)
    assert 5 < fit_eg["half_life"] < 40


def test_half_life_sensitivity_has_one_row_per_window(ou_spread):
    table = ou.half_life_sensitivity(ou_spread, windows=(252, 1000, 10**6))
    assert list(table.index) == [252, 1000]  # windows longer than the data are skipped
    assert {"kappa", "half_life", "ar1_b"} <= set(table.columns)


def test_fit_ou_needs_data():
    with pytest.raises(ValueError):
        ou.fit_ou(pd.Series([1.0, 2.0]))
