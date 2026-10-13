"""Cointegration tests and hedge-ratio estimation (Issue 5).

Confirm each log-price is I(1) (ADF), then test the pair with Engle-Granger
(use EG critical values on the estimated residual) and Johansen. Estimate the
hedge ratio statically (OLS) and dynamically (rolling / Kalman) and compare
stability across sub-periods -- the GC/SI link is known to break in places.
Free libs only: statsmodels, (optional) pykalman.
"""
from __future__ import annotations

import pandas as pd


def adf_is_i1(series: pd.Series) -> bool:
    """True if `series` is I(1): unit root in levels, stationary in diffs."""
    raise NotImplementedError


def engle_granger(log_a: pd.Series, log_b: pd.Series) -> dict:
    """Two-step EG test. Returns hedge ratio beta, residual, and p-value."""
    raise NotImplementedError


def johansen(log_a: pd.Series, log_b: pd.Series) -> dict:
    """Johansen trace / max-eigenvalue test; returns rank and hedge weights."""
    raise NotImplementedError


def kalman_hedge_ratio(log_a: pd.Series, log_b: pd.Series) -> pd.Series:
    """Time-varying hedge ratio via a Kalman filter (handles drift)."""
    raise NotImplementedError
