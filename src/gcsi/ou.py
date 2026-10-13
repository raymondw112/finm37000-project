"""Ornstein-Uhlenbeck fit and half-life (Issue 6).

Model the spread X_t with dX = kappa(theta - X)dt + sigma dW. Fit via the
exact AR(1) discretization (OLS) or scipy MLE, recover kappa, theta, sigma,
and the half-life = ln(2)/kappa. Estimate on OUR data; do not borrow a
literature number. Report sensitivity to the estimation window.
"""
from __future__ import annotations

import pandas as pd


def fit_ou(spread: pd.Series) -> dict:
    """Return {'kappa', 'theta', 'sigma', 'sigma_eq', 'half_life'}."""
    raise NotImplementedError


def half_life(kappa: float) -> float:
    """ln(2) / kappa, in the sampling units of the spread."""
    raise NotImplementedError
