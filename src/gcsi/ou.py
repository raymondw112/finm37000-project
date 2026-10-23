"""Ornstein-Uhlenbeck fit and half-life (#8).

Model the spread X_t with dX = kappa (theta - X) dt + sigma dW. Sampled daily
(dt = 1 trading day) the exact discretization is an AR(1),

    X_t = a + b X_{t-1} + e_t,   b = exp(-kappa),   a = theta (1 - b),
    Var(e_t) = sigma^2 (1 - b^2) / (2 kappa),

so an OLS fit of X_t on X_{t-1} recovers kappa = -ln(b), theta = a / (1 - b) and
sigma. The half-life ln(2) / kappa is in **trading days**. If b is not in (0, 1)
the spread is not mean-reverting at this frequency: kappa, theta, sigma and
sigma_eq are NaN and the half-life is inf rather than a misleading number.

Estimate on our own spread (from #7), not a literature number, and check how
much the half-life moves with the estimation window (`half_life_sensitivity`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def half_life(kappa: float) -> float:
    """ln(2) / kappa, in the sampling units of the spread (trading days for daily data).

    Returns inf for kappa <= 0 or NaN (no mean reversion).
    """
    if not np.isfinite(kappa) or kappa <= 0:
        return float("inf")
    return float(np.log(2.0) / kappa)


def fit_ou(spread: pd.Series) -> dict:
    """Fit an OU process to a daily spread via its exact AR(1) form.

    Returns {'kappa', 'theta', 'sigma', 'sigma_eq', 'half_life', 'ar1_b', 'nobs'}:
      kappa: mean-reversion speed per trading day,
      theta: long-run mean of the spread,
      sigma: instantaneous volatility (per sqrt trading day),
      sigma_eq: stationary standard deviation sigma / sqrt(2 kappa),
      half_life: ln(2) / kappa in trading days,
      ar1_b: the fitted AR(1) slope, nobs: observations used.
    """
    x = spread.dropna().to_numpy(dtype=float)
    if len(x) < 3:
        raise ValueError("need at least 3 observations to fit an AR(1)")
    x_prev, x_next = x[:-1], x[1:]
    b, a = np.polyfit(x_prev, x_next, 1)
    resid = x_next - (a + b * x_prev)
    step_var = float(np.var(resid, ddof=2))
    out = {"ar1_b": float(b), "nobs": int(len(x))}
    if not 0.0 < b < 1.0:
        return out | {"kappa": np.nan, "theta": np.nan, "sigma": np.nan,
                      "sigma_eq": np.nan, "half_life": float("inf")}
    kappa = -np.log(b)
    sigma = np.sqrt(step_var * 2.0 * kappa / (1.0 - b**2))
    return out | {
        "kappa": float(kappa),
        "theta": float(a / (1.0 - b)),
        "sigma": float(sigma),
        "sigma_eq": float(sigma / np.sqrt(2.0 * kappa)),
        "half_life": half_life(kappa),
    }


def half_life_sensitivity(spread: pd.Series,
                          windows: tuple[int, ...] = (126, 252, 504, 1008)) -> pd.DataFrame:
    """Half-life fitted on the most recent N observations, for each N in `windows`.

    Windows longer than the series are skipped. Returns one row per window
    with columns kappa, half_life, ar1_b, indexed by window length.
    """
    s = spread.dropna()
    rows = {}
    for n in windows:
        if n > len(s):
            continue
        fit = fit_ou(s.iloc[-n:])
        rows[n] = {"kappa": fit["kappa"], "half_life": fit["half_life"], "ar1_b": fit["ar1_b"]}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("window")
