"""Cointegration tests and hedge-ratio estimation (#7).

Direction (used everywhere in this module): **log GC is regressed on log SI**,

    log GC_t = alpha + beta * log SI_t + resid_t,

so the spread handed to #9 is ``log GC - alpha - beta * log SI``. Inputs should
be log prices of the *ratio-adjusted* continuous series from #5
(``rolls.build_continuous(..., adjustment="ratio")["close"]``); raw closes have
roll gaps that would show up as fake breaks in the residual.

What is here:
  - ``adf_is_i1``: check that a log-price is I(1) before testing cointegration.
  - ``engle_granger``: static OLS hedge ratio plus the Engle-Granger test with
    MacKinnon critical values (``statsmodels.tsa.stattools.coint``). Plain
    ``adfuller`` on the OLS residual is too optimistic because beta is
    estimated first.
  - ``johansen``: trace / max-eigenvalue rank test and the normalised
    cointegrating vector, as a cross-check on the EG beta.
  - ``rolling_hedge_ratio``: trailing-window OLS, shifted one day so alpha and
    beta on day t use data up to t-1 only. Drop-in for the placeholder in the
    #10 backtest skeleton.
  - ``kalman_hedge_ratio``: random-walk alpha/beta via a Kalman filter. Also
    shifted (we return the one-step-ahead prediction), and it exposes the beta
    path so #9/#10 can see how much re-sizing a drifting beta causes.
  - ``subperiod_engle_granger``: EG beta and p-value per calendar period, the
    evidence for (in)stability across sub-periods that #7 asks for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def _align(log_gc: pd.Series, log_si: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Inner-join two series on their index and drop rows with NaN in either."""
    both = pd.concat({"gc": log_gc, "si": log_si}, axis=1, join="inner").dropna()
    return both["gc"], both["si"]


def adf_is_i1(series: pd.Series, alpha: float = 0.05) -> bool:
    """True if `series` is I(1): a unit root in levels, stationary in first differences."""
    x = series.dropna()
    p_level = adfuller(x, autolag="AIC", result_object=False)[1]
    p_diff = adfuller(x.diff().dropna(), autolag="AIC", result_object=False)[1]
    return bool(p_level > alpha and p_diff < alpha)


def engle_granger(log_gc: pd.Series, log_si: pd.Series) -> dict:
    """Static hedge ratio and Engle-Granger cointegration test of log GC on log SI.

    Returns a dict with:
      alpha, beta: OLS intercept and slope of log GC on log SI,
      resid: the spread log GC - alpha - beta * log SI (pd.Series),
      tstat, pvalue: EG test statistic and MacKinnon p-value,
      crit: {"1%", "5%", "10%"} critical values for the EG statistic,
      nobs: number of observations used.
    A small p-value rejects "no cointegration".
    """
    y, x = _align(log_gc, log_si)
    beta, alpha = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
    resid = (y - alpha - beta * x).rename("spread")
    tstat, pvalue, crit = coint(y, x, trend="c", autolag="aic")
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "resid": resid,
        "tstat": float(tstat),
        "pvalue": float(pvalue),
        "crit": dict(zip(("1%", "5%", "10%"), map(float, crit))),
        "nobs": int(len(y)),
    }


def johansen(log_gc: pd.Series, log_si: pd.Series, det_order: int = 0,
             k_ar_diff: int = 1) -> dict:
    """Johansen trace and max-eigenvalue tests on (log GC, log SI).

    det_order=0 puts a constant in the cointegrating relation; k_ar_diff is the
    number of lagged differences in the VECM. Returns:
      rank_trace, rank_max_eig: cointegration rank implied at the 5% level,
      trace_stat, trace_crit_5, max_eig_stat, max_eig_crit_5: per hypothesis r=0, r<=1,
      beta: hedge ratio from the first cointegrating vector, normalised so the
            GC weight is 1 (comparable with engle_granger's beta),
      vector: that normalised vector (GC weight, SI weight).
    """
    y, x = _align(log_gc, log_si)
    res = coint_johansen(pd.concat([y, x], axis=1).to_numpy(), det_order, k_ar_diff)
    # statsmodels can return tiny imaginary parts from the eigen-solver; drop them
    trace, trace_crit = np.real(res.lr1), res.cvt[:, 1]      # column 1 = 5%
    max_eig, max_eig_crit = np.real(res.lr2), res.cvm[:, 1]
    evec = np.real(res.evec)
    vec = evec[:, 0] / evec[0, 0]                            # GC weight normalised to 1

    def rank(stats, crits):
        r = 0
        while r < len(stats) and stats[r] > crits[r]:
            r += 1
        return r

    return {
        "rank_trace": rank(trace, trace_crit),
        "rank_max_eig": rank(max_eig, max_eig_crit),
        "trace_stat": trace.tolist(),
        "trace_crit_5": trace_crit.tolist(),
        "max_eig_stat": max_eig.tolist(),
        "max_eig_crit_5": max_eig_crit.tolist(),
        "beta": float(-vec[1]),
        "vector": (1.0, float(vec[1])),
    }


def rolling_hedge_ratio(log_gc: pd.Series, log_si: pd.Series, window: int) -> pd.DataFrame:
    """Trailing-window OLS of log GC on log SI, shifted one day (no look-ahead).

    alpha and beta on day t are estimated on days t-window .. t-1 only, so the
    first `window` rows are NaN. Returns a DataFrame with columns alpha, beta on
    the inputs' common index. Same interface as the placeholder in #10.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    y, x = _align(log_gc, log_si)
    beta = y.rolling(window).cov(x) / x.rolling(window).var()
    alpha = y.rolling(window).mean() - beta * x.rolling(window).mean()
    return pd.DataFrame({"alpha": alpha, "beta": beta}).shift(1)


def kalman_hedge_ratio(log_gc: pd.Series, log_si: pd.Series, delta: float = 1e-5,
                       obs_var: float | None = None, warmup: int = 60) -> pd.DataFrame:
    """Time-varying alpha and beta via a Kalman filter, with no look-ahead.

    State [alpha_t, beta_t] follows a random walk with covariance
    delta / (1 - delta) * I; observation log GC_t = alpha_t + beta_t log SI_t + e_t
    with Var(e_t) = obs_var (default: residual variance of an OLS fit on the
    first `warmup` days). The value returned for day t is the *prediction*
    made before seeing day t, i.e. it uses data up to t-1, like
    rolling_hedge_ratio. Smaller delta means a smoother beta path.

    Returns a DataFrame with columns alpha, beta (NaN for the first `warmup` days).
    """
    y, x = _align(log_gc, log_si)
    if len(y) <= warmup:
        raise ValueError("need more observations than `warmup`")
    yv, xv = y.to_numpy(), x.to_numpy()
    b0, a0 = np.polyfit(xv[:warmup], yv[:warmup], 1)
    if obs_var is None:
        obs_var = float(np.var(yv[:warmup] - a0 - b0 * xv[:warmup], ddof=2))
    q = delta / (1.0 - delta) * np.eye(2)
    state, cov = np.array([a0, b0]), np.eye(2)
    out = np.full((len(y), 2), np.nan)
    for t in range(warmup, len(y)):
        cov = cov + q                      # predict: random-walk state
        out[t] = state                     # prediction for day t, uses data to t-1
        h = np.array([1.0, xv[t]])
        err = yv[t] - h @ state            # update with day t
        s = h @ cov @ h + obs_var
        gain = cov @ h / s
        state = state + gain * err
        cov = cov - np.outer(gain, h) @ cov
    return pd.DataFrame(out, index=y.index, columns=["alpha", "beta"])


def subperiod_engle_granger(log_gc: pd.Series, log_si: pd.Series, freq: str = "YE",
                            min_obs: int = 120) -> pd.DataFrame:
    """Engle-Granger beta and p-value per calendar period (default: per year).

    Periods with fewer than `min_obs` observations are skipped. Returns one row
    per period with columns start, end, nobs, beta, pvalue, cointegrated_5pct.
    """
    y, x = _align(log_gc, log_si)
    rows = []
    for key, sub in y.groupby(pd.Grouper(freq=freq)):
        if len(sub) < min_obs:
            continue
        res = engle_granger(sub, x.loc[sub.index])
        rows.append({"period": key, "start": sub.index.min(), "end": sub.index.max(),
                     "nobs": res["nobs"], "beta": res["beta"], "pvalue": res["pvalue"],
                     "cointegrated_5pct": res["pvalue"] < 0.05})
    return pd.DataFrame(rows).set_index("period") if rows else pd.DataFrame(
        columns=["start", "end", "nobs", "beta", "pvalue", "cointegrated_5pct"])
