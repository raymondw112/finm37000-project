"""Signal generation and position sizing (#9).

This is the bridge from statistics to trades: a spread built from the hedge
ratio (#7), a z-score that only uses past data, a {-1, 0, +1} signal, and the
conversion of that signal into real contract counts.

Conventions (shared with #7 and #10):
  - spread_t = log GC_t - alpha_t - beta_t * log SI_t, on ratio-adjusted
    closes (rolls.build_continuous(..., adjustment="ratio")["close"]).
  - alpha_t and beta_t are estimated with data up to t-1 (that is #7's job).
  - z_t uses the spread at t, but its mean and standard deviation come only
    from the window ending t-1, so nothing on day t is used to scale itself.
  - signal = +1 is long the spread: long GC, short SI. -1 is the reverse.
  - Contract sizes come from gcsi.contracts, never typed by hand.

Sizing default is beta-neutral: for every $1 of gold, hold $beta of silver,
which is what the log-price regression log GC = alpha + beta log SI implies.
Dollar-neutral (equal dollars per leg) is available as an option. With
micros=True each leg is filled with standard contracts first and micros
(MGC 10 oz, SIL 1,000 oz) for the remainder, so the hedge ratio is matched
far more closely than whole contracts allow; the leftover ounces are reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gcsi import contracts

LEGS = ("GC", "MGC", "SI", "SIL")
MODES = ("beta", "dollar")


# --------------------------------------------------------------------------- #
# Spread and z-score
# --------------------------------------------------------------------------- #
def spread(log_gc: pd.Series, log_si: pd.Series, alpha: pd.Series | float,
           beta: pd.Series | float) -> pd.Series:
    """log GC - alpha - beta * log SI. alpha and beta may be scalars or series."""
    return (log_gc - alpha - beta * log_si).rename("spread")


def zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """z-score of `spread` with mean and std from the window ending t-1.

    NaN until `lookback` past values exist. Matches the placeholder in the
    #10 backtest skeleton, so it can be swapped in directly.
    """
    _check_lookback(lookback)
    mean = spread.rolling(lookback).mean().shift(1)
    std = spread.rolling(lookback).std().shift(1)
    return ((spread - mean) / std).rename("z")


def consistent_zscore(log_gc: pd.Series, log_si: pd.Series, alpha: pd.Series,
                      beta: pd.Series, lookback: int) -> pd.DataFrame:
    """Spread and z-score where the trailing window uses today's hedge ratio.

    `zscore(spread(...))` compares today's spread with past spreads that were
    each built with their own day's beta, so a drifting beta moves the mean
    and std even when the relationship hasn't changed. Here, on day t, the
    past window t-lookback .. t-1 is re-expressed with alpha_t and beta_t:

        mean_t = mean(log GC) - alpha_t - beta_t * mean(log SI)
        var_t  = var(log GC) + beta_t^2 var(log SI) - 2 beta_t cov(log GC, log SI)

    with the moments taken over the window ending t-1, which is exact and
    vectorised. Returns columns `spread` and `z`.
    """
    _check_lookback(lookback)
    m_gc = log_gc.rolling(lookback).mean().shift(1)
    m_si = log_si.rolling(lookback).mean().shift(1)
    v_gc = log_gc.rolling(lookback).var().shift(1)
    v_si = log_si.rolling(lookback).var().shift(1)
    c = log_gc.rolling(lookback).cov(log_si).shift(1)
    s = spread(log_gc, log_si, alpha, beta)
    mean = m_gc - alpha - beta * m_si
    var = (v_gc + beta**2 * v_si - 2 * beta * c).clip(lower=0)
    std = np.sqrt(var).where(var > 0)
    return pd.DataFrame({"spread": s, "z": (s - mean) / std})


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def generate_signals(z: pd.Series, entry: float = 2.0, exit_: float = 0.0,
                     stop: float = 4.0) -> pd.Series:
    """Turn z-scores into target spread positions in {-1, 0, +1}.

    - Flat: go short the spread (-1) when z >= entry, long (+1) when z <= -entry.
    - Long: exit when z rises to exit_ or above. Short: exit when z falls to exit_ or below.
    - Stop: |z| >= stop flattens, and we stay flat until |z| is back below
      entry (so we don't re-enter straight into a blow-out).
    - NaN z (warm-up) means flat.

    Same state machine as the #10 skeleton's placeholder.
    """
    if not 0 <= abs(exit_) < entry < stop:
        raise ValueError(f"need |exit_| < entry < stop, got exit_={exit_}, entry={entry}, stop={stop}")
    out = np.zeros(len(z), dtype=int)
    pos, stopped = 0, False
    for i, zi in enumerate(z.to_numpy(dtype=float)):
        if np.isnan(zi):
            pos = 0
        elif abs(zi) >= stop:
            pos, stopped = 0, True
        elif stopped:
            stopped = abs(zi) >= entry
        elif pos == 1 and zi >= exit_:
            pos = 0
        elif pos == -1 and zi <= exit_:
            pos = 0
        elif pos == 0 and zi >= entry:
            pos = -1
        elif pos == 0 and zi <= -entry:
            pos = 1
        out[i] = pos
    return pd.Series(out, index=z.index, name="signal")


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #
def target_ounces(signal: int, beta: float, gc_price: float, si_price: float,
                  gc_notional: float, mode: str = "beta") -> tuple[float, float]:
    """Signed troy ounces per leg for one day. +1 gives long GC, short SI.

    GC ounces = signal * gc_notional / gc_price.
    SI ounces = -signal * k * gc_notional / si_price, with k = beta ("beta"
    mode) or 1 ("dollar" mode). Flat or unknown beta gives (0, 0).
    """
    _check_mode(mode)
    if signal == 0 or not np.isfinite(beta) or not np.isfinite(gc_price) or not np.isfinite(si_price):
        return 0.0, 0.0
    k = beta if mode == "beta" else 1.0
    return signal * gc_notional / gc_price, -signal * k * gc_notional / si_price


def target_contracts(signal: int, beta: float, gc_price: float, si_price: float,
                     gc_notional: float, mode: str = "beta", micros: bool = True) -> dict[str, int]:
    """One day's contract counts per product: {"GC", "MGC", "SI", "SIL"}.

    micros=True: whole standard contracts, then micros for the remainder
    (contracts.ounce_neutral_mix). micros=False: round each leg to the nearest
    standard contract. Same signature as the #10 skeleton's placeholder.
    """
    gc_oz, si_oz = target_ounces(signal, beta, gc_price, si_price, gc_notional, mode)
    if micros:
        gc, mgc = contracts.ounce_neutral_mix(gc_oz, "GC", "MGC")
        si, sil = contracts.ounce_neutral_mix(si_oz, "SI", "SIL")
    else:
        gc, mgc = int(round(gc_oz / contracts.GC.multiplier)), 0
        si, sil = int(round(si_oz / contracts.SI.multiplier)), 0
    return {"GC": gc, "MGC": mgc, "SI": si, "SIL": sil}


def size_positions(signal: pd.Series, beta: pd.Series, gc_price: pd.Series,
                   si_price: pd.Series, gc_notional: float, mode: str = "beta",
                   micros: bool = True) -> pd.DataFrame:
    """Daily contract counts per product, plus the ounces left unhedged.

    Inputs are aligned on `signal`'s index; prices are actual (raw) closes of
    the contracts held. Returns columns GC, MGC, SI, SIL (signed contracts),
    gc_resid_oz and si_resid_oz (target ounces minus ounces actually held),
    and si_dollar_ratio (SI notional / GC notional held, NaN when flat).
    """
    _check_mode(mode)
    idx = signal.index
    b, pg, ps = beta.reindex(idx), gc_price.reindex(idx), si_price.reindex(idx)
    rows, resid = [], []
    for s, bi, g, si_ in zip(signal.to_numpy(), b.to_numpy(float), pg.to_numpy(float), ps.to_numpy(float)):
        counts = target_contracts(int(s), bi, g, si_, gc_notional, mode, micros)
        want_gc, want_si = target_ounces(int(s), bi, g, si_, gc_notional, mode)
        held_gc = counts["GC"] * contracts.GC.multiplier + counts["MGC"] * contracts.MGC.multiplier
        held_si = counts["SI"] * contracts.SI.multiplier + counts["SIL"] * contracts.SIL.multiplier
        rows.append(counts)
        resid.append((want_gc - held_gc, want_si - held_si,
                      abs(held_si * si_) / abs(held_gc * g) if held_gc else np.nan))
    out = pd.DataFrame(rows, index=idx, columns=list(LEGS)).astype(int)
    out[["gc_resid_oz", "si_resid_oz", "si_dollar_ratio"]] = pd.DataFrame(resid, index=idx)
    return out


# --------------------------------------------------------------------------- #
def _check_lookback(lookback: int) -> None:
    if lookback < 2:
        raise ValueError(f"lookback must be at least 2, got {lookback}")


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
