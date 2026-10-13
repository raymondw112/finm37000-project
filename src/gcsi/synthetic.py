"""Synthetic data with known true parameters, for tests and notebooks (Issue #14).

Every generator is deterministic given ``seed`` and returns the parameters it
was built from, so a test can check that an estimator recovers them. Nothing
here touches Databento or the network.

Generators
    simulate_ou            OU spread with known kappa / theta / sigma        (-> #8)
    cointegrated_prices    GC/SI prices with a known hedge ratio beta        (-> #7, #9)
    per_expiry_bars        Databento-style per-expiry daily bars + known rolls (-> #4, #5)
    contract_definitions   matching instrument_id -> symbol / expiration     (-> #4, #5)
    top_of_book            daily bid/ask with an exact one-tick spread       (-> #10)
    toy_positions          GC/SI contract counts with known trades           (-> #10)

Conventions
    * Time steps are trading days (dt = 1); kappa is per trading day.
    * Timestamps are tz-aware UTC, like Databento's ``ts_event``.
    * Contract specs below are for the synthetic data only. The project's
      authoritative specs will live in the contract-specs module (Konark's N3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GC_MULTIPLIER = 100.0   # troy oz per GC contract
SI_MULTIPLIER = 5000.0  # troy oz per SI contract
TICK_SIZE = {"GC": 0.10, "SI": 0.005}

DEFAULT_START = "2025-01-02"


def _trading_days(n: int, start: str = DEFAULT_START) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, tz="UTC", name="ts_event")


def _round_to_tick(x: np.ndarray, tick: float) -> np.ndarray:
    return np.round(np.asarray(x) / tick) * tick


# --------------------------------------------------------------------------- #
# Ornstein-Uhlenbeck spread
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OUParams:
    """dX = kappa (theta - X) dt + sigma dW, with dt in trading days.

    Defaults give a half-life of ln(2)/0.05 ~= 13.9 trading days.
    """

    kappa: float = 0.05
    theta: float = 0.0
    sigma: float = 0.01

    @property
    def half_life(self) -> float:
        return np.log(2.0) / self.kappa

    @property
    def ar1_b(self) -> float:
        """Slope b in the exact AR(1) form X_t = a + b X_{t-1} + e_t (dt = 1)."""
        return float(np.exp(-self.kappa))

    @property
    def ar1_a(self) -> float:
        return self.theta * (1.0 - self.ar1_b)

    @property
    def step_std(self) -> float:
        """Std of the AR(1) innovation e_t."""
        return self.sigma * np.sqrt((1.0 - np.exp(-2.0 * self.kappa)) / (2.0 * self.kappa))

    @property
    def stationary_std(self) -> float:
        return self.sigma / np.sqrt(2.0 * self.kappa)


def simulate_ou(params: OUParams = OUParams(), n: int = 2520, seed: int = 0,
                x0: float | None = None, start: str = DEFAULT_START) -> pd.Series:
    """Simulate an OU path with the exact discretization (no Euler bias).

    Starts at ``theta`` unless ``x0`` is given. Returns a Series named 'spread'.
    """
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = params.theta if x0 is None else x0
    shocks = rng.standard_normal(n - 1) * params.step_std
    for t in range(1, n):
        x[t] = params.ar1_a + params.ar1_b * x[t - 1] + shocks[t - 1]
    return pd.Series(x, index=_trading_days(n, start), name="spread")


# --------------------------------------------------------------------------- #
# Cointegrated GC / SI prices
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CointParams:
    """log(GC) = alpha + beta * log(SI) + spread, spread ~ OU, log(SI) ~ random walk.

    Defaults start near GC = 2000, SI = 25 (gold/silver ratio ~80).
    """

    beta: float = 0.8
    gc0: float = 2000.0
    si0: float = 25.0
    si_daily_vol: float = 0.018
    spread: OUParams = field(default_factory=OUParams)

    @property
    def alpha(self) -> float:
        return float(np.log(self.gc0) - self.beta * np.log(self.si0))


@dataclass(frozen=True)
class SyntheticPair:
    """Output of ``cointegrated_prices``: prices plus the truth used to build them."""

    prices: pd.DataFrame  # columns GC, SI (price levels)
    spread: pd.Series     # true spread = log GC - alpha - beta log SI
    params: CointParams


def cointegrated_prices(params: CointParams = CointParams(), n: int = 2520,
                        seed: int = 0, start: str = DEFAULT_START) -> SyntheticPair:
    """Simulate GC/SI daily prices that are cointegrated with a known beta."""
    rng = np.random.default_rng(seed)
    idx = _trading_days(n, start)
    log_si = np.log(params.si0) + np.concatenate(
        [[0.0], np.cumsum(rng.standard_normal(n - 1) * params.si_daily_vol)])
    # independent child seed so the spread does not reuse the SI shocks
    spread = simulate_ou(params.spread, n=n, seed=int(rng.integers(2**31)), start=start)
    log_gc = params.alpha + params.beta * log_si + spread.to_numpy()
    prices = pd.DataFrame({"GC": np.exp(log_gc), "SI": np.exp(log_si)}, index=idx)
    return SyntheticPair(prices=prices, spread=spread, params=params)


# --------------------------------------------------------------------------- #
# Per-expiry bars (Databento ohlcv-1d style) with known roll dates
# --------------------------------------------------------------------------- #
# GC and SI list different active months, so their rolls fall on different dates.
_CONTRACTS = {
    "GC": [("GCG5", "2025-02-26"), ("GCJ5", "2025-04-28"), ("GCM5", "2025-06-26")],
    "SI": [("SIH5", "2025-03-27"), ("SIK5", "2025-05-28"), ("SIN5", "2025-07-29")],
}
_FIRST_ID = {"GC": 1000, "SI": 2000}
_START_PRICE = {"GC": 2000.0, "SI": 25.0}
ROLL_LEAD_DAYS = 22  # volume crossover this many trading days before expiration


def contract_definitions(root: str) -> pd.DataFrame:
    """Instrument definitions for the synthetic contracts of ``root`` ('GC' or 'SI').

    Columns: instrument_id, raw_symbol, expiration (UTC). Mirrors the fields we
    need from Databento's ``definition`` schema.
    """
    rows = [
        {"instrument_id": _FIRST_ID[root] + k, "raw_symbol": sym,
         "expiration": pd.Timestamp(exp, tz="UTC")}
        for k, (sym, exp) in enumerate(_CONTRACTS[root])
    ]
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PerExpiry:
    """Output of ``per_expiry_bars``."""

    bars: pd.DataFrame         # long format, one row per (ts_event, instrument_id)
    definitions: pd.DataFrame  # see contract_definitions
    roll_dates: pd.Series      # from_symbol -> first date the next contract has more volume
    roll_ratios: pd.Series     # from_symbol -> close(next) / close(current) on roll date
    annual_carry: float


def per_expiry_bars(root: str = "GC", seed: int = 0, annual_carry: float = 0.04,
                    daily_vol: float = 0.01, start: str = "2024-12-02") -> PerExpiry:
    """Databento-style daily bars for three consecutive expiries of ``root``.

    All contracts share one spot path; contract k trades at
    spot * exp(annual_carry * years_to_expiry_k), so every roll has a known
    price jump. The front contract has the most volume until exactly
    ``ROLL_LEAD_DAYS`` trading days before its expiration, when the next
    contract takes over. Contracts stop trading after their expiration.
    """
    if root not in _CONTRACTS:
        raise ValueError(f"root must be one of {sorted(_CONTRACTS)}")
    rng = np.random.default_rng(seed)
    defs = contract_definitions(root)
    last = defs["expiration"].max()
    idx = pd.bdate_range(pd.Timestamp(start, tz="UTC"), last.normalize(), name="ts_event")
    n = len(idx)
    spot = _START_PRICE[root] * np.exp(
        np.concatenate([[0.0], np.cumsum(rng.standard_normal(n - 1) * daily_vol)]))

    # roll (volume crossover) dates: ROLL_LEAD_DAYS trading days before each expiry
    roll_dates = {}
    for k in range(len(defs) - 1):
        exp_pos = idx.searchsorted(defs["expiration"].iloc[k].normalize())
        roll_dates[defs["raw_symbol"].iloc[k]] = idx[exp_pos - ROLL_LEAD_DAYS]
    # index of the most-traded contract on each day (switches ON the roll date)
    roll_naive = pd.DatetimeIndex(list(roll_dates.values())).tz_convert(None).to_numpy()
    active = np.searchsorted(roll_naive, idx.tz_convert(None).to_numpy(), side="right")

    tick = TICK_SIZE[root]
    frames = []
    for k, row in defs.iterrows():
        alive = idx <= row["expiration"]
        years = (row["expiration"] - idx).days / 365.25
        close = _round_to_tick(spot * np.exp(annual_carry * years), tick)
        noise = np.abs(rng.standard_normal((3, n))) * daily_vol / 4
        opn = _round_to_tick(close * (1 + rng.choice([-1, 1], n) * noise[0]), tick)
        high = _round_to_tick(np.maximum(opn, close) * (1 + noise[1]), tick)
        low = _round_to_tick(np.minimum(opn, close) * (1 - noise[2]), tick)
        base = np.where(active == k, 50_000, np.where(active == k - 1, 5_000, 500))
        volume = (base * rng.uniform(0.9, 1.1, n)).astype(np.int64)
        frames.append(pd.DataFrame({
            "ts_event": idx, "rtype": 35, "publisher_id": 1,
            "instrument_id": row["instrument_id"],
            "open": opn, "high": np.maximum(high, np.maximum(opn, close)),
            "low": np.minimum(low, np.minimum(opn, close)), "close": close,
            "volume": volume, "symbol": row["raw_symbol"],
        })[alive])
    bars = (pd.concat(frames, ignore_index=True)
            .sort_values(["ts_event", "instrument_id"], ignore_index=True))

    closes = bars.pivot(index="ts_event", columns="symbol", values="close")
    syms = list(defs["raw_symbol"])
    ratios = {s: closes.loc[d, syms[i + 1]] / closes.loc[d, s]
              for i, (s, d) in enumerate(roll_dates.items())}
    return PerExpiry(bars=bars, definitions=defs,
                     roll_dates=pd.Series(roll_dates, name="roll_date"),
                     roll_ratios=pd.Series(ratios, name="roll_ratio"),
                     annual_carry=annual_carry)


# --------------------------------------------------------------------------- #
# Top of book and positions (for the cost model / backtest)
# --------------------------------------------------------------------------- #
def top_of_book(mid: pd.Series, root: str, seed: int = 0) -> pd.DataFrame:
    """Daily best bid/ask around ``mid`` with an exact one-tick spread.

    Columns: bid_px, ask_px, bid_sz, ask_sz. The bid is ``mid`` floored to the
    tick grid, so ask_px - bid_px == TICK_SIZE[root] on every row.
    """
    rng = np.random.default_rng(seed)
    tick = TICK_SIZE[root]
    bid = np.floor(np.round(mid.to_numpy() / tick, 9)) * tick
    n = len(mid)
    return pd.DataFrame({
        "bid_px": bid, "ask_px": bid + tick,
        "bid_sz": rng.integers(1, 50, n), "ask_sz": rng.integers(1, 50, n),
    }, index=mid.index)


def toy_positions(index: pd.DatetimeIndex, gc_lots: int = 5, si_lots: int = 8) -> pd.DataFrame:
    """GC/SI contract counts with two known round trips (needs len(index) >= 80).

    Long spread (+GC, -SI) on rows [10, 30), short spread on rows [50, 70),
    flat otherwise. 5 GC : 8 SI is roughly dollar-neutral at a ratio of ~80.
    """
    if len(index) < 80:
        raise ValueError("toy_positions needs at least 80 timestamps")
    gc = np.zeros(len(index), dtype=int)
    gc[10:30] = gc_lots
    gc[50:70] = -gc_lots
    si = -np.sign(gc) * si_lots
    return pd.DataFrame({"GC": gc, "SI": si}, index=index)
