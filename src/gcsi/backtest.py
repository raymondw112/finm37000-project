"""Walk-forward backtest loop (#10). SKELETON.

Costs and the PnL ledger were split out of #10 into #24 (cost model) and #25
(ledger). This module is the walk-forward loop that ties the pieces together:

  1. hedge ratio on day t from data up to t-1             (#7, placeholder here)
  2. spread and trailing z-score, then a {-1, 0, +1} signal (#9, placeholder here)
  3. signal -> contract counts in GC, MGC, SI, SIL          (#9, placeholder here)
  4. fills at the raw (unadjusted) close, daily mark-to-market on raw closes,
     rolls closed and reopened so the roll gap is never PnL (#25 replaces this)
  5. a flat USD-per-contract-per-side cost                  (#24 replaces this)

Every placeholder is marked `TODO(#N)` with the issue that replaces it, and has
the same inputs and outputs as the interface agreed for that issue, so swapping
one in is a one-line change.

Timing (no look-ahead):
  - beta_t and alpha_t are fitted on log closes up to t-1.
  - z_t = (spread_t - mean) / std, with mean and std from the window ending t-1.
  - The target position is decided at the close of day t from information
    known at that close and filled at that close (raw_close_t). It earns PnL
    from day t+1 on. So the PnL of day t uses only positions decided at t-1.

Rolls: `gc` and `si` are `rolls.build_continuous` outputs. When the held
contract changes between day t and t+1, the position is closed in the old
contract and reopened in the new one at the close of day t (the roll log's
`exec_date`). The PnL of day t+1 is then measured from the new contract's
price on day t, so the gap between contracts never shows up as PnL. The new
contract's day-t price comes from `roll_logs` (`rolls.roll_log`) if given,
otherwise it is backed out of the adjusted close (ratio or back adjustment).
Micros are assumed to trade at the same price as their standard contract and
roll with it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from gcsi import contracts

LEGS = ("GC", "MGC", "SI", "SIL")
_ROOT_OF_LEG = {"GC": "GC", "MGC": "GC", "SI": "SI", "SIL": "SI"}
_REQUIRED = ("close", "raw_close", "contract")


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters of one backtest run.

    window:        trailing days for the hedge-ratio fit.
    z_lookback:    trailing days for the spread mean and std.
    entry, exit_, stop: z-score thresholds (see `signals.generate_signals`).
    gc_notional:   USD notional of the GC leg when in a position.
    mode:          "beta" (SI notional = beta x GC notional) or "dollar".
    micros:        trade MGC / SIL for the remainder after whole contracts.
    cost_per_contract: USD per contract per side, any product.
                   PLACEHOLDER until the #24 cost model lands.
    initial_equity: starting equity in USD.

    Warm-up: beta needs `window` prior days and the z-score a further
    `z_lookback` days of spread, so no trade is possible before row
    `window + z_lookback` (312 trading days, about 15 months, with the
    defaults). The flat start of a run is warm-up, not a result.
    """

    window: int = 252
    z_lookback: int = 60
    entry: float = 2.0
    exit_: float = 0.0
    stop: float = 4.0
    gc_notional: float = 1_000_000.0
    mode: str = "beta"
    micros: bool = True
    cost_per_contract: float = 0.0  # TODO(#24): replace with the cost model
    initial_equity: float = 1_000_000.0


# --------------------------------------------------------------------------- #
# Placeholders. Each keeps the interface of the module that replaces it.
# --------------------------------------------------------------------------- #
def _rolling_hedge_ratio(log_gc: pd.Series, log_si: pd.Series, window: int) -> pd.DataFrame:
    """TODO(#7): replace with `cointegration.rolling_hedge_ratio`.

    Trailing OLS of log GC on log SI, shifted one day: alpha and beta on day t
    use days t-window .. t-1 only. NaN for the first `window` days.
    """
    beta = log_gc.rolling(window).cov(log_si) / log_si.rolling(window).var()
    alpha = log_gc.rolling(window).mean() - beta * log_si.rolling(window).mean()
    return pd.DataFrame({"alpha": alpha, "beta": beta}).shift(1)


def _spread(log_gc: pd.Series, log_si: pd.Series, alpha: pd.Series, beta: pd.Series) -> pd.Series:
    """TODO(#9): replace with `signals.spread`. log GC - alpha - beta * log SI."""
    return log_gc - alpha - beta * log_si


def _zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """TODO(#9): replace with `signals.zscore`. Mean and std from the window ending t-1."""
    mean = spread.rolling(lookback).mean().shift(1)
    std = spread.rolling(lookback).std().shift(1)
    return (spread - mean) / std


def _generate_signals(z: pd.Series, entry: float, exit_: float, stop: float) -> pd.Series:
    """TODO(#9): replace with `signals.generate_signals`.

    +1 = long spread (long GC, short SI), entered when z <= -entry; -1 when
    z >= entry. Exit when z crosses `exit_`. Flatten on |z| >= stop and stay
    flat until |z| < entry again. NaN z -> flat.
    """
    out = np.zeros(len(z), dtype=int)
    pos, stopped = 0, False
    for i, zi in enumerate(z.to_numpy()):
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


def _size_positions(signal: int, beta: float, gc_price: float, si_price: float,
                    gc_notional: float, mode: str, micros: bool) -> dict[str, int]:
    """TODO(#9): replace with `signals.size_positions`.

    One day's target contract counts. +1 gives positive GC and negative SI.
    GC ounces = gc_notional / gc_price; SI ounces = beta * gc_notional / si_price
    ("beta") or gc_notional / si_price ("dollar").
    """
    if signal == 0 or np.isnan(beta):
        return dict.fromkeys(LEGS, 0)
    gc_oz = signal * gc_notional / gc_price
    si_oz = -signal * (beta if mode == "beta" else 1.0) * gc_notional / si_price
    if micros:
        gc, mgc = contracts.ounce_neutral_mix(gc_oz, "GC", "MGC")
        si, sil = contracts.ounce_neutral_mix(si_oz, "SI", "SIL")
    else:
        gc = int(round(gc_oz / contracts.GC.multiplier))
        si = int(round(si_oz / contracts.SI.multiplier))
        mgc = sil = 0
    return {"GC": gc, "MGC": mgc, "SI": si, "SIL": sil}


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def _check(frame: pd.DataFrame, name: str) -> None:
    missing = [c for c in _REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}; pass a rolls.build_continuous output")


def _roll_entry_prices(leg: pd.DataFrame, roll_log: pd.DataFrame | None) -> pd.Series:
    """Price of the NEW contract on the day before each switch, indexed by switch date.

    Taken from `roll_log` (`to_close` by `switch_date`) if given, else backed out
    of the adjusted close: on a switch day the adjusted return is the new
    contract's own return, so new_price(t-1) = raw(t) / (adj(t) / adj(t-1)) for
    ratio adjustment, raw(t) - (adj(t) - adj(t-1)) for back adjustment.
    """
    switch = leg["contract"].ne(leg["contract"].shift()) & leg["contract"].shift().notna()
    if roll_log is not None and len(roll_log):
        prices = roll_log.set_index("switch_date")["to_close"]
        return prices.reindex(leg.index[switch])
    adjustment = leg.attrs.get("adjustment", "ratio")
    raw, adj = leg["raw_close"], leg["close"]
    if adjustment == "ratio":
        implied = raw / (adj / adj.shift())
    elif adjustment == "back":
        implied = raw - adj.diff()
    else:
        raise ValueError("A raw-adjusted series cannot recover roll prices; pass roll_logs")
    return implied[switch]


def simulate(positions: pd.DataFrame, gc: pd.DataFrame, si: pd.DataFrame,
             cost_per_contract: float = 0.0, initial_equity: float = 0.0,
             roll_logs: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Mark given end-of-day positions to market on raw closes, with rolls and costs.

    positions: signed contract counts per leg (any of GC, MGC, SI, SIL; missing
        legs are 0), indexed like `gc` / `si`. The row for day t is the holding
        after trading at the close of day t.
    gc, si: `rolls.build_continuous` outputs (close, raw_close, contract).
    cost_per_contract: USD per contract per side. TODO(#24): cost model.
    roll_logs: optional {"GC": rolls.roll_log(...), "SI": ...}.

    Returns a daily frame: the position columns, gross_pnl, costs, net_pnl,
    equity (all USD). gross_pnl on day t = sum over legs of
    position(t-1) x (raw_close(t) - entry price) x multiplier, where the entry
    price is raw_close(t-1), or the new contract's day t-1 price after a roll.
    Trades at the close of day t cost cost_per_contract x |contracts traded|;
    a roll counts as closing the old contract and opening the new one.
    TODO(#25): replace with the PnL ledger (fills, attribution).
    """
    roll_logs = roll_logs or {}
    _check(gc, "gc")
    _check(si, "si")
    pos = positions.reindex(columns=list(LEGS)).fillna(0).astype(int)
    frames = {"GC": gc, "SI": si}
    raw = {r: f["raw_close"].reindex(pos.index).to_numpy(float) for r, f in frames.items()}
    held = {r: f["contract"].reindex(pos.index).to_numpy(object) for r, f in frames.items()}
    roll_px = {r: _roll_entry_prices(f, roll_logs.get(r)).reindex(pos.index).to_numpy(float)
               for r, f in frames.items()}
    mult = {leg: contracts.get_spec(leg).multiplier for leg in LEGS}

    n = len(pos)
    p = pos.to_numpy()
    gross, costs = np.zeros(n), np.zeros(n)
    for t in range(n):
        for j, leg in enumerate(LEGS):
            root = _ROOT_OF_LEG[leg]
            if t > 0 and p[t - 1, j]:
                rolled = held[root][t] != held[root][t - 1]
                entry = roll_px[root][t] if rolled else raw[root][t - 1]
                gross[t] += p[t - 1, j] * (raw[root][t] - entry) * mult[leg]
            prev = p[t - 1, j] if t > 0 else 0
            rolls_next = t + 1 < n and held[root][t + 1] != held[root][t]
            traded = abs(prev) + abs(p[t, j]) if rolls_next else abs(p[t, j] - prev)
            costs[t] += traded * cost_per_contract

    out = pos.copy()
    out["gross_pnl"] = gross
    out["costs"] = costs
    out["net_pnl"] = gross - costs
    out["equity"] = initial_equity + out["net_pnl"].cumsum()
    return out


def run(gc: pd.DataFrame, si: pd.DataFrame, config: BacktestConfig | dict | None = None,
        roll_logs: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Walk-forward backtest of the GC/SI spread.

    gc, si: `rolls.build_continuous` outputs for each leg. `close` (adjusted)
        drives the signal, `raw_close` (actual prices) drives fills and PnL.
        Joined on dates where both legs traded.
    config: `BacktestConfig` or a dict of its fields; defaults if None.
    roll_logs: optional {"GC": rolls.roll_log(...), "SI": ...} for exact roll prices.

    Returns a daily frame: positions per leg (GC, MGC, SI, SIL), alpha, beta,
    spread, z, signal (+1 = long GC / short SI), gross_pnl, costs, net_pnl,
    equity (USD).
    The first `window + z_lookback` rows are warm-up and always flat (see
    `BacktestConfig`); start evaluation windows after them.
    """
    if config is None:
        config = BacktestConfig()
    elif isinstance(config, dict):
        config = BacktestConfig(**config)
    _check(gc, "gc")
    _check(si, "si")
    idx = gc.index.intersection(si.index).sort_values()
    gc, si = _restrict(gc, idx), _restrict(si, idx)

    log_gc, log_si = np.log(gc["close"]), np.log(si["close"])
    hr = _rolling_hedge_ratio(log_gc, log_si, config.window)  # TODO(#7)
    spread = _spread(log_gc, log_si, hr["alpha"], hr["beta"])  # TODO(#9)
    z = _zscore(spread, config.z_lookback)  # TODO(#9)
    signal = _generate_signals(z, config.entry, config.exit_, config.stop)  # TODO(#9)

    # walk forward: the target on day t uses only what is known at the close of t
    targets = [
        _size_positions(int(signal.iloc[t]), float(hr["beta"].iloc[t]),  # TODO(#9)
                        float(gc["raw_close"].iloc[t]), float(si["raw_close"].iloc[t]),
                        config.gc_notional, config.mode, config.micros)
        for t in range(len(idx))
    ]
    positions = pd.DataFrame(targets, index=idx, columns=list(LEGS))

    out = simulate(positions, gc, si, config.cost_per_contract, config.initial_equity, roll_logs)
    diag = pd.DataFrame({"alpha": hr["alpha"], "beta": hr["beta"], "spread": spread,
                         "z": z, "signal": signal})
    out = pd.concat([out[list(LEGS)], diag, out.drop(columns=list(LEGS))], axis=1)
    out.attrs["config"] = asdict(config)
    return out


def _restrict(frame: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Rows on `idx`, keeping `attrs` (the adjustment type lives there)."""
    out = frame.reindex(idx)
    out.attrs = dict(frame.attrs)
    return out
