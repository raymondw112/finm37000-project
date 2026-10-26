"""PnL and position ledger with attribution (#25).

Takes end-of-day contract positions per leg (GC, MGC, SI, SIL) and the two
`rolls.build_continuous` frames, and produces:

  - `daily`: positions, PnL per leg, gross PnL, trade and roll costs, net PnL,
    equity, and (if target ounces are given) the PnL split into the ideal
    spread PnL and the sizing residual from whole contracts.
  - `fills`: one row per leg per trade: date, leg, contract, signed quantity,
    price, kind ("trade", "roll_close", "roll_open"), cost.

Conventions (same as the #10 skeleton it replaces):
  - The row for day t is the holding after trading at the close of day t, at
    the raw (unadjusted) close of the contract held that day.
  - PnL of day t = position(t-1) x (raw_close(t) - entry) x multiplier, where
    entry is raw_close(t-1), or the new contract's day t-1 price after a roll,
    so the price gap between contracts is never PnL.
  - A roll happens at the close of the last day in the old contract: the old
    position is closed at the old contract's close and the new position opened
    at the new contract's close. If the target also changes that day, the part
    of the trade that keeps the same position is a roll; the rest is a trade.
  - Micros trade at their standard contract's price and roll with it.

Attribution: with `target_oz` (columns gc_oz, si_oz, the exact fractional
ounces the signal wanted), `ideal_pnl` is what those exact ounces would have
earned, and `sizing_residual = gross_pnl - ideal_pnl` is what rounding to whole
contracts cost or made. Then net = ideal + sizing residual - trade costs - roll costs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gcsi import contracts
from gcsi.costs import CostModel

LEGS = ("GC", "MGC", "SI", "SIL")
ROOT_OF_LEG = {"GC": "GC", "MGC": "GC", "SI": "SI", "SIL": "SI"}
_REQUIRED = ("close", "raw_close", "contract")


@dataclass
class Ledger:
    daily: pd.DataFrame
    fills: pd.DataFrame

    def attribution(self) -> dict[str, float]:
        """Totals in USD: gross, ideal spread PnL, sizing residual, costs, net."""
        d = self.daily
        out = {"gross_pnl": d["gross_pnl"].sum(), "trade_costs": d["trade_costs"].sum(),
               "roll_costs": d["roll_costs"].sum(), "net_pnl": d["net_pnl"].sum()}
        if "ideal_pnl" in d:
            out["ideal_pnl"] = d["ideal_pnl"].sum()
            out["sizing_residual"] = d["sizing_residual"].sum()
        out["n_trades"] = int((self.fills["kind"] == "trade").sum()) if len(self.fills) else 0
        out["n_rolls"] = int((self.fills["kind"] == "roll_close").sum()) if len(self.fills) else 0
        return {k: float(v) if not isinstance(v, int) else v for k, v in out.items()}


def check_leg(frame: pd.DataFrame, name: str) -> None:
    missing = [c for c in _REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}; pass a rolls.build_continuous output")


def roll_entry_prices(leg: pd.DataFrame, roll_log: pd.DataFrame | None) -> pd.Series:
    """Price of the NEW contract on the day before each switch, indexed by switch date.

    From `roll_log` (`to_close` by `switch_date`) if given, else backed out of the
    adjusted close: on a switch day the adjusted return is the new contract's own
    return, so new_price(t-1) = raw(t) / (adj(t) / adj(t-1)) for ratio adjustment,
    raw(t) - (adj(t) - adj(t-1)) for back adjustment.
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


def build(positions: pd.DataFrame, gc: pd.DataFrame, si: pd.DataFrame,
          cost_model: CostModel | None = None, initial_equity: float = 0.0,
          roll_logs: dict[str, pd.DataFrame] | None = None,
          target_oz: pd.DataFrame | None = None) -> Ledger:
    """Mark end-of-day positions to market on raw closes, with rolls, costs and attribution.

    positions: signed contract counts per leg (missing legs are 0), indexed like gc/si.
    gc, si: `rolls.build_continuous` outputs (close, raw_close, contract).
    cost_model: `costs.CostModel`; default is the #24 default model.
    roll_logs: optional {"GC": rolls.roll_log(...), "SI": ...} for exact roll prices.
    target_oz: optional frame with gc_oz, si_oz (exact ounces wanted) for attribution.
    """
    cost_model = cost_model if cost_model is not None else CostModel()
    roll_logs = roll_logs or {}
    check_leg(gc, "gc")
    check_leg(si, "si")
    pos = positions.reindex(columns=list(LEGS)).fillna(0).astype(int)
    idx = pos.index
    frames = {"GC": gc, "SI": si}
    raw = {r: f["raw_close"].reindex(idx).to_numpy(float) for r, f in frames.items()}
    held = {r: f["contract"].reindex(idx).to_numpy(object) for r, f in frames.items()}
    roll_px = {r: roll_entry_prices(f, roll_logs.get(r)).reindex(idx).to_numpy(float)
               for r, f in frames.items()}
    mult = {leg: contracts.get_spec(leg).multiplier for leg in LEGS}
    unit = {leg: cost_model.per_contract(leg) for leg in LEGS}

    n, p = len(pos), pos.to_numpy()
    leg_pnl = np.zeros((n, len(LEGS)))
    trade_cost, roll_cost = np.zeros(n), np.zeros(n)
    fills = []

    def entry_price(root: str, t: int) -> float:
        rolled = held[root][t] != held[root][t - 1]
        return roll_px[root][t] if rolled else raw[root][t - 1]

    for t in range(n):
        for j, leg in enumerate(LEGS):
            root = ROOT_OF_LEG[leg]
            if t > 0 and p[t - 1, j]:
                leg_pnl[t, j] = p[t - 1, j] * (raw[root][t] - entry_price(root, t)) * mult[leg]
            prev, now = (p[t - 1, j] if t > 0 else 0), p[t, j]
            rolls_next = t + 1 < n and held[root][t + 1] != held[root][t]
            day = idx[t]
            if rolls_next:
                kept = min(abs(prev), abs(now)) if np.sign(prev) == np.sign(now) else 0
                total = abs(prev) + abs(now)          # close all old, open all new
                roll_n, trade_n = 2 * kept, total - 2 * kept
                roll_cost[t] += roll_n * unit[leg]
                trade_cost[t] += trade_n * unit[leg]
                new_px = roll_px[root][t + 1]
                if prev:
                    fills.append((day, leg, held[root][t], -prev, raw[root][t],
                                  "roll_close" if kept else "trade",
                                  abs(prev) * unit[leg]))
                if now:
                    fills.append((day, leg, held[root][t + 1], now, new_px,
                                  "roll_open" if kept else "trade", abs(now) * unit[leg]))
            elif now != prev:
                trade_cost[t] += abs(now - prev) * unit[leg]
                fills.append((day, leg, held[root][t], now - prev, raw[root][t], "trade",
                              abs(now - prev) * unit[leg]))

    daily = pos.copy()
    for j, leg in enumerate(LEGS):
        daily[f"pnl_{leg}"] = leg_pnl[:, j]
    daily["gross_pnl"] = leg_pnl.sum(axis=1)
    if target_oz is not None:
        oz = target_oz.reindex(idx).fillna(0.0)
        ideal = np.zeros(n)
        for t in range(1, n):
            for root, col in (("GC", "gc_oz"), ("SI", "si_oz")):
                q = oz[col].iloc[t - 1]
                if q:
                    ideal[t] += q * (raw[root][t] - entry_price(root, t))
        daily["ideal_pnl"] = ideal
        daily["sizing_residual"] = daily["gross_pnl"] - daily["ideal_pnl"]
    daily["trade_costs"] = trade_cost
    daily["roll_costs"] = roll_cost
    daily["costs"] = trade_cost + roll_cost
    daily["net_pnl"] = daily["gross_pnl"] - daily["costs"]
    daily["equity"] = initial_equity + daily["net_pnl"].cumsum()
    fills = pd.DataFrame(fills, columns=["date", "leg", "contract", "qty", "price", "kind", "cost"])
    return Ledger(daily=daily, fills=fills)
