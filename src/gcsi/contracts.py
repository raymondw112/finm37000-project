"""Contract specifications for the COMEX precious-metals futures we trade.

One source of truth for contract terms, so sizing (#9), PnL and rolls (#5)
never hard-code numbers like 100 oz or 5,000 oz.

Sources (CME Group):
  - Gold GC / Micro Gold MGC fact card:
    https://www.cmegroup.com/trading/metals/files/fact-card-gold-futures-options.pdf
  - COMEX Rulebook Ch. 112 (Silver SI):
    https://www.cmegroup.com/content/dam/cmegroup/rulebook/COMEX/1a/112.pdf
  - COMEX Rulebook Ch. 121 (Micro Silver SIL):
    https://www.cmegroup.com/content/dam/cmegroup/rulebook/COMEX/1a/121.pdf

Calendar rules used here:
  - Delivery can happen on any business day of the delivery month, starting
    on its first business day.
  - First notice day (FND) is taken as the last business day of the month
    before the delivery month. Longs still holding after this can be assigned
    delivery, so a strategy must roll out before it.
  - Last trading day is the third-last business day of the delivery month.

Active months: the liquid months we roll between, not every listed month.
Gold uses the CME "active months" from the fact card (Feb, Apr, Jun, Aug, Dec).
Silver uses Mar, May, Jul, Sep, Dec. Jan and Oct are listed but thinner; open
question for #5 whether to include them.

Business days here are weekdays minus an optional holiday list. The CME
holiday calendar (e.g. Good Friday) is not built in; pass `holidays` if exact
dates matter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

# Futures month codes, e.g. GCZ6 = December gold.
MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
               7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}


@dataclass(frozen=True)
class ContractSpec:
    """Static terms of one futures product."""

    root: str                 # Globex root symbol, e.g. "GC"
    name: str
    multiplier: float         # troy ounces per contract
    tick_size: float          # minimum price move, USD per troy ounce (outrights)
    active_months: tuple[int, ...]  # liquid delivery months we roll between
    underlying: str           # "gold" or "silver", used to pair standard with micro

    @property
    def tick_value(self) -> float:
        """USD value of a one-tick move for one contract."""
        return round(self.multiplier * self.tick_size, 10)


GC = ContractSpec("GC", "COMEX Gold", 100.0, 0.10, (2, 4, 6, 8, 12), "gold")
MGC = ContractSpec("MGC", "COMEX Micro Gold", 10.0, 0.10, (2, 4, 6, 8, 12), "gold")
SI = ContractSpec("SI", "COMEX Silver", 5000.0, 0.005, (3, 5, 7, 9, 12), "silver")
SIL = ContractSpec("SIL", "COMEX Micro Silver", 1000.0, 0.005, (3, 5, 7, 9, 12), "silver")

SPECS: dict[str, ContractSpec] = {s.root: s for s in (GC, MGC, SI, SIL)}


def get_spec(root: str) -> ContractSpec:
    """Look up a spec by root symbol (case-insensitive)."""
    try:
        return SPECS[root.upper()]
    except KeyError:
        raise KeyError(f"Unknown contract root {root!r}; known: {sorted(SPECS)}") from None


def notional(root: str, price: float, contracts: float = 1.0) -> float:
    """USD notional of `contracts` contracts at `price` (USD per troy ounce)."""
    return get_spec(root).multiplier * price * contracts


def pnl(root: str, entry_price: float, exit_price: float, contracts: float) -> float:
    """USD PnL of a position; `contracts` is signed (+long, -short)."""
    return get_spec(root).multiplier * (exit_price - entry_price) * contracts


# ---- calendar helpers -------------------------------------------------------

def _bdays(holidays: Iterable | None) -> pd.offsets.CustomBusinessDay:
    return pd.offsets.CustomBusinessDay(holidays=list(holidays or []))


def first_notice_day(year: int, month: int, holidays: Iterable | None = None) -> pd.Timestamp:
    """Last business day of the month before the delivery month (year, month)."""
    first_of_delivery = pd.Timestamp(year=year, month=month, day=1)
    return first_of_delivery - _bdays(holidays)


def last_trading_day(year: int, month: int, holidays: Iterable | None = None) -> pd.Timestamp:
    """Third-last business day of the delivery month."""
    first_of_next = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthBegin(1)
    return first_of_next - 3 * _bdays(holidays)


def contract_symbol(root: str, year: int, month: int) -> str:
    """Globex-style outright symbol, e.g. ('GC', 2026, 12) -> 'GCZ6'."""
    get_spec(root)  # validate root
    return f"{root.upper()}{MONTH_CODES[month]}{year % 10}"


def active_contracts(root: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> list[tuple[int, int]]:
    """Active (year, month) delivery months whose FND falls in [start, end + 1 year].

    The extra year makes sure the contract we roll into near `end` is included.
    """
    spec = get_spec(root)
    start, end = pd.Timestamp(start), pd.Timestamp(end) + pd.DateOffset(years=1)
    out = []
    for year in range(start.year, end.year + 1):
        for month in spec.active_months:
            if start <= first_notice_day(year, month) <= end:
                out.append((year, month))
    return out


def ounce_neutral_mix(ounces: float, standard: str, micro: str) -> tuple[int, int]:
    """Split a target in troy ounces into (standard, micro) contract counts.

    Uses as many standard contracts as fit, then rounds the remainder to the
    nearest micro. Signed: a negative target gives negative counts.
    E.g. 12,345 oz of silver -> (2 SI, 2 SIL) = 12,000 oz.
    """
    std, mic = get_spec(standard), get_spec(micro)
    if std.underlying != mic.underlying:
        raise ValueError(f"{standard} and {micro} have different underlyings")
    sign = 1 if ounces >= 0 else -1
    remaining = abs(ounces)
    n_std = int(remaining // std.multiplier)
    n_mic = int(round((remaining - n_std * std.multiplier) / mic.multiplier))
    return sign * n_std, sign * n_mic
