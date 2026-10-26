"""Transaction cost model (#24).

Every contract traded costs, per side:

    (half_spread_ticks + slippage_ticks) x tick value  +  exchange fee  +  commission

Tick values come from `gcsi.contracts` (GC $10, MGC $1, SI $25, SIL $5), never
typed here. A roll is two trades of the same leg (close the old contract, open
the new one) and is charged the same way; the ledger (#25) reports roll costs
separately from signal trades.

Defaults (per contract, per side):
  - half_spread_ticks = 0.5: crossing a one-tick market, which is the usual
    state of the front GC and SI books during US hours. Our fills are at the
    daily close, so this is a modelling assumption, not a measured number.
  - exchange_fee: COMEX non-member fees as passed through by Interactive
    Brokers (GC and SI $1.65, MGC $0.70); SIL is not on that schedule and is
    assumed equal to MGC. Source:
    https://brokerage.ibkr.com/en/accounts/fees/COMEX.php
  - commission: $0.85 standard, $0.25 micro, a typical retail broker rate.
    Assumption; change it to your own broker's.

`CostModel.flat(usd)` charges the same USD per contract per side on every leg,
which is what the backtest skeleton used before this module existed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from gcsi import contracts

LEGS = ("GC", "MGC", "SI", "SIL")


@dataclass(frozen=True)
class LegCost:
    """Cost terms for one product, per contract per side."""

    half_spread_ticks: float = 0.5
    exchange_fee: float = 0.0
    commission: float = 0.0
    slippage_ticks: float = 0.0
    flat: float | None = None  # if set, overrides everything above

    def per_contract(self, root: str) -> float:
        """USD cost of trading one contract of `root` once (one side)."""
        if self.flat is not None:
            return float(self.flat)
        ticks = self.half_spread_ticks + self.slippage_ticks
        return ticks * contracts.get_spec(root).tick_value + self.exchange_fee + self.commission


def _default_legs() -> dict[str, LegCost]:
    return {
        "GC": LegCost(0.5, exchange_fee=1.65, commission=0.85),
        "MGC": LegCost(0.5, exchange_fee=0.70, commission=0.25),
        "SI": LegCost(0.5, exchange_fee=1.65, commission=0.85),
        "SIL": LegCost(0.5, exchange_fee=0.70, commission=0.25),  # fee assumed = MGC
    }


@dataclass(frozen=True)
class CostModel:
    """Per-leg costs. `legs` maps GC, MGC, SI, SIL to a `LegCost`."""

    legs: dict[str, LegCost] = field(default_factory=_default_legs)

    @classmethod
    def flat(cls, usd_per_contract: float) -> "CostModel":
        """Same USD per contract per side on every leg (the skeleton's placeholder)."""
        return cls({leg: LegCost(flat=usd_per_contract) for leg in LEGS})

    @classmethod
    def zero(cls) -> "CostModel":
        return cls.flat(0.0)

    def per_contract(self, leg: str) -> float:
        """USD per contract per side for `leg`."""
        return self.legs[leg].per_contract(leg)

    def cost(self, leg: str, contracts_traded: float) -> float:
        """USD cost of trading `contracts_traded` contracts of `leg` (sign ignored)."""
        return abs(contracts_traded) * self.per_contract(leg)

    def scaled(self, factor: float) -> "CostModel":
        """Every cost multiplied by `factor` (for cost-sensitivity runs)."""
        out = {}
        for leg, c in self.legs.items():
            if c.flat is not None:
                out[leg] = replace(c, flat=c.flat * factor)
            else:
                out[leg] = LegCost(c.half_spread_ticks * factor, c.exchange_fee * factor,
                                   c.commission * factor, c.slippage_ticks * factor)
        return CostModel(out)

    def table(self) -> dict[str, float]:
        """{leg: USD per contract per side}, for reports."""
        return {leg: round(self.per_contract(leg), 4) for leg in LEGS}

    def round_trip_spread_cost(self, n_gc: float, n_si: float) -> float:
        """USD to open and close a spread of n_gc GC and n_si SI contracts."""
        return 2 * (self.cost("GC", n_gc) + self.cost("SI", n_si))
