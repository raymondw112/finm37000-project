"""Walk-forward backtest engine with realistic futures costs (Issue 8).

Calibrate hedge ratio and thresholds only on data prior to each trade.
PnL on the actual-price series (not the ratio-adjusted one), in dollars per
contract. Cost model: four bid-ask crossings per round trip, monthly roll
cost, SPAN spread-margin credit; fills from top-of-book, not mid.
"""
from __future__ import annotations

import pandas as pd

GC_MULTIPLIER = 100.0    # oz per GC contract
SI_MULTIPLIER = 5000.0   # oz per SI contract


def run(positions: pd.DataFrame, prices: pd.DataFrame, costs: dict) -> pd.DataFrame:
    """Simulate the strategy and return a daily-marked PnL / equity frame.

    positions: GC/SI contract counts over time (from signals.size_positions).
    prices: actual (unadjusted) GC/SI prices with roll dates/costs.
    costs: {'bid_ask', 'commission', 'roll_cost', ...}. See Issue 8.
    """
    raise NotImplementedError
