"""Signal generation and position sizing (Issue 7).

Standardize the spread to a z-score (parameters estimated only on prior data),
enter at |z| ~ 2, exit near 0, optional stop at |z| ~ 3-4. Size positions in
CME contracts (GC = 100 oz, SI = 5000 oz); state the neutrality choice
(dollar- vs beta- vs ounce-neutral) and how fractional contracts are rounded.
"""
from __future__ import annotations

import pandas as pd


def zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """Rolling z-score using only trailing data (no look-ahead)."""
    raise NotImplementedError


def generate_signals(z: pd.Series, entry: float = 2.0, exit_: float = 0.0,
                     stop: float = 4.0) -> pd.Series:
    """Map z-scores to target spread positions in {-1, 0, +1}."""
    raise NotImplementedError


def size_positions(signal: pd.Series, beta: pd.Series, mode: str = "dollar") -> pd.DataFrame:
    """Translate a spread signal into GC/SI contract counts. See Issue 7."""
    raise NotImplementedError
