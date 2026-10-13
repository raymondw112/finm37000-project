"""Continuous-contract construction and roll adjustment (Issue 3).

Databento continuous prices are UNADJUSTED. We build the continuous series
ourselves and produce TWO series per instrument:
  - ratio-adjusted  -> used for SIGNALS (preserves returns/ratios)
  - actual-price    -> used for PnL (with logged roll dates/costs)
Never mix them. Roll before First Notice Day (GC/SI are physically delivered).
"""
from __future__ import annotations

from enum import Enum

import pandas as pd


class Adjustment(str, Enum):
    RAW = "raw"
    BACK = "back"      # Panama / additive
    RATIO = "ratio"    # multiplicative


def build_continuous(expiries: pd.DataFrame, month: int = 0, roll_days_before_fnd: int = 5,
                     adjustment: Adjustment = Adjustment.RATIO) -> pd.DataFrame:
    """Stitch per-expiry data into one continuous series.

    month: 0 = front (CL1 / .c.0), 1 = second (CL2 / .c.1).
    Returns the adjusted series plus a column of roll dates. See Issue 3.
    """
    raise NotImplementedError


def roll_schedule(expiries: pd.DataFrame, roll_days_before_fnd: int = 5) -> pd.DataFrame:
    """Compute roll dates (volume/OI crossover or N days before FND)."""
    raise NotImplementedError
