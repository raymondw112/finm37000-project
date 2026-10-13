"""Performance evaluation (Issue 9).

Summarize a backtest equity curve: total and annualized return, Sharpe,
max drawdown, hit rate, turnover. Stress the strategy out-of-sample across
the 2008, 2011, and 2020 regimes. Write tables and figures to output/.
"""
from __future__ import annotations

import pandas as pd


def summarize(equity: pd.Series) -> dict:
    """Return headline metrics for an equity curve."""
    raise NotImplementedError


def stress_periods(equity: pd.Series, periods: dict) -> pd.DataFrame:
    """Per-regime metrics (e.g. {'2008': (start, end), ...})."""
    raise NotImplementedError
