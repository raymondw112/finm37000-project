"""Databento ingestion for GC and SI futures (Issue 2).

Pull per-expiry data with parent symbology (GC.FUT, SI.FUT) from GLBX.MDP3,
cache locally so requests are not re-billed. Reuse finm37000.db_env_util for
the API key (read from the DATABENTO_API_KEY environment variable, never
hard-coded).
"""
from __future__ import annotations

import pandas as pd

DATASET = "GLBX.MDP3"


def pull_expiries(symbol: str, start: str, end: str, schema: str = "ohlcv-1d") -> pd.DataFrame:
    """Download per-expiry data for a parent symbol (e.g. 'GC.FUT').

    Returns a tidy frame indexed by (ts, instrument_id). Caches raw downloads
    under data/ (gitignored). See Issue 2.
    """
    raise NotImplementedError


def load_cached(symbol: str) -> pd.DataFrame:
    """Load a previously cached pull from data/, or raise if absent."""
    raise NotImplementedError
