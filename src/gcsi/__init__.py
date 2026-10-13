"""gcsi: Gold (GC) / Silver (SI) futures pairs-trading pipeline.

Component map (each module maps to a GitHub Issue):
    data         Databento ingestion + local cache            (Issue 2)
    rolls        continuous-contract build + roll adjustment   (Issue 3)
    cointegration  ADF / Engle-Granger / Johansen + hedge ratio (Issue 5)
    ou           Ornstein-Uhlenbeck fit + half-life            (Issue 6)
    signals      z-score entry/exit + position sizing          (Issue 7)
    backtest     walk-forward engine with costs                (Issue 8)
    metrics      performance evaluation                        (Issue 9)

Data layer reuses the course `finm37000` package where possible.
All functions are stubs for now; see docs/DESIGN.md for the pipeline.
"""

__all__ = ["data", "rolls", "cointegration", "ou", "signals", "backtest", "metrics"]
