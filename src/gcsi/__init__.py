"""gcsi: Gold (GC) / Silver (SI) futures pairs-trading pipeline.

Component map (each module maps to a GitHub Issue):
    contracts    contract specs (GC, MGC, SI, SIL) + calendar  (#18)
    data         Databento ingestion + local cache            (#4)
    rolls        continuous-contract build + roll adjustment   (#5)
    cointegration  ADF / Engle-Granger / Johansen + hedge ratio (#7)
    ou           Ornstein-Uhlenbeck fit + half-life            (#8)
    signals      z-score entry/exit + position sizing          (#9)
    backtest     walk-forward engine with costs                (#10)
    metrics      performance evaluation                        (#11)
    synthetic    synthetic data with known parameters, for tests (#14)

Data layer reuses the course `finm37000` package where possible.
All functions are stubs for now; see docs/DESIGN.md for the pipeline.
"""

__all__ = ["contracts", "data", "rolls", "cointegration", "ou", "signals", "backtest", "metrics", "synthetic"]
