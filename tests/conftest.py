"""Shared pytest fixtures: synthetic data with known parameters + a mock Databento client (Issue #14).

Use these to build and test a module before real data or upstream issues land.
Each fixture's docstring states the true values it was generated with; the
underlying generators live in ``gcsi.synthetic`` so notebooks can reuse them.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from gcsi import synthetic as syn

SEED = 20261013  # fixed so every run (and every teammate) gets identical data


@pytest.fixture(scope="session")
def ou_params() -> syn.OUParams:
    """True OU parameters: kappa=0.05/day, theta=0, sigma=0.01 -> half-life ~13.86 days."""
    return syn.OUParams(kappa=0.05, theta=0.0, sigma=0.01)


@pytest.fixture(scope="session")
def ou_spread(ou_params) -> pd.Series:
    """5,000-day OU path from ``ou_params`` (seed fixed). For #8."""
    return syn.simulate_ou(ou_params, n=5000, seed=SEED)


@pytest.fixture(scope="session")
def coint_pair() -> syn.SyntheticPair:
    """2,520 days of GC/SI prices with log GC = alpha + 0.8 log SI + OU spread.

    True values in ``coint_pair.params`` (beta=0.8, spread kappa=0.05) and the
    true spread in ``coint_pair.spread``. For #7 and #9.
    """
    return syn.cointegrated_prices(n=2520, seed=SEED)


@pytest.fixture(scope="session")
def gc_expiries() -> syn.PerExpiry:
    """GCG5/GCJ5/GCM5 daily bars, 4% annual carry, known ``roll_dates``/``roll_ratios``. For #5."""
    return syn.per_expiry_bars("GC", seed=SEED)


@pytest.fixture(scope="session")
def si_expiries() -> syn.PerExpiry:
    """SIH5/SIK5/SIN5 daily bars; rolls on different dates from GC. For #5."""
    return syn.per_expiry_bars("SI", seed=SEED + 1)


@pytest.fixture(scope="session")
def gc_top_of_book(coint_pair) -> pd.DataFrame:
    """Daily GC bid/ask around the synthetic GC price, spread exactly 0.10 (one tick). For #10."""
    return syn.top_of_book(coint_pair.prices["GC"], "GC", seed=SEED)


@pytest.fixture(scope="session")
def si_top_of_book(coint_pair) -> pd.DataFrame:
    """Daily SI bid/ask, spread exactly 0.005 (one tick). For #10."""
    return syn.top_of_book(coint_pair.prices["SI"], "SI", seed=SEED)


@pytest.fixture(scope="session")
def toy_positions(coint_pair) -> pd.DataFrame:
    """5 GC : 8 SI spread positions: long rows 10-29, short rows 50-69, else flat. For #10."""
    return syn.toy_positions(coint_pair.prices.index[:100])


# --------------------------------------------------------------------------- #
# Mock Databento client
# --------------------------------------------------------------------------- #
@dataclass
class _Store:
    """Stands in for databento.DBNStore: only ``to_df`` is implemented."""

    df: pd.DataFrame

    def to_df(self) -> pd.DataFrame:
        return self.df.copy()


class _Timeseries:
    def __init__(self, client: "MockDatabentoClient"):
        self._client = client

    def get_range(self, dataset, symbols, schema, start=None, end=None,
                  stype_in="raw_symbol", **kwargs):
        """Mimics ``databento.Historical().timeseries.get_range``.

        Supports parent symbols 'GC.FUT' / 'SI.FUT' with schemas 'ohlcv-1d' and
        'definition'. Every call is recorded in ``client.calls``.
        """
        self._client.calls.append(dict(dataset=dataset, symbols=symbols, schema=schema,
                                       start=start, end=end, stype_in=stype_in, **kwargs))
        if dataset != "GLBX.MDP3":
            raise ValueError(f"mock only serves GLBX.MDP3, got {dataset!r}")
        roots = [s.split(".")[0] for s in ([symbols] if isinstance(symbols, str) else symbols)]
        frames = []
        for root in roots:
            data = self._client.data[root]
            if schema == "ohlcv-1d":
                df = data.bars.set_index("ts_event")
                if start is not None:
                    df = df[df.index >= pd.Timestamp(start, tz="UTC")]
                if end is not None:
                    df = df[df.index < pd.Timestamp(end, tz="UTC")]
            elif schema == "definition":
                df = data.definitions
            else:
                raise ValueError(f"mock does not serve schema {schema!r}")
            frames.append(df)
        return _Store(pd.concat(frames))


class MockDatabentoClient:
    """Drop-in for ``databento.Historical`` in tests: no network, no billing.

    ``calls`` lists every request, so a test can assert that a second run hit
    the local cache instead of the client (``len(client.calls)`` unchanged).
    """

    def __init__(self, data: dict[str, syn.PerExpiry]):
        self.data = data
        self.calls: list[dict] = []
        self.timeseries = _Timeseries(self)


@pytest.fixture
def mock_databento_client(gc_expiries, si_expiries) -> MockDatabentoClient:
    """Fresh mock client per test, serving the GC/SI per-expiry fixtures. For #4."""
    return MockDatabentoClient({"GC": gc_expiries, "SI": si_expiries})
