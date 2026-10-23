"""Databento ingestion for GC and SI futures (#4).

Pulls per-contract daily bars from GLBX.MDP3 with parent symbology (GC.FUT,
SI.FUT), keeps outright futures only, joins each contract's expiration from
the `definition` schema, and caches everything under data/ (git-ignored) so a
request is never billed or downloaded twice.

    bars = load_bars("GC", "2010-06-06", "2026-10-20")   # feeds rolls.build_continuous

Robustness:
  - Key: DATABENTO_API_KEY (environment or .env) first, else the course helper
    `finm37000.get_databento_api_key()` (~/.databento_api_key). Never printed.
  - Cost guard: `metadata.get_cost` before every pull; raises above `max_cost`
    USD unless `yes=True`.
  - One request per calendar year, so a failure resumes instead of restarting.
    A year that is complete is cached once; the current year is re-pulled only
    when a later end date is asked for.
  - Validation: duplicates, non-positive prices and negative volume are dropped
    and reported; missing active months (from `gcsi.contracts`) are reported.

CLI:
    python -m gcsi.data --start 2010-06-06 --end 2026-10-20 [--roots GC SI] [--yes]
    python -m gcsi.data --start 2010-06-06 --end 2026-10-20 --cost-only
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from gcsi import contracts

DATASET = "GLBX.MDP3"
DATASET_START = pd.Timestamp("2010-06-06", tz="UTC")  # first day of GLBX.MDP3 (#19)
BAR_SCHEMA = "ohlcv-1d"
DEF_SCHEMA = "definition"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "databento"
DEFAULT_MAX_COST = 5.0  # USD for one call; daily bars for GC + SI are far below this

BAR_COLUMNS = ["ts_event", "instrument_id", "symbol", "expiration",
               "open", "high", "low", "close", "volume"]
DEF_COLUMNS = ["instrument_id", "symbol", "expiration"]
_FUTURE_CLASS = "F"  # Databento InstrumentClass.FUTURE; spreads are "S"


class CostLimitExceeded(RuntimeError):
    """The estimated cost of the pulls is above `max_cost` and `yes` was not given."""


@dataclass
class ValidationReport:
    """What `validate_bars` dropped or found missing. Printed by `load_bars`."""

    root: str
    rows: int = 0
    contracts: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    missing_months: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"{self.root}: {self.rows:,} bars, {self.contracts} contracts"]
        for reason, n in self.dropped.items():
            if n:
                lines.append(f"  dropped {n:,} rows: {reason}")
        if self.missing_months:
            lines.append(f"  missing active months: {', '.join(self.missing_months)}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Client and key
# --------------------------------------------------------------------------- #
def get_api_key() -> str:
    """DATABENTO_API_KEY from the environment (or .env), else ~/.databento_api_key.

    Returns the key as a `finm37000` Secret where possible, so printing it shows
    only asterisks. Raises RuntimeError with setup instructions if neither exists.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:  # python-dotenv is a dependency, but don't fail without it
        pass
    key = os.environ.get("DATABENTO_API_KEY")
    try:
        from finm37000 import get_databento_api_key
        from finm37000.db_env_util import Secret
    except ImportError:
        get_databento_api_key, Secret = None, str
    if key:
        return Secret(key)
    if get_databento_api_key is not None:
        try:
            return get_databento_api_key()
        except OSError:
            pass
    raise RuntimeError("No Databento API key: set DATABENTO_API_KEY (or put it in .env), "
                       "or save it to ~/.databento_api_key.")


def get_client():
    """A real `databento.Historical` client built from `get_api_key()`."""
    import databento as db
    return db.Historical(str.__str__(get_api_key()))


# --------------------------------------------------------------------------- #
# Yearly chunks, cost guard and cache
# --------------------------------------------------------------------------- #
def _utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _year_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    """Calendar-year request windows [s, e) covering [start, end), clipped to the dataset."""
    chunks = []
    for year in range(start.year, end.year + 1):
        s = max(pd.Timestamp(year=year, month=1, day=1, tz="UTC"), DATASET_START)
        e = min(pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC"), end)
        if s < e:
            chunks.append((year, s, e))
    return chunks


def _cache_path(cache_dir: Path, root: str, schema: str, year: int) -> Path:
    return cache_dir / f"{root}_{schema}_{year}.parquet"


def _cached_end(path: Path) -> pd.Timestamp | None:
    """End (exclusive) of what a cache file covers, or None if there is no cache."""
    meta = path.with_suffix(".json")
    if not (path.exists() and meta.exists()):
        return None
    return pd.Timestamp(json.loads(meta.read_text())["end"])


def _write_cache(path: Path, df: pd.DataFrame, end: pd.Timestamp) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({"end": end.isoformat()}))


def _clip_end(client: Any, end: pd.Timestamp) -> pd.Timestamp:
    """Clip `end` to the last available date (free metadata call), if the client offers it."""
    get_range = getattr(getattr(client, "metadata", None), "get_dataset_range", None)
    if get_range is None:
        return end
    try:
        available = _utc(get_range(dataset=DATASET)["end"])
    except Exception:  # metadata hiccup: keep the requested end, the pull will say if it's wrong
        return end
    return min(end, available)


def _plan(roots: Iterable[str], schemas: Iterable[str], start: pd.Timestamp, end: pd.Timestamp,
          cache_dir: Path) -> list[dict]:
    """Every (root, schema, year) request not already covered by the cache."""
    todo = []
    for root in roots:
        for schema in schemas:
            for year, s, e in _year_chunks(start, end):
                path = _cache_path(cache_dir, root, schema, year)
                covered = _cached_end(path)
                if covered is None or covered < e:
                    todo.append(dict(root=root, schema=schema, year=year, start=s, end=e, path=path))
    return todo


def estimate_cost(client: Any, todo: list[dict]) -> float:
    """Sum of `metadata.get_cost` (USD) over planned requests. The cost query itself is free."""
    total = 0.0
    for req in todo:
        total += float(client.metadata.get_cost(
            dataset=DATASET, symbols=[f"{req['root']}.FUT"], stype_in="parent",
            schema=req["schema"], start=req["start"].strftime("%Y-%m-%d"),
            end=req["end"].strftime("%Y-%m-%d")))
    return total


def _check_cost(client: Any, todo: list[dict], max_cost: float | None, yes: bool,
                verbose: bool) -> None:
    if not todo or max_cost is None:
        return
    cost = estimate_cost(client, todo)
    if verbose:
        print(f"Databento cost estimate for {len(todo)} uncached request(s): ${cost:,.2f}")
    if cost > max_cost and not yes:
        raise CostLimitExceeded(f"Estimated cost ${cost:,.2f} is above max_cost ${max_cost:,.2f}; "
                                f"pass yes=True (CLI: --yes) to pull anyway.")


def _pull(client: Any, req: dict) -> pd.DataFrame:
    """One yearly request, as a flat frame (ts_event becomes a column for bars)."""
    store = client.timeseries.get_range(
        dataset=DATASET, symbols=[f"{req['root']}.FUT"], stype_in="parent", schema=req["schema"],
        start=req["start"].strftime("%Y-%m-%d"), end=req["end"].strftime("%Y-%m-%d"))
    df = store.to_df()
    if req["schema"] == BAR_SCHEMA:
        df = df.reset_index()
        keep = ["ts_event", "instrument_id", "symbol", "open", "high", "low", "close", "volume"]
    else:
        df = _outright_definitions(df.reset_index(drop=True), req["root"])
        keep = DEF_COLUMNS
    missing = set(keep) - set(df.columns)
    if missing:
        raise ValueError(f"Databento {req['schema']} response for {req['root']} lacks {sorted(missing)}")
    return df[keep]


def _fetch(client: Any, root: str, schema: str, start: pd.Timestamp, end: pd.Timestamp,
           cache_dir: Path) -> pd.DataFrame:
    """Read every yearly chunk from cache, pulling the ones that are missing or stale."""
    frames = []
    for year, s, e in _year_chunks(start, end):
        path = _cache_path(cache_dir, root, schema, year)
        covered = _cached_end(path)
        if covered is None or covered < e:
            df = _pull(client, dict(root=root, schema=schema, year=year, start=s, end=e))
            _write_cache(path, df, e)
        else:
            df = pd.read_parquet(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Definitions, bars, validation
# --------------------------------------------------------------------------- #
def _is_outright(symbol: str, root: str) -> bool:
    """'GCZ6' for root 'GC': root + month code + one digit. Spreads ('GCG5-GCJ5') fail."""
    return (symbol[:-2] == root and len(symbol) == len(root) + 2
            and symbol[-2] in contracts.MONTH_CODES.values() and symbol[-1].isdigit())


def _outright_definitions(defs: pd.DataFrame, root: str) -> pd.DataFrame:
    """Outright futures of `root` from a `definition` frame, one row per instrument_id."""
    if "raw_symbol" in defs:  # real to_df also has a mapped `symbol` column; raw_symbol wins
        defs = defs.drop(columns="symbol", errors="ignore").rename(columns={"raw_symbol": "symbol"})
    if "instrument_class" in defs:
        defs = defs[defs["instrument_class"].astype(str) == _FUTURE_CLASS]
    defs = defs[defs["symbol"].astype(str).map(lambda s: _is_outright(s, root))]
    # the definition schema re-sends every instrument each session: keep the latest
    return (defs.drop_duplicates("instrument_id", keep="last")
            .assign(instrument_id=lambda d: d["instrument_id"].astype("int64"),
                    expiration=lambda d: pd.to_datetime(d["expiration"], utc=True))
            [DEF_COLUMNS].reset_index(drop=True))


def load_definitions(root: str, start: str | pd.Timestamp, end: str | pd.Timestamp,
                     client: Any = None, *, cache_dir: Path | str = DEFAULT_CACHE_DIR,
                     max_cost: float | None = DEFAULT_MAX_COST, yes: bool = False,
                     verbose: bool = True) -> pd.DataFrame:
    """Outright futures definitions for `root` ('GC' or 'SI') listed in [start, end).

    Columns: instrument_id (int64), symbol (e.g. 'GCZ6'), expiration (UTC).
    Calendar spreads, which parent symbology also returns, are removed.
    """
    root = contracts.get_spec(root).root
    start, end = _utc(start), _utc(end)
    client = client if client is not None else get_client()
    end = _clip_end(client, end)
    cache_dir = Path(cache_dir)
    _check_cost(client, _plan([root], [DEF_SCHEMA], start, end, cache_dir), max_cost, yes, verbose)
    defs = _fetch(client, root, DEF_SCHEMA, start, end, cache_dir)
    if defs.empty:
        return pd.DataFrame(columns=DEF_COLUMNS)
    return defs.drop_duplicates("instrument_id", keep="last").reset_index(drop=True)


def validate_bars(bars: pd.DataFrame, root: str, start: pd.Timestamp | str | None = None,
                  end: pd.Timestamp | str | None = None) -> tuple[pd.DataFrame, ValidationReport]:
    """Drop bad rows and report what was dropped and which active months are missing.

    Drops: duplicate (instrument_id, ts_event), missing or non-positive prices,
    negative volume. Missing months are active months (from `gcsi.contracts`)
    whose last trading day falls inside [start, end) but have no bars at all.
    """
    report = ValidationReport(root=root)
    prices = ["open", "high", "low", "close"]

    dup = bars.duplicated(["instrument_id", "ts_event"], keep="last")
    report.dropped["duplicate (contract, day)"] = int(dup.sum())
    bars = bars[~dup]
    bad_price = bars[prices].isna().any(axis=1) | (bars[prices] <= 0).any(axis=1)
    report.dropped["missing or non-positive price"] = int(bad_price.sum())
    bars = bars[~bad_price]
    bad_vol = bars["volume"] < 0
    report.dropped["negative volume"] = int(bad_vol.sum())
    bars = bars[~bad_vol]

    if start is not None and end is not None:
        present = set(zip(bars["expiration"].dt.year, bars["expiration"].dt.month))
        s, e = _utc(start).tz_localize(None), _utc(end).tz_localize(None)
        spec = contracts.get_spec(root)
        for year in range(s.year, e.year + 1):
            for month in spec.active_months:
                if s <= contracts.last_trading_day(year, month) < e and (year, month) not in present:
                    report.missing_months.append(contracts.contract_symbol(root, year, month) + f" ({year})")

    report.rows, report.contracts = len(bars), bars["instrument_id"].nunique()
    return bars.reset_index(drop=True), report


def load_bars(root: str, start: str | pd.Timestamp, end: str | pd.Timestamp,
              client: Any = None, *, cache_dir: Path | str = DEFAULT_CACHE_DIR,
              max_cost: float | None = DEFAULT_MAX_COST, yes: bool = False,
              verbose: bool = True) -> pd.DataFrame:
    """Daily bars for every outright `root` contract ('GC' or 'SI') in [start, end).

    Returns a long frame sorted by (ts_event, instrument_id) with columns
    ts_event (UTC), instrument_id (int64), symbol, expiration (UTC), open, high,
    low, close (USD per troy ounce), volume (contracts). Outrights only: calendar
    spreads and any instrument without a definition are dropped. This is the
    input `rolls.build_continuous(bars, root)` expects.

    `client=None` builds a real Databento client; tests pass the mock from #14.
    `max_cost` (USD) caps the estimated cost of the uncached pulls, checked
    before anything is downloaded; `None` turns the check off. `yes=True`
    pulls even when the estimate is above it. A second identical call is
    served from `cache_dir` and makes no requests.
    """
    root = contracts.get_spec(root).root
    start, end = _utc(start), _utc(end)
    if end <= start:
        raise ValueError(f"end {end.date()} must be after start {start.date()}")
    client = client if client is not None else get_client()
    end = _clip_end(client, end)
    cache_dir = Path(cache_dir)
    _check_cost(client, _plan([root], [DEF_SCHEMA, BAR_SCHEMA], start, end, cache_dir),
                max_cost, yes, verbose)

    defs = load_definitions(root, start, end, client, cache_dir=cache_dir, max_cost=None,
                            verbose=False)
    raw = _fetch(client, root, BAR_SCHEMA, start, end, cache_dir)
    raw["ts_event"] = pd.to_datetime(raw["ts_event"], utc=True)
    raw = raw[(raw["ts_event"] >= start) & (raw["ts_event"] < end)]
    raw = raw.astype({"instrument_id": "int64"})

    bars = raw.drop(columns="symbol").merge(defs, on="instrument_id", how="inner")
    no_def = len(raw) - len(bars)
    bars = bars.astype({"volume": "int64", **{c: "float64" for c in ["open", "high", "low", "close"]}})
    bars, report = validate_bars(bars, root, start, end)
    report.dropped = {"spread or no outright definition": no_def, **report.dropped}
    if verbose:
        print(report)
    return bars[BAR_COLUMNS].sort_values(["ts_event", "instrument_id"], ignore_index=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    today = pd.Timestamp.now(tz="UTC").normalize()
    parser = argparse.ArgumentParser(description="Pull and cache GC/SI daily bars from Databento.")
    parser.add_argument("--start", default=DATASET_START.strftime("%Y-%m-%d"))
    parser.add_argument("--end", default=today.strftime("%Y-%m-%d"), help="exclusive")
    parser.add_argument("--roots", nargs="+", default=["GC", "SI"])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST, help="USD")
    parser.add_argument("--yes", action="store_true", help="pull even above --max-cost")
    parser.add_argument("--cost-only", action="store_true",
                        help="print the (free) cost estimate for uncached pulls and exit")
    args = parser.parse_args(argv)

    client = get_client()
    start, end = _utc(args.start), _clip_end(client, _utc(args.end))
    if args.cost_only:
        todo = _plan(args.roots, [DEF_SCHEMA, BAR_SCHEMA], start, end, args.cache_dir)
        print(f"{len(todo)} uncached request(s), {start.date()} to {end.date()}: "
              f"${estimate_cost(client, todo):,.2f}")
        return
    todo = _plan(args.roots, [DEF_SCHEMA, BAR_SCHEMA], start, end, args.cache_dir)
    _check_cost(client, todo, args.max_cost, args.yes, verbose=True)
    for root in args.roots:
        bars = load_bars(root, start, end, client, cache_dir=args.cache_dir, max_cost=None)
        print(f"{root}: {bars['ts_event'].min().date()} to {bars['ts_event'].max().date()}")
    print(f"Cached under {args.cache_dir}")


if __name__ == "__main__":
    main()
