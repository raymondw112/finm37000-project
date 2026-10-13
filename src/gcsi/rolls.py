"""Continuous-contract construction and roll schedule (#5).

Databento continuous prices are UNADJUSTED and its default `.c.0` rule holds
the front month into delivery. We build our own roll schedule and use the
course package (`finm37000.continuous`) for the actual splicing.

Two series per instrument, never mixed:
  - ratio-adjusted  -> SIGNALS. Returns across a roll are the new contract's
                       own return, so roll gaps do not look like price moves.
  - raw (unadjusted) -> PnL, together with `roll_log`, which records each
                       roll's prices so its cost can be charged explicitly.

Decisions (see #5 for the discussion):
  - Contracts: active months only (`gcsi.contracts`), front month first.
  - Roll trigger: "fnd" rolls a fixed number of business days before first
    notice day. "volume" rolls on volume crossover, but never later than the
    "fnd" date. GC/SI are physically delivered, so we never hold into FND.
  - No look-ahead: a roll executes at the close of day t-1 using information
    known at that close (the volume of t-1), and the series uses the new
    contract from day t.
  - Each leg gets its own schedule (gold and silver have different cycles).
  - Pair alignment: daily, inner join on the session date (`align_pair`).
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable

import pandas as pd
from finm37000.continuous import multiplicative_splice, additive_splice, splice_unadjusted

from gcsi import contracts

DATE_COL = "ts_event"
_CODE_TO_MONTH = {code: month for month, code in contracts.MONTH_CODES.items()}


class Adjustment(str, Enum):
    RAW = "raw"
    BACK = "back"      # additive (Panama)
    RATIO = "ratio"    # multiplicative


class RollRule(str, Enum):
    FND = "fnd"
    VOLUME = "volume"


def parse_symbol(symbol: str, ref_date: pd.Timestamp) -> tuple[str, int, int]:
    """'GCZ6' -> ('GC', 2026, 12). The one-digit year is resolved near `ref_date`."""
    root, code, digit = symbol[:-2], symbol[-2], symbol[-1]
    if not root.isalpha() or code not in _CODE_TO_MONTH or not digit.isdigit():
        raise ValueError(f"Not an outright futures symbol: {symbol!r}")
    ref_year = pd.Timestamp(ref_date).year
    year = ref_year - ref_year % 10 + int(digit)
    if year < ref_year - 2:
        year += 10
    elif year > ref_year + 7:
        year -= 10
    return root, year, _CODE_TO_MONTH[code]


def _outrights(bars: pd.DataFrame, root: str) -> pd.DataFrame:
    """Rows for outright contracts of `root` in its active months.

    Drops calendar spreads (symbols like 'GCZ6-GCG7'), other roots, and
    non-active months.
    """
    spec = contracts.get_spec(root)
    first_date = bars[DATE_COL].min()
    meta = {}
    for sym in bars["symbol"].unique():
        if "-" in sym or " " in sym:
            continue
        try:
            r, year, month = parse_symbol(sym, first_date)
        except ValueError:
            continue
        if r == spec.root and month in spec.active_months:
            meta[sym] = (year, month)
    out = bars[bars["symbol"].isin(meta)].copy()
    out["year"] = out["symbol"].map(lambda s: meta[s][0])
    out["month"] = out["symbol"].map(lambda s: meta[s][1])
    return out


def roll_schedule(bars: pd.DataFrame, root: str, rule: RollRule | str = RollRule.FND,
                  days_before_fnd: int = 5, holidays: Iterable | None = None) -> pd.DataFrame:
    """Which contract the continuous series holds on which dates.

    `bars`: long per-expiry frame with `ts_event`, `instrument_id`, `symbol`,
    `close`, `volume` (Databento ohlcv-1d layout).

    Returns one row per contract held: symbol, instrument_id, year, month,
    d0 (first date held), d1 (first date NOT held, i.e. the switch date), and
    fnd. Every d1 is a date on which the next contract also traded the day
    before, so the adjustment ratio is always computable.
    """
    rule = RollRule(rule)
    out = _outrights(bars, root)
    if out.empty:
        raise ValueError(f"No active-month outrights for {root} in the data")
    tz = out[DATE_COL].dt.tz
    dates = pd.DatetimeIndex(sorted(out[DATE_COL].unique()))
    vol = out.pivot_table(index=DATE_COL, columns="symbol", values="volume", aggfunc="sum")
    chain = (out[["symbol", "instrument_id", "year", "month"]].drop_duplicates("symbol")
             .sort_values(["year", "month"]).reset_index(drop=True))
    bday = pd.offsets.CustomBusinessDay(holidays=list(holidays or []))

    rows = []
    d0 = vol[chain.loc[0, "symbol"]].dropna().index.min()
    for i, cur in chain.iterrows():
        fnd = contracts.first_notice_day(cur["year"], cur["month"], holidays)
        last_hold = pd.Timestamp(fnd - days_before_fnd * bday).tz_localize(tz)
        latest_switch = dates[dates > last_hold].min() if (dates > last_hold).any() else None

        switch = latest_switch
        if i + 1 < len(chain):
            nxt = chain.loc[i + 1, "symbol"]
            both = vol[[cur["symbol"], nxt]].dropna()
            if rule is RollRule.VOLUME:
                # decide at close of t-1 with t-1 volume, switch on t
                crossed = both.index[(both[nxt] > both[cur["symbol"]]) & (both.index >= d0)]
                for day in crossed:
                    later = dates[dates > day]
                    if len(later) and (latest_switch is None or later[0] <= latest_switch):
                        switch = later[0]
                        break
            # the next contract must have traded on the day before the switch
            if switch is not None:
                prev_day = dates[dates < switch].max()
                if prev_day not in both.index:
                    raise ValueError(f"{nxt} has no bar on {prev_day.date()}; cannot roll "
                                     f"from {cur['symbol']} there")
        rows.append({"symbol": cur["symbol"], "instrument_id": int(cur["instrument_id"]),
                     "year": int(cur["year"]), "month": int(cur["month"]), "fnd": fnd,
                     "d0": d0, "d1": switch if switch is not None else dates.max() + pd.Timedelta(days=1)})
        if switch is None or i + 1 == len(chain):
            break
        d0 = switch
    sched = pd.DataFrame(rows)
    return sched[sched["d0"] < sched["d1"]].reset_index(drop=True)


def to_roll_spec(schedule: pd.DataFrame) -> list[dict[str, str]]:
    """Schedule -> Databento/course-style roll spec [{'d0', 'd1', 's'}]."""
    return [{"d0": r.d0.strftime("%Y-%m-%d"), "d1": r.d1.strftime("%Y-%m-%d"),
             "s": str(r.instrument_id)} for r in schedule.itertuples()]


def build_continuous(bars: pd.DataFrame, root: str,
                     adjustment: Adjustment | str = Adjustment.RATIO,
                     rule: RollRule | str = RollRule.FND, days_before_fnd: int = 5,
                     holidays: Iterable | None = None,
                     schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stitch per-expiry bars into one continuous daily series for `root`.

    Returns a frame indexed by `ts_event` with columns: close (adjusted as
    requested), raw_close (the held contract's actual price), symbol,
    instrument_id, roll (True on the first day of each new contract).
    Adjustment is forward (the first contract keeps its real prices), which is
    how the course package splices; returns are identical either way.
    """
    adjustment = Adjustment(adjustment)
    if schedule is None:
        schedule = roll_schedule(bars, root, rule, days_before_fnd, holidays)
    spec = to_roll_spec(schedule)
    df = bars[bars["instrument_id"].isin(schedule["instrument_id"])].copy()
    df["instrument_id"] = df["instrument_id"].astype(int)
    raw = splice_unadjusted(spec, df, DATE_COL).set_index(DATE_COL)
    if adjustment is Adjustment.RAW:
        adj = raw["close"]
    elif adjustment is Adjustment.RATIO:
        adj = multiplicative_splice(spec, df, DATE_COL).set_index(DATE_COL)["close"]
    else:
        adj = additive_splice(spec, df, DATE_COL).set_index(DATE_COL)["close"]
    out = pd.DataFrame({"close": adj, "raw_close": raw["close"], "symbol": raw["symbol"],
                        "instrument_id": raw["instrument_id"]})
    out["roll"] = out["symbol"].ne(out["symbol"].shift()) & out["symbol"].shift().notna()
    out.attrs.update(root=root, adjustment=adjustment.value)
    return out


def roll_log(bars: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per roll: execution date (close of the day before the switch),
    from/to contract, both closes, the price gap and the ratio.

    PnL uses this to charge the roll explicitly on the raw series.
    """
    closes = bars.pivot_table(index=DATE_COL, columns="symbol", values="close")
    rows = []
    for a, b in zip(schedule.itertuples(), schedule.iloc[1:].itertuples()):
        exec_day = closes.index[closes.index < b.d0].max()
        fc, tc = closes.loc[exec_day, a.symbol], closes.loc[exec_day, b.symbol]
        rows.append({"exec_date": exec_day, "switch_date": b.d0, "from_symbol": a.symbol,
                     "to_symbol": b.symbol, "from_close": fc, "to_close": tc,
                     "gap": tc - fc, "ratio": tc / fc})
    return pd.DataFrame(rows)


def align_pair(gc: pd.DataFrame, si: pd.DataFrame, column: str = "close") -> pd.DataFrame:
    """Daily GC/SI frame on dates where both traded (inner join)."""
    return pd.concat({"GC": gc[column], "SI": si[column]}, axis=1, join="inner").dropna()
