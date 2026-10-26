"""Performance evaluation (#11).

Everything is measured on daily USD PnL against a fixed capital (the backtest's
initial equity), because the strategy runs a constant gold notional rather than
compounding:

    daily return r_t = net_pnl_t / capital

`summarize` gives the headline numbers, `trade_stats` counts round trips from
the signal, `stress_periods` repeats `summarize` on the four windows fixed in
#19 (Aug-Dec 2011, Apr 2013, Mar 2020, 2022), and `by_year` gives the calendar
table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

STRESS_WINDOWS = {
    "Aug-Dec 2011": ("2011-08-01", "2011-12-31"),
    "Apr 2013": ("2013-04-01", "2013-04-30"),
    "Mar 2020": ("2020-03-01", "2020-03-31"),
    "2022": ("2022-01-01", "2022-12-31"),
}


def _returns(pnl: pd.Series, capital: float) -> pd.Series:
    if capital <= 0:
        raise ValueError("capital must be positive")
    return pnl.astype(float) / capital


def max_drawdown(pnl: pd.Series, capital: float) -> float:
    """Largest peak-to-trough fall of capital + cumulative PnL, as a fraction (negative)."""
    eq = capital + pnl.cumsum()
    peak = np.maximum.accumulate(np.r_[capital, eq.to_numpy()])[1:]
    return float((eq.to_numpy() / peak - 1).min()) if len(eq) else 0.0


def summarize(pnl: pd.Series, capital: float = 1_000_000.0) -> dict:
    """Headline metrics of daily net PnL (USD) on fixed `capital`.

    Returns days, total_pnl, ann_return, ann_vol, sharpe, sortino,
    max_drawdown, calmar, hit_rate (share of active days with positive PnL)
    and active_share (share of days with nonzero PnL).
    """
    r = _returns(pnl, capital)
    n = len(r)
    if n == 0:
        return {"days": 0}
    mean, sd = r.mean(), r.std()
    downside = r[r < 0].std()
    active = r[r != 0]
    ann_ret = mean * TRADING_DAYS
    mdd = max_drawdown(pnl, capital)
    return {
        "days": int(n),
        "total_pnl": float(pnl.sum()),
        "ann_return": float(ann_ret),
        "ann_vol": float(sd * np.sqrt(TRADING_DAYS)) if n > 1 else float("nan"),
        "sharpe": float(np.sqrt(TRADING_DAYS) * mean / sd) if sd > 0 else float("nan"),
        "sortino": float(np.sqrt(TRADING_DAYS) * mean / downside) if downside > 0 else float("nan"),
        "max_drawdown": mdd,
        "calmar": float(ann_ret / -mdd) if mdd < 0 else float("nan"),
        "hit_rate": float((active > 0).mean()) if len(active) else float("nan"),
        "active_share": float(len(active) / n),
    }


def trade_stats(signal: pd.Series, pnl: pd.Series, fill_lag: int = 1) -> dict:
    """Round trips from the signal: count, per year, average holding days, win rate.

    A trade runs from the first day a nonzero signal is held to the day it
    goes flat or flips; its PnL is the net PnL from the day after entry
    through the day after the last day held (the entry day's cost is left
    out so a flip isn't counted twice). Signals are held `fill_lag` days
    after they are decided.
    """
    held = signal.shift(fill_lag).fillna(0).astype(int)
    trade_id = (held != held.shift()).cumsum()
    trades = []
    for _, block in held.groupby(trade_id):
        side = int(block.iloc[0])
        if side == 0:
            continue
        # PnL of a holding shows up the day after it is held, through the exit day
        days = block.index
        nxt = pnl.index.get_indexer([days[-1]])[0] + 1
        end = pnl.index[min(nxt, len(pnl.index) - 1)]
        trades.append({"side": side, "start": days[0], "days": len(days),
                       "pnl": float(pnl.loc[days[0]:end].iloc[1:].sum())})
    years = max(len(signal) / TRADING_DAYS, 1e-9)
    if not trades:
        return {"n_trades": 0, "trades_per_year": 0.0, "avg_holding_days": float("nan"),
                "win_rate": float("nan"), "avg_trade_pnl": float("nan")}
    t = pd.DataFrame(trades)
    return {"n_trades": int(len(t)), "trades_per_year": float(len(t) / years),
            "avg_holding_days": float(t["days"].mean()), "win_rate": float((t["pnl"] > 0).mean()),
            "avg_trade_pnl": float(t["pnl"].mean()), "long_trades": int((t["side"] == 1).sum()),
            "short_trades": int((t["side"] == -1).sum())}


def stress_periods(pnl: pd.Series, periods: dict | None = None, capital: float = 1_000_000.0,
                   beta: pd.Series | None = None) -> pd.DataFrame:
    """`summarize` per window ({name: (start, end)}), plus the beta range if given."""
    periods = periods or STRESS_WINDOWS
    rows = {}
    tz = pnl.index.tz
    for name, (a, b) in periods.items():
        a, b = pd.Timestamp(a, tz=tz), pd.Timestamp(b, tz=tz) + pd.Timedelta(days=1)
        part = pnl[(pnl.index >= a) & (pnl.index < b)]
        if part.empty:
            rows[name] = {"days": 0}
            continue
        row = summarize(part, capital)
        if beta is not None:
            bb = beta[(beta.index >= a) & (beta.index < b)].dropna()
            row["beta_min"], row["beta_max"] = (float(bb.min()), float(bb.max())) if len(bb) else (np.nan, np.nan)
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def by_year(pnl: pd.Series, capital: float = 1_000_000.0) -> pd.DataFrame:
    """Calendar-year PnL, return, Sharpe and max drawdown."""
    rows = {}
    for y, part in pnl.groupby(pnl.index.year):
        s = summarize(part, capital)
        rows[int(y)] = {"pnl": s["total_pnl"], "return": s["total_pnl"] / capital,
                        "sharpe": s["sharpe"], "max_drawdown": s["max_drawdown"]}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("year")
