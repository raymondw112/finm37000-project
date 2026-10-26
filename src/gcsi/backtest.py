"""Walk-forward backtest of the GC/SI spread (#10).

Pipeline for one parameter set (`run`):

  1. hedge ratio alpha_t, beta_t from data up to t-1      cointegration (#7):
     trailing OLS (default) or a one-step-ahead Kalman filter
  2. spread and z-score with mean/std from the window ending t-1   signals (#9):
     the "consistent" z-score (past window re-priced with today's beta) by
     default, or the plain rolling z-score
  3. {-1, 0, +1} signal and contract counts in GC, MGC, SI, SIL    signals (#9)
  4. orders decided at the close of day t are filled `fill_lag` days later at
     that day's raw close (default 1: the ohlcv close is the last trade, not
     the settlement, so we don't assume we traded at the price that made
     the signal)
  5. mark-to-market on raw closes, rolls, costs and attribution    ledger (#25),
     costs (#24)

`walk_forward` adds out-of-sample parameter choice: every test year, pick the
parameter set with the best Sharpe on the preceding `train_years` and use it
for that year only. Because every `run` is causal (day t uses data up to t), a
full-history run per parameter set followed by stitching the chosen positions
year by year is the same as re-running the strategy each year, and the stitched
positions are re-marked through the ledger so switching costs are charged.

Timing (no look-ahead):
  - beta_t and alpha_t are fitted on log closes up to t-1.
  - z_t uses the spread at t with mean and std from the window ending t-1.
  - The target is decided at the close of t; with fill_lag = L it is held from
    the close of t+L, so the PnL of any day uses only decisions made earlier.

Rolls: `gc` and `si` are `rolls.build_continuous` outputs. Rolling is handled
by the ledger: the position is closed in the old contract and reopened in the
new one at the close of the last day held, so the gap between contracts is
never PnL, and both legs of the roll are charged as roll costs.

Warm-up: the hedge ratio needs `window` prior days. The consistent z-score is
defined from then on; the plain z-score needs a further `z_lookback` days of
spread. Positions are flat until both exist, plus `fill_lag` days.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from gcsi import cointegration, signals
from gcsi.costs import CostModel
from gcsi.ledger import LEGS, Ledger, build as build_ledger, check_leg

HEDGES = ("rolling", "kalman")
ZSCORES = ("consistent", "simple")


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters of one backtest run.

    window:        trailing days for the rolling hedge-ratio fit (and the Kalman warm-up).
    z_lookback:    trailing days for the spread mean and std.
    entry, exit_, stop: z-score thresholds (see `signals.generate_signals`).
    gc_notional:   USD notional of the GC leg when in a position.
    mode:          "beta" (SI notional = beta x GC notional) or "dollar".
    micros:        trade MGC / SIL for the remainder after whole contracts.
    hedge:         "rolling" (trailing OLS) or "kalman".
    kalman_delta:  state noise of the Kalman filter (smaller = smoother beta).
    zscore:        "consistent" (past window re-priced with today's beta) or "simple".
    fill_lag:      days between the signal close and the fill close (0 = same close).
    cost_per_contract: if set, a flat USD per contract per side on every leg;
                   None uses the `costs.CostModel` default (ticks + fees + commission).
    initial_equity: starting equity in USD (also the capital returns are measured on).
    """

    window: int = 252
    z_lookback: int = 60
    entry: float = 2.0
    exit_: float = 0.0
    stop: float = 4.0
    gc_notional: float = 1_000_000.0
    mode: str = "beta"
    micros: bool = True
    hedge: str = "rolling"
    kalman_delta: float = 1e-5
    zscore: str = "consistent"
    fill_lag: int = 1
    cost_per_contract: float | None = None
    initial_equity: float = 1_000_000.0

    def __post_init__(self):
        if self.hedge not in HEDGES:
            raise ValueError(f"hedge must be one of {HEDGES}, got {self.hedge!r}")
        if self.zscore not in ZSCORES:
            raise ValueError(f"zscore must be one of {ZSCORES}, got {self.zscore!r}")
        if self.fill_lag < 0:
            raise ValueError("fill_lag must be >= 0")

    @property
    def warm_up(self) -> int:
        """Rows that are always flat at the start of a run."""
        z_extra = self.z_lookback if self.zscore == "simple" else 0
        return self.window + z_extra + self.fill_lag

    def cost_model(self) -> CostModel:
        return CostModel() if self.cost_per_contract is None else CostModel.flat(self.cost_per_contract)


def _config(config: BacktestConfig | dict | None) -> BacktestConfig:
    if config is None:
        return BacktestConfig()
    if isinstance(config, dict):
        return BacktestConfig(**config)
    return config


def _restrict(frame: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Rows on `idx`, keeping `attrs` (the adjustment type lives there)."""
    out = frame.reindex(idx)
    out.attrs = dict(frame.attrs)
    return out


def align(gc: pd.DataFrame, si: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both legs on the dates where both traded."""
    check_leg(gc, "gc")
    check_leg(si, "si")
    idx = gc.index.intersection(si.index).sort_values()
    return _restrict(gc, idx), _restrict(si, idx)


def hedge_ratio(log_gc: pd.Series, log_si: pd.Series, config: BacktestConfig) -> pd.DataFrame:
    """alpha, beta on each day from data up to the day before."""
    if config.hedge == "kalman":
        return cointegration.kalman_hedge_ratio(log_gc, log_si, delta=config.kalman_delta,
                                                warmup=config.window)
    return cointegration.rolling_hedge_ratio(log_gc, log_si, config.window)


def signal_frame(gc: pd.DataFrame, si: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """alpha, beta, spread, z, signal on the aligned dates (no positions yet)."""
    log_gc, log_si = np.log(gc["close"]), np.log(si["close"])
    hr = hedge_ratio(log_gc, log_si, config).reindex(gc.index)
    if config.zscore == "consistent":
        zz = signals.consistent_zscore(log_gc, log_si, hr["alpha"], hr["beta"], config.z_lookback)
        spread, z = zz["spread"], zz["z"]
    else:
        spread = signals.spread(log_gc, log_si, hr["alpha"], hr["beta"])
        z = signals.zscore(spread, config.z_lookback)
    sig = signals.generate_signals(z, config.entry, config.exit_, config.stop)
    return pd.DataFrame({"alpha": hr["alpha"], "beta": hr["beta"], "spread": spread,
                         "z": z, "signal": sig})


def target_positions(diag: pd.DataFrame, gc: pd.DataFrame, si: pd.DataFrame,
                     config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Contract counts decided at each close, and the exact ounces they aim for."""
    sized = signals.size_positions(diag["signal"], diag["beta"], gc["raw_close"], si["raw_close"],
                                   config.gc_notional, config.mode, config.micros)
    oz = []
    for s, b, g, x in zip(diag["signal"].to_numpy(), diag["beta"].to_numpy(float),
                          gc["raw_close"].to_numpy(float), si["raw_close"].to_numpy(float)):
        oz.append(signals.target_ounces(int(s), b, g, x, config.gc_notional, config.mode))
    oz = pd.DataFrame(oz, index=diag.index, columns=["gc_oz", "si_oz"])
    return sized[list(LEGS)], oz


def _lag(frame: pd.DataFrame, lag: int, fill) -> pd.DataFrame:
    return frame.shift(lag).fillna(fill) if lag else frame


def run(gc: pd.DataFrame, si: pd.DataFrame, config: BacktestConfig | dict | None = None,
        roll_logs: dict[str, pd.DataFrame] | None = None,
        cost_model: CostModel | None = None, return_ledger: bool = False):
    """Backtest of the GC/SI spread for one parameter set.

    gc, si: `rolls.build_continuous` outputs for each leg. `close` (adjusted)
        drives the signal, `raw_close` (actual prices) drives fills and PnL.
        Joined on dates where both legs traded.
    config: `BacktestConfig` or a dict of its fields; defaults if None.
    roll_logs: optional {"GC": rolls.roll_log(...), "SI": ...} for exact roll prices.
    cost_model: overrides the config's costs if given.

    Returns a daily frame: positions held per leg (GC, MGC, SI, SIL), alpha,
    beta, spread, z, signal (decided at that close; +1 = long GC / short SI),
    pnl per leg, gross_pnl, ideal_pnl, sizing_residual, trade_costs,
    roll_costs, costs, net_pnl, equity (USD). With return_ledger=True,
    returns (frame, ledger) so the fills are available too.
    """
    config = _config(config)
    gc, si = align(gc, si)
    diag = signal_frame(gc, si, config)
    target, oz = target_positions(diag, gc, si, config)
    held = _lag(target, config.fill_lag, 0).astype(int)
    held_oz = _lag(oz, config.fill_lag, 0.0)
    led = build_ledger(held, gc, si, cost_model or config.cost_model(), config.initial_equity,
                       roll_logs, target_oz=held_oz)
    d = led.daily
    out = pd.concat([d[list(LEGS)], diag, d.drop(columns=list(LEGS))], axis=1)
    out.attrs["config"] = asdict(config)
    return (out, led) if return_ledger else out


def simulate(positions: pd.DataFrame, gc: pd.DataFrame, si: pd.DataFrame,
             cost_per_contract: float = 0.0, initial_equity: float = 0.0,
             roll_logs: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Mark given end-of-day positions to market with a flat cost per contract per side.

    Kept for compatibility with the skeleton; `ledger.build` does the work.
    Returns positions, gross_pnl, costs, net_pnl, equity.
    """
    led = build_ledger(positions, gc, si, CostModel.flat(cost_per_contract), initial_equity,
                       roll_logs)
    d = led.daily
    return d[list(LEGS) + ["gross_pnl", "costs", "net_pnl", "equity"]]


# --------------------------------------------------------------------------- #
# Walk-forward parameter selection
# --------------------------------------------------------------------------- #
DEFAULT_GRID = {"entry": [1.5, 2.0, 2.5], "z_lookback": [40, 60, 120]}


def sharpe(pnl: pd.Series, capital: float) -> float:
    """Annualized Sharpe of daily USD PnL on fixed capital; NaN if flat."""
    r = pnl / capital
    sd = r.std()
    return float(np.sqrt(252) * r.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")


def walk_forward(gc: pd.DataFrame, si: pd.DataFrame, base: BacktestConfig | dict | None = None,
                 grid: dict | None = None, train_years: int = 3,
                 roll_logs: dict[str, pd.DataFrame] | None = None,
                 cost_model: CostModel | None = None) -> dict:
    """Out-of-sample backtest with parameters re-chosen every calendar year.

    For each test year Y with `train_years` full years of live (post warm-up)
    history before it, every parameter set in `grid` (keys are BacktestConfig
    fields) is scored by its net Sharpe on years Y-train_years .. Y-1, and the
    best one trades year Y. Positions are stitched and re-marked through the
    ledger, so a change of parameters at a year boundary pays its trading cost.

    Returns {"daily": stitched daily frame (test years only, plus the signal
    diagnostics of the chosen set), "choices": one row per test year with the
    chosen parameters and their train Sharpe, "runs": {params: full run}}.
    """
    base = _config(base)
    grid = grid or DEFAULT_GRID
    cost_model = cost_model or base.cost_model()
    gc, si = align(gc, si)
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    runs = {}
    for c in combos:
        cfg = replace(base, **c)
        runs[tuple(c.items())] = run(gc, si, cfg, roll_logs, cost_model)

    warm = max(replace(base, **c).warm_up for c in combos)
    live_start = gc.index[min(warm, len(gc.index) - 1)]
    years = sorted(set(gc.index.year))
    first_test = live_start.year + train_years + (0 if live_start.dayofyear < 10 else 1)
    choices, pieces = [], []
    for y in years:
        if y < first_test:
            continue
        train = (gc.index.year >= y - train_years) & (gc.index.year < y)
        test = gc.index.year == y
        scored = []
        for key, out in runs.items():
            s = sharpe(out.loc[train, "net_pnl"], base.initial_equity)
            scored.append((s if np.isfinite(s) else -np.inf, key))
        best_s, best = max(scored, key=lambda x: x[0])
        choices.append({"year": y, **dict(best), "train_sharpe": best_s})
        pieces.append(runs[best].loc[test])
    if not pieces:
        raise ValueError("Not enough history for one test year; shorten train_years or the windows")
    stitched = pd.concat(pieces)
    held = stitched[list(LEGS)]
    # re-mark the stitched positions so parameter switches pay their costs; the
    # first test day opens whatever the chosen set holds (no PnL before it)
    led = build_ledger(held, _restrict(gc, stitched.index), _restrict(si, stitched.index),
                       cost_model, base.initial_equity, roll_logs)
    daily = led.daily.copy()
    daily["ideal_pnl"] = stitched["ideal_pnl"]
    daily["sizing_residual"] = daily["gross_pnl"] - daily["ideal_pnl"]
    daily = pd.concat([daily, stitched[["alpha", "beta", "spread", "z", "signal"]]], axis=1)
    return {"daily": daily, "ledger": led, "choices": pd.DataFrame(choices).set_index("year"),
            "runs": runs}
