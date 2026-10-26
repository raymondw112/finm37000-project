"""End-to-end run: data -> continuous series -> analysis -> backtests -> report (#6, #10, #11).

    python -m gcsi.report --source databento --start 2010-06-06 [--end YYYY-MM-DD] [--yes]
    python -m gcsi.report --source synthetic          # no key needed, known answer

Writes to output/<source>/:
    report.md        the results in words and tables (start here)
    results.json     every number in report.md
    eda_yearly.csv   Engle-Granger beta and p-value per year
    wf_choices.csv   parameters chosen each walk-forward year
    by_year.csv      out-of-sample PnL by calendar year
    stress.csv       the four stress windows (#19)
    sensitivity.csv  costs x0/x2, same-close fills, Kalman beta, dollar-neutral, no micros
    daily.csv        out-of-sample daily positions, PnL and attribution
    fills.csv        every simulated fill (trades and rolls)
    fig_*.png        prices and ratio, spread and z, hedge ratio, equity, drawdown

Two backtests are reported:
  - Baseline: the pre-committed parameters (window 252, z lookback 60, entry
    2, exit 0, stop 4, chosen before any data was seen), run over the whole
    history after warm-up. No fitting, so the full period counts.
  - Walk-forward (headline): every year the best of a small grid (entry x z
    lookback) on the previous 3 years trades the next year. Out of sample only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from gcsi import backtest, cointegration, metrics, ou, rolls
from gcsi.backtest import BacktestConfig
from gcsi.costs import CostModel

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT_DIR / "output"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class _CacheOnly:
    """Stand-in Databento client for --cache-only: anything not cached is an error, never a pull."""

    metadata = None

    @property
    def timeseries(self):
        raise RuntimeError("Data not in the cache for this window. Run `python -m gcsi.data` "
                           "first, or pass an --end on or before the cached end date.")


def load_databento(start: str, end: str | None, yes: bool, cache_dir: Path | None = None,
                   cache_only: bool = False) -> dict:
    from gcsi import data
    end = end or pd.Timestamp.now(tz="UTC").normalize().strftime("%Y-%m-%d")
    kw = {"yes": yes}
    if cache_dir is not None:
        kw["cache_dir"] = cache_dir
    if cache_only:
        kw.update(client=_CacheOnly(), max_cost=None)
    return {root: data.load_bars(root, start, end, **kw) for root in ("GC", "SI")}


def load_synthetic(years: int = 16, seed: int = 7, start: str = "2010-06-07") -> dict:
    from gcsi import synthetic
    pair = synthetic.cointegrated_prices(n=252 * years, seed=seed, start=start)
    bars = {"GC": synthetic.per_expiry_chain(pair.prices["GC"], "GC"),
            "SI": synthetic.per_expiry_chain(pair.prices["SI"], "SI")}
    bars["truth"] = {"beta": pair.params.beta,
                     "half_life_days": float(np.log(2) / pair.params.spread.kappa)}
    return bars


def drop_weekends(bars: pd.DataFrame) -> pd.DataFrame:
    """Remove Saturday/Sunday bars.

    Globex reopens on Sunday evening, so a daily bar keyed by UTC date can show
    a short Sunday session. It is a few hours of thin trading, not a trading day.
    """
    return bars[bars["ts_event"].dt.dayofweek < 5]


def continuous(bars: dict, days_before_fnd: int = 5):
    legs, logs, n_rolls = {}, {}, {}
    for root in ("GC", "SI"):
        b = drop_weekends(bars[root])
        sched = rolls.roll_schedule(b, root, rule="fnd", days_before_fnd=days_before_fnd)
        legs[root] = rolls.build_continuous(b, root, adjustment="ratio", schedule=sched)
        logs[root] = rolls.roll_log(b, sched)
        n_rolls[root] = int(len(logs[root]))
    return legs["GC"], legs["SI"], logs, n_rolls


# --------------------------------------------------------------------------- #
# Exploratory analysis (#6)
# --------------------------------------------------------------------------- #
def eda(gc: pd.DataFrame, si: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    pair = rolls.align_pair(gc, si)
    lg, ls = np.log(pair["GC"]), np.log(pair["SI"])
    eg = cointegration.engle_granger(lg, ls)
    jo = cointegration.johansen(lg, ls)
    fit = ou.fit_ou(eg["resid"])
    sens = ou.half_life_sensitivity(eg["resid"])
    yearly = cointegration.subperiod_engle_granger(lg, ls)
    rets = np.log(pair).diff().dropna()
    raw = rolls.align_pair(gc, si, column="raw_close")
    ratio = raw["GC"] / raw["SI"]
    out = {
        "start": str(pair.index.min().date()), "end": str(pair.index.max().date()),
        "days": int(len(pair)),
        "gc_is_I1": cointegration.adf_is_i1(lg), "si_is_I1": cointegration.adf_is_i1(ls),
        "eg_beta": eg["beta"], "eg_alpha": eg["alpha"], "eg_pvalue": eg["pvalue"],
        "eg_tstat": eg["tstat"], "eg_crit_5pct": eg["crit"]["5%"],
        "johansen_rank_trace": jo["rank_trace"], "johansen_beta": jo["beta"],
        "ou_half_life_days": fit["half_life"], "ou_kappa": fit["kappa"], "ou_sigma_eq": fit["sigma_eq"],
        "half_life_by_window": {int(k): float(v) for k, v in sens["half_life"].items()},
        "years_cointegrated_5pct": int(yearly["cointegrated_5pct"].sum()) if len(yearly) else 0,
        "years_tested": int(len(yearly)),
        "yearly_beta_min": float(yearly["beta"].min()) if len(yearly) else float("nan"),
        "yearly_beta_max": float(yearly["beta"].max()) if len(yearly) else float("nan"),
        "return_corr": float(rets["GC"].corr(rets["SI"])),
        "ratio_last": float(ratio.iloc[-1]), "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()), "ratio_mean": float(ratio.mean()),
    }
    return out, yearly


# --------------------------------------------------------------------------- #
# Backtests
# --------------------------------------------------------------------------- #
SENSITIVITIES = {
    "baseline": {},
    "no costs": {"costs": 0.0},
    "costs x2": {"costs": 2.0},
    "fill at signal close": {"fill_lag": 0},
    "Kalman hedge ratio": {"hedge": "kalman"},
    "simple z-score": {"zscore": "simple"},
    "dollar-neutral": {"mode": "dollar"},
    "no micros": {"micros": False},
}


def run_backtests(gc, si, roll_logs, base: BacktestConfig, cost_model: CostModel,
                  grid: dict | None = None, train_years: int = 3) -> dict:
    cap = base.initial_equity
    out = {}
    # baseline, full history after warm-up
    base_df, base_led = backtest.run(gc, si, base, roll_logs, cost_model, return_ledger=True)
    live = base_df.iloc[base.warm_up:]
    out["baseline"] = {"df": base_df, "ledger": base_led, "live": live,
                       "summary": metrics.summarize(live["net_pnl"], cap),
                       "gross": metrics.summarize(live["gross_pnl"], cap),
                       "trades": metrics.trade_stats(live["signal"], live["net_pnl"], base.fill_lag),
                       "attribution": base_led.attribution(),
                       "stress": metrics.stress_periods(live["net_pnl"], capital=cap,
                                                        beta=base_df["beta"])}
    # sensitivities around the baseline
    rows = {}
    for name, change in SENSITIVITIES.items():
        change = dict(change)
        cm = cost_model.scaled(change.pop("costs")) if "costs" in change else cost_model
        cfg = replace(base, **change)
        df = base_df if name == "baseline" else backtest.run(gc, si, cfg, roll_logs, cm)
        lv = df.iloc[cfg.warm_up:]
        s = metrics.summarize(lv["net_pnl"], cap)
        t = metrics.trade_stats(lv["signal"], lv["net_pnl"], cfg.fill_lag)
        rows[name] = {"sharpe": s["sharpe"], "ann_return": s["ann_return"],
                      "max_drawdown": s["max_drawdown"], "trades_per_year": t["trades_per_year"],
                      "costs_usd": float(lv["costs"].sum()), "net_pnl_usd": s["total_pnl"]}
    out["sensitivity"] = pd.DataFrame.from_dict(rows, orient="index")
    # walk-forward, out of sample
    wf = backtest.walk_forward(gc, si, base, grid, train_years, roll_logs, cost_model)
    d = wf["daily"]
    out["walk_forward"] = {"daily": d, "ledger": wf["ledger"], "choices": wf["choices"],
                           "summary": metrics.summarize(d["net_pnl"], cap),
                           "gross": metrics.summarize(d["gross_pnl"], cap),
                           "trades": metrics.trade_stats(d["signal"], d["net_pnl"], base.fill_lag),
                           "attribution": wf["ledger"].attribution() | {
                               "ideal_pnl": float(d["ideal_pnl"].sum()),
                               "sizing_residual": float(d["sizing_residual"].sum())},
                           "by_year": metrics.by_year(d["net_pnl"], cap),
                           "stress": metrics.stress_periods(d["net_pnl"], capital=cap, beta=d["beta"])}
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def figures(gc, si, res: dict, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    navy, gold, slate, grey = "#14213D", "#B8862B", "#6B7B8C", "#9AA3AD"
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
    raw = rolls.align_pair(gc, si, column="raw_close")
    base = res["baseline"]["df"]
    wf = res["walk_forward"]["daily"]
    cap = base.attrs["config"]["initial_equity"]

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(raw.index, raw["GC"], color=gold, lw=1, label="Gold GC ($/oz)")
    ax0b = ax[0].twinx()
    ax0b.plot(raw.index, raw["SI"], color=slate, lw=1, label="Silver SI ($/oz, right)")
    ax[0].set_title("Front-contract prices (raw)")
    ax[0].legend(loc="upper left"); ax0b.legend(loc="lower right")
    ax[1].plot(raw.index, raw["GC"] / raw["SI"], color=navy, lw=1)
    ax[1].set_title("Gold / silver price ratio")
    fig.tight_layout(); fig.savefig(out_dir / "fig_prices_ratio.png"); plt.close(fig)
    files.append("fig_prices_ratio.png")

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(base.index, base["spread"], color=navy, lw=0.8)
    ax[0].set_title("Spread  log GC - alpha - beta log SI  (rolling 1y hedge ratio)")
    ax[1].plot(base.index, base["z"], color=grey, lw=0.6)
    for lvl, ls_ in ((2, "--"), (-2, "--"), (4, ":"), (-4, ":")):
        ax[1].axhline(lvl, color=gold, lw=0.8, ls=ls_)
    longs, shorts = base["signal"] == 1, base["signal"] == -1
    ax[1].scatter(base.index[longs], base["z"][longs], s=2, color="#2E6FBA", label="long spread")
    ax[1].scatter(base.index[shorts], base["z"][shorts], s=2, color="#C8642B", label="short spread")
    ax[1].set_ylim(-6, 6); ax[1].legend(loc="upper left", markerscale=4)
    ax[1].set_title("z-score with entry (2) and stop (4) levels")
    fig.tight_layout(); fig.savefig(out_dir / "fig_spread_z.png"); plt.close(fig)
    files.append("fig_spread_z.png")

    lg, ls = np.log(gc["close"]), np.log(si["close"])
    lg, ls = lg.align(ls, join="inner")
    kal = cointegration.kalman_hedge_ratio(lg, ls, warmup=252)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(base.index, base["beta"], color=navy, lw=1, label="rolling 1y OLS")
    ax.plot(kal.index, kal["beta"], color=gold, lw=1, label="Kalman")
    ax.set_title("Hedge ratio beta (log GC on log SI), estimated with data up to the day before")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "fig_beta.png"); plt.close(fig)
    files.append("fig_beta.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    live = res["baseline"]["live"]
    ax.plot(live.index, live["gross_pnl"].cumsum() / 1e3, color=grey, lw=1, label="baseline, before costs")
    ax.plot(live.index, live["net_pnl"].cumsum() / 1e3, color=navy, lw=1.2, label="baseline, after costs")
    anchor = live["net_pnl"].cumsum().reindex(wf.index[:1]).fillna(0).iloc[0]
    ax.plot(wf.index, (anchor + wf["net_pnl"].cumsum()) / 1e3, color=gold, lw=1.4,
            label="walk-forward (out of sample), after costs, from the baseline's level")
    for name, (a, b) in metrics.STRESS_WINDOWS.items():
        ax.axvspan(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"), color=gold, alpha=0.10)
    ax.set_ylabel("cumulative PnL ($k)")
    ax.set_title(f"Cumulative PnL on ${cap/1e6:.0f}m gold notional (shaded: stress windows)")
    ax.legend(loc="lower left", fontsize=8); fig.tight_layout(); fig.savefig(out_dir / "fig_equity.png"); plt.close(fig)
    files.append("fig_equity.png")

    fig, ax = plt.subplots(figsize=(10, 2.8))
    for series, color, label in ((live["net_pnl"], navy, "baseline"), (wf["net_pnl"], gold, "walk-forward")):
        eq = cap + series.cumsum()
        dd = eq / np.maximum.accumulate(np.r_[cap, eq.to_numpy()])[1:] - 1
        ax.fill_between(series.index, dd * 100, 0, color=color, alpha=0.35, label=label)
    ax.set_ylabel("drawdown (%)"); ax.legend(loc="lower left")
    ax.set_title("Drawdown after costs")
    fig.tight_layout(); fig.savefig(out_dir / "fig_drawdown.png"); plt.close(fig)
    files.append("fig_drawdown.png")
    return files


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


def _num(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:,.{d}f}"


def _usd(x):
    return "n/a" if x is None or not np.isfinite(x) else f"${x:,.0f}"


def write_report(source: str, e: dict, yearly: pd.DataFrame, res: dict, n_rolls: dict,
                 base: BacktestConfig, cost_model: CostModel, out_dir: Path, figs: list[str],
                 truth: dict | None = None) -> dict:
    wf, bl = res["walk_forward"], res["baseline"]
    s, g, t, a = wf["summary"], wf["gross"], wf["trades"], wf["attribution"]
    bs, bt = bl["summary"], bl["trades"]
    results = {"source": source, "eda": e, "rolls": n_rolls, "config": asdict(base),
               "cost_per_contract_usd": cost_model.table(),
               "walk_forward": {"summary": s, "gross": g, "trades": t, "attribution": a,
                                "first_test_year": int(wf["choices"].index.min()),
                                "choices": wf["choices"].reset_index().to_dict("records")},
               "baseline": {"summary": bs, "gross": bl["gross"], "trades": bt,
                            "attribution": bl["attribution"]},
               "truth": truth}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    yearly.to_csv(out_dir / "eda_yearly.csv")
    wf["choices"].to_csv(out_dir / "wf_choices.csv")
    wf["by_year"].to_csv(out_dir / "by_year.csv")
    stress = bl["stress"].add_prefix("baseline_").join(wf["stress"].add_prefix("wf_"))
    stress.to_csv(out_dir / "stress.csv")
    res["sensitivity"].to_csv(out_dir / "sensitivity.csv")
    wf["daily"].to_csv(out_dir / "daily.csv")
    bl["ledger"].fills.to_csv(out_dir / "fills.csv", index=False)

    first = int(wf["choices"].index.min())
    lines = [
        f"# GC/SI pairs backtest: results ({source} data)", "",
        f"Data {e['start']} to {e['end']}, {e['days']:,} trading days with both legs. "
        f"Rolls: {n_rolls['GC']} gold, {n_rolls['SI']} silver (5 business days before first notice).", "",
    ]
    if truth:
        lines += [f"Synthetic data with a known answer: true beta {truth['beta']}, true half-life "
                  f"{truth['half_life_days']:.1f} days. A test of the pipeline, not of the market.", ""]
    lines += [
        "## Headline: walk-forward, out of sample, after costs", "",
        f"Test years {first} to {e['end'][:4]}; each year uses the best (entry, z lookback) on the "
        "previous 3 years.", "",
        "| Metric | After costs | Before costs |", "| --- | --- | --- |",
        f"| Sharpe | {_num(s['sharpe'])} | {_num(g['sharpe'])} |",
        f"| Annual return on $1m | {_pct(s['ann_return'])} | {_pct(g['ann_return'])} |",
        f"| Annual volatility | {_pct(s['ann_vol'])} | {_pct(g['ann_vol'])} |",
        f"| Max drawdown | {_pct(s['max_drawdown'])} | {_pct(g['max_drawdown'])} |",
        f"| Total PnL | {_usd(s['total_pnl'])} | {_usd(g['total_pnl'])} |", "",
        f"Trades: {t['n_trades']} round trips ({_num(t['trades_per_year'], 1)} a year), average "
        f"holding {_num(t['avg_holding_days'], 1)} days, win rate {_pct(t['win_rate'])}.", "",
        "PnL attribution (USD): "
        f"ideal spread {_usd(a['ideal_pnl'])}, whole-contract sizing {_usd(a['sizing_residual'])}, "
        f"trade costs -{_usd(a['trade_costs'])}, roll costs -{_usd(a['roll_costs'])}, "
        f"net {_usd(a['net_pnl'])}.", "",
        "## Baseline: pre-committed parameters, full history after warm-up", "",
        f"Sharpe {_num(bs['sharpe'])}, annual return {_pct(bs['ann_return'])}, max drawdown "
        f"{_pct(bs['max_drawdown'])}, {bt['n_trades']} round trips, average holding "
        f"{_num(bt['avg_holding_days'], 1)} days. Costs: trades {_usd(bl['attribution']['trade_costs'])}, "
        f"rolls {_usd(bl['attribution']['roll_costs'])}, against gross PnL {_usd(bl['attribution']['gross_pnl'])}.", "",
        "## Is the pair cointegrated?", "",
        f"Full sample Engle-Granger: beta {_num(e['eg_beta'], 3)}, p-value {_num(e['eg_pvalue'], 4)} "
        f"(t {_num(e['eg_tstat'])} vs 5% critical {_num(e['eg_crit_5pct'])}). Johansen rank "
        f"{e['johansen_rank_trace']}, beta {_num(e['johansen_beta'], 3)}. Cointegrated at 5% in "
        f"{e['years_cointegrated_5pct']} of {e['years_tested']} single years; yearly beta from "
        f"{_num(e['yearly_beta_min'], 2)} to {_num(e['yearly_beta_max'], 2)}. Daily return correlation "
        f"{_num(e['return_corr'], 2)}.", "",
        f"OU half-life of the full-sample spread: {_num(e['ou_half_life_days'], 1)} trading days. "
        "By window (last N days): " + ", ".join(f"{k}d {_num(v, 1)}" for k, v in e["half_life_by_window"].items()) + ".", "",
        f"Gold/silver ratio: last {_num(e['ratio_last'], 1)}, range {_num(e['ratio_min'], 1)} to "
        f"{_num(e['ratio_max'], 1)}.", "",
        "## Stress windows (baseline)", "",
        "| Window | PnL | Sharpe | Max DD | Beta range |", "| --- | --- | --- | --- | --- |",
    ]
    for name, r in bl["stress"].iterrows():
        if r.get("days", 0):
            lines.append(f"| {name} | {_usd(r['total_pnl'])} | {_num(r['sharpe'])} | "
                         f"{_pct(r['max_drawdown'])} | {_num(r.get('beta_min'), 2)} to {_num(r.get('beta_max'), 2)} |")
        else:
            lines.append(f"| {name} | outside data | | | |")
    lines += ["", "## Sensitivity (baseline parameters)", "",
              "| Variant | Sharpe | Annual return | Max DD | Trades/yr | Costs |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name, r in res["sensitivity"].iterrows():
        lines.append(f"| {name} | {_num(r['sharpe'])} | {_pct(r['ann_return'])} | {_pct(r['max_drawdown'])} | "
                     f"{_num(r['trades_per_year'], 1)} | {_usd(r['costs_usd'])} |")
    lines += ["", "## Out-of-sample PnL by year", "", "| Year | PnL | Return | Sharpe | Chosen " + " / ".join(c for c in wf["choices"].columns if c != "train_sharpe") + " |",
              "| --- | --- | --- | --- | --- |"]
    ch = wf["choices"]
    for y, r in wf["by_year"].iterrows():
        c = ch.loc[y] if y in ch.index else None
        pick = " / ".join(f"{c[k]:g}" for k in ch.columns if k != "train_sharpe") if c is not None else ""
        lines.append(f"| {y} | {_usd(r['pnl'])} | {_pct(r['return'])} | {_num(r['sharpe'])} | {pick} |")
    lines += ["", "## Assumptions", "",
              f"Gold notional ${base.gc_notional:,.0f} per trade, beta-neutral with micros, fills "
              f"{base.fill_lag} day after the signal at the raw close. Cost per contract per side (USD): "
              + ", ".join(f"{k} {v:.2f}" for k, v in cost_model.table().items())
              + " (half a tick + exchange fee + commission; see gcsi/costs.py).", "",
              "## Figures", ""] + [f"![{f[4:-4]}]({f})" for f in figs]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    return results


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Run the full GC/SI pipeline and write a report.")
    p.add_argument("--source", choices=["databento", "synthetic"], default="databento")
    p.add_argument("--start", default="2010-06-06")
    p.add_argument("--end", default=None)
    p.add_argument("--yes", action="store_true", help="allow Databento pulls above the cost cap")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--cache-only", action="store_true",
                   help="read only the local cache; never call Databento (no key needed)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--train-years", type=int, default=3)
    args = p.parse_args(argv)

    if args.source == "synthetic":
        bars = load_synthetic()
        truth = bars.pop("truth")
    else:
        bars = load_databento(args.start, args.end, args.yes, args.cache_dir, args.cache_only)
        truth = None
    out_dir = args.out / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Building continuous series ...")
    gc, si, roll_logs, n_rolls = continuous(bars)
    print("Analysis (cointegration, OU) ...")
    e, yearly = eda(gc, si)
    base, cost_model = BacktestConfig(), CostModel()
    print("Backtests (baseline, sensitivities, walk-forward) ...")
    res = run_backtests(gc, si, roll_logs, base, cost_model, train_years=args.train_years)
    print("Figures ...")
    figs = figures(gc, si, res, out_dir)
    results = write_report(args.source, e, yearly, res, n_rolls, base, cost_model, out_dir, figs, truth)
    s = results["walk_forward"]["summary"]
    print(f"Walk-forward out of sample: Sharpe {s['sharpe']:.2f}, annual return "
          f"{100 * s['ann_return']:.1f}%, max drawdown {100 * s['max_drawdown']:.1f}%")
    print(f"Report: {out_dir / 'report.md'}")
    return results


if __name__ == "__main__":
    main()
