# Gold vs. silver pairs trading on CME futures: a cost-aware backtest

[![tests](https://github.com/ryotahidaka/finm37000-project/actions/workflows/tests.yml/badge.svg)](https://github.com/ryotahidaka/finm37000-project/actions/workflows/tests.yml)

**Bottom line: a daily mean-reversion trade on the gold/silver spread loses money out of sample, and the cause is the pair, not trading costs.**

We built a reproducible pipeline that downloads every COMEX gold (GC) and silver (SI) futures contract from Databento (`GLBX.MDP3`, June 2010 to October 2026), rolls contracts before first notice, tests for cointegration, and backtests a z-score spread strategy with realistic costs, micro contracts and walk-forward parameter selection. One command reproduces every number below. On synthetic data where the spread truly mean-reverts, the same code makes money, so the negative result reflects the market rather than a bug.

## Headline results

Out of sample, after costs, $1m gold notional, 4,248 trading days with both legs, 81 gold and 82 silver rolls:

| | Walk-forward, 2015 to 2026 | Baseline, 2011 to 2026 (fixed parameters) |
| --- | --- | --- |
| Sharpe | -0.85 | -0.21 |
| Annual return | -6.8% | -1.7% |
| Max drawdown | -85.7% | -47.6% |
| Round trips (per year) | 43 (3.5) | 72 (4.5) |
| Win rate | 37% | 58% |
| Total PnL | -$830k | -$267k |
| Trading + roll costs | $23k | $31k |

![Cumulative PnL of the baseline and walk-forward backtests](docs/results/fig_equity.png)

## What we found

**1. Gold and silver are correlated but not cointegrated.** Daily returns have a correlation of 0.79, yet the full-sample Engle-Granger test gives a p-value of 0.87 (t-statistic -1.15 against a 5% critical value of -3.34), and the Johansen test finds no cointegrating relation. Tested one year at a time, the pair is cointegrated at 5% in only 2 of 17 years (2014 and 2015), and the yearly hedge ratio ranges from 0.15 to 0.79.

**2. Mean reversion exists only on short windows.** Fitting an Ornstein-Uhlenbeck process to the spread gives a half-life of 12 to 14 trading days on the last 6 to 12 months, 37 days on 2 years, 98 days on 4 years and 940 days on the full sample. A strategy assuming a stable mean keeps betting on reversions that arrive years later, if at all.

**3. The gold/silver ratio trends for years.** It fell to 32 in 2011, peaked at 122 in March 2020, and dropped below 50 in early 2026 (chart: `docs/results/fig_prices_ratio.png`). The worst out-of-sample years are 2025 (-34%) and 2020 (-17%), when the spread kept moving away from its mean.

**4. Costs are not the problem.** Out of sample, trading and roll costs total about $23k against a gross loss of $807k. In the baseline, doubling every cost only moves the Sharpe from -0.21 to -0.24.

**5. No variation rescues the trade.** Every variant keeps the baseline parameters:

| Variant | Sharpe | Annual return | Max drawdown |
| --- | --- | --- | --- |
| Baseline | -0.21 | -1.7% | -47.6% |
| No costs | -0.19 | -1.5% | -45.4% |
| Fill at the signal close instead of the next day | -0.23 | -1.8% | -49.8% |
| Kalman-filter hedge ratio | -0.10 | -0.8% | -36.7% |
| No micro contracts | -0.19 | -1.5% | -45.6% |
| Dollar-neutral instead of beta-neutral | 0.15 | 2.0% | -32.8% |

Dollar-neutral sizing is the only positive variant, too weak and unrelated to the cointegration hypothesis to count as an edge.

**6. Tuning on the past does not help.** Each year the walk-forward picks the entry threshold and z-score lookback with the best Sharpe over the previous three years. Those settings did not carry over, and the walk-forward does worse than the fixed baseline.

Stress windows (baseline): -$169k in March 2020, -$50k in August to December 2011, -$47k in 2022, +$12k in April 2013.

## How it works

1. **Data** (`gcsi.data`): `definition` and `ohlcv-1d` for every GC and SI contract, with a cost check and a local cache.
2. **Continuous series** (`gcsi.rolls`): each leg rolls 5 business days before first notice. Signals use ratio-adjusted prices so roll gaps never look like moves; PnL uses the contract actually held, with each roll charged as two trades.
3. **Statistics** (`gcsi.cointegration`, `gcsi.ou`): Engle-Granger and Johansen tests, rolling and Kalman hedge ratios, OU half-life, no look-ahead.
4. **Signals** (`gcsi.signals`): the spread is log GC minus α minus β times log SI, with β from a trailing one-year OLS shifted one day. The z-score uses the past 60 days re-priced at today's β. Enter at |z| ≥ 2, exit at 0, stop at |z| ≥ 4.
5. **Backtest** (`gcsi.backtest`, `gcsi.costs`, `gcsi.ledger`): $1m of gold, silver sized at β times that in dollars, with micros (MGC, SIL) for the remainder. Orders fill at the next day's close. Costs per contract per side are half a tick plus exchange fee plus commission (GC $7.50, MGC $1.45, SI $15.00, SIL $3.45). The ledger attributes PnL to spread, sizing, trades and rolls.
6. **Report** (`gcsi.metrics`, `gcsi.report`): performance metrics, stress windows, sensitivities and figures.

**Pipeline check:** on 16 years of synthetic data with a known hedge ratio of 0.8 and a half-life of 13.9 days (`--source synthetic`), the walk-forward earns a Sharpe of 1.15 after costs.

## Limitations and next steps

The backtest trades daily closes with a one-day delay and does not model intraday execution. Next steps: a regime filter that trades only while a trailing cointegration test passes, settlement-price fills from the `statistics` schema, and an intraday study of working the spread as one order. Earlier scope, roles and the roadmap are in [docs/PLANNING.md](docs/PLANNING.md); the component design is in [docs/DESIGN.md](docs/DESIGN.md).

## Running the project

**Requirements:** Python 3.12, git and a Databento API key with CME access.

```bash
git clone https://github.com/ryotahidaka/finm37000-project.git
cd finm37000-project
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # pinned versions; installs src/gcsi in editable mode

cp .env.example .env                 # paste your key into .env (git-ignored)
pytest                               # Databento tests are skipped unless you run pytest -m db
```

The key is read from the `DATABENTO_API_KEY` environment variable or `.env`. If neither is set, the course helper `finm37000.get_databento_api_key()` reads `~/.databento_api_key`.

```bash
# 1. Download and cache GC and SI daily bars (every contract since the dataset start)
python -m gcsi.data --cost-only                 # free: prints the estimated cost
python -m gcsi.data --start 2010-06-06          # pulls, validates and caches under data/databento/
#    Refuses to pull above --max-cost (default $5) unless you add --yes.

# 2. Run everything: continuous series, statistics, both backtests, sensitivities, figures
python -m gcsi.report --source databento --start 2010-06-06
#    --end YYYY-MM-DD (exclusive) fixes the last date; --end 2026-10-24 uses the same
#    window as the committed results. --cache-only reads data/databento/ and never calls Databento.

# 3. Results: output/databento/report.md, plus results.json, CSV tables and fig_*.png

# No key? The whole pipeline on synthetic data with a known answer:
python -m gcsi.report --source synthetic        # writes output/synthetic/
```

`output/` is git-ignored. The results quoted above are a committed copy of one run, in [docs/results/](docs/results/report.md).

## Repository layout

```
src/gcsi/        contracts, data, rolls, cointegration, ou, signals, costs, ledger,
                 backtest, metrics, report (+ synthetic data for tests)
tests/           pytest suite; GitHub Actions runs it on Linux and Windows
docs/results/    committed results: report.md, results.json, CSV tables, figures
docs/DESIGN.md   pipeline and design decisions
docs/PLANNING.md original planning README (scope, roles, roadmap)
data/, output/   Databento cache and generated results (git-ignored)
```

## Team

Group 13, FINM 37000, University of Chicago, autumn 2026: Ryota Hidaka, Abinaya Ankur Raut, Konark Gupta, Raymond Wang.

## References

- Engle, R. and Granger, C. (1987), Co-integration and error correction, *Econometrica* 55(2)
- Gatev, E., Goetzmann, W. and Rouwenhorst, K. G. (2006), Pairs trading: performance of a relative-value arbitrage rule, *Review of Financial Studies* 19(3)
- Vidyamurthy, G. (2004), *Pairs Trading: Quantitative Methods and Analysis*, Wiley
- [Databento documentation](https://databento.com/docs) and [CME gold and silver contract specifications](https://www.cmegroup.com/markets/metals/precious/gold.contractSpecs.html)
