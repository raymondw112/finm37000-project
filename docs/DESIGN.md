# Project Design: Gold/Silver Futures Pairs Trading

> Design and component overview for the FINM 37000 project. Companion to the README: the README states *what* we deliver and how to run it; this document shows *how the pieces fit together* and *why*. Maintained as part of the Design-leader work; the task breakdown lives in the repo Issues.

## 1. Goal in one line

Build a reproducible pipeline that pulls CME Gold (GC) and Silver (SI) futures from Databento, constructs a statistically valid spread, generates mean-reversion signals, and backtests the strategy with realistic futures costs, then reports honestly whether an edge survives.

## 2. Pipeline (data to result)

```mermaid
flowchart TD
    A[Databento GLBX.MDP3<br/>GC.FUT and SI.FUT per-expiry] --> B[Ingestion and local cache<br/>trades + top-of-book]
    B --> C[Continuous-contract build<br/>roll on volume ~5d pre-FND]
    C --> D1[Ratio-adjusted series<br/>for SIGNALS]
    C --> D2[Actual-price series + logged roll costs<br/>for PnL]
    D1 --> E[Exploratory analysis<br/>ratio, regimes, ADF I 1 tests]
    E --> F[Cointegration + hedge ratio<br/>Engle-Granger, Johansen, Kalman beta]
    F --> G[Spread + OU model<br/>kappa, theta, half-life]
    G --> H[Signal generation<br/>z-score entry/exit, position sizing]
    H --> I[Backtest engine<br/>walk-forward, 4-leg costs, roll, SPAN margin]
    D2 --> I
    I --> J[Evaluation + report<br/>Sharpe, drawdown, stress 2008/2011/2020]
    J --> K[Results in output/<br/>tables + figures]
```

## 3. Components

```mermaid
flowchart LR
    subgraph course[Course package finm37000 - reused]
        c1[db_env_util.py<br/>Databento key handling]
        c2[continuous.py<br/>continuous/roll helpers]
        c3[market_data.py / futures.py]
    end
    subgraph ours[Our package src/]
        s1[data/<br/>ingestion + roll adjustment]
        s2[pairs/<br/>cointegration + hedge ratio]
        s3[signals/<br/>OU, z-score, sizing]
        s4[backtest/<br/>engine + costs + metrics]
    end
    course --> s1
    s1 --> s2 --> s3 --> s4
    s4 --> out[output/ results]
```

We reuse the professor's `finm37000` package for the data and roll plumbing rather than rebuilding it; our package holds the pairs-trading logic on top.

## 4. Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Pair | Gold (GC) vs Silver (SI) | Economically linked precious metals; course-permitted (energy mean-reversion is not) |
| Selection method | Cointegration backbone, distance as benchmark | A single known pair needs no ML search; cointegration gives a tradable spread |
| Hedge ratio | Rolling / Kalman, not static OLS | The GC/SI relationship drifts and breaks in some regimes; a static beta is unsafe |
| Signal | OU half-life + z-score bands | Standard, interpretable; half-life estimated on our data, not borrowed from papers |
| Continuous series | Two series: ratio-adjusted (signals) + real-price (PnL) | Databento continuous prices are unadjusted; mixing adjustment methods corrupts either the signal or the PnL |
| Costs | 4-leg bid-ask + monthly roll + SPAN margin | A spread crosses the spread on both legs each entry and exit; ignoring this inflates returns |
| Libraries | statsmodels, scipy, pykalman (all free) | No dependency on paid tooling (ArbitrageLab is paid; excluded) |
| Advanced model | Copula optional; skip PCA and ML selection | Copula is genuinely bivariate; PCA/ML add complexity without payoff for one pair |

## 5. How the Issues map to components

| Issue | Component | Produces |
|---|---|---|
| Structure + env | repo root | installable environment |
| Databento ingestion | `data/` | cached GC/SI frames |
| Continuous + roll | `data/` | ratio-adjusted + real-price series |
| EDA | notebook | tradability assessment |
| Cointegration + hedge ratio | `pairs/` | spread definition |
| OU + half-life | `signals/` | reversion parameters |
| Signal generation | `signals/` | dated positions |
| Backtest | `backtest/` | equity curve with costs |
| Evaluation | `backtest/` + `output/` | metrics + figures |
| Copula (stretch) | `signals/` | benchmark comparison |

## 6. Central risk (state it up front)

The gold/silver long-run relationship is **not stable**. The empirical literature finds cointegration in some periods and breakdown in others (notably the 1990s, and blow-outs in 2011 and March 2020). We therefore treat "the pair is cointegrated" as a hypothesis to test across sub-periods, use strictly out-of-sample calibration, and report whether the edge survives transaction costs rather than assuming it does. A clear negative result, honestly shown, is a valid and strong outcome for this project.
