"""End-to-end smoke test of the report on synthetic data (#6, #10, #11)."""
import json

import pytest

from gcsi import report, synthetic


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    bars = report.load_synthetic(years=7, seed=3)
    truth = bars.pop("truth")
    gc, si, logs, n_rolls = report.continuous(bars)
    e, yearly = report.eda(gc, si)
    base, cm = report.BacktestConfig(window=126), report.CostModel()
    res = report.run_backtests(gc, si, logs, base, cm, grid={"entry": [1.5, 2.0]}, train_years=2)
    figs = report.figures(gc, si, res, out)
    results = report.write_report("synthetic", e, yearly, res, n_rolls, base, cm, out, figs, truth)
    return out, results, n_rolls, e


def test_files_written(run):
    out, results, _, _ = run
    for name in ("report.md", "results.json", "daily.csv", "fills.csv", "stress.csv",
                 "sensitivity.csv", "by_year.csv", "wf_choices.csv", "fig_equity.png"):
        assert (out / name).exists(), name
    assert json.loads((out / "results.json").read_text())["source"] == "synthetic"


def test_known_answer_is_recovered(run):
    _, results, n_rolls, e = run
    assert abs(e["eg_beta"] - 0.8) < 0.1 and e["eg_pvalue"] < 0.01
    assert n_rolls["GC"] >= 30 and n_rolls["SI"] >= 30   # 5 a year each
    assert results["walk_forward"]["summary"]["sharpe"] > 0.5


def test_chain_generator_matches_load_bars_columns():
    pair = synthetic.cointegrated_prices(n=300, seed=1)
    bars = synthetic.per_expiry_chain(pair.prices["GC"], "GC")
    assert list(bars.columns) == ["ts_event", "instrument_id", "symbol", "expiration",
                                  "open", "high", "low", "close", "volume"]
    assert bars.groupby("symbol")["instrument_id"].nunique().eq(1).all()
