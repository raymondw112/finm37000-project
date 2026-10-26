"""Tests for the transaction cost model (#24)."""
import pytest

from gcsi import contracts
from gcsi.costs import CostModel, LegCost


def test_default_per_contract_costs_by_hand():
    m = CostModel()
    # half a tick + exchange fee + commission
    assert m.per_contract("GC") == pytest.approx(0.5 * 10.0 + 1.65 + 0.85)   # 7.50
    assert m.per_contract("SI") == pytest.approx(0.5 * 25.0 + 1.65 + 0.85)   # 15.00
    assert m.per_contract("MGC") == pytest.approx(0.5 * 1.0 + 0.70 + 0.25)   # 1.45
    assert m.per_contract("SIL") == pytest.approx(0.5 * 5.0 + 0.70 + 0.25)   # 3.45


def test_tick_values_come_from_contract_specs():
    one_tick = CostModel({leg: LegCost(1.0) for leg in ("GC", "MGC", "SI", "SIL")})
    for leg in ("GC", "MGC", "SI", "SIL"):
        assert one_tick.per_contract(leg) == contracts.get_spec(leg).tick_value


def test_sign_ignored_and_linear_in_contracts():
    m = CostModel()
    assert m.cost("SI", -4) == m.cost("SI", 4) == pytest.approx(4 * m.per_contract("SI"))
    assert m.cost("GC", 0) == 0


def test_flat_zero_and_scaled():
    assert CostModel.flat(2.5).per_contract("SIL") == 2.5
    assert CostModel.zero().cost("GC", 10) == 0
    m = CostModel()
    assert m.scaled(2.0).per_contract("GC") == pytest.approx(2 * m.per_contract("GC"))
    assert CostModel.flat(3.0).scaled(0.5).per_contract("MGC") == pytest.approx(1.5)


def test_slippage_adds_ticks():
    m = CostModel({"GC": LegCost(0.5, 1.65, 0.85, slippage_ticks=1.0)})
    assert m.per_contract("GC") == pytest.approx(1.5 * 10 + 1.65 + 0.85)


def test_round_trip_spread_cost():
    m = CostModel()
    # 5 GC + 6 SI opened and closed
    assert m.round_trip_spread_cost(5, 6) == pytest.approx(2 * (5 * 7.5 + 6 * 15.0))
