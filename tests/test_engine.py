"""T3 — deterministic quote engine. Every acceptance rule of the spec maps to a test here."""
import math
from decimal import ROUND_HALF_UP, Decimal

import pytest
from pydantic import ValidationError

from oficio.core.engine import CONTINUOUS_UNITS, compute_quote
from oficio.core.pricebook import load_pricebook
from oficio.core.schemas import JobSpec, LineItemRequest


@pytest.fixture(scope="module")
def pb():
    return load_pricebook()


def li(item_id: str, qty: float, quote: str = "customer asked for it") -> LineItemRequest:
    return LineItemRequest(item_id=item_id, qty=qty, source_quote=quote)


# ---------- basic correctness ----------

def test_single_continuous_line_exact_math(pb):
    item = pb.get("wall_plastering")
    q = compute_quote(JobSpec(line_items=[li("wall_plastering", 10)]), pb)
    expected_sub = Decimal(str(item.labor_cost)) * 10
    assert q.subtotal == float(expected_sub.quantize(Decimal("0.01")))
    expected_total = (expected_sub * Decimal("1.25")).quantize(Decimal("0.01"), ROUND_HALF_UP)
    assert q.total == float(expected_total)
    assert q.status == "ok"
    assert q.margin_applied == 0.25
    assert q.currency == "USD"


def test_multi_line_subtotal_is_sum_of_lines(pb):
    spec = JobSpec(line_items=[li("wall_plastering", 12), li("painting_3_coats", 30), li("toilet", 1)])
    q = compute_quote(spec, pb)
    assert q.subtotal == pytest.approx(sum(line.subtotal for line in q.lines), abs=0.02)
    assert len(q.lines) == 3


def test_labor_plus_material_unit_cost(pb):
    """unit_cost must be labor + material for every line."""
    q = compute_quote(JobSpec(line_items=[li("ceramic_tile_standard", 5)]), pb)
    item = pb.get("ceramic_tile_standard")
    assert q.lines[0].unit_cost == pytest.approx(item.labor_cost + item.material_cost)


# ---------- discrete vs continuous rounding ----------

def test_discrete_units_round_up(pb):
    q = compute_quote(JobSpec(line_items=[li("cement_bag", 3.2)]), pb)
    line = q.lines[0]
    assert line.billed_qty == 4
    assert line.requested_qty == 3.2
    assert any("whole" in a for a in q.assumptions)


def test_discrete_whole_qty_no_assumption(pb):
    q = compute_quote(JobSpec(line_items=[li("cement_bag", 3)]), pb)
    assert q.lines[0].billed_qty == 3
    assert not any("whole" in a for a in q.assumptions)


def test_continuous_units_bill_fractional(pb):
    q = compute_quote(JobSpec(line_items=[li("wall_plastering", 12.5)]), pb)
    assert q.lines[0].billed_qty == 12.5


def test_every_pricebook_unit_is_classified(pb):
    """A new unit added to the price book must be consciously classified."""
    discrete = {"unit", "bag", "bucket_5gal", "gallon", "set"}
    for item in pb.items:
        assert item.unit in CONTINUOUS_UNITS or item.unit in discrete, item.unit


# ---------- fail-closed: unknown items ----------

def test_unknown_item_goes_to_needs_info_never_priced(pb):
    spec = JobSpec(line_items=[li("gold_plated_jacuzzi", 1, "I want a gold jacuzzi")])
    q = compute_quote(spec, pb)
    assert q.status == "needs_info"
    assert q.lines == []
    assert q.subtotal == 0.0
    assert "gold_plated_jacuzzi" in q.needs_info[0]
    assert "gold jacuzzi" in q.needs_info[0]  # evidence is carried through


def test_partial_quote_known_items_still_priced(pb):
    spec = JobSpec(line_items=[li("toilet", 1), li("unknown_thing", 2)])
    q = compute_quote(spec, pb)
    assert q.status == "needs_info"
    assert len(q.lines) == 1
    assert q.subtotal > 0


def test_missing_info_from_agent_propagates(pb):
    spec = JobSpec(line_items=[li("toilet", 1)], missing_info=["countertop material not specified"])
    q = compute_quote(spec, pb)
    assert q.status == "needs_info"
    assert "countertop material not specified" in q.needs_info


# ---------- margin rules ----------

def test_default_margin_applied(pb):
    q = compute_quote(JobSpec(line_items=[li("toilet", 1)]), pb)
    assert q.margin_applied == pb.rules.default_margin


def test_margin_below_floor_is_clamped_and_declared(pb):
    q = compute_quote(JobSpec(line_items=[li("toilet", 1)]), pb, margin=0.05)
    assert q.margin_applied == pb.rules.margin_floor
    assert any("floor" in a for a in q.assumptions)


def test_margin_above_floor_respected(pb):
    q = compute_quote(JobSpec(line_items=[li("toilet", 1)]), pb, margin=0.35)
    assert q.margin_applied == 0.35


def test_total_is_subtotal_times_margin(pb):
    q = compute_quote(JobSpec(line_items=[li("wall_plastering", 20), li("cement_bag", 5)]), pb)
    expected = Decimal(str(q.subtotal)) * Decimal("1.25")
    assert q.total == pytest.approx(float(expected), abs=0.02)


# ---------- budget flag ----------

def test_over_budget_flagged_not_trimmed(pb):
    spec = JobSpec(line_items=[li("water_heater", 1)], client_budget=1.0)
    q = compute_quote(spec, pb)
    assert q.over_budget is True
    assert len(q.lines) == 1  # scope was NOT silently trimmed
    assert any("budget" in a.lower() for a in q.assumptions)


def test_within_budget_not_flagged(pb):
    spec = JobSpec(line_items=[li("gas_valve", 1)], client_budget=10_000.0)
    q = compute_quote(spec, pb)
    assert q.over_budget is False


# ---------- determinism & reproducibility ----------

def test_same_input_same_quote_id_and_totals(pb):
    spec = JobSpec(line_items=[li("wall_plastering", 12), li("cement_bag", 3)])
    a, b = compute_quote(spec, pb), compute_quote(spec, pb)
    assert a.quote_id == b.quote_id
    assert a.total == b.total


def test_different_input_different_quote_id(pb):
    a = compute_quote(JobSpec(line_items=[li("toilet", 1)]), pb)
    b = compute_quote(JobSpec(line_items=[li("toilet", 2)]), pb)
    assert a.quote_id != b.quote_id


def test_quote_carries_pricebook_version(pb):
    q = compute_quote(JobSpec(line_items=[li("toilet", 1)]), pb)
    assert q.pricebook_version == pb.version


# ---------- contract validation (schema edge) ----------

def test_evidence_is_mandatory():
    with pytest.raises(ValidationError):
        LineItemRequest(item_id="toilet", qty=1, source_quote="")


def test_qty_must_be_positive():
    with pytest.raises(ValidationError):
        LineItemRequest(item_id="toilet", qty=0, source_quote="a toilet please")


def test_empty_spec_yields_ok_empty_quote(pb):
    q = compute_quote(JobSpec(), pb)
    assert q.status == "ok" and q.lines == [] and q.total == 0.0


# ---------- realistic end-to-end scenario ----------

def test_kitchen_remodel_scenario(pb):
    """12 m2 kitchen: plaster + skim + paint + tile floor with adhesive + faucet."""
    spec = JobSpec(
        space_type="kitchen",
        area_m2=12,
        line_items=[
            li("wall_plastering", 34, "plaster the kitchen walls"),
            li("wall_skim_coat", 34, "smooth finish on walls"),
            li("painting_3_coats", 46, "paint walls and ceiling"),
            li("tile_install_thinset", 12, "new floor tile"),
            li("ceramic_tile_standard", 12, "new floor tile"),
            li("tile_adhesive_25kg", 2.4, "new floor tile"),
            li("kitchen_mixer_tap", 1, "replace the tap"),
        ],
    )
    q = compute_quote(spec, pb)
    assert q.status == "ok"
    assert len(q.lines) == 7
    adhesive = next(line for line in q.lines if line.item_id == "tile_adhesive_25kg")
    assert adhesive.billed_qty == 3  # 2.4 bags -> 3 whole bags
    assert q.total > q.subtotal > 0
    computed = Decimal(str(q.subtotal)) * Decimal("1.25")
    assert q.total == pytest.approx(float(computed), abs=0.02)


def test_ceil_matches_math_ceil_for_many_qtys(pb):
    for qty in [0.1, 0.5, 1.0, 1.01, 2.999, 7.5]:
        q = compute_quote(JobSpec(line_items=[li("cement_bag", qty)]), pb)
        assert q.lines[0].billed_qty == math.ceil(qty)
