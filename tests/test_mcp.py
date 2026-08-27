"""T11 — the MCP surface. Tools are plain callables, so the contract is testable
without a running server; what is verified here is exactly what an agent gets."""
import pytest

from oficio.core.pricebook import load_pricebook
from oficio.service.mcp_tools import (
    TOOL_SPECS,
    call_tool,
    create_quote,
    explain_quote,
    get_catalog,
)


@pytest.fixture(scope="module")
def pb():
    return load_pricebook()


def line(item_id="wall_plastering", qty=10, quote="plaster the walls"):
    return {"item_id": item_id, "qty": qty, "source_quote": quote}


# ---------- tool declarations ----------

def test_three_tools_declared():
    assert {s["name"] for s in TOOL_SPECS} == {"get_catalog", "create_quote", "explain_quote"}


def test_every_spec_has_schema_and_description():
    for spec in TOOL_SPECS:
        assert spec["description"] and spec["inputSchema"]["type"] == "object"


def test_create_quote_requires_evidence_in_its_schema():
    spec = next(s for s in TOOL_SPECS if s["name"] == "create_quote")
    required = spec["inputSchema"]["properties"]["line_items"]["items"]["required"]
    assert "source_quote" in required  # evidence is part of the public contract


# ---------- get_catalog ----------

def test_catalog_lists_all_items(pb):
    result = get_catalog(pricebook=pb)
    assert len(result["items"]) == len(pb.items)
    assert result["pricebook_version"] == pb.version
    assert result["currency"] == "USD"


def test_catalog_filters_by_category(pb):
    result = get_catalog(category="plumbing", pricebook=pb)
    assert result["items"]
    assert all(i["category"] == "plumbing" for i in result["items"])


def test_catalog_never_exposes_costs(pb):
    """An agent gets what it can order, not the business's cost structure."""
    for item in get_catalog(pricebook=pb)["items"]:
        assert "labor_cost" not in item and "material_cost" not in item


# ---------- create_quote ----------

def test_create_quote_prices_a_job(pb):
    result = create_quote(pricebook=pb, line_items=[line()])
    assert result["status"] == "ok"
    assert result["total"] > result["subtotal"] > 0
    assert result["quote_id"].startswith("q_")


def test_create_quote_is_deterministic(pb):
    payload = {"line_items": [line(), line("toilet", 2, "two toilets")]}
    assert create_quote(pricebook=pb, **payload) == create_quote(pricebook=pb, **payload)


def test_create_quote_fails_closed_on_unknown_item(pb):
    result = create_quote(pricebook=pb, line_items=[line("gold_jacuzzi", 1, "a gold jacuzzi")])
    assert result["status"] == "needs_info"
    assert result["lines"] == [] and result["total"] == 0.0


def test_create_quote_rejects_missing_evidence(pb):
    with pytest.raises(Exception):
        create_quote(pricebook=pb, line_items=[{"item_id": "toilet", "qty": 1}])


def test_create_quote_clamps_margin_below_floor(pb):
    result = create_quote(pricebook=pb, line_items=[line()], margin=0.01)
    assert result["margin_applied"] == pb.rules.margin_floor


def test_create_quote_passes_budget_flag(pb):
    result = create_quote(pricebook=pb, line_items=[line("water_heater", 1, "a water heater")],
                          client_budget=1.0)
    assert result["over_budget"] is True


# ---------- explain_quote ----------

def test_explain_shows_arithmetic_per_line(pb):
    quote = create_quote(pricebook=pb, line_items=[line()])
    explained = explain_quote(quote)
    assert len(explained["steps"]) == 1
    assert "x USD" in explained["steps"][0]
    assert "margin" in explained["summary"]
    assert "no figure is estimated" in explained["note"]


def test_explain_flags_whole_unit_billing(pb):
    quote = create_quote(pricebook=pb, line_items=[line("cement_bag", 3.2, "cement bags")])
    step = explain_quote(quote)["steps"][0]
    assert "billed in whole units" in step


def test_explain_marks_incomplete_quotes(pb):
    quote = create_quote(pricebook=pb, line_items=[line("unknown_x", 1, "something odd")])
    explained = explain_quote(quote)
    assert explained["status"] == "needs_info"
    assert explained["unanswered"]
    assert "must be answered" in explained["note"]


def test_explain_invents_no_numbers(pb):
    """Explanation is formatting, not a second opinion."""
    quote = create_quote(pricebook=pb, line_items=[line(), line("toilet", 1, "one toilet")])
    explained = explain_quote(quote)
    assert f"{quote['total']:.2f}" in explained["summary"]
    assert f"{quote['subtotal']:.2f}" in explained["summary"]


# ---------- dispatch: the path an MCP client actually takes ----------

def test_dispatch_round_trip(pb):
    catalog = call_tool("get_catalog", {}, pricebook=pb)
    item_id = catalog["items"][0]["item_id"]
    quote = call_tool("create_quote",
                      {"line_items": [line(item_id, 1, "I need this")]}, pricebook=pb)
    explained = call_tool("explain_quote", {"quote": quote}, pricebook=pb)
    assert explained["quote_id"] == quote["quote_id"]


def test_dispatch_rejects_unknown_tool(pb):
    with pytest.raises(KeyError):
        call_tool("drop_database", {}, pricebook=pb)
