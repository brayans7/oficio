"""MCP surface — the same engine, exposed so *other agents* can quote.

A buying agent should not have to scrape a web form to get a price. These three
tools are the machine-readable contract:

- ``get_catalog``  — what can be quoted at all, and in which units.
- ``create_quote`` — price a JobSpec. Same determinism, same fail-closed rules.
- ``explain_quote``— why a total is what it is, line by line, with the customer's
  own words attached. An agent that cannot explain a number should not send it.

The tool functions are plain callables returning plain dicts, so they are testable
without a running MCP server; ``build_server()`` binds them to the protocol.
"""
from __future__ import annotations

from typing import Any

from ..core.engine import compute_quote
from ..core.pricebook import Pricebook, load_pricebook
from ..core.schemas import JobSpec

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_catalog",
        "description": "List every quotable item with its unit and category. "
                       "Item ids from this list are the only valid input to create_quote.",
        "inputSchema": {
            "type": "object",
            "properties": {"category": {"type": "string",
                                        "description": "optional filter, e.g. 'plumbing'"}},
        },
    },
    {
        "name": "create_quote",
        "description": "Price a job. Every line must carry source_quote: the literal customer "
                       "text justifying it. Unknown items are returned as questions, never priced.",
        "inputSchema": {
            "type": "object",
            "required": ["line_items"],
            "properties": {
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["item_id", "qty", "source_quote"],
                        "properties": {
                            "item_id": {"type": "string"},
                            "qty": {"type": "number"},
                            "source_quote": {"type": "string"},
                        },
                    },
                },
                "space_type": {"type": "string"},
                "area_m2": {"type": "number"},
                "client_budget": {"type": "number"},
                "margin": {"type": "number",
                           "description": "optional; values below the price book floor are clamped"},
            },
        },
    },
    {
        "name": "explain_quote",
        "description": "Explain a quote in plain language: per-line arithmetic, the margin applied, "
                       "declared assumptions and anything still unanswered.",
        "inputSchema": {
            "type": "object",
            "required": ["quote"],
            "properties": {"quote": {"type": "object", "description": "a create_quote result"}},
        },
    },
]


def get_catalog(category: str | None = None, pricebook: Pricebook | None = None) -> dict:
    pb = pricebook or load_pricebook()
    items = [i for i in pb.items if category is None or i.category == category]
    return {
        "pricebook_version": pb.version,
        "currency": pb.currency,
        "categories": sorted({i.category for i in pb.items}),
        "items": [{"item_id": i.id, "name": i.name, "unit": i.unit,
                   "category": i.category, "kind": i.kind} for i in items],
    }


def create_quote(pricebook: Pricebook | None = None, **payload: Any) -> dict:
    pb = pricebook or load_pricebook()
    margin = payload.pop("margin", None)
    spec = JobSpec.model_validate(payload)
    return compute_quote(spec, pb, margin=margin).model_dump()


def explain_quote(quote: dict) -> dict:
    """Render a quote as an auditable explanation. Pure formatting — no new numbers."""
    lines = quote.get("lines", [])
    currency = quote.get("currency", "USD")
    steps = [
        f"{ln['name']}: {ln['billed_qty']} {ln['unit']} x {currency} {ln['unit_cost']:.2f} "
        f"= {currency} {ln['subtotal']:.2f}"
        + (f"  (requested {ln['requested_qty']}, billed in whole units)"
           if ln["billed_qty"] != ln["requested_qty"] else "")
        for ln in lines
    ]
    margin = quote.get("margin_applied", 0.0)
    summary = (
        f"Subtotal {currency} {quote.get('subtotal', 0):.2f}"
        f" + {margin * 100:.0f}% margin"
        f" = {currency} {quote.get('total', 0):.2f}"
    )
    return {
        "quote_id": quote.get("quote_id"),
        "status": quote.get("status"),
        "steps": steps,
        "summary": summary,
        "assumptions": quote.get("assumptions", []),
        "unanswered": quote.get("needs_info", []),
        "over_budget": quote.get("over_budget", False),
        "note": ("This quote is incomplete: the questions above must be answered before it is valid."
                 if quote.get("status") == "needs_info"
                 else "Every line is priced from the versioned price book; no figure is estimated."),
    }


HANDLERS = {"get_catalog": get_catalog, "create_quote": create_quote, "explain_quote": explain_quote}


def call_tool(name: str, arguments: dict, pricebook: Pricebook | None = None) -> dict:
    """Dispatch used by both the MCP server and the tests."""
    if name not in HANDLERS:
        raise KeyError(f"Unknown tool {name!r}. Available: {sorted(HANDLERS)}")
    if name == "explain_quote":
        return explain_quote(**arguments)
    return HANDLERS[name](pricebook=pricebook, **arguments)


def build_server():  # pragma: no cover — requires the optional mcp dependency
    """Bind the handlers to a stdio MCP server."""
    from mcp.server import Server
    from mcp.types import TextContent, Tool
    import json

    server = Server("oficio")

    @server.list_tools()
    async def _list() -> list[Tool]:
        return [Tool(**spec) for spec in TOOL_SPECS]

    @server.call_tool()
    async def _call(name: str, arguments: dict) -> list[TextContent]:
        return [TextContent(type="text", text=json.dumps(call_tool(name, arguments), indent=2))]

    return server


def main() -> None:  # pragma: no cover
    import asyncio

    from mcp.server.stdio import stdio_server

    async def _run():
        server = build_server()
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
