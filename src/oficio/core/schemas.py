"""Contracts between the agent layer and the deterministic engine.

Money semantics (single source of truth):
- Every line carries `unit_cost` (labor + material per unit, PRE-margin) and
  `subtotal` = unit_cost x billed_qty.
- The quote's `subtotal` is the sum of line subtotals (pre-margin).
- Margin is applied once, at the end: `total` = subtotal x (1 + margin_applied),
  rounded half-up to the cent. No hidden per-line markups.

Evidence rule: every requested line MUST carry `source_quote` — the literal
customer text that justifies it. A line without evidence is an eval failure
upstream; the engine refuses to accept an empty one.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class LineItemRequest(BaseModel):
    """One thing the customer asked for, as extracted by the agent."""

    item_id: str
    qty: float = Field(gt=0)
    source_quote: str = Field(min_length=3, description="Literal customer text justifying this line")


class JobSpec(BaseModel):
    """Structured job description. Produced by the agent, consumed by the engine."""

    line_items: list[LineItemRequest] = Field(default_factory=list)
    space_type: str | None = None
    area_m2: float | None = Field(default=None, gt=0)
    client_budget: float | None = Field(default=None, gt=0)
    missing_info: list[str] = Field(default_factory=list)


class QuoteLine(BaseModel):
    """One priced line. billed_qty may exceed requested qty for discrete units
    (you cannot buy half a bag of cement)."""

    item_id: str
    name: str
    requested_qty: float
    billed_qty: float
    unit: str
    unit_cost: float = Field(description="labor + material per unit, pre-margin")
    subtotal: float


class QuoteResult(BaseModel):
    quote_id: str
    pricebook_version: str
    currency: str
    status: str = Field(pattern="^(ok|needs_info)$")
    lines: list[QuoteLine]
    subtotal: float
    margin_applied: float
    total: float
    assumptions: list[str] = Field(default_factory=list)
    needs_info: list[str] = Field(default_factory=list)
    over_budget: bool = False
