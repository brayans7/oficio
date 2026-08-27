"""The deterministic quote engine. Pure: same JobSpec + same price book -> same quote, always.

Design rules this module enforces:
- Unknown item -> `needs_info`. The engine NEVER estimates a price it doesn't have.
- Discrete units (bags, buckets, fixtures) are billed in whole units, rounded UP —
  you cannot buy 3.2 bags of cement. Continuous units (m2, m3) bill as requested.
- Margin is applied once at the end. A margin below the floor is clamped up to the
  floor and declared in `assumptions` — never silently accepted.
- Money math uses Decimal; floats only cross the boundary at the schema edge.
- quote_id is a content hash: identical inputs yield the identical quote id,
  which makes every quote reproducible and auditable.
"""
from __future__ import annotations

import hashlib
import json
import math
from decimal import ROUND_HALF_UP, Decimal

from .pricebook import Pricebook
from .schemas import JobSpec, QuoteLine, QuoteResult

#: Units billed exactly as requested (fractions allowed).
CONTINUOUS_UNITS = frozenset({"m2", "m3", "dwelling"})

_CENT = Decimal("0.01")


def _money(value: Decimal) -> float:
    return float(value.quantize(_CENT, rounding=ROUND_HALF_UP))


def _quote_id(spec: JobSpec, pricebook_version: str, margin: Decimal) -> str:
    payload = json.dumps(
        {
            "spec": spec.model_dump(),
            "pricebook": pricebook_version,
            "margin": str(margin),
        },
        sort_keys=True,
    )
    return "q_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def compute_quote(
    spec: JobSpec,
    pricebook: Pricebook,
    margin: float | None = None,
) -> QuoteResult:
    """Price a JobSpec against a price book. Deterministic and fail-closed."""
    rules = pricebook.rules
    assumptions: list[str] = []
    needs_info: list[str] = list(spec.missing_info)

    requested_margin = Decimal(str(margin if margin is not None else rules.default_margin))
    floor = Decimal(str(rules.margin_floor))
    if requested_margin < floor:
        assumptions.append(
            f"Requested margin {requested_margin} is below the floor {floor}; "
            f"floor applied. Margins below the floor are a business rule violation."
        )
        applied_margin = floor
    else:
        applied_margin = requested_margin

    lines: list[QuoteLine] = []
    subtotal = Decimal("0")
    known_ids = pricebook.item_ids

    for req in spec.line_items:
        if req.item_id not in known_ids:
            needs_info.append(
                f"'{req.item_id}' is not in price book v{pricebook.version} — "
                f"a price must be added or confirmed before quoting it. (customer said: {req.source_quote!r})"
            )
            continue

        item = pricebook.get(req.item_id)
        if item.unit in CONTINUOUS_UNITS:
            billed_qty = Decimal(str(req.qty))
        else:
            billed_qty = Decimal(math.ceil(req.qty))
            if billed_qty != Decimal(str(req.qty)):
                assumptions.append(
                    f"{item.name}: billed {billed_qty} whole {item.unit}(s) for a request of "
                    f"{req.qty} — discrete units are never sold fractionally."
                )

        unit_cost = Decimal(str(item.labor_cost)) + Decimal(str(item.material_cost))
        line_subtotal = unit_cost * billed_qty
        subtotal += line_subtotal
        lines.append(
            QuoteLine(
                item_id=item.id,
                name=item.name,
                requested_qty=req.qty,
                billed_qty=float(billed_qty),
                unit=item.unit,
                unit_cost=_money(unit_cost),
                subtotal=_money(line_subtotal),
            )
        )

    total = subtotal * (Decimal("1") + applied_margin)
    status = "needs_info" if needs_info else "ok"

    over_budget = False
    if spec.client_budget is not None and _money(total) > spec.client_budget:
        over_budget = True
        assumptions.append(
            f"Quoted total {_money(total)} exceeds the stated budget {spec.client_budget}. "
            f"Flagged for an explicit conversation — the engine does not trim scope on its own."
        )

    return QuoteResult(
        quote_id=_quote_id(spec, pricebook.version, applied_margin),
        pricebook_version=pricebook.version,
        currency=pricebook.currency,
        status=status,
        lines=lines,
        subtotal=_money(subtotal),
        margin_applied=float(applied_margin),
        total=_money(total),
        assumptions=assumptions,
        needs_info=needs_info,
        over_budget=over_budget,
    )
