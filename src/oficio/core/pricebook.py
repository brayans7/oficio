"""Versioned price book: loading and validation.

Fail-closed by design: an unknown item id is a lookup error, never an estimate.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class PriceItem(BaseModel):
    id: str
    name: str
    category: str
    kind: str = Field(pattern="^(labor|material)$")
    unit: str
    labor_cost: float = Field(ge=0)
    material_cost: float = Field(ge=0)
    materials_included: bool


class PricebookRules(BaseModel):
    default_margin: float = Field(gt=0, lt=1)
    margin_floor: float = Field(gt=0, lt=1)
    rounding: str
    tile_price_cap_per_m2: float | None = None


class Pricebook(BaseModel):
    version: str
    currency: str
    description: str = ""
    items: list[PriceItem]
    rules: PricebookRules

    def get(self, item_id: str) -> PriceItem:
        for item in self.items:
            if item.id == item_id:
                return item
        raise UnknownItemError(item_id)

    @property
    def item_ids(self) -> set[str]:
        return {item.id for item in self.items}


class UnknownItemError(KeyError):
    """Raised when an item id is not in the price book. The engine surfaces this
    as `needs_info` — it never guesses a price."""

    def __init__(self, item_id: str) -> None:
        self.item_id = item_id
        super().__init__(f"Unknown price book item: {item_id!r}. Refusing to estimate.")


DEFAULT_PRICEBOOK_PATH = Path(__file__).resolve().parents[3] / "data" / "pricebook.v1.json"


def load_pricebook(path: str | Path = DEFAULT_PRICEBOOK_PATH) -> Pricebook:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Pricebook.model_validate(raw)
