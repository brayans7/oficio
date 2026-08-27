"""Model pricing — fail-closed.

A model with no tariff entry raises. It never costs $0.

Silent $0 is the most expensive bug in LLM systems: spend looks fine on the
dashboard while the bill grows. Rates are USD per million tokens and are
declared here explicitly, with the date they were checked, so a stale rate is
visible instead of invisible.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

#: Rates checked 2026-08. Update deliberately; never guess.
PRICING: dict[str, tuple[str, str]] = {
    # model id: (input $/Mtok, output $/Mtok)
    "claude-haiku-4-5": ("1.00", "5.00"),
    "claude-sonnet-4-6": ("3.00", "15.00"),
    "claude-opus-4-8": ("15.00", "75.00"),
}

_MILLION = Decimal("1000000")


class PricingError(RuntimeError):
    """Raised when a model has no declared tariff. Fail-closed by design."""


@dataclass(frozen=True)
class Usage:
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        try:
            rate_in, rate_out = PRICING[self.model]
        except KeyError as exc:
            raise PricingError(
                f"No tariff declared for model {self.model!r}. Refusing to record a $0 call — "
                f"add it to PRICING with the date you checked the rate."
            ) from exc
        cost = (
            Decimal(self.input_tokens) * Decimal(rate_in)
            + Decimal(self.output_tokens) * Decimal(rate_out)
        ) / _MILLION
        return float(round(cost, 6))
