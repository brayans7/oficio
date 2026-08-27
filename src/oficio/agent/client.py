"""Hardened model client: budget, retries, cost telemetry.

Every call is metered and traced. The transport is injectable, so the whole
control layer — budgets, retries, tracing, routing — is tested without a
network or an API key; the real Anthropic SDK is just one transport
implementation among them.

Guarantees:
- A call that would exceed the daily budget is refused BEFORE it is made.
- Transient failures (429/5xx/overloaded) retry with exponential backoff;
  everything else fails immediately — retrying a 400 only burns money.
- Every completed call appends one JSONL trace line with model, tokens, cost,
  latency and route. No call is invisible.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .pricing import Usage

#: Task -> model. Cheap model reads; expensive model checks. Overridable by env.
ROUTES: dict[str, str] = {
    "extract": os.getenv("OFICIO_MODEL_EXTRACT", "claude-haiku-4-5"),
    "verify": os.getenv("OFICIO_MODEL_VERIFY", "claude-sonnet-4-6"),
}

RETRYABLE_MARKERS = ("429", "500", "502", "503", "529", "overloaded", "rate_limit", "timeout")


class BudgetExceeded(RuntimeError):
    """Raised before a call when the daily budget is already spent."""


class UpstreamError(RuntimeError):
    """Raised after retries are exhausted, or immediately on a non-retryable error."""


@dataclass
class Reply:
    text: str
    usage: Usage
    route: str
    latency_ms: int
    attempts: int

    @property
    def cost_usd(self) -> float:
        return self.usage.cost_usd


class Transport(Protocol):
    """Minimal seam over a model API: prompt in, (text, in_tokens, out_tokens) out."""

    def __call__(self, model: str, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        ...


@dataclass
class ModelClient:
    transport: Transport
    daily_budget_usd: float = field(default_factory=lambda: float(os.getenv("DAILY_COST_LIMIT", "5.0")))
    trace_path: Path | None = None
    max_attempts: int = 3
    base_backoff_s: float = 0.5
    _sleep: Callable[[float], None] = time.sleep
    _now_ms: Callable[[], int] = lambda: int(time.monotonic() * 1000)
    spent_usd: float = 0.0
    calls: int = 0

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.daily_budget_usd - self.spent_usd)

    def complete(self, route: str, system: str, user: str, max_tokens: int = 4096) -> Reply:
        if route not in ROUTES:
            raise ValueError(f"Unknown route {route!r}. Declared routes: {sorted(ROUTES)}")
        model = ROUTES[route]

        if self.remaining_usd <= 0:
            raise BudgetExceeded(
                f"Daily budget of ${self.daily_budget_usd} is spent (${self.spent_usd:.4f} used "
                f"over {self.calls} calls). Refusing the call before it is made."
            )

        started = self._now_ms()
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                text, in_tok, out_tok = self.transport(model, system, user, max_tokens)
                break
            except Exception as exc:  # noqa: BLE001 — classified below
                last_error = exc
                if not _is_retryable(exc) or attempt == self.max_attempts:
                    raise UpstreamError(
                        f"Model call failed on attempt {attempt}/{self.max_attempts}: {exc}"
                    ) from exc
                self._sleep(self.base_backoff_s * (2 ** (attempt - 1)))
        else:  # pragma: no cover — loop always breaks or raises
            raise UpstreamError(str(last_error))

        usage = Usage(model=model, input_tokens=in_tok, output_tokens=out_tok)
        cost = usage.cost_usd  # raises PricingError for an untariffed model — before we bank it
        self.spent_usd += cost
        self.calls += 1
        reply = Reply(
            text=text,
            usage=usage,
            route=route,
            latency_ms=self._now_ms() - started,
            attempts=attempt,
        )
        self._trace(reply)
        return reply

    def _trace(self, reply: Reply) -> None:
        if self.trace_path is None:
            return
        record = {
            "route": reply.route,
            "model": reply.usage.model,
            "input_tokens": reply.usage.input_tokens,
            "output_tokens": reply.usage.output_tokens,
            "cost_usd": reply.cost_usd,
            "latency_ms": reply.latency_ms,
            "attempts": reply.attempts,
            "cumulative_cost_usd": round(self.spent_usd, 6),
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def _is_retryable(exc: Exception) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in RETRYABLE_MARKERS)


def anthropic_transport() -> Transport:
    """The real transport. Imported lazily so `core` and tests never need the SDK."""

    def _call(model: str, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        from anthropic import Anthropic  # local import: optional dependency

        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise UpstreamError("ANTHROPIC_API_KEY is not set. Failing closed rather than degrading.")
        resp = Anthropic(api_key=key).messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    return _call
