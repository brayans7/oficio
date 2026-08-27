"""T5 — hardened model client: budget, retries, cost telemetry, fail-closed pricing."""
import json

import pytest

from oficio.agent.client import (
    ROUTES,
    BudgetExceeded,
    ModelClient,
    UpstreamError,
    _is_retryable,
)
from oficio.agent.pricing import PRICING, PricingError, Usage


def fake_transport(text="ok", in_tok=1000, out_tok=500):
    def _t(model, system, user, max_tokens):
        return text, in_tok, out_tok

    return _t


def flaky_transport(failures: int, error: Exception, then_text="ok"):
    state = {"n": 0}

    def _t(model, system, user, max_tokens):
        state["n"] += 1
        if state["n"] <= failures:
            raise error
        return then_text, 10, 10

    _t.state = state
    return _t


def client(**kw) -> ModelClient:
    kw.setdefault("transport", fake_transport())
    kw.setdefault("_sleep", lambda s: None)
    return ModelClient(**kw)


# ---------- pricing: fail-closed ----------

def test_untariffed_model_raises_never_zero():
    with pytest.raises(PricingError, match="No tariff"):
        Usage(model="claude-imaginary-9", input_tokens=1000, output_tokens=1000).cost_usd


def test_known_model_cost_is_exact():
    # haiku: $1/Mtok in, $5/Mtok out -> 1M in + 1M out = $6.00
    u = Usage(model="claude-haiku-4-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert u.cost_usd == 6.0


def test_zero_tokens_cost_zero():
    assert Usage(model="claude-haiku-4-5", input_tokens=0, output_tokens=0).cost_usd == 0.0


def test_every_declared_route_has_a_tariff():
    """A route pointing at an unpriced model would fail at runtime — catch it here."""
    for route, model in ROUTES.items():
        assert model in PRICING, f"route {route} -> {model} has no tariff"


# ---------- budget ----------

def test_call_within_budget_succeeds():
    c = client(daily_budget_usd=1.0)
    reply = c.complete("extract", "sys", "user")
    assert reply.text == "ok"
    assert c.spent_usd > 0
    assert c.calls == 1


def test_budget_exhausted_refuses_before_calling():
    called = {"n": 0}

    def counting(model, system, user, max_tokens):
        called["n"] += 1
        return "ok", 1000, 1000

    c = ModelClient(transport=counting, daily_budget_usd=0.0)
    with pytest.raises(BudgetExceeded):
        c.complete("extract", "sys", "user")
    assert called["n"] == 0  # the refusal happened BEFORE the network call


def test_budget_depletes_across_calls_then_blocks():
    # haiku 1000 in + 500 out = $0.0035 per call; budget for ~2 calls
    c = client(daily_budget_usd=0.006)
    c.complete("extract", "s", "u")
    c.complete("extract", "s", "u")
    assert c.remaining_usd == 0.0
    with pytest.raises(BudgetExceeded):
        c.complete("extract", "s", "u")
    assert c.calls == 2


def test_remaining_never_negative():
    c = client(daily_budget_usd=0.001)
    c.complete("extract", "s", "u")
    assert c.remaining_usd == 0.0


# ---------- routing ----------

def test_routes_pick_different_models():
    seen = {}

    def spy(model, system, user, max_tokens):
        seen[model] = seen.get(model, 0) + 1
        return "ok", 10, 10

    c = ModelClient(transport=spy, daily_budget_usd=1.0)
    c.complete("extract", "s", "u")
    c.complete("verify", "s", "u")
    assert seen == {ROUTES["extract"]: 1, ROUTES["verify"]: 1}
    assert ROUTES["extract"] != ROUTES["verify"]  # cheap reads, expensive checks


def test_unknown_route_rejected():
    with pytest.raises(ValueError, match="Unknown route"):
        client().complete("summarize_the_universe", "s", "u")


# ---------- retries ----------

def test_retryable_error_is_retried_then_succeeds():
    t = flaky_transport(2, RuntimeError("529 overloaded_error"))
    c = ModelClient(transport=t, daily_budget_usd=1.0, _sleep=lambda s: None)
    reply = c.complete("extract", "s", "u")
    assert reply.attempts == 3
    assert t.state["n"] == 3


def test_retries_exhausted_raises_upstream():
    t = flaky_transport(99, RuntimeError("503 service unavailable"))
    c = ModelClient(transport=t, daily_budget_usd=1.0, _sleep=lambda s: None, max_attempts=3)
    with pytest.raises(UpstreamError):
        c.complete("extract", "s", "u")
    assert t.state["n"] == 3


def test_non_retryable_error_fails_immediately():
    t = flaky_transport(99, ValueError("400 invalid_request: bad schema"))
    c = ModelClient(transport=t, daily_budget_usd=1.0, _sleep=lambda s: None)
    with pytest.raises(UpstreamError):
        c.complete("extract", "s", "u")
    assert t.state["n"] == 1  # no money burned on retries that cannot succeed


def test_backoff_is_exponential():
    slept: list[float] = []
    t = flaky_transport(2, RuntimeError("429 rate_limit"))
    c = ModelClient(transport=t, daily_budget_usd=1.0, _sleep=slept.append, base_backoff_s=0.5)
    c.complete("extract", "s", "u")
    assert slept == [0.5, 1.0]


@pytest.mark.parametrize(
    "message,expected",
    [
        ("429 rate_limit exceeded", True),
        ("529 overloaded", True),
        ("503 upstream", True),
        ("connection timeout", True),
        ("400 invalid_request", False),
        ("401 authentication_error", False),
        ("nonsense", False),
    ],
)
def test_retryable_classification(message, expected):
    assert _is_retryable(RuntimeError(message)) is expected


# ---------- tracing ----------

def test_trace_line_written_per_call(tmp_path):
    trace = tmp_path / "traces" / "calls.jsonl"
    c = client(daily_budget_usd=1.0, trace_path=trace)
    c.complete("extract", "s", "u")
    c.complete("verify", "s", "u")
    lines = [json.loads(x) for x in trace.read_text().strip().splitlines()]
    assert len(lines) == 2
    first = lines[0]
    assert first["route"] == "extract"
    assert first["model"] == ROUTES["extract"]
    assert first["input_tokens"] == 1000
    assert first["cost_usd"] > 0
    assert first["latency_ms"] >= 0
    assert lines[1]["cumulative_cost_usd"] > first["cumulative_cost_usd"]


def test_no_trace_path_means_no_file(tmp_path):
    c = client(daily_budget_usd=1.0, trace_path=None)
    c.complete("extract", "s", "u")
    assert list(tmp_path.iterdir()) == []


def test_failed_call_is_not_billed():
    t = flaky_transport(99, RuntimeError("503"))
    c = ModelClient(transport=t, daily_budget_usd=1.0, _sleep=lambda s: None)
    with pytest.raises(UpstreamError):
        c.complete("extract", "s", "u")
    assert c.spent_usd == 0.0 and c.calls == 0
