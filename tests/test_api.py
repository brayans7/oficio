"""T12 — the HTTP surface.

The deterministic endpoints are tested for real; /chat is tested for the two
things that matter without a key: it refuses clearly when extraction is
unavailable, and it surfaces the evidence panel when it is.
"""
import json

import pytest
from fastapi.testclient import TestClient

from oficio.service import api


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(api, "TRACE_PATH", tmp_path / "calls.jsonl")
    return TestClient(api.app)


def spec(item_id="wall_plastering", qty=10, quote="plaster the walls"):
    return {"spec": {"line_items": [{"item_id": item_id, "qty": qty, "source_quote": quote}]}}


# ---------- health & catalog ----------

def test_health_reports_pricebook_and_capability(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["catalog_items"] >= 50
    assert body["extraction_available"] is False  # no key in this environment


def test_catalog_endpoint(client):
    body = client.get("/catalog").json()
    assert len(body["items"]) >= 50
    assert body["currency"] == "USD"


def test_catalog_category_filter(client):
    body = client.get("/catalog?category=plumbing").json()
    assert all(i["category"] == "plumbing" for i in body["items"])


# ---------- deterministic quoting: works with no key ----------

def test_quote_without_api_key(client):
    body = client.post("/quote", json=spec()).json()
    assert body["quote"]["status"] == "ok"
    assert body["quote"]["total"] > 0
    assert body["explanation"]["steps"]


def test_quote_is_reproducible(client):
    a = client.post("/quote", json=spec()).json()["quote"]
    b = client.post("/quote", json=spec()).json()["quote"]
    assert a["quote_id"] == b["quote_id"] and a["total"] == b["total"]


def test_quote_unknown_item_needs_info(client):
    body = client.post("/quote", json=spec("gold_jacuzzi", 1, "a gold jacuzzi")).json()
    assert body["quote"]["status"] == "needs_info"
    assert body["quote"]["total"] == 0.0


def test_quote_rejects_line_without_evidence(client):
    payload = {"spec": {"line_items": [{"item_id": "toilet", "qty": 1}]}}
    assert client.post("/quote", json=payload).status_code == 422


def test_quote_honours_margin_floor(client):
    payload = spec() | {"margin": 0.01}
    body = client.post("/quote", json=payload).json()
    assert body["quote"]["margin_applied"] == 0.20


# ---------- chat: fails closed without a key ----------

def test_chat_refuses_without_key(client):
    response = client.post("/chat", json={"message": "I need 10 m2 of plastering"})
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
    assert "/quote still prices" in response.json()["detail"]


def test_chat_returns_evidence_panel(monkeypatch, tmp_path):
    """With a key and a scripted model, /chat exposes exactly what was seen."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(api, "TRACE_PATH", tmp_path / "calls.jsonl")
    payload = {"line_items": [{"item_id": "wall_plastering", "qty": 10,
                               "source_quote": "10 m2 of plastering"}]}
    monkeypatch.setattr(api, "_client", lambda: api.ModelClient(
        transport=lambda m, s, u, mt: (json.dumps(payload), 400, 120),
        daily_budget_usd=1.0, trace_path=tmp_path / "calls.jsonl"))

    body = TestClient(api.app).post("/chat", json={"message": "I need 10 m2 of plastering"}).json()
    saw = body["what_the_agent_saw"]
    assert saw["lines"][0]["evidence"] == "10 m2 of plastering"
    assert saw["dropped"] == []
    assert body["quote"]["total"] > 0
    assert body["cost_usd"] > 0


def test_chat_reports_dropped_lines(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(api, "TRACE_PATH", tmp_path / "calls.jsonl")
    payload = {"line_items": [{"item_id": "invented_item", "qty": 1,
                               "source_quote": "10 m2 of plastering"}]}
    monkeypatch.setattr(api, "_client", lambda: api.ModelClient(
        transport=lambda m, s, u, mt: (json.dumps(payload), 400, 120), daily_budget_usd=1.0))

    body = TestClient(api.app).post("/chat", json={"message": "I need 10 m2 of plastering"}).json()
    assert body["what_the_agent_saw"]["dropped"]
    assert body["quote"]["lines"] == []


# ---------- traces ----------

def test_traces_empty_by_default(client):
    assert client.get("/traces/summary").json() == {"calls": 0, "total_cost_usd": 0.0, "by_model": {}}


def test_traces_summarize_spend(client, monkeypatch, tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        {"model": "claude-haiku-4-5", "input_tokens": 1000, "output_tokens": 500,
         "cost_usd": 0.0035, "latency_ms": 900},
        {"model": "claude-haiku-4-5", "input_tokens": 500, "output_tokens": 200,
         "cost_usd": 0.0015, "latency_ms": 700},
    ]) + "\n")
    monkeypatch.setattr(api, "TRACE_PATH", path)
    body = TestClient(api.app).get("/traces/summary").json()
    assert body["calls"] == 2
    assert body["total_cost_usd"] == 0.005
    assert body["by_model"]["claude-haiku-4-5"]["calls"] == 2
    assert body["avg_latency_ms"] == 800.0


# ---------- the page itself ----------

def test_index_is_self_contained(client):
    html = client.get("/").text
    assert "Oficio" in html
    assert "what the agent saw" in html.lower()
    assert "http://" not in html.replace("http://www.w3.org", "")  # no external assets
