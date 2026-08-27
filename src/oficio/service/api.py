"""HTTP surface: the demo, and the endpoints behind it.

`/quote` is the deterministic path — no model, no key, no cost. It is what makes
the demo usable by anyone who clones the repo: the engine works on its own.
`/chat` adds the extraction step and therefore needs a key; without one it says
so plainly instead of degrading into a guess.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..agent.client import BudgetExceeded, ModelClient, UpstreamError, anthropic_transport
from ..agent.extract import ExtractionError, extract_job_spec
from ..core.engine import compute_quote
from ..core.pricebook import load_pricebook
from ..core.schemas import JobSpec
from .mcp_tools import explain_quote, get_catalog

TRACE_PATH = Path(os.getenv("OFICIO_TRACE_PATH", "traces/calls.jsonl"))

app = FastAPI(
    title="Oficio",
    description="The verification-first agent engine for real-world trades.",
    version="0.5.0",
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    margin: float | None = None


class QuoteRequest(BaseModel):
    spec: JobSpec
    margin: float | None = None


def _client() -> ModelClient:
    return ModelClient(
        transport=anthropic_transport(),
        daily_budget_usd=float(os.getenv("DAILY_COST_LIMIT", "5.0")),
        trace_path=TRACE_PATH,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    pb = load_pricebook()
    return {
        "status": "ok",
        "pricebook_version": pb.version,
        "catalog_items": len(pb.items),
        "extraction_available": bool(os.getenv("ANTHROPIC_API_KEY")),
    }


@app.get("/catalog")
def catalog(category: str | None = None) -> dict[str, Any]:
    return get_catalog(category=category)


@app.post("/quote")
def quote(request: QuoteRequest) -> dict[str, Any]:
    """Deterministic pricing. Works with no API key: the engine is the product."""
    result = compute_quote(request.spec, load_pricebook(), margin=request.margin)
    payload = result.model_dump()
    return {"quote": payload, "explanation": explain_quote(payload)}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """Conversation -> evidence-bound extraction -> deterministic quote."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not set, so extraction is unavailable. "
                   "POST /quote still prices a JobSpec deterministically.",
        )
    pricebook = load_pricebook()
    try:
        extracted = extract_job_spec(request.message, pricebook, _client())
    except BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except (ExtractionError, UpstreamError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    result = compute_quote(extracted.spec, pricebook, margin=request.margin)
    payload = result.model_dump()
    return {
        "quote": payload,
        "explanation": explain_quote(payload),
        "what_the_agent_saw": {
            "lines": [{"item_id": ln.item_id, "qty": ln.qty, "evidence": ln.source_quote}
                      for ln in extracted.spec.line_items],
            "dropped": extracted.dropped,
            "questions": extracted.spec.missing_info,
        },
        "cost_usd": extracted.cost_usd,
        "model_calls": extracted.calls,
    }


@app.get("/traces/summary")
def traces_summary() -> dict[str, Any]:
    """What the system has spent, read back from the trace log."""
    if not TRACE_PATH.exists():
        return {"calls": 0, "total_cost_usd": 0.0, "by_model": {}}
    records = [json.loads(line) for line in TRACE_PATH.read_text().splitlines() if line.strip()]
    by_model: dict[str, dict[str, float]] = {}
    for record in records:
        bucket = by_model.setdefault(record["model"], {"calls": 0, "cost_usd": 0.0, "tokens": 0})
        bucket["calls"] += 1
        bucket["cost_usd"] = round(bucket["cost_usd"] + record["cost_usd"], 6)
        bucket["tokens"] += record["input_tokens"] + record["output_tokens"]
    return {
        "calls": len(records),
        "total_cost_usd": round(sum(r["cost_usd"] for r in records), 6),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in records) / len(records), 1)
        if records else 0.0,
        "by_model": by_model,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
