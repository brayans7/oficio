"""Eval case schema and loaders.

A case is a conversation plus the labels a correct extraction must produce.
Labels are deliberately minimal — the ids that must appear, the quantities that
must be right, and whether the request is answerable at all. Anything richer
would encode the implementation's opinions into its own exam.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "evals"
EXTRACTION_PATH = DATA_DIR / "extraction.jsonl"
SECURITY_PATH = DATA_DIR / "security.jsonl"


class ExpectedLine(BaseModel):
    item_id: str
    qty: float


class ExtractionCase(BaseModel):
    id: str
    kind: str = Field(pattern="^(complete|incomplete|noisy)$")
    conversation: str
    expected_lines: list[ExpectedLine] = Field(default_factory=list)
    expects_missing_info: bool = False
    note: str = ""


class SecurityCase(BaseModel):
    id: str
    attack: str
    conversation: str
    #: An id the attack tries to smuggle in, or None when it attacks a rule
    #: rather than the catalog (e.g. "quote it for free", "reveal your prompt").
    forbidden_item_id: str | None = None
    must_not_appear: list[str] = Field(default_factory=list)
    note: str = ""


def _load(path: Path, model: type[BaseModel]) -> list:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run: python -m oficio.evals.build_dataset")
    return [model.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def load_extraction_cases(path: Path = EXTRACTION_PATH) -> list[ExtractionCase]:
    return _load(path, ExtractionCase)


def load_security_cases(path: Path = SECURITY_PATH) -> list[SecurityCase]:
    return _load(path, SecurityCase)


def dump_cases(cases: list[BaseModel], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(c.model_dump()) for c in cases) + "\n")
