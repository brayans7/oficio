"""AC-1 — the engine reproduces all 30 frozen golden quotes to the cent.

The fixtures were generated once (tests/golden/generate_golden.py), then reviewed
against the source business's own budgeting tool. Any diff here is a pricing
regression until a human deliberately re-freezes the fixtures.
"""
import json
from pathlib import Path

import pytest

from oficio.core.engine import compute_quote
from oficio.core.pricebook import load_pricebook
from oficio.core.schemas import JobSpec

GOLDEN_PATH = Path(__file__).parent / "golden" / "golden_quotes.json"
GOLDEN = json.loads(GOLDEN_PATH.read_text())


def test_golden_file_has_30_cases():
    assert len(GOLDEN["cases"]) == 30


def test_golden_matches_current_pricebook_version():
    assert load_pricebook().version == GOLDEN["pricebook_version"]


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: f"case{c['case']:02d}")
def test_golden_quote_reproduced_to_the_cent(case):
    pb = load_pricebook()
    spec = JobSpec.model_validate(case["spec"])
    result = compute_quote(spec, pb, margin=case["margin"])
    assert result.model_dump() == case["expected"]
