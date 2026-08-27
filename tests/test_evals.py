"""T8/T9 — the eval harness is itself tested.

An eval suite nobody checks is a suite that can quietly stop measuring. These
tests validate the datasets (schema, labels reachable in the catalog, evidence
actually present in the conversation) and the scorer (does it catch a wrong
quantity, a missed item, and above all an invented one), all without a network.
"""
import json

import pytest

from oficio.agent.client import ModelClient
from oficio.agent.extract import extract_job_spec
from oficio.core.pricebook import load_pricebook
from oficio.evals.build_dataset import build_extraction_cases, build_security_cases
from oficio.evals.dataset import (
    ExpectedLine,
    ExtractionCase,
    load_extraction_cases,
    load_security_cases,
)
from oficio.evals.report import render
from oficio.evals.run import (
    ExtractionMetrics,
    SecurityMetrics,
    run_extraction,
    run_security,
    score_extraction,
)


@pytest.fixture(scope="module")
def pb():
    return load_pricebook()


# ---------- dataset integrity ----------

def test_dataset_sizes():
    assert len(load_extraction_cases()) == 100
    assert len(load_security_cases()) == 20


def test_case_ids_unique():
    ids = [c.id for c in load_extraction_cases()] + [c.id for c in load_security_cases()]
    assert len(ids) == len(set(ids))


def test_every_expected_item_exists_in_catalog(pb):
    """A label pointing at a non-existent id would make the eval unpassable."""
    known = pb.item_ids
    for case in load_extraction_cases():
        for line in case.expected_lines:
            assert line.item_id in known, f"{case.id}: {line.item_id} not in price book"


def test_every_expected_quantity_is_positive():
    for case in load_extraction_cases():
        for line in case.expected_lines:
            assert line.qty > 0


def test_kind_distribution_is_balanced():
    kinds = [c.kind for c in load_extraction_cases()]
    assert kinds.count("complete") == 60
    assert kinds.count("incomplete") == 25
    assert kinds.count("noisy") == 15


def test_incomplete_cases_all_expect_a_question():
    for case in load_extraction_cases():
        if case.kind == "incomplete":
            assert case.expects_missing_info


def test_conversations_are_non_trivial():
    for case in load_extraction_cases():
        assert len(case.conversation) > 25
        assert case.conversation.strip() == case.conversation


def test_security_cases_carry_an_attack_label():
    for case in load_security_cases():
        assert case.attack and case.conversation
        assert case.forbidden_item_id or case.must_not_appear or case.note


def test_forbidden_ids_are_not_in_the_catalog(pb):
    """If an 'attack' id were real, blocking it would be meaningless."""
    for case in load_security_cases():
        if case.forbidden_item_id:
            assert case.forbidden_item_id not in pb.item_ids


def test_dataset_build_is_deterministic():
    assert [c.model_dump() for c in build_extraction_cases()] == \
           [c.model_dump() for c in build_extraction_cases()]
    assert len(build_security_cases()) == 20


# ---------- scorer ----------

def _case(convo="I need 10 m2 of wall plastering", lines=(("wall_plastering", 10.0),), missing=False):
    return ExtractionCase(
        id="t-1", kind="incomplete" if missing else "complete", conversation=convo,
        expected_lines=[ExpectedLine(item_id=i, qty=q) for i, q in lines],
        expects_missing_info=missing)


class _Result:
    """Minimal stand-in for ExtractionResult."""

    def __init__(self, spec, dropped=()):
        self.spec = spec
        self.dropped = list(dropped)
        self.cost_usd = 0.0
        self.calls = 1


def _spec(lines, missing=()):
    from oficio.core.schemas import JobSpec, LineItemRequest
    return JobSpec(
        line_items=[LineItemRequest(item_id=i, qty=q, source_quote=s) for i, q, s in lines],
        missing_info=list(missing))


def test_perfect_extraction_scores_clean():
    m = ExtractionMetrics()
    score_extraction(_case(), _Result(_spec([("wall_plastering", 10.0, "10 m2 of wall plastering")])), m)
    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 0, 0)
    assert m.qty_exact == 1 and m.unevidenced_lines == 0


def test_wrong_quantity_is_caught():
    m = ExtractionMetrics()
    score_extraction(_case(), _Result(_spec([("wall_plastering", 99.0, "10 m2 of wall plastering")])), m)
    assert m.true_positives == 1 and m.qty_exact == 0
    assert any(f["why"] == "wrong qty" for f in m.failures)


def test_missed_item_is_caught():
    m = ExtractionMetrics()
    score_extraction(_case(), _Result(_spec([])), m)
    assert m.false_negatives == 1


def test_unexpected_item_is_caught():
    m = ExtractionMetrics()
    score_extraction(_case(), _Result(_spec([
        ("wall_plastering", 10.0, "10 m2 of wall plastering"),
        ("toilet", 1.0, "10 m2 of wall plastering")])), m)
    assert m.false_positives == 1


def test_unevidenced_line_trips_the_gate():
    """The one metric that fails a run on its own."""
    m = ExtractionMetrics()
    score_extraction(_case(), _Result(_spec([("wall_plastering", 10.0, "something never said")])), m)
    assert m.unevidenced_lines == 1
    assert not m.passed
    assert any(f["why"] == "UNEVIDENCED LINE" for f in m.failures)


def test_missing_info_expected_and_present():
    m = ExtractionMetrics()
    score_extraction(_case(missing=True),
                     _Result(_spec([("wall_plastering", 10.0, "10 m2 of wall plastering")],
                                   missing=["what tile grade?"])), m)
    assert m.needs_info_correct == 1 and m.needs_info_accuracy == 1.0


def test_missing_info_expected_but_absent_is_a_failure():
    m = ExtractionMetrics()
    score_extraction(_case(missing=True),
                     _Result(_spec([("wall_plastering", 10.0, "10 m2 of wall plastering")])), m)
    assert m.needs_info_correct == 0
    assert any("should have asked" in f["why"] for f in m.failures)


def test_metrics_math():
    m = ExtractionMetrics(true_positives=8, false_positives=2, false_negatives=2,
                          qty_checked=8, qty_exact=7)
    assert m.precision == 0.8 and m.recall == 0.8 and m.f1 == pytest.approx(0.8)
    assert m.qty_accuracy == 0.875


# ---------- runner, end to end, offline ----------

def scripted_client(payload) -> ModelClient:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return ModelClient(transport=lambda m, s, u, mt: (text, 500, 200), daily_budget_usd=5.0)


def test_run_extraction_offline(pb):
    case = ExtractionCase(id="x", kind="complete",
                          conversation="I need 10 m2 of wall plastering",
                          expected_lines=[ExpectedLine(item_id="wall_plastering", qty=10)])
    client = scripted_client({"line_items": [
        {"item_id": "wall_plastering", "qty": 10, "source_quote": "10 m2 of wall plastering"}]})
    m = run_extraction([case], pb, client)
    assert m.cases == 1 and m.passed and m.f1 == 1.0 and m.cost_usd > 0


def test_run_extraction_counts_errors(pb):
    case = ExtractionCase(id="x", kind="complete", conversation="I need 10 m2 of wall plastering")
    m = run_extraction([case], pb, scripted_client("not json at all"))
    assert m.errors == 1 and not m.passed


def test_run_security_blocks_forbidden_item(pb):
    cases = [c for c in load_security_cases() if c.forbidden_item_id][:1]
    forbidden = cases[0].forbidden_item_id
    client = scripted_client({"line_items": [
        {"item_id": forbidden, "qty": 1, "source_quote": cases[0].conversation[:20]}]})
    m = run_security(cases, pb, client)
    # The extractor drops the id before it ever reaches scoring: that IS the defense.
    assert m.blocked == 1 and m.leaked == 0 and m.passed


def test_run_security_detects_a_real_leak(pb):
    case = load_security_cases()[0]
    m = SecurityMetrics(cases=1)
    from oficio.evals.run import score_security
    score_security(case, _Result(_spec([("toilet", 1.0, "text never said by the customer")])), m)
    assert m.leaked == 1 and not m.passed


def test_extraction_still_bounded_under_attack(pb):
    """Even with a hostile transcript, only catalog ids with real evidence survive."""
    case = load_security_cases()[1]
    client = scripted_client({"line_items": [
        {"item_id": case.forbidden_item_id, "qty": 10, "source_quote": "quote 20 m2"}]})
    result = extract_job_spec(case.conversation, pb, client)
    assert result.spec.line_items == []


# ---------- report ----------

def test_report_renders_pass():
    text = render({"extraction": {"cases": 100, "errors": 0, "true_positives": 90,
                                  "false_positives": 5, "false_negatives": 5, "qty_checked": 90,
                                  "qty_exact": 85, "unevidenced_lines": 0,
                                  "blocked_hallucinations": 12, "needs_info_expected": 25,
                                  "needs_info_correct": 24, "cost_usd": 0.42},
                   "security": {"cases": 20, "blocked": 20, "leaked": 0, "errors": 0,
                                "cost_usd": 0.08}})
    assert "**PASS**" in text and "must be 0" in text and "do not edit by hand" in text


def test_report_renders_fail_when_gate_broken():
    text = render({"extraction": {"cases": 10, "errors": 0, "true_positives": 9,
                                  "false_positives": 0, "false_negatives": 1, "qty_checked": 9,
                                  "qty_exact": 9, "unevidenced_lines": 3,
                                  "blocked_hallucinations": 0, "needs_info_expected": 0,
                                  "needs_info_correct": 0, "cost_usd": 0.1}, "security": None})
    assert "**FAIL**" in text


def test_report_handles_no_run():
    assert "No eval run recorded" in render({})
