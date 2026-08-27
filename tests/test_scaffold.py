"""T1/T2 smoke tests: the package imports and the anonymized price book is valid."""
import pytest

from oficio.core.pricebook import UnknownItemError, load_pricebook


def test_package_imports():
    import oficio

    assert oficio.__version__


def test_pricebook_loads_and_validates():
    pb = load_pricebook()
    assert pb.version == "1.0.0"
    assert pb.currency == "USD"
    assert len(pb.items) >= 50
    assert pb.rules.margin_floor < pb.rules.default_margin


def test_pricebook_has_labor_and_materials():
    pb = load_pricebook()
    kinds = {i.kind for i in pb.items}
    assert kinds == {"labor", "material"}
    assert all(i.labor_cost > 0 or i.material_cost > 0 for i in pb.items)


def test_unknown_item_fails_closed():
    pb = load_pricebook()
    with pytest.raises(UnknownItemError):
        pb.get("gold_plated_jacuzzi")


def test_no_sensitive_terms_in_pricebook():
    """Grep-gate: the public price book must never leak business provenance."""
    from oficio.core.pricebook import DEFAULT_PRICEBOOK_PATH

    text = DEFAULT_PRICEBOOK_PATH.read_text(encoding="utf-8").lower()
    for term in ["tuobra", "bogota", "medellin", "antioquia", "colombia", "cop"]:
        assert term not in text, f"sensitive term leaked: {term}"
