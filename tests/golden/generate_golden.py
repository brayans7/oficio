"""Regenerates the golden quote fixtures — run ONLY on a deliberate pricebook/engine change.

The 30 cases cover: single lines, multi-line remodels, fractional discrete units,
margin clamping, custom margins, unknown items, budget flags and empty specs.
Frozen output = the contract. CI reproduces every case to the cent; any diff is a
regression until a human re-freezes on purpose.

Usage:  python tests/golden/generate_golden.py
"""
from __future__ import annotations

import json
from pathlib import Path

from oficio.core.engine import compute_quote
from oficio.core.pricebook import load_pricebook
from oficio.core.schemas import JobSpec, LineItemRequest


def li(item_id: str, qty: float, sq: str) -> dict:
    return LineItemRequest(item_id=item_id, qty=qty, source_quote=sq).model_dump()


CASES: list[dict] = [
    # --- singles, continuous units (1-8) ---
    {"line_items": [li("wall_plastering", 10, "plaster 10 m2 of walls")]},
    {"line_items": [li("wall_skim_coat", 25, "skim coat the living room")]},
    {"line_items": [li("painting_3_coats", 48, "paint everything, 3 coats")]},
    {"line_items": [li("tile_install_mortar_bed", 18, "tile the floor over mortar")]},
    {"line_items": [li("masonry_wall_build", 14.5, "build the dividing wall")]},
    {"line_items": [li("plastering_new_build_facade", 60, "plaster the facade")]},
    {"line_items": [li("plumbing_full_unit", 1, "redo all the plumbing")]},
    {"line_items": [li("electrical_full_unit", 1, "rewire the whole unit")]},
    # --- singles, discrete units incl. fractional requests (9-14) ---
    {"line_items": [li("cement_bag", 3.2, "cement for the render")]},
    {"line_items": [li("tile_adhesive_25kg", 5.5, "adhesive for the tiles")]},
    {"line_items": [li("paint_premium_5gal", 1.1, "premium paint")]},
    {"line_items": [li("clay_block_10cm", 420.3, "blocks for the wall")]},
    {"line_items": [li("toilet", 2, "two new toilets")]},
    {"line_items": [li("glass_shower_sliding", 1, "sliding glass shower door")]},
    # --- multi-line jobs (15-21) ---
    {"space_type": "kitchen", "area_m2": 12, "line_items": [
        li("wall_plastering", 34, "plaster the kitchen walls"),
        li("wall_skim_coat", 34, "smooth finish"),
        li("painting_3_coats", 46, "paint walls and ceiling"),
        li("tile_install_thinset", 12, "new floor tile"),
        li("ceramic_tile_standard", 12, "new floor tile"),
        li("tile_adhesive_25kg", 2.4, "new floor tile"),
        li("kitchen_mixer_tap", 1, "replace the tap")]},
    {"space_type": "bathroom", "area_m2": 5, "line_items": [
        li("tile_install_mortar_bed", 22, "tile floor and walls"),
        li("ceramic_tile_standard", 22, "tile floor and walls"),
        li("toilet", 1, "new toilet"),
        li("bathroom_sink", 1, "new sink"),
        li("sink_faucet", 1, "new faucet"),
        li("shower_mixer", 1, "new shower mixer"),
        li("bath_accessory_set", 1, "towel bars and accessories")]},
    {"space_type": "bedroom", "area_m2": 16, "line_items": [
        li("wall_skim_coat", 42, "smooth the walls"),
        li("painting_3_coats", 58, "repaint bedroom"),
        li("paint_standard_5gal", 0.8, "standard paint is fine")]},
    {"space_type": "full_unit", "line_items": [
        li("demolition_job", 1, "demolish the old kitchen"),
        li("debris_removal", 1, "haul away the debris"),
        li("material_hauling_in", 1, "bring materials up"),
        li("post_construction_cleaning", 1, "leave it clean")]},
    {"space_type": "kitchen", "line_items": [
        li("kitchen_countertop", 1, "new countertop"),
        li("kitchen_sink_inox", 1, "stainless sink"),
        li("kitchen_mixer_tap", 1, "mixer tap"),
        li("range_hood", 1, "range hood")]},
    {"space_type": "gas_works", "line_items": [
        li("gas_heater_install", 1, "install the gas heater"),
        li("water_heater", 1, "the heater itself"),
        li("gas_valve", 1, "gas valve"),
        li("gas_valve_install", 1, "install the valve"),
        li("gas_certification", 1, "get it certified")]},
    {"space_type": "new_build", "line_items": [
        li("plastering_new_build_interior", 220, "plaster interior walls"),
        li("plastering_new_build_bath", 36, "plaster the bathrooms"),
        li("tile_install_bath_new_build", 36, "tile the bathrooms"),
        li("cement_bag", 48, "cement"),
        li("render_sand_hauled_m3", 6.5, "sand delivered")]},
    # --- margin behaviour (22-25) ---
    {"line_items": [li("toilet", 1, "a toilet")], "_margin": 0.35},
    {"line_items": [li("toilet", 1, "a toilet")], "_margin": 0.20},
    {"line_items": [li("toilet", 1, "a toilet")], "_margin": 0.05},   # below floor -> clamped
    {"line_items": [li("wall_plastering", 100, "big job")], "_margin": 0.30},
    # --- fail-closed & edge cases (26-30) ---
    {"line_items": [li("gold_plated_jacuzzi", 1, "I want a gold jacuzzi")]},
    {"line_items": [li("toilet", 1, "a toilet"), li("smart_mirror", 1, "a smart mirror")]},
    {"line_items": [li("toilet", 1, "a toilet")], "missing_info": ["tile grade not specified"]},
    {"line_items": [li("water_heater", 1, "water heater")], "client_budget": 50.0},
    {"line_items": []},
]


def main() -> None:
    pb = load_pricebook()
    golden = []
    for i, case in enumerate(CASES, 1):
        margin = case.pop("_margin", None)
        spec = JobSpec.model_validate(case)
        result = compute_quote(spec, pb, margin=margin)
        golden.append({"case": i, "margin": margin, "spec": spec.model_dump(),
                       "expected": result.model_dump()})
    out = Path(__file__).parent / "golden_quotes.json"
    out.write_text(json.dumps({"pricebook_version": pb.version, "cases": golden}, indent=2) + "\n")
    print(f"Froze {len(golden)} golden quotes -> {out}")


if __name__ == "__main__":
    main()
