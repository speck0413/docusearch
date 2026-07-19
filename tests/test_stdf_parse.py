"""STDF v4 file parse (R-STDF-1): a synthetic file → per-test records with conditions attached,
proving the DTR COND engine end to end through real pystdf parsing."""

from __future__ import annotations

from pathlib import Path

from harness.stdf_synth import sample_conditioned_run

from docusearch import stdf


def test_parse_stdf_attaches_conditions_per_part(tmp_path: Path) -> None:
    data = sample_conditioned_run(tmp_path / "s.stdf").read_bytes()
    run = stdf.parse_stdf_tests(data, scope="part")

    assert run.lot_id == "LOTA" and run.job_nam == "PROG1"
    assert len(run.tests) == 4  # VMIN, VMAX, IDDQ (part 1) + VMIN (part 2)

    vmin1, vmax1, iddq1, vmin2 = run.tests
    # part 1: COND corner+temp set, then COND_OFF corner drops corner for IDDQ
    assert vmin1.test_txt == "VMIN_core" and vmin1.conditions == {"corner": "slow", "temp": "125C"}
    assert vmax1.conditions == {"corner": "slow", "temp": "125C"}
    assert iddq1.conditions == {"temp": "125C"}  # corner cleared, temp remains
    # part 2: per-part scope reset conditions at the part-1 PRR → clean slate
    assert vmin2.conditions == {} and vmin2.passed is False
    assert vmin1.part_id == "1" and vmin2.part_id == "2"


def test_parse_stdf_run_scope_persists_across_parts(tmp_path: Path) -> None:
    data = sample_conditioned_run(tmp_path / "s.stdf").read_bytes()
    run = stdf.parse_stdf_tests(data, scope="run")
    vmin2 = run.tests[3]
    # run scope: conditions persist across the part boundary; temp (still set) carries into part 2
    assert vmin2.conditions == {"temp": "125C"}


def test_stdf_test_text_is_searchable(tmp_path: Path) -> None:
    data = sample_conditioned_run(tmp_path / "s.stdf").read_bytes()
    run = stdf.parse_stdf_tests(data)
    text = stdf.stdf_test_text(run.tests[0])
    assert "VMIN_core" in text and "COND corner=slow" in text and "PASS" in text
