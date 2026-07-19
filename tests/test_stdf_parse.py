"""STDF v4 file parse (R-STDF-1): a synthetic file → per-test records with conditions attached,
proving the DTR COND engine end to end through real pystdf parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
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


def test_parse_mpr_expands_pins_and_ftr_functional() -> None:
    from harness.stdf_synth import StdfBuilder

    b = StdfBuilder().far().mir(lot_id="L", test_cod="WS1")
    b.pir()
    b.ptr(1000, "VMIN", 0.72, lo=0.70, hi=0.85, units="V")
    b.mpr(2000, "IDDQ_pins", [1e-6, 2e-6, 1.5e-6, 3e-6], lo=0.0, hi=5e-6, units="A")
    b.ftr(3000, "SCAN_pass")
    b.ftr(3001, "SCAN_fail", fail=True)
    b.prr(part_id="1", hard_bin=1)
    b.mrr()
    run = stdf.parse_stdf_tests(b.to_bytes())

    by_rec: dict[str, list[stdf.StdfTest]] = {}
    for t in run.tests:
        by_rec.setdefault(t.rec_type, []).append(t)

    assert len(by_rec["PTR"]) == 1 and by_rec["PTR"][0].test_txt == "VMIN"
    # MPR → one analyzable sub-test per pin, each carrying the scalar limits
    mpr = by_rec["MPR"]
    assert len(mpr) == 4 and [t.pin for t in mpr] == [0, 1, 2, 3]
    assert mpr[0].test_txt == "IDDQ_pins[0]" and mpr[0].hi_limit is not None
    assert mpr[3].result is not None
    # FTR → functional, no numeric result, pass/fail from the flag
    ftr = {t.test_txt: t for t in by_rec["FTR"]}
    assert ftr["SCAN_pass"].result is None and ftr["SCAN_pass"].passed
    assert not ftr["SCAN_fail"].passed


def test_optional_fields_inherited_from_first_record_by_test_number() -> None:
    """STDF v4: a test's name/limits/units live on its FIRST record; later records omit them or flag
    them invalid and inherit by test number (R-STDF-1)."""
    from harness.stdf_synth import StdfBuilder

    b = StdfBuilder().far().mir(lot_id="L", test_cod="WS1")
    # part 1 — first record for each test carries the static fields
    b.pir()
    b.ptr(1000, "VMIN", 0.71, lo=0.70, hi=0.85, units="V")
    b.ptr(1001, "IREF", 5.0, lo=1.0, hi=9.0, units="mA")
    b.prr(part_id="1", hard_bin=1)
    # part 2 — later records omit / flag the static fields
    b.pir()
    b.ptr(1000, "", 0.72)                              # bare: no name, no limits → inherit all
    b.ptr(1001, "", 6.0, lo=999.0, hi=9.0, opt_flag=0x10)  # OPT_FLAG bit4: LO_LIMIT invalid → inherit
    b.prr(part_id="2", hard_bin=1)
    # part 3 — explicit "no low limit"
    b.pir()
    b.ptr(1001, "", 7.0, lo=999.0, hi=9.0, opt_flag=0x40)  # OPT_FLAG bit6: no low limit at all
    b.prr(part_id="3", hard_bin=1)
    b.mrr()
    run = stdf.parse_stdf_tests(b.to_bytes())

    vmin = [t for t in run.tests if t.test_num == 1000]
    iref = [t for t in run.tests if t.test_num == 1001]
    # bare later VMIN inherits name + both limits + units from the first record (float32 approx)
    assert vmin[1].test_txt == "VMIN" and vmin[1].units == "V"
    assert vmin[1].lo_limit == pytest.approx(0.70, abs=1e-5)
    assert vmin[1].hi_limit == pytest.approx(0.85, abs=1e-5)
    # IREF part 2: name inherited; LO invalid→inherited 1.0 (NOT the bogus 999); HI kept
    assert iref[1].test_txt == "IREF"
    assert iref[1].lo_limit == pytest.approx(1.0) and iref[1].hi_limit == pytest.approx(9.0)
    # IREF part 3: bit6 → genuinely no low limit
    assert iref[2].lo_limit is None and iref[2].hi_limit == pytest.approx(9.0)
