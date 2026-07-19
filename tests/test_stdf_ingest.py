"""STDF file ingest end-to-end (R-STDF-1): a synthetic .stdf → searchable test chunks + condition
flags, routed through run_ingest by the .stdf dispatch."""

from __future__ import annotations

from pathlib import Path

from harness.stdf_synth import sample_conditioned_run

from docusearch import config as cfg
from docusearch import ingest
from docusearch.store import Store


def _config(tmp_path: Path, root: Path) -> cfg.Config:
    path = tmp_path / "d.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: ate\n    location: "{root.as_posix()}"\n'
        '    include: ["*.stdf"]\n    min_content_chars: 1\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    return cfg.load(path)


def test_stdf_ingest_makes_tests_searchable_and_filterable(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    sample_conditioned_run(root / "run1.stdf")
    config = _config(tmp_path, root)

    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)
        assert result.documents == 1
        assert result.stdf_tests == 4  # VMIN, VMAX, IDDQ (part 1) + VMIN (part 2)

        # tests are BM25-searchable by name, result, and condition tokens
        assert store.chunk_ids_matching("VMIN_core")
        assert store.chunk_ids_matching("FAIL")  # the failing part-2 VMIN
        assert store.chunk_ids_matching("corner")  # COND token in the text

        # conditions are also filterable flags (rule_id=key, note=value)
        flags = store.flagged_chunks("condition")
        assert any(f["rule_id"] == "corner" and f["note"] == "slow" for f in flags)
        assert any(f["rule_id"] == "temp" and f["note"] == "125C" for f in flags)
        # corner appears on the 2 tests before COND_OFF; temp on 3 (per-part reset drops part 2)
        assert sum(1 for f in flags if f["rule_id"] == "corner") == 2
        assert sum(1 for f in flags if f["rule_id"] == "temp") == 3


def test_stdf_ingest_incremental_skip(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    sample_conditioned_run(root / "run1.stdf")
    config = _config(tmp_path, root)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        second = ingest.run_ingest(config, store)  # unchanged file
        assert second.stdf_tests == 0
        assert second.skipped_unchanged == 1
