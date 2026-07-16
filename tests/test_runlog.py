"""Async JSONL logging tests (§13, R-LOG-1..3).

The logger must (1) write structured JSONL records carrying run_id/ts/event/fields,
(2) never block the caller on disk I/O — a background thread does the writing,
(3) flush everything on close / process exit, and (4) gate volume by level.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from docusearch import runlog


def _read_records(log_dir: Path) -> list[dict[str, object]]:
    files = sorted(log_dir.glob("*.jsonl"))
    records: list[dict[str, object]] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def test_run_id_is_stable_and_nonempty() -> None:
    assert isinstance(runlog.RUN_ID, str)
    assert runlog.RUN_ID
    assert runlog.RUN_ID == runlog.RUN_ID  # module-level constant, not regenerated


def test_log_writes_structured_jsonl(tmp_path: Path) -> None:
    with runlog.RunLog(tmp_path) as rl:
        rl.log("ingest.stage", stage="filter", files=10, doc_ids=[1, 2, 3], took_ms=4.2)
    records = _read_records(tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["event"] == "ingest.stage"
    assert rec["stage"] == "filter"
    assert rec["files"] == 10
    assert rec["doc_ids"] == [1, 2, 3]
    assert rec["run_id"] == rl.run_id
    assert "ts" in rec and rec["level"] == "info"


def test_log_file_is_dated(tmp_path: Path) -> None:
    with runlog.RunLog(tmp_path) as rl:
        rl.log("hello")
    expected = tmp_path / f"{date.today().isoformat()}.jsonl"
    assert expected.exists()


def test_level_gating_drops_below_threshold(tmp_path: Path) -> None:
    with runlog.RunLog(tmp_path, level="warning") as rl:
        rl.log("noisy", level="debug")
        rl.log("informational", level="info")
        rl.log("important", level="warning")
    records = _read_records(tmp_path)
    events = [r["event"] for r in records]
    assert events == ["important"]


def test_records_preserve_order(tmp_path: Path) -> None:
    with runlog.RunLog(tmp_path) as rl:
        for i in range(5):
            rl.log("step", i=i)
    records = _read_records(tmp_path)
    assert [r["i"] for r in records] == [0, 1, 2, 3, 4]


def test_log_does_not_block_and_flush_drains(tmp_path: Path) -> None:
    rl = runlog.RunLog(tmp_path)
    for i in range(100):
        rl.log("burst", i=i)  # returns immediately; writer drains in background
    rl.flush()
    assert len(_read_records(tmp_path)) == 100
    rl.close()


def test_disabled_logger_writes_nothing(tmp_path: Path) -> None:
    with runlog.RunLog(tmp_path, enabled=False) as rl:
        rl.log("ignored", level="warning")
    assert _read_records(tmp_path) == []
    assert list(tmp_path.glob("*.jsonl")) == []


def test_custom_run_id(tmp_path: Path) -> None:
    with runlog.RunLog(tmp_path, run_id="RUN-XYZ") as rl:
        rl.log("event")
    assert _read_records(tmp_path)[0]["run_id"] == "RUN-XYZ"


def test_module_level_configure_and_shutdown(tmp_path: Path) -> None:
    runlog.configure(tmp_path, level="info")
    try:
        runlog.log("cli.init", wrote="docusearch.yaml")
        runlog.flush()
        records = _read_records(tmp_path)
        assert records and records[0]["event"] == "cli.init"
    finally:
        runlog.shutdown()
    # After shutdown, module-level log is a no-op (no active logger).
    runlog.log("dropped")


def test_new_run_id_is_unique() -> None:
    assert runlog.new_run_id() != runlog.new_run_id()
