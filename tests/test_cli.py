"""CLI skeleton tests (R-CFG-2, R-LOG-1).

Phase 0 ships three commands: ``init`` (write the config template), ``ingest
--dry-run`` (preview the plan without touching an index), and ``gate`` (write a
Part A / Part B sign-off checklist). Each is a thin wrapper over the modules.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from docusearch import cli


def test_init_writes_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["init"])
    assert rc == 0
    assert (tmp_path / "docusearch.yaml").exists()
    assert "Wrote config" in capsys.readouterr().out


def test_init_does_not_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docusearch.yaml").write_text("mode: server\n", encoding="utf-8")
    rc = cli.main(["init"])
    assert rc == 0
    assert (tmp_path / "docusearch.yaml").read_text(encoding="utf-8") == "mode: server\n"
    assert "already exists" in capsys.readouterr().out


def test_init_force_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docusearch.yaml").write_text("mode: server\n", encoding="utf-8")
    assert cli.main(["init", "--force"]) == 0
    assert "docusearch configuration" in (tmp_path / "docusearch.yaml").read_text("utf-8")


def test_init_custom_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(["init", "--config", "custom.yaml"]) == 0
    assert (tmp_path / "custom.yaml").exists()


def test_ingest_dry_run_lists_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    cli.main(["init"])
    capsys.readouterr()
    rc = cli.main(["ingest", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "vendor-html" in out
    assert "include" in out
    assert "audience" in out


def _write_corpus_config(tmp_path: Path) -> None:
    """A small real corpus + a config that points at it (written as docusearch.yaml)."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "spi.html").write_text(
        "<body><h1>SPI</h1><p>The SPI timing nonce ZZQ42 configures the peripheral bus.</p></body>",
        encoding="utf-8",
    )
    (tmp_path / "docusearch.yaml").write_text(
        "paths:\n"
        '  staging_dir: "./staging"\n'
        '  db_path: "./catalog.db"\n'
        '  tmp_dir: "./tmp"\n'
        "sources:\n"
        '  - name: docs\n    location: "./docs"\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )


def test_ingest_real_writes_audit_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    _write_corpus_config(tmp_path)
    rc = cli.main(["ingest"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Ingested 1 docs" in out
    assert list((tmp_path / "tmp" / "reports").glob("ingest-audit-*.md"))


def test_search_cli_finds_needle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    _write_corpus_config(tmp_path)
    cli.main(["ingest"])
    capsys.readouterr()
    rc = cli.main(["search", "ZZQ42"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "D:" in out and "SPI" in out


def test_audit_cli_prints_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    _write_corpus_config(tmp_path)
    cli.main(["ingest"])
    capsys.readouterr()
    rc = cli.main(["audit"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "documents" in out and "Anomalies" in out


def test_show_cli_prints_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    _write_corpus_config(tmp_path)
    cli.main(["ingest"])
    capsys.readouterr()
    rc = cli.main(["show", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ZZQ42" in out


def test_show_cli_missing_doc_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    _write_corpus_config(tmp_path)
    cli.main(["ingest"])
    capsys.readouterr()
    assert cli.main(["show", "999"]) == 1


def test_search_batch_file_grades_goldens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    _write_corpus_config(tmp_path)  # spi.html contains the needle ZZQ42
    cli.main(["ingest"])
    capsys.readouterr()
    (tmp_path / "goldens.yaml").write_text(
        "- id: g1\n  query: ZZQ42\n  expect_docs: [spi.html]\n  notes: the needle\n"
        "- id: g2\n  query: absenttopicxyz\n  expect_docs: [nope.html]\n",
        encoding="utf-8",
    )
    rc = cli.main(["search", "--batch-file", "goldens.yaml", "--out", "report.md"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Graded 2" in out and "1 PASS" in out
    body = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "[PASS] g1" in body and "[FAIL] g2" in body


def test_gate_writes_checklist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    cli.main(["init"])
    capsys.readouterr()
    rc = cli.main(["gate", "1"])
    assert rc == 0
    files = list((tmp_path / "tmp" / "gates").glob("GATE-1-*.md"))
    assert len(files) == 1
    body = files[0].read_text(encoding="utf-8")
    for needle in ("Part A", "Part B", "PASS", "FAIL", "Signed"):
        assert needle in body


def test_gate_with_explicit_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cli.main(["init"])
    assert cli.main(["gate", "4a", "--name", "pdf"]) == 0
    assert (tmp_path / "tmp" / "gates" / "GATE-4a-pdf.md").exists()


def test_version_flag_exits_zero(capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "docusearch" in capsys.readouterr().out


def test_no_command_returns_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main([]) == 2


def test_cli_writes_a_log_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cli.main(["init"])
    log_file = tmp_path / "tmp" / "logs" / f"{date.today().isoformat()}.jsonl"
    assert log_file.exists()
    events = [json.loads(line)["event"] for line in log_file.read_text("utf-8").splitlines()]
    assert "cli.init" in events
