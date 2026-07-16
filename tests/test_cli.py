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


def test_ingest_without_dry_run_is_not_yet_implemented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    cli.main(["init"])
    capsys.readouterr()
    rc = cli.main(["ingest"])
    assert rc == 2
    assert "Phase 1" in capsys.readouterr().out


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
