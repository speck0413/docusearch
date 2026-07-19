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


def test_show_cli_prints_full_chunk_not_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    from docusearch.store import Store

    monkeypatch.chdir(tmp_path)
    _write_corpus_config(tmp_path)
    cli.main(["ingest"])
    needle = "ENDNEEDLE_QZX"
    big = ("word " * 260) + needle  # ~1300 chars; the needle sits well past 800
    with Store.open("./catalog.db") as s:
        doc_id = next(iter(s.document_path_to_id().values()))
        s.add_enrichment_chunk(doc_id, big, "loc")
    capsys.readouterr()
    rc = cli.main(["show", str(doc_id)])
    out = capsys.readouterr().out
    assert rc == 0
    assert needle in out  # full text printed — no 800-char cap (the reported bug)


def test_show_cli_max_chars_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    from docusearch.store import Store

    monkeypatch.chdir(tmp_path)
    _write_corpus_config(tmp_path)
    cli.main(["ingest"])
    needle = "TAILNEEDLE_ZZ"
    big = ("word " * 260) + needle
    with Store.open("./catalog.db") as s:
        doc_id = next(iter(s.document_path_to_id().values()))
        s.add_enrichment_chunk(doc_id, big, "loc")
    capsys.readouterr()
    cli.main(["show", str(doc_id), "--max-chars", "50"])
    out = capsys.readouterr().out
    assert needle not in out and "…" in out  # opt-in cap truncates with an ellipsis


def test_show_cli_missing_doc_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    _write_corpus_config(tmp_path)
    cli.main(["ingest"])
    capsys.readouterr()
    assert cli.main(["show", "999"]) == 1


def test_remove_cli_purges_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    _write_corpus_config(tmp_path)  # source label is "docs"
    cli.main(["ingest"])
    capsys.readouterr()
    rc = cli.main(["remove", "docs", "--yes"])  # --yes skips the confirmation prompt
    out = capsys.readouterr().out
    assert rc == 0
    assert "Removed 1 documents" in out
    capsys.readouterr()
    cli.main(["audit"])
    assert "documents: **0**" in capsys.readouterr().out  # index is empty again


def test_remove_cli_unknown_source_lists_known(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    _write_corpus_config(tmp_path)
    cli.main(["ingest"])
    capsys.readouterr()
    rc = cli.main(["remove", "delete_me_next", "--yes"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No documents found" in out
    assert "Known sources" in out and "docs" in out


def _vision_config(tmp_path: Path) -> None:
    """The corpus config plus vision enabled (embed stays 'none' so tests need no model)."""
    _write_corpus_config(tmp_path)
    text = (tmp_path / "docusearch.yaml").read_text(encoding="utf-8")
    text += 'enrich:\n  vision_images: true\n  vision_model: "claude-opus-4-8"\n'
    (tmp_path / "docusearch.yaml").write_text(text, encoding="utf-8")


class _StubVision:
    model_id = "stub-vision-1"

    def describe(self, image_path, *, media_type, alt="", caption="", context=""):  # type: ignore[no-untyped-def]
        from docusearch.vision import ImageInsight

        return ImageInsight(text="OCR nonce VZX9", description="a block diagram", model=self.model_id)


def _stage_image(tmp_path: Path) -> None:
    """Attach one image (row + staged original) to the ingested document."""
    import hashlib

    from docusearch.store import Store

    data = b"\x89PNG\r\n\x1a\n stub"
    sha = hashlib.sha256(data).hexdigest()
    images = tmp_path / "staging" / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / f"{sha}.png").write_bytes(data)
    with Store.open("./catalog.db") as store:
        doc_id = next(iter(store.document_path_to_id().values()))
        store.add_image(
            sha256=sha, ext="png", doc_id=doc_id, locator="Fig", alt="", caption="", num_bytes=len(data)
        )


def test_vision_cli_off_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    _write_corpus_config(tmp_path)  # vision off by default
    cli.main(["ingest"])
    capsys.readouterr()
    rc = cli.main(["vision", "--yes"])
    assert rc == 1
    assert "vision_images is off" in capsys.readouterr().err


def test_vision_cli_nothing_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    _vision_config(tmp_path)
    cli.main(["ingest"])  # corpus html has no images
    capsys.readouterr()
    rc = cli.main(["vision", "--yes"])
    assert rc == 0
    assert "No images need vision" in capsys.readouterr().out


def test_vision_cli_enriches_with_stub_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    from docusearch import vision

    monkeypatch.chdir(tmp_path)
    _vision_config(tmp_path)
    cli.main(["ingest"])
    _stage_image(tmp_path)
    monkeypatch.setattr(vision, "make_vision_provider", lambda enrich: _StubVision())
    capsys.readouterr()
    rc = cli.main(["vision", "--yes"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Enriched 1 images" in out
    capsys.readouterr()
    cli.main(["search", "VZX9"])  # the enrichment chunk is BM25-searchable
    assert "VZX9" in capsys.readouterr().out


class _FailVision:
    model_id = "fail"

    def describe(self, image_path, *, media_type, alt="", caption="", context=""):  # type: ignore[no-untyped-def]
        raise RuntimeError("backend down")


def test_vision_cli_total_failure_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    from docusearch import vision

    monkeypatch.chdir(tmp_path)
    _vision_config(tmp_path)
    cli.main(["ingest"])
    _stage_image(tmp_path)
    monkeypatch.setattr(vision, "make_vision_provider", lambda enrich: _FailVision())
    capsys.readouterr()
    rc = cli.main(["vision", "--yes"])  # every image fails -> non-zero exit for automation
    assert rc == 1
    assert "Enriched 0 images" in capsys.readouterr().out


def test_search_json_emits_structured_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    _write_corpus_config(tmp_path)
    cli.main(["ingest"])
    capsys.readouterr()
    rc = cli.main(["search", "ZZQ42", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["hits"], "expected at least one hit"
    hit = payload["hits"][0]
    assert {"doc_id", "chunk_id", "citation", "snippet"} <= hit.keys()
    assert hit["citation"] == f"D:{hit['doc_id']}#{hit['chunk_id']}"


def test_report_cli_renders_html_with_verified_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cli.main(["init"])  # config gives base_url + embed model label
    (tmp_path / "answer.yaml").write_text(
        'title: "PA overview"\n'
        'body: "PA is controlled over the nWire bus [D:12#3]. A general fact [GK]."\n'
        "evidence:\n  - [12, 3]\n"
        'audience: ["engineering"]\n',
        encoding="utf-8",
    )
    rc = cli.main(["report", "--spec", "answer.yaml", "--format", "html", "--out", "r.html"])
    assert rc == 0
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "<html" in html.lower() and "PA overview" in html
    assert "documents/12?chunk=3" in html  # citation resolved to a reference URL


def test_report_cli_refuses_hallucinated_citation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    cli.main(["init"])
    (tmp_path / "answer.yaml").write_text(
        'title: "x"\nbody: "claim [D:99#9]"\nevidence:\n  - [1, 1]\n', encoding="utf-8"
    )
    rc = cli.main(["report", "--spec", "answer.yaml", "--out", "r.md"])
    err = capsys.readouterr().err
    assert rc == 1
    assert err.startswith("error:") and "evidence" in err  # refused, cleanly


def test_cli_prints_clean_error_not_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    # a known, actionable failure (bad enum here; model mismatch in the wild) should print
    # a one-line "error: …" with guidance, and exit 1 — not dump a Python traceback.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docusearch.yaml").write_text("embed:\n  device: gpu\n", encoding="utf-8")
    rc = cli.main(["ingest"])
    err = capsys.readouterr().err
    assert rc == 1
    assert err.startswith("error:")
    assert "cuda" in err  # names the accepted options so the user can fix it


def test_self_heal_thread_starts_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docusearch import config as cfg
    from docusearch.catalog import Catalog

    monkeypatch.chdir(tmp_path)
    cli.main(["init"])
    cat = Catalog(cfg.load(tmp_path / "docusearch.yaml"))
    assert cli._start_self_heal(cat, 0) is None  # disabled
    thread = cli._start_self_heal(cat, 60)  # enabled -> a running daemon thread
    assert thread is not None and thread.is_alive() and thread.daemon


def test_serve_config_has_self_heal_interval(tmp_path: Path) -> None:
    from docusearch import config as cfg

    assert cfg.default().serve.self_heal_minutes == 60  # default: hourly
    path = tmp_path / "docusearch.yaml"
    path.write_text("serve:\n  self_heal_minutes: 0\n", encoding="utf-8")
    assert cfg.load(path).serve.self_heal_minutes == 0  # 0 disables it


def test_models_cli_lists_cache_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    cache = tmp_path / "hfcache"
    cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))  # isolate from the real model cache
    rc = cli.main(["models"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Model cache:" in out and str(cache) in out
    assert "delete-cache" in out  # tells the user how to purge


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


def _preflight_config(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.html").write_text("<body><h1>H</h1><p>Do NOT.</p></body>", encoding="utf-8")
    cfg_path = tmp_path / "d.yaml"
    cfg_path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 1\n'
        f'embed:\n  model: "none"\n'
        f'enrich:\n  preflight_rules: "{(tmp_path / "preflight_rules.yaml").as_posix()}"\n',
        encoding="utf-8",
    )
    return cfg_path


def test_preflight_writes_rules_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    from docusearch import enrich

    cfg_path = _preflight_config(tmp_path)
    out = tmp_path / "preflight_rules.yaml"
    seen: dict[str, object] = {}

    def fake_run_preflight(config, *, out_path, model, seed):  # type: ignore[no-untyped-def]
        seen.update(out_path=str(out_path), model=model, seed=seed)
        rules = enrich.PreflightRules(
            approved=False, gotcha_patterns=[enrich.GotchaPattern(r"do NOT", "warning")], sampled=1
        )
        enrich.write_preflight_rules(rules, out_path)
        return rules

    monkeypatch.setattr(enrich, "run_preflight", fake_run_preflight)
    rc = cli.main(["preflight", "--config", str(cfg_path), "--model", "m", "--seed", "3"])
    assert rc == 0
    assert seen == {"out_path": str(out), "model": "m", "seed": 3}
    assert out.is_file()
    combined = capsys.readouterr()
    assert "warning" in (combined.out + combined.err) and str(out) in combined.out


def test_related_cli_walks_nhop(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    from docusearch import config as cfg
    from docusearch import ingest
    from docusearch.store import Store

    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.html").write_text(
        '<body><h1>A</h1><p>Alpha overview.</p><a href="b.html">B</a></body>', encoding="utf-8"
    )
    (root / "b.html").write_text(
        '<body><h1>B</h1><p>Beta details.</p><a href="c.html">C</a></body>', encoding="utf-8"
    )
    (root / "c.html").write_text("<body><h1>C</h1><p>Gamma reference.</p></body>", encoding="utf-8")
    cfg_path = tmp_path / "d.yaml"
    cfg_path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    config = cfg.load(cfg_path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        a = store.document_id_for_path((root / "a.html").resolve().as_posix())
    assert cli.main(["related", str(a), "--direction", "out", "--depth", "2", "--config", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "1·out" in out and "B" in out  # direct neighbour
    assert "2·out" in out and "C" in out  # 2-hop neighbour


def test_discrepancies_cli_reports_and_persists(
    tmp_path: Path, capsys
) -> None:  # type: ignore[no-untyped-def]
    from docusearch import config as cfg
    from docusearch import ingest
    from docusearch.store import Store

    root = tmp_path / "docs"
    root.mkdir()
    body = "<body><h1>H</h1><p>The identical shared paragraph of text here.</p></body>"
    (root / "one.html").write_text(body, encoding="utf-8")
    (root / "two.html").write_text(body, encoding="utf-8")  # dup
    cfg_path = tmp_path / "d.yaml"
    cfg_path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    config = cfg.load(cfg_path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
    assert cli.main(["discrepancies", "--persist", "--config", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "Duplicate active documents: **1**" in out
    with Store.open(config.paths.db_path) as store:
        assert store.count_flags("discrepancy") == 2  # one flag per dup doc, persisted


def test_preflight_refuses_to_clobber_approved_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docusearch import enrich

    cfg_path = _preflight_config(tmp_path)
    out = tmp_path / "preflight_rules.yaml"
    enrich.write_preflight_rules(
        enrich.PreflightRules(approved=True, gotcha_patterns=[enrich.GotchaPattern("x", "l")]), out
    )
    calls = {"n": 0}

    def fake_run_preflight(config, **kw):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return enrich.PreflightRules(approved=False)

    monkeypatch.setattr(enrich, "run_preflight", fake_run_preflight)
    assert cli.main(["preflight", "--config", str(cfg_path)]) == 1  # refuses without --yes
    assert calls["n"] == 0  # never touched the approved file
    assert cli.main(["preflight", "--config", str(cfg_path), "--yes"]) == 0  # --yes overrides
    assert calls["n"] == 1


def test_cli_writes_a_log_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cli.main(["init"])
    log_file = tmp_path / "tmp" / "logs" / f"{date.today().isoformat()}.jsonl"
    assert log_file.exists()
    events = [json.loads(line)["event"] for line in log_file.read_text("utf-8").splitlines()]
    assert "cli.init" in events
