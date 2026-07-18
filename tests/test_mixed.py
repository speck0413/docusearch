"""Phase 4d — Mixed suite (R-TEST-2, §15.4): one store, four formats interleaved by index%4."""

from __future__ import annotations

from pathlib import Path

from docusearch import config
from docusearch.catalog import Catalog
from docusearch.convert import convert_mixed, convert_source_mixed


def _mixed_config(tmp_path: Path, corpus: Path):  # type: ignore[no-untyped-def]
    from docusearch import config

    path = tmp_path / "mixed.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "m.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: mixed\n    location: "{corpus.as_posix()}"\n'
        '    include: ["**/*.html", "**/*.md", "**/*.docx", "**/*.pdf"]\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    return config.load(path)


def test_convert_mixed_modulus_assignment(tmp_path: Path) -> None:
    src = tmp_path / "html"
    src.mkdir()
    # 8 files -> index%4 -> html, md, docx, pdf, html, md, docx, pdf (sorted by name)
    for i in range(8):
        (src / f"doc{i}.html").write_text(
            f"<body><h1>Doc {i}</h1><p>needle NONCE{i}00X here in ordinary prose text.</p></body>",
            encoding="utf-8",
        )
    dst = tmp_path / "mixed"
    result = convert_mixed(src, dst)
    assert result.converted == 8 and not result.errors
    got = sorted(p.suffix for p in dst.rglob("*.*"))
    # each of the four formats appears twice
    assert got.count(".html") == 2 and got.count(".md") == 2
    assert got.count(".docx") == 2 and got.count(".pdf") == 2

    # one store ingests all four formats; a needle from each format is recoverable
    cfg = _mixed_config(tmp_path, dst)
    cat = Catalog(cfg)
    res = cat.ingest()
    assert res.documents == 8 and set(res.per_extension) >= {"html", "md", "docx", "pdf"}
    for i in range(8):
        hits = cat.search(f"NONCE{i}00X")
        assert hits, f"needle from doc{i} (format {['html','md','docx','pdf'][i % 4]}) not recovered"


def test_convert_source_mixed_honors_selector_and_spreads_formats(tmp_path: Path) -> None:
    # Mixed suite on a real source (R-TEST-2): the derived files carry CLEAN content_selector text
    # (converted formats), html copies are cleaned at ingest, and every format is represented.
    src = tmp_path / "site"
    src.mkdir()
    for i in range(8):
        (src / f"page{i}.html").write_text(
            f"<html><body><nav>CHROME-NAV</nav><article><h1>Doc {i}</h1>"
            f"<p>MIXNONCE{i}77 the calibration procedure.</p></article>"
            "<footer>CHROME-FOOT</footer></body></html>",
            encoding="utf-8",
        )
    source = config.SourceConfig(
        type="fs", name="s", version="", location=str(src),
        include=["**/*.html"], exclude=[],
        content_selector="article", strip_selectors=["nav", "footer"],
        min_content_chars=5, audience=[],
    )
    dst = tmp_path / "mixed"
    result = convert_source_mixed(source, dst)
    assert result.converted == 8 and not result.errors
    got = sorted(p.suffix for p in dst.rglob("*.*"))
    assert got.count(".html") == 2 and got.count(".md") == 2
    assert got.count(".docx") == 2 and got.count(".pdf") == 2

    # a converted (non-html) file must carry clean article text, no chrome baked in
    md_text = next(dst.rglob("*.md")).read_text(encoding="utf-8")
    assert "calibration procedure" in md_text and "CHROME" not in md_text


def test_convert_mixed_reused_dst_has_no_stale_files(tmp_path: Path) -> None:
    # Red-team M1: re-running convert_mixed into a dir that already holds a PREVIOUS run's output
    # (from a larger/differently-sized source) must not leave stale files behind — they would be
    # ingested as phantom documents. The output must reflect ONLY the current source.
    src = tmp_path / "html"
    src.mkdir()
    for i in range(8):
        (src / f"doc{i}.html").write_text(
            f"<body><h1>Doc {i}</h1><p>STALECHK{i} body text here for indexing.</p></body>",
            encoding="utf-8",
        )
    dst = tmp_path / "mixed"
    assert convert_mixed(src, dst).converted == 8
    assert len(list(dst.rglob("*.*"))) == 8

    # source shrinks to 5 files; re-run into the SAME dst
    for i in range(5, 8):
        (src / f"doc{i}.html").unlink()
    assert convert_mixed(src, dst).converted == 5
    # dst must now hold exactly the 5 current files — no doc5/6/7 leftovers from run 1
    assert len(list(dst.rglob("*.*"))) == 5
    cfg = _mixed_config(tmp_path, dst)
    res = Catalog(cfg).ingest()
    assert res.documents == 5  # not 8 — no phantom stale documents
