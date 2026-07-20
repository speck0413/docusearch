"""`docusearch bootstrap` (task #35): scan a mixed repo, detect its format mix, and emit a starter
docusearch.yaml with the right store_type + includes + inline hints. The output must load cleanly."""

from __future__ import annotations

from pathlib import Path

from docusearch import bootstrap
from docusearch import config as cfg


def _touch(root: Path, rel: str, text: str = "x") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_scan_counts_and_skips_noise(tmp_path: Path) -> None:
    _touch(tmp_path, "a.py")
    _touch(tmp_path, "pkg/b.py")
    _touch(tmp_path, "web/index.html", "<html></html>")
    _touch(tmp_path, ".git/config", "ignore me")
    _touch(tmp_path, "node_modules/dep/x.js", "ignore me")
    scan = bootstrap.scan_repo(tmp_path)
    assert scan.counts["code"] == 2 and scan.counts["doc"] == 1
    assert scan.code_languages["python"] == 2
    assert "js" not in scan.extensions  # node_modules skipped


def test_scan_survives_symlink_cycle(tmp_path: Path) -> None:
    # red-team portability note: a symlink loop must never hang or raise (os.walk followlinks=False)
    import os

    _touch(tmp_path, "real.py")
    sub = tmp_path / "sub"
    sub.mkdir()
    try:
        os.symlink(tmp_path, sub / "loop")  # sub/loop -> tmp_path (a cycle)
    except (OSError, NotImplementedError):
        import pytest
        pytest.skip("symlinks not supported here")
    scan = bootstrap.scan_repo(tmp_path)  # must return, not loop
    assert scan.counts["code"] == 1


def test_recommend_store_type(tmp_path: Path) -> None:
    code = tmp_path / "code_repo"
    for i in range(5):
        _touch(code, f"m{i}.py")
    _touch(code, "README.md")
    assert bootstrap.recommend_store_type(bootstrap.scan_repo(code)) == "code"

    docs = tmp_path / "doc_repo"
    for i in range(4):
        _touch(docs, f"p{i}.html", "<html></html>")
    assert bootstrap.recommend_store_type(bootstrap.scan_repo(docs)) == "document"

    data = tmp_path / "data_repo"
    for i in range(3):
        _touch(data, f"d{i}.csv", "a,b\n1,2\n")
    assert bootstrap.recommend_store_type(bootstrap.scan_repo(data)) == "data"


def test_bootstrap_config_is_valid_and_typed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    for i in range(4):
        _touch(repo, f"src/m{i}.py")
    _touch(repo, "docs/guide.html", "<html></html>")
    text = bootstrap.bootstrap_config(repo, name="myrepo")
    assert 'store_type: "code"' in text
    assert '"*.py"' in text and "myrepo" in text
    # it loads cleanly (a real, valid config) with no unknown-key warnings
    out = tmp_path / "docusearch.yaml"
    out.write_text(text, encoding="utf-8")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = cfg.load(out)
    assert config.store_type == "code"
    assert config.sources[0].name == "myrepo"
    assert "*.py" in config.sources[0].include


def test_cli_bootstrap_writes_and_guards(tmp_path: Path) -> None:
    from docusearch.cli import main

    repo = tmp_path / "repo"
    for i in range(3):
        _touch(repo, f"m{i}.py")
    out = tmp_path / "docusearch.yaml"
    assert main(["bootstrap", str(repo), "--name", "r", "--out", str(out)]) == 0
    assert out.exists() and 'store_type: "code"' in out.read_text(encoding="utf-8")
    # second write without --force is refused; with --force it overwrites
    assert main(["bootstrap", str(repo), "--out", str(out)]) == 1
    assert main(["bootstrap", str(repo), "--out", str(out), "--force"]) == 0
    # a non-directory is a clean error
    assert main(["bootstrap", str(tmp_path / "nope")]) == 2


def test_bootstrap_hints_html_and_notes_secondary(tmp_path: Path) -> None:
    repo = tmp_path / "docs"
    for i in range(5):
        _touch(repo, f"p{i}.html", "<html></html>")
    for i in range(2):
        _touch(repo, f"m{i}.py")  # secondary content
    text = bootstrap.bootstrap_config(repo, name="site")
    assert 'store_type: "document"' in text
    assert "docusearch inspect" in text          # HTML selector hint
    assert "code" in text and "2" in text         # notes the secondary code files


def test_bootstrap_config_resists_yaml_injection(tmp_path: Path) -> None:
    # red-team #H3: a crafted --name (or a hostile directory path/filename) must not inject YAML keys
    import warnings

    repo = tmp_path / "repo"
    _touch(repo, "a.py")
    evil_name = 'ok\n    tier: "internal"\n  - name: sneaky'
    text = bootstrap.bootstrap_config(repo, name=evil_name)
    out = tmp_path / "c.yaml"
    out.write_text(text, encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = cfg.load(out)
    assert len(config.sources) == 1                       # no injected second source
    assert all(s.tier != "internal" for s in config.sources)  # tier not elevated by the payload
    assert config.sources[0].name == "ok_____tier___internal____-_name__sneaky"  # sanitised to a label

    # a hostile directory name (quote + newline) is JSON-encoded into the location, still valid YAML
    hostile = tmp_path / 'weird"\n    tier: "internal'
    hostile.mkdir()
    _touch(hostile, "g.py")
    out2 = tmp_path / "c2.yaml"
    out2.write_text(bootstrap.bootstrap_config(hostile, name="clean"), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c2 = cfg.load(out2)
    assert c2.sources[0].tier == "vendor" and c2.sources[0].location.endswith("tier: \"internal")


def test_bootstrap_pdf_font_hint(tmp_path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    repo = tmp_path / "manuals"
    repo.mkdir()
    c = canvas.Canvas(str(repo / "m.pdf"), pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, 740, "Chapter")
    c.setFont("Helvetica", 11)
    for i, y in enumerate(range(700, 460, -20)):
        c.drawString(72, y, f"Body line {i} at the ordinary size for this manual.")
    c.showPage()
    c.save()
    _touch(repo, "index.html", "<html></html>")  # keep it a document store
    text = bootstrap.bootstrap_config(repo, name="manuals")
    assert 'store_type: "document"' in text
    assert "PDF headings inferred from font size" in text and "H1" in text
