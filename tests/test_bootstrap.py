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
