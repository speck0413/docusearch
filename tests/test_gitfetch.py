"""Phase 9 / GATE 9 — fetch a git/GitHub repo source via the user's own `git` (no tokens in
docusearch). A local `file://` repo stands in for a remote one so the real `git clone` runs offline."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from docusearch import gitfetch


def _local_repo(tmp_path: Path, name: str = "src") -> Path:
    """A real git repo on disk (a stand-in for a remote), with one commit and two files."""
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    (repo / "a.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    (repo / "README.md").write_text("# demo\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t.co", "-c", "user.name=t",
                    "commit", "-q", "-m", "init"], check=True)
    return repo


def test_is_remote() -> None:
    for loc in ("https://github.com/o/r", "http://gitlab.com/o/r", "ssh://git@h/o/r",
                "git://h/o/r", "git@github.com:o/r.git", "github.com/o/r", "gitlab.com/o/r",
                "file:///tmp/x"):
        assert gitfetch.is_remote(loc), loc
    for loc in ("/tmp/repo", "./repo", "../x", "C:\\repo", "relative/path", ""):
        assert not gitfetch.is_remote(loc), loc


def test_normalize_url() -> None:
    assert gitfetch.normalize_url("github.com/o/r") == "https://github.com/o/r"
    assert gitfetch.normalize_url("https://github.com/o/r") == "https://github.com/o/r"
    assert gitfetch.normalize_url("git@github.com:o/r.git") == "git@github.com:o/r.git"


def test_fetch_clones_a_local_repo(tmp_path: Path) -> None:
    src = _local_repo(tmp_path)
    dest_root = tmp_path / "cache"
    got = gitfetch.fetch_repo(f"file://{src}", dest_root)
    assert got.is_dir() and (got / "a.py").exists() and (got / "README.md").exists()
    assert got.parent == dest_root  # cached under the dest root, name derived from the URL


def test_fetch_reuses_cache_then_refreshes(tmp_path: Path) -> None:
    src = _local_repo(tmp_path)
    dest_root = tmp_path / "cache"
    gitfetch.fetch_repo(f"file://{src}", dest_root)  # first clone (cached)
    # add a new file to the source and commit
    (src / "b.py").write_text("def bye():\n    return 2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(src), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(src), "-c", "user.email=t@t.co", "-c", "user.name=t",
                    "commit", "-q", "-m", "more"], check=True)
    # without refresh: the cached clone is reused (no b.py)
    assert not (gitfetch.fetch_repo(f"file://{src}", dest_root) / "b.py").exists()
    # with refresh: re-fetched (b.py now present)
    assert (gitfetch.fetch_repo(f"file://{src}", dest_root, refresh=True) / "b.py").exists()


def test_fetch_specific_branch(tmp_path: Path) -> None:
    src = _local_repo(tmp_path)
    subprocess.run(["git", "-C", str(src), "checkout", "-q", "-b", "feature"], check=True)
    (src / "feat.py").write_text("def f():\n    return 3\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(src), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(src), "-c", "user.email=t@t.co", "-c", "user.name=t",
                    "commit", "-q", "-m", "feat"], check=True)
    got = gitfetch.fetch_repo(f"file://{src}", tmp_path / "cache", ref="feature")
    assert (got / "feat.py").exists()


def test_rejects_dangerous_urls(tmp_path: Path) -> None:
    for bad in ("-oProxyCommand=evil", "--upload-pack=touch /tmp/x",
                "ext::sh -c 'touch /tmp/pwn'", "fd::7"):
        with pytest.raises(gitfetch.GitFetchError):
            gitfetch.fetch_repo(bad, tmp_path / "cache")


def test_missing_repo_fails_clean(tmp_path: Path) -> None:
    with pytest.raises(gitfetch.GitFetchError, match="clone"):
        gitfetch.fetch_repo(f"file://{tmp_path / 'nope'}", tmp_path / "cache")


def test_cache_name_cannot_escape_dest_root(tmp_path: Path) -> None:
    # a URL crafted with ../ must not let the clone target escape the cache root
    for hostile in ("https://h/../../../etc/evil", "https://h/a/../../b", "file:///../../x"):
        name = gitfetch._safe_name(hostile)  # noqa: SLF001
        assert "/" not in name and ".." not in name and "\\" not in name
        assert (tmp_path / "cache" / name).resolve().parent == (tmp_path / "cache").resolve()


def test_ingest_code_store_from_git_remote(tmp_path: Path) -> None:
    # the full pipeline: a `store_type: code` source whose location is a git URL is cloned, then
    # parsed into symbols exactly like a local folder ("GitHub sources treated the same")
    from docusearch import config as cfg
    from docusearch import ingest
    from docusearch.store import Store

    src = _local_repo(tmp_path)
    path = tmp_path / "d.yaml"
    path.write_text(
        f'store_type: "code"\npaths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: remote\n    location: "file://{src}"\n'
        '    include: ["*.py"]\n    min_content_chars: 1\n'
        'embed:\n  model: "none"\n', encoding="utf-8")
    config = cfg.load(path)
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)
        assert result.documents == 1 and result.code_symbols == 1  # a.py -> hello()
        assert store.code_symbols_query()[0]["qualname"] == "hello"
    # a bad remote is a clean ingest error, not a crash
    path.write_text(
        f'store_type: "code"\npaths:\n  staging_dir: "{(tmp_path / "s2").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c2.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t2").as_posix()}"\n'
        f'sources:\n  - name: bad\n    location: "file://{tmp_path / "does-not-exist"}"\n'
        '    include: ["*.py"]\n    min_content_chars: 1\n'
        'embed:\n  model: "none"\n', encoding="utf-8")
    config = cfg.load(path)
    with Store.open(config.paths.db_path) as store:
        r2 = ingest.run_ingest(config, store)
        assert r2.documents == 0 and any("git fetch failed" in msg for _, msg in r2.errors)
