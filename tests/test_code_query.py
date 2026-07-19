"""Phase 9 / GATE 9 — the code store's non-AI query surface: Service.list_code + code_style_guide,
the substrate the `code` CLI and MCP tools share (parity proven in the gate checkout)."""

from __future__ import annotations

from pathlib import Path

from docusearch import config as cfg
from docusearch import ingest
from docusearch.server import Service
from docusearch.store import Store

PY = '''\
def connect(url: str) -> bool:
    """Open a connection."""
    return bool(url)


class Client:
    """A client."""

    def send(self, payload: bytes) -> int:
        return len(payload)
'''
JS = "function helper(x) { return x + 1 }\n"


def _svc(tmp_path: Path) -> Service:
    d = tmp_path / "repo"
    d.mkdir()
    (d / "client.py").write_text(PY, encoding="utf-8")
    (d / "util.js").write_text(JS, encoding="utf-8")
    path = tmp_path / "d.yaml"
    path.write_text(
        f'store_type: "code"\npaths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: repo\n    location: "{d.as_posix()}"\n'
        '    include: ["*.py", "*.js"]\n    min_content_chars: 1\n'
        'embed:\n  model: "none"\n', encoding="utf-8")
    config = cfg.load(path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
    return Service(config)


def test_list_code(tmp_path: Path) -> None:
    svc = _svc(tmp_path)
    res = svc.list_code()
    assert res["count"] == 4
    quals = {s["qualname"] for s in res["symbols"]}
    assert quals == {"connect", "Client", "Client.send", "helper"}

    methods = svc.list_code(kind="method")
    assert methods["count"] == 1 and methods["symbols"][0]["qualname"] == "Client.send"

    py = svc.list_code(language="python")
    assert py["count"] == 3 and all(s["language"] == "python" for s in py["symbols"])

    named = svc.list_code(name_like="conn%")
    assert [s["qualname"] for s in named["symbols"]] == ["connect"]


def test_code_style_guide(tmp_path: Path) -> None:
    svc = _svc(tmp_path)
    res = svc.code_style_guide()
    assert set(res["languages"]) == {"python", "javascript"}
    assert res["html"].startswith("<!doctype html>")
    assert "Style guide" in res["html"] and "snake_case" in res["html"]

    one = svc.code_style_guide(language="python")
    assert one["languages"] == ["python"]


def test_list_code_absurd_doc_id_is_clean(tmp_path: Path) -> None:
    # red-team #M1: an out-of-int64 doc_id must return empty, not raise OverflowError (which would
    # escape the MCP tool's except (PermissionError, ValueError) wrapper)
    svc = _svc(tmp_path)
    assert svc.list_code(doc_id=2**64) == {"symbols": [], "count": 0}
    assert svc.list_code(doc_id=-(2**70)) == {"symbols": [], "count": 0}


def test_cli_code_ls_strips_terminal_escapes() -> None:
    # red-team #M2: file-derived text printed by `code ls` must not carry ANSI escape bytes that
    # spoof the reviewer's terminal
    from docusearch.cli import _safe_term
    assert _safe_term("a\x1b[31mERROR\x1b[0m/x.py") == "a[31mERROR[0m/x.py"
    assert "\x1b" not in _safe_term("\x1b]0;title\x07evil") and "\x07" not in _safe_term("\x07")
