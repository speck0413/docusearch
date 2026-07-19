"""`docusearch stdf` CLI tools as LIVE MCP clients, plus the service/store layer they drive
(R-API-1, R-STDF-2).

Two layers of coverage:
  * service/helpers in-process — SKU (= source label) filing, glob + SKU listing, upload guards;
  * one real over-the-wire round-trip — start `serve` in a thread and run upload → ls → report →
    audit through the actual MCP client, proving CLI↔MCP parity end to end.
"""

from __future__ import annotations

import io
import socket
import tarfile
import threading
import time
from pathlib import Path

import pytest
from harness.stdf_synth import StdfBuilder

from docusearch import config as cfg
from docusearch.server import Service, _match_glob


def _stdf_bytes(vmins: list[float], *, hi: float = 0.85, cod: str = "WS1") -> bytes:
    b = StdfBuilder().far().mir(lot_id="LOTZ", test_cod=cod)
    for i, v in enumerate(vmins):
        b.pir()
        b.ptr(1000, "VMIN", v, lo=0.70, hi=hi, units="V", fail=(v < 0.70))
        b.ptr(2000, "IDDQ", 1e-6 + i * 1e-7, lo=0.0, hi=2e-6, units="A")
        b.prr(part_id=str(i + 1), hard_bin=1 if v >= 0.70 else 5)
    b.mrr()
    return b.to_bytes()


def _targz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _config(tmp_path: Path, *, port: int = 8321) -> Path:
    src = tmp_path / "seed"
    src.mkdir()
    path = tmp_path / "docusearch.yaml"
    path.write_text(
        f'store_type: "data"\n'
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f"serve:\n  host: \"127.0.0.1\"\n  port: {port}\n  mcp_path: \"/mcp\"\n"
        f'sources:\n  - name: seed\n    location: "{src.as_posix()}"\n'
        '    include: ["*.stdf"]\n    min_content_chars: 1\n    insertion: "FT"\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------- glob helper


def test_match_glob_basename_and_path() -> None:
    assert _match_glob("/data/ate/lotZ_run2.stdf", "*run2*")
    assert _match_glob("/data/ate/lotZ_run2.stdf", "lotZ_run2.stdf")
    assert _match_glob("/data/ate/lotZ_run2.stdf", "*/ate/lotZ_*.stdf")  # full-path glob
    assert not _match_glob("/data/ate/lotZ_run1.stdf", "*run2*")
    assert _match_glob(r"C:\ate\lotZ_run2.stdf", "*run2*")  # windows path, posix pattern


# --------------------------------------------------------------- service layer (no network)


def test_upload_archive_files_under_sku_then_lists(tmp_path: Path) -> None:
    svc = Service(cfg.load(_config(tmp_path)))
    bundle = _targz({
        "lotZ_run1.stdf": _stdf_bytes([0.71, 0.72, 0.90, 0.91]),
        "lotZ_run2.stdf": _stdf_bytes([0.68, 0.75, 0.88, 0.93], hi=0.80, cod="WS2"),
    })
    res = svc.upload_archive(data=bundle, filename="upload.tar.gz", sku="WIDGET_A", insertion="WS1")
    assert res["documents"] == 2
    assert res["uploaded_by"] == ""

    listing = svc.list_stdf_documents()
    assert {d["sku"] for d in listing["documents"]} == {"WIDGET_A"}  # filed under the SKU
    assert listing["skus"] == ["WIDGET_A"]
    assert all(d["tests"] > 0 for d in listing["documents"])
    # glob narrows to one file; a foreign SKU narrows to none
    assert len(svc.list_stdf_documents(glob="*run2*")["documents"]) == 1
    assert svc.list_stdf_documents(sku="NOPE")["documents"] == []


def test_upload_archive_requires_sku_and_rejects_junk(tmp_path: Path) -> None:
    svc = Service(cfg.load(_config(tmp_path)))
    good = _targz({"a.stdf": _stdf_bytes([0.71, 0.72])})
    with pytest.raises(ValueError, match="SKU"):
        svc.upload_archive(data=good, filename="u.tar.gz", sku="   ")
    with pytest.raises(ValueError, match="zip or .tar.gz"):
        svc.upload_archive(data=good, filename="notarchive.stdf", sku="X")
    with pytest.raises(ValueError, match="cap"):
        svc.upload_archive(data=good, filename="u.tar.gz", sku="X", max_bytes=10)


# --------------------------------------------------------------- CLI helpers (pure)


def test_bundle_globs_dedupes_and_flattens(tmp_path: Path) -> None:
    from docusearch.cli import _bundle_globs

    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    (tmp_path / "d1" / "lot.stdf").write_bytes(_stdf_bytes([0.71]))
    (tmp_path / "d2" / "lot.stdf").write_bytes(_stdf_bytes([0.72]))  # same basename, different dir
    data, names = _bundle_globs([str(tmp_path / "**" / "*.stdf")])
    assert len(names) == 2 and len(set(names)) == 2  # both kept, names disambiguated
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        assert sorted(tar.getnames()) == sorted(names)


def test_mcp_url_from_config_and_override(tmp_path: Path) -> None:
    from docusearch.cli import _mcp_url

    conf = cfg.load(_config(tmp_path, port=9999))
    assert _mcp_url(conf, None) == "http://127.0.0.1:9999/mcp"
    assert _mcp_url(conf, "http://host:1/mcp") == "http://host:1/mcp"


# --------------------------------------------------------------- MCP client result parsing


class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Result:
    def __init__(self, *, structured: object = None, content: list[object] | None = None,
                 is_error: bool = False) -> None:
        self.structuredContent = structured
        self.content = content or []
        self.isError = is_error


def test_mcp_unwrap_structured_and_wrapped_and_text() -> None:
    from docusearch.mcp_client import MCPError, _unwrap

    assert _unwrap(_Result(structured={"documents": []}), "list_stdf") == {"documents": []}
    assert _unwrap(_Result(structured={"result": [1, 2]}), "x") == [1, 2]  # FastMCP scalar wrap
    assert _unwrap(_Result(content=[_Block('{"a": 1}')]), "x") == {"a": 1}  # text JSON fallback
    assert _unwrap(_Result(content=[_Block("plain")]), "x") == "plain"  # non-JSON text
    with pytest.raises(MCPError, match="returned an error"):
        _unwrap(_Result(content=[_Block("boom")], is_error=True), "x")


def test_mcp_friendly_messages() -> None:
    from docusearch.mcp_client import _flatten, _friendly

    refused = _friendly("http://h/mcp", ConnectionRefusedError("Connection refused"))
    assert "cannot reach" in refused and "docusearch serve" in refused
    other = _friendly("http://h/mcp", ValueError("bad protocol"))
    assert "failed" in other and "cannot reach" not in other
    grp = ExceptionGroup("g", [ValueError("a"), OSError("b")])
    assert len(_flatten(grp)) == 2


# --------------------------------------------------------------- LIVE over-the-wire round-trip


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _wait_port(host: str, port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"server on {host}:{port} did not come up")


# A real server thread (uvicorn/anyio/mcp) emits teardown noise we must not fail the suite on:
# the streamable-HTTP transport leaves anyio memory streams for __del__ (cosmetic ResourceWarning),
# and third-party libs may deprecation-warn. Same rationale as the PyMuPDF/starlette exemptions.
@pytest.mark.filterwarnings("default")
@pytest.mark.filterwarnings("ignore::ResourceWarning")
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_stdf_cli_end_to_end_over_live_mcp(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import uvicorn

    from docusearch.cli import main
    from docusearch.server import create_app

    port = _free_port()
    config_path = _config(tmp_path, port=port)
    ate = tmp_path / "ate"
    ate.mkdir()
    (ate / "lotZ_run1.stdf").write_bytes(_stdf_bytes([0.71, 0.72, 0.90, 0.91]))
    (ate / "lotZ_run2.stdf").write_bytes(_stdf_bytes([0.68, 0.75, 0.88, 0.93], hi=0.80, cod="WS2"))

    server = uvicorn.Server(uvicorn.Config(
        create_app(cfg.load(config_path)), host="127.0.0.1", port=port, log_level="warning"
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _wait_port("127.0.0.1", port)
        cn = ["--config", str(config_path)]

        assert main(["stdf", "upload", "WIDGET_A", str(ate / "*.stdf"),
                     "--insertion", "WS1", *cn]) == 0
        assert "Uploaded to SKU 'WIDGET_A'" in capsys.readouterr().out

        assert main(["stdf", "ls", *cn]) == 0
        out = capsys.readouterr().out
        assert "WIDGET_A" in out and "lotZ_run1.stdf" in out and "lotZ_run2.stdf" in out

        assert main(["stdf", "ls", "*run2*", *cn]) == 0
        out = capsys.readouterr().out
        assert "run2" in out and "run1" not in out

        assert main(["stdf", "report", "*run1*", "--test", "1000", *cn]) == 0
        rep_out = capsys.readouterr().out
        assert "Wrote" in rep_out
        assert Path(rep_out.split("Wrote", 1)[1].split("(")[0].strip()).is_file()

        assert main(["stdf", "audit", "*.stdf", *cn]) == 0
        aud_out = capsys.readouterr().out
        assert "audit dashboard" in aud_out
        audit_html = Path(aud_out.split("dashboard", 1)[1].split("(")[0].strip())
        body = audit_html.read_text()
        assert body.startswith("<!doctype html>")
        for tab in ("Diff", "Q-Q", "Histograms", "Trend", "Site"):
            assert f">{tab}<" in body

        # an unreachable endpoint yields a clean, actionable error (not a traceback)
        assert main(["stdf", "ls", "--url", "http://127.0.0.1:1/mcp", *cn]) == 1
        assert "cannot reach" in capsys.readouterr().err
    finally:
        server.should_exit = True
        thread.join(timeout=15)


@pytest.mark.filterwarnings("default")
@pytest.mark.filterwarnings("ignore::ResourceWarning")
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_data_cli_over_live_mcp(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`docusearch data ls/plot` drives the same live MCP server on a generic CSV data store."""
    import uvicorn

    from docusearch import ingest
    from docusearch.cli import main
    from docusearch.server import create_app
    from docusearch.store import Store

    port = _free_port()
    d = tmp_path / "tables"
    d.mkdir()
    (d / "sensors.csv").write_text(
        "vmin,site\n" + "".join(f"{0.70 + 0.002 * i},{1 + i % 2}\n" for i in range(30)),
        encoding="utf-8")
    config_path = tmp_path / "docusearch.yaml"
    config_path.write_text(
        f'store_type: "data"\npaths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'serve:\n  host: "127.0.0.1"\n  port: {port}\n  mcp_path: "/mcp"\n'
        f'sources:\n  - name: data\n    location: "{d.as_posix()}"\n'
        '    include: ["*.csv"]\n    min_content_chars: 1\n'
        '    csv:\n      group: "site"\nembed:\n  model: "none"\n', encoding="utf-8")
    config = cfg.load(config_path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)

    server = uvicorn.Server(uvicorn.Config(
        create_app(config), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _wait_port("127.0.0.1", port)
        cn = ["--config", str(config_path)]

        assert main(["data", "ls", *cn]) == 0
        out = capsys.readouterr().out
        assert "vmin" in out and "sensors" in out

        assert main(["data", "plot", "vmin", "--kind", "histogram", *cn]) == 0
        plot_out = capsys.readouterr().out
        assert "Wrote" in plot_out
        assert Path(plot_out.split("Wrote", 1)[1].split("(")[0].strip()).is_file()

        # by-group whisker also works over the wire
        assert main(["data", "plot", "sensors.vmin", "--kind", "whisker", "--by-group", *cn]) == 0
        assert "Wrote" in capsys.readouterr().out
    finally:
        server.should_exit = True
        thread.join(timeout=15)
