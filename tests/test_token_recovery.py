"""Token-recovery harness: extraction fidelity on synopsis-style markup (§16.4).

Guards the red-team Finding 1 regression: inline <div>/<span>/<code> synopsis text must
survive extraction into the chunks, and nothing may glue across element boundaries.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from docusearch import Catalog
from docusearch import config as cfg
from docusearch.store import Store

_ROOT = Path(__file__).resolve().parents[1]

SYNOPSIS_DOC = """<html><head><title>API</title></head><body><h1>API</h1>
<div class="methodsynopsis"><span class="modifier">public</span>
<span class="modifier">function</span>
<span class="methodname"><strong>Gmagick::getimagegamma</strong></span>():
<span class="type"><a href="float.html">float</a></span></div>
<div class="classsynopsisinfo"><code>readonly</code> <code>public</code> int
<var>Gmagick::endcolumn</var></div>
<ul><li><a href="a.html">GnuPG Functions</a>
<ul><li><a href="b.html">gnupg_adddecryptkey</a> - Add a key for decryption</li></ul></li></ul>
<p>Ordinary paragraph about configuration and timing and the peripheral interface.</p>
</body></html>"""


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "harness_tokrec", _ROOT / "harness" / "token_recovery.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tr = _load()


def test_visible_tokens_include_synopsis_terms() -> None:
    toks = tr.visible_tokens(SYNOPSIS_DOC)
    for needed in ("public", "function", "readonly", "endcolumn", "gnupg_adddecryptkey"):
        assert needed in toks


def test_token_recovery_is_perfect_on_synopsis(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "api.html").write_text(SYNOPSIS_DOC, encoding="utf-8")
    config_path = tmp_path / "docusearch.yaml"
    config_path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n'
        f'  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{docs.as_posix()}"\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    config = cfg.load(config_path)
    Catalog(config).ingest()
    with Store.open(config.paths.db_path) as store:
        report = tr.measure(store, content_selector="", sample=10, seed=1)
    assert report.passes
    assert report.min_recovery >= 0.98
