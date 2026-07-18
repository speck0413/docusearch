"""Config round-trip + validation tests (R-CFG-1..4).

These are the red tests for config.py: template generation from a single
source-of-truth, YAML round-trip to typed defaults, auto-creation, and
unknown-key / bad-enum validation.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from docusearch import config as cfg


def test_default_config_path_is_docusearch_yaml() -> None:
    assert Path("docusearch.yaml") == cfg.DEFAULT_CONFIG_PATH


def test_render_is_valid_yaml_and_has_header() -> None:
    text = cfg.render_template()
    # Header banner + friendly guidance (R-CFG-3).
    assert "docusearch configuration" in text
    assert "edit freely" in text
    # Parses cleanly as YAML.
    data = yaml.safe_load(text)
    assert isinstance(data, dict)


def test_render_documents_every_field_and_options() -> None:
    text = cfg.render_template()
    # A representative sweep of the documented options / curated choices (R-CFG-3/4).
    for needle in (
        "standalone",
        "server",
        "client",
        "BM25-only",  # embed.model: none explanation (R-CFG-4)
        "sentence-transformers/all-MiniLM-L6-v2",  # curated model id
        "sentence-transformers id",  # note that any HF id works
        "DOCUSEARCH_ROLES",  # roles come from env, not config (R-CFG-1)
        "content_selector",
        "strip_selectors",
        "min_content_chars",
    ):
        assert needle in text, f"template missing documentation for {needle!r}"


def test_render_roundtrips_to_typed_defaults(tmp_path: Path) -> None:
    """A freshly generated file loads back to exactly Config.default()."""
    path = tmp_path / "docusearch.yaml"
    cfg.write_template(path)
    loaded = cfg.load(path)
    assert loaded == cfg.default()


def test_default_values_match_contract() -> None:
    c = cfg.default()
    assert c.mode == "standalone"
    assert c.paths.db_path == "./catalog.db"
    assert c.embed.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert c.embed.batch_size == 128
    assert c.index.chunk_tokens == 350
    assert c.search.rrf_k == 60
    assert c.serve.port == 8321
    assert c.serve.mcp_path == "/mcp"
    assert c.logging.level == "info"
    # sources default is one example filesystem source
    assert len(c.sources) == 1
    assert c.sources[0].type == "fs"
    assert c.sources[0].include == ["**/*.html"]


def test_write_template_creates_missing_parent(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "docusearch.yaml"
    written = cfg.write_template(path)
    assert written is True
    assert path.exists()


def test_write_template_no_overwrite_without_force(tmp_path: Path) -> None:
    path = tmp_path / "docusearch.yaml"
    path.write_text("mode: server\n", encoding="utf-8")
    written = cfg.write_template(path)  # force defaults to False
    assert written is False
    assert path.read_text(encoding="utf-8") == "mode: server\n"


def test_write_template_force_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "docusearch.yaml"
    path.write_text("mode: server\n", encoding="utf-8")
    written = cfg.write_template(path, force=True)
    assert written is True
    assert "docusearch configuration" in path.read_text(encoding="utf-8")


def test_load_autocreates_when_missing(tmp_path: Path) -> None:
    """R-CFG-2: loading a nonexistent path writes the template, then loads it."""
    path = tmp_path / "docusearch.yaml"
    assert not path.exists()
    loaded = cfg.load(path)
    assert path.exists()
    assert loaded == cfg.default()


def test_unknown_key_warns_but_loads(tmp_path: Path) -> None:
    path = tmp_path / "docusearch.yaml"
    path.write_text("mode: standalone\nnonsense_key: 3\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="nonsense_key"):
        loaded = cfg.load(path)
    assert loaded.mode == "standalone"


def test_unknown_nested_key_warns(tmp_path: Path) -> None:
    path = tmp_path / "docusearch.yaml"
    path.write_text("embed:\n  modle: none\n", encoding="utf-8")  # typo'd key
    with pytest.warns(UserWarning, match="modle"):
        cfg.load(path)


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ("mode: bogus\n", "standalone"),
        ("logging:\n  level: LOUD\n", "info"),
        ("embed:\n  device: gpu\n", "cuda"),
    ],
)
def test_bad_enum_errors_with_accepted_options(tmp_path: Path, body: str, needle: str) -> None:
    path = tmp_path / "docusearch.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load(path)
    # The error names the accepted options so the user can fix it (R-CFG-3).
    assert needle in str(exc.value)


@pytest.mark.parametrize("model", ["none", "auto", "BAAI/bge-small-en-v1.5"])
def test_embed_model_accepts_none_auto_and_hf_ids(tmp_path: Path, model: str) -> None:
    """R-CFG-4: embed.model is open (none|auto|any HF id), not a closed enum."""
    path = tmp_path / "docusearch.yaml"
    path.write_text(f'embed:\n  model: "{model}"\n', encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = cfg.load(path)
    assert loaded.embed.model == model


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "mps"])
def test_device_accepts_mps_for_apple_silicon(tmp_path: Path, device: str) -> None:
    """R-EMB-1: macOS GPU (Metal) is a first-class device alongside cpu/cuda."""
    path = tmp_path / "docusearch.yaml"
    path.write_text(f"embed:\n  device: {device}\n", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = cfg.load(path)
    assert loaded.embed.device == device


def test_template_renders_lists_as_block_sequences() -> None:
    """List fields (include/exclude/audience) render as easy-to-edit block sequences,
    not inline flow arrays."""
    text = cfg.render_template()
    # block style: a "key:" line followed by dash items, not include: ["**/*.html"]
    assert '\n      - "**/*.html"' in text  # include item as a block sequence entry
    assert '\n      - "**/nav/**"' in text  # exclude item likewise
    assert "include: [" not in text  # the old inline flow form is gone
    # an empty list still renders inline (there is no item to dash)
    assert "strip_selectors: []" in text
    # and it still round-trips to the same typed defaults
    data = yaml.safe_load(text)
    assert data["sources"][0]["include"] == ["**/*.html"]


def test_partial_source_inherits_field_defaults(tmp_path: Path) -> None:
    """A source entry with only name/location still gets sane field defaults."""
    path = tmp_path / "docusearch.yaml"
    path.write_text(
        "sources:\n  - name: mine\n    location: /data/docs\n",
        encoding="utf-8",
    )
    loaded = cfg.load(path)
    assert len(loaded.sources) == 1
    src = loaded.sources[0]
    assert src.name == "mine"
    assert src.location == "/data/docs"
    assert src.type == "fs"  # inherited default
    assert src.min_content_chars == 200  # inherited default
    assert src.version == ""  # inherited default (blank = untracked)


def test_source_version_loads_and_documents(tmp_path: Path) -> None:
    assert "version" in cfg.render_template()  # documented in the template
    path = tmp_path / "docusearch.yaml"
    path.write_text(
        'sources:\n  - name: mine\n    version: "2024.3"\n    location: /data/docs\n',
        encoding="utf-8",
    )
    assert cfg.load(path).sources[0].version == "2024.3"


def test_scalar_override_loads(tmp_path: Path) -> None:
    path = tmp_path / "docusearch.yaml"
    path.write_text("mode: server\nserve:\n  port: 9000\n", encoding="utf-8")
    loaded = cfg.load(path)
    assert loaded.mode == "server"
    assert loaded.serve.port == 9000


def test_config_is_frozen() -> None:
    c = cfg.default()
    with pytest.raises((AttributeError, TypeError)):
        c.mode = "server"  # type: ignore[misc]


def test_non_mapping_config_errors(tmp_path: Path) -> None:
    path = tmp_path / "docusearch.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(cfg.ConfigError, match="mapping"):
        cfg.load(path)
