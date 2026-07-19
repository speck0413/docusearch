"""Vision enrichment stage — image OCR + description (enrich.vision_images, R-ING-9).

Determinism (R-SRCH-5) here is by *persistence*, not sampling: the model is called once at
enrichment time and the result is stored; search never re-calls it, so ranked results over the
stored text stay identical. The three backends are exercised with injected fakes (a subprocess
runner for the CLI, a stub client for the API, a stub pipeline for local) — no network, key,
or model download; the orchestration is exercised with an injected fake provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from docusearch import config, vision
from docusearch.store import Store
from docusearch.vision import (
    AnthropicVisionProvider,
    ClaudeCliVisionProvider,
    ImageInsight,
    LocalVisionProvider,
    VisionError,
    _local_reply_text,
    enrich_images,
    make_vision_provider,
)

PNG = b"\x89PNG\r\n\x1a\n fake-png-bytes"


def _png_file(tmp_path: Path) -> Path:
    p = tmp_path / "img.png"
    p.write_bytes(PNG)
    return p


# --------------------------------------------------------------------------- fakes


class FakeVision:
    """Deterministic in-memory provider implementing the VisionProvider protocol."""

    model_id = "fake-vision-1"

    def __init__(self) -> None:
        self.calls: list[tuple[Path, str, str, str, str]] = []

    def describe(
        self,
        image_path: Path,
        *,
        media_type: str,
        alt: str = "",
        caption: str = "",
        context: str = "",
    ) -> ImageInsight:
        self.calls.append((image_path, media_type, alt, caption, context))
        return ImageInsight(
            text=f"OCR[{caption or alt}]",
            description=f"block diagram at {context}",
            model=self.model_id,
        )


def _seed_image(store: Store, staging: Path, sha: str, ext: str = "png") -> int:
    doc_id = store.add_document(
        path=f"/docs/{sha}.html",
        source="vendor",
        source_version="",
        title="Doc",
        content_hash=sha,
        content_type="documentation",
        fmt="html",
        audience=["engineering"],
        mtime=0.0,
        status="active",
    )
    store.add_image(
        sha256=sha,
        ext=ext,
        doc_id=doc_id,
        locator="Setup > Block Diagram",
        alt="",
        caption="Figure 1",
        num_bytes=len(PNG),
    )
    images = staging / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / f"{sha}.{ext}").write_bytes(PNG)
    return doc_id


# --------------------------------------------------------------------------- config


def test_vision_defaults(tmp_path: Path) -> None:
    cfg = config.load(tmp_path / "docusearch.yaml")
    assert cfg.enrich.vision_images is False
    assert cfg.enrich.vision_provider == "claude-cli"  # no-key default
    assert cfg.enrich.vision_model == "claude-opus-4-8"


def test_template_documents_vision(tmp_path: Path) -> None:
    config.load(tmp_path / "docusearch.yaml")
    text = (tmp_path / "docusearch.yaml").read_text()
    assert "vision_provider" in text and "vision_model" in text


def test_bad_vision_provider_rejected(tmp_path: Path) -> None:
    path = tmp_path / "docusearch.yaml"
    path.write_text("enrich:\n  vision_provider: nope\n", encoding="utf-8")
    with pytest.raises(config.ConfigError):
        config.load(path)


# ----------------------------------------------------------------------- store v3


def test_migration_adds_vision_columns() -> None:
    with Store.open(":memory:") as store:
        assert store.schema_version >= 3
        cols = {row[1] for row in store._conn.execute("PRAGMA table_info(images)")}
        assert {"vision_text", "vision_model"} <= cols


def test_images_needing_vision_lifecycle(tmp_path: Path) -> None:
    with Store.open(":memory:") as store:
        _seed_image(store, tmp_path, "a" * 64)
        needing = store.images_needing_vision()
        assert [r["sha256"] for r in needing] == ["a" * 64]
        store.set_image_vision("a" * 64, "some text", "fake-vision-1")
        assert store.images_needing_vision() == []
        row = store.get_image("a" * 64)
        assert row is not None and row["vision_model"] == "fake-vision-1"


def test_add_enrichment_chunk_is_searchable(tmp_path: Path) -> None:
    with Store.open(":memory:") as store:
        doc_id = _seed_image(store, tmp_path, "b" * 64)
        store.add_enrichment_chunk(doc_id, "unique_nonce_zqx signal ADC", "loc")
        assert store.chunk_ids_matching("unique_nonce_zqx")


# ------------------------------------------------------------- provider dispatch


def _enrich_on(tmp_path: Path, provider: str) -> Any:
    cfg = config.load(tmp_path / "docusearch.yaml")
    return cfg.enrich.__class__(
        preflight_sample=cfg.enrich.preflight_sample,
        preflight_rules=cfg.enrich.preflight_rules,
        ai_summaries=cfg.enrich.ai_summaries,
        vision_images=True,
        vision_provider=provider,
        vision_model="m-1",
    )


def test_make_vision_provider_gating(tmp_path: Path) -> None:
    cfg = config.load(tmp_path / "docusearch.yaml")
    assert make_vision_provider(cfg.enrich) is None  # off by default
    assert isinstance(make_vision_provider(_enrich_on(tmp_path, "claude-cli")), ClaudeCliVisionProvider)
    assert isinstance(make_vision_provider(_enrich_on(tmp_path, "anthropic")), AnthropicVisionProvider)
    assert isinstance(make_vision_provider(_enrich_on(tmp_path, "local")), LocalVisionProvider)


def test_image_insight_searchable_text() -> None:
    assert ImageInsight("txt", "desc", "m").searchable_text() == "desc\n\ntxt"
    assert ImageInsight("", "desc", "m").searchable_text() == "desc"
    assert ImageInsight("", "", "m").searchable_text() == ""


# ---------------------------------------------------------------- orchestration


def test_enrich_images_basic(tmp_path: Path) -> None:
    with Store.open(":memory:") as store:
        _seed_image(store, tmp_path, "c" * 64)
        _seed_image(store, tmp_path, "d" * 64)
        prov = FakeVision()
        result = enrich_images(store, prov, staging_dir=tmp_path)
        assert result.enriched == 2
        assert result.failed == 0
        assert len(prov.calls) == 2
        assert prov.calls[0][1] == "image/png"  # media_type resolved from ext
        assert prov.calls[0][0].name.endswith(".png")  # a path is passed, not bytes
        assert store.images_needing_vision() == []
        assert store.chunk_ids_matching("diagram")


def test_enrich_images_idempotent(tmp_path: Path) -> None:
    with Store.open(":memory:") as store:
        _seed_image(store, tmp_path, "e" * 64)
        prov = FakeVision()
        enrich_images(store, prov, staging_dir=tmp_path)
        chunks_after_first = store.count_chunks()
        second = enrich_images(store, prov, staging_dir=tmp_path)
        assert second.enriched == 0
        assert len(prov.calls) == 1
        assert store.count_chunks() == chunks_after_first


def test_enrich_images_skips_unsupported_ext(tmp_path: Path) -> None:
    with Store.open(":memory:") as store:
        _seed_image(store, tmp_path, "f" * 64, ext="svg")
        prov = FakeVision()
        result = enrich_images(store, prov, staging_dir=tmp_path)
        assert result.enriched == 0
        assert result.skipped == 1
        assert prov.calls == []


def test_enrich_images_missing_file(tmp_path: Path) -> None:
    with Store.open(":memory:") as store:
        _seed_image(store, tmp_path, "0" * 64)
        (tmp_path / "images" / f"{'0' * 64}.png").unlink()
        result = enrich_images(store, FakeVision(), staging_dir=tmp_path)
        assert result.failed == 1
        assert result.errors and result.errors[0][0] == "0" * 64


def test_enrich_images_records_vision_error(tmp_path: Path) -> None:
    class _Boom:
        model_id = "boom"

        def describe(self, image_path, *, media_type, alt="", caption="", context=""):  # type: ignore[no-untyped-def]
            raise VisionError("provider exploded")

    with Store.open(":memory:") as store:
        _seed_image(store, tmp_path, "7" * 64)
        result = enrich_images(store, _Boom(), staging_dir=tmp_path)
        assert result.failed == 1 and "exploded" in result.errors[0][1]


def test_enrich_images_confines_path_traversal(tmp_path: Path) -> None:
    # red-team M2: a poisoned images row must not read a file resolved outside staging/images/
    (tmp_path / "images").mkdir()
    (tmp_path / "secret.png").write_bytes(b"secret")  # a sibling of images/, must stay unread
    with Store.open(":memory:") as store:
        doc_id = store.add_document(
            path="/d.html", source="v", source_version="", title="D", content_hash="h",
            content_type="documentation", fmt="html", audience=["e"], mtime=0.0, status="active",
        )
        store.add_image(
            sha256="../secret", ext="png", doc_id=doc_id, locator="", alt="", caption="", num_bytes=6
        )
        prov = FakeVision()
        result = enrich_images(store, prov, staging_dir=tmp_path)
        assert result.failed == 1
        assert prov.calls == []  # the escaping path was refused before any provider call


def test_enrich_images_survives_non_visionerror(tmp_path: Path) -> None:
    # red-team follow-up: a provider raising ANY exception (not just VisionError) must not abort
    # the batch — the failure is recorded and the pass continues (operability).
    class _Raw:
        model_id = "raw"

        def __init__(self) -> None:
            self.calls = 0

        def describe(self, image_path, *, media_type, alt="", caption="", context=""):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                raise FileNotFoundError("claude")  # a raw, non-VisionError exception
            return ImageInsight("t", "d", self.model_id)

    with Store.open(":memory:") as store:
        _seed_image(store, tmp_path, "5" * 64)
        _seed_image(store, tmp_path, "6" * 64)
        prov = _Raw()
        result = enrich_images(store, prov, staging_dir=tmp_path)
        assert prov.calls == 2  # did NOT abort after the first image raised
        assert result.enriched == 1 and result.failed == 1
        assert "FileNotFoundError" in result.errors[0][1]


def test_enrich_images_deterministic(tmp_path: Path) -> None:
    def run(root: Path) -> str:
        with Store.open(":memory:") as store:
            doc_id = _seed_image(store, root, "9" * 64)
            enrich_images(store, FakeVision(), staging_dir=root)
            rows = store.chunks_for_document(doc_id)
            return "\n".join(r["text"] for r in rows if r["kind"] == "enrichment")

    a = run(tmp_path / "one")
    b = run(tmp_path / "two")
    assert a == b and a != ""


def test_enrich_images_progress_callback(tmp_path: Path) -> None:
    with Store.open(":memory:") as store:
        _seed_image(store, tmp_path, "1" * 64)
        seen: list[tuple[str, int, int]] = []
        enrich_images(
            store, FakeVision(), staging_dir=tmp_path, progress=lambda *a: seen.append(a)
        )
        assert seen and seen[-1] == ("vision", 1, 1)


# ------------------------------------------------------- claude CLI backend


def _cli_ok(result_obj: dict[str, str]) -> Any:
    """A runner that emits a `claude -p --output-format json` success envelope."""
    captured: dict[str, Any] = {}

    def runner(argv: list[str]) -> tuple[int, str, str]:
        captured["argv"] = argv
        return 0, json.dumps({"type": "result", "is_error": False, "result": json.dumps(result_obj)}), ""

    runner.captured = captured  # type: ignore[attr-defined]
    return runner


def test_claude_cli_builds_argv_and_parses(tmp_path: Path) -> None:
    runner = _cli_ok({"text": "PA nWire SPI", "description": "an SPI block"})
    prov = ClaudeCliVisionProvider("claude-opus-4-8", runner=runner)
    img = _png_file(tmp_path)
    insight = prov.describe(img, media_type="image/png", context="Setup")
    assert insight.text == "PA nWire SPI" and insight.description == "an SPI block"
    assert insight.model == "claude-cli:claude-opus-4-8"
    argv = runner.captured["argv"]  # type: ignore[attr-defined]
    assert argv[0] == "claude" and "-p" in argv
    assert argv[argv.index("--model") + 1] == "claude-opus-4-8"
    assert argv[argv.index("--allowedTools") + 1] == "Read"
    assert argv[argv.index("--output-format") + 1] == "json"
    assert str(img) in argv[argv.index("-p") + 1]  # the image path is in the prompt


def test_claude_cli_nonzero_exit_raises(tmp_path: Path) -> None:
    prov = ClaudeCliVisionProvider("m", runner=lambda argv: (1, "", "boom"))
    with pytest.raises(VisionError, match="exit 1"):
        prov.describe(_png_file(tmp_path), media_type="image/png")


def test_claude_cli_error_envelope_raises(tmp_path: Path) -> None:
    def runner(argv: list[str]) -> tuple[int, str, str]:
        return 0, json.dumps({"type": "result", "is_error": True, "result": "quota"}), ""

    with pytest.raises(VisionError):
        ClaudeCliVisionProvider("m", runner=runner).describe(_png_file(tmp_path), media_type="image/png")


def test_claude_cli_wraps_exec_failure(tmp_path: Path) -> None:
    # red-team H1: a missing `claude` binary / timeout must become VisionError, not crash the
    # batch. Injected runner raising:
    def boom(argv: list[str]) -> tuple[int, str, str]:
        raise FileNotFoundError("no such file: claude")

    with pytest.raises(VisionError, match="invocation failed"):
        ClaudeCliVisionProvider("m", runner=boom).describe(_png_file(tmp_path), media_type="image/png")
    # and the REAL subprocess path with a nonexistent binary (zero cost, no quota):
    prov = ClaudeCliVisionProvider("m", cli="claude-does-not-exist-xyz", timeout=5)
    with pytest.raises(VisionError):
        prov.describe(_png_file(tmp_path), media_type="image/png")


def test_claude_cli_raw_json_without_envelope(tmp_path: Path) -> None:
    # some invocations print the bare reply, not a result envelope — still parsed
    def runner(argv: list[str]) -> tuple[int, str, str]:
        return 0, '{"text": "T", "description": "D"}', ""

    insight = ClaudeCliVisionProvider("m", runner=runner).describe(
        _png_file(tmp_path), media_type="image/png"
    )
    assert insight.text == "T" and insight.description == "D"


# ------------------------------------------------------- anthropic backend (stubbed)


class _StubMessages:
    def __init__(self, outer: _StubClient) -> None:
        self._outer = outer

    def create(self, **kwargs: Any) -> Any:
        self._outer.last = kwargs
        if self._outer.raises is not None:
            raise self._outer.raises
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._outer.reply)])


class _StubClient:
    def __init__(self, reply: str, raises: Exception | None = None) -> None:
        self.reply = reply
        self.raises = raises
        self.last: dict[str, Any] = {}
        self.messages = _StubMessages(self)


def test_anthropic_builds_request_and_parses(tmp_path: Path) -> None:
    client = _StubClient(json.dumps({"text": "PA nWire SPI", "description": "an SPI block"}))
    prov = AnthropicVisionProvider("claude-opus-4-8", client=client)
    insight = prov.describe(_png_file(tmp_path), media_type="image/png", caption="Fig 2")
    assert insight.text == "PA nWire SPI" and insight.model == "claude-opus-4-8"
    sent = client.last
    assert sent["model"] == "claude-opus-4-8"
    assert "temperature" not in sent  # Opus 4.8 / Sonnet 5 reject it
    blocks = sent["messages"][0]["content"]
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["media_type"] == "image/png"
    assert blocks[1]["type"] == "text"
    assert sent["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_wraps_api_error(tmp_path: Path) -> None:
    prov = AnthropicVisionProvider("m", client=_StubClient("", raises=RuntimeError("boom")))
    with pytest.raises(VisionError):
        prov.describe(_png_file(tmp_path), media_type="image/png")


def test_anthropic_rejects_non_json(tmp_path: Path) -> None:
    prov = AnthropicVisionProvider("m", client=_StubClient("not json at all"))
    with pytest.raises(VisionError):
        prov.describe(_png_file(tmp_path), media_type="image/png")


def test_anthropic_missing_file_is_vision_error(tmp_path: Path) -> None:
    # red-team M1: a missing image file must be VisionError, not an uncaught FileNotFoundError
    prov = AnthropicVisionProvider("m", client=_StubClient('{"text":"T","description":"D"}'))
    with pytest.raises(VisionError):
        prov.describe(tmp_path / "nope.png", media_type="image/png")


# ------------------------------------------------------- local backend


def test_local_reply_text_extracts_from_shapes() -> None:
    # chat-list shape (last assistant turn)
    chat = [{"generated_text": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "A"}]}]
    assert _local_reply_text(chat) == "A"
    # plain-string shape
    assert _local_reply_text([{"generated_text": "hello"}]) == "hello"


def test_local_provider_parses_stub_pipeline(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")  # noqa: N806 - pillow is a [vision-local] extra
    img = tmp_path / "real.png"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(img)

    def pipe(messages: Any, max_new_tokens: int = 0) -> Any:
        return [{"generated_text": [{"role": "assistant", "content": '{"text": "T", "description": "D"}'}]}]

    prov = LocalVisionProvider("google/gemma-3-4b-it", pipeline=pipe)
    insight = prov.describe(img, media_type="image/png", context="Setup")
    assert insight.text == "T" and insight.description == "D"
    assert insight.model == "google/gemma-3-4b-it"


def test_local_provider_wraps_failure(tmp_path: Path) -> None:
    pytest.importorskip("PIL.Image")

    def boom(messages: Any, max_new_tokens: int = 0) -> Any:
        raise RuntimeError("cuda oom")

    with pytest.raises(VisionError):
        LocalVisionProvider("m", pipeline=boom).describe(_png_file(tmp_path), media_type="image/png")


def test_media_type_map_covers_common_raster() -> None:
    assert vision.MEDIA_TYPES["png"] == "image/png"
    assert vision.MEDIA_TYPES["jpg"] == "image/jpeg"
    assert "svg" not in vision.MEDIA_TYPES
