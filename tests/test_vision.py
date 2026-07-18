"""Vision enrichment stage — cloud image OCR + description (enrich.vision_images).

Determinism (R-SRCH-5) here is by *persistence*, not temperature: Opus 4.8 / Sonnet 5
reject sampling params, so the model is called once at enrichment time and the result is
stored; search never re-calls it, so ranked results over the stored text stay identical.
The real Anthropic provider is exercised with a stub client (no network, no key); the
orchestration is exercised with an injected fake provider.
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
    ImageInsight,
    VisionError,
    enrich_images,
    make_vision_provider,
)

PNG = b"\x89PNG\r\n\x1a\n fake-png-bytes"


# --------------------------------------------------------------------------- fakes


class FakeVision:
    """Deterministic in-memory provider implementing the VisionProvider protocol."""

    model_id = "fake-vision-1"

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str, str, str, str]] = []

    def describe(
        self,
        image_bytes: bytes,
        *,
        media_type: str,
        alt: str = "",
        caption: str = "",
        context: str = "",
    ) -> ImageInsight:
        self.calls.append((image_bytes, media_type, alt, caption, context))
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


def test_vision_model_default_is_opus(tmp_path: Path) -> None:
    cfg = config.load(tmp_path / "docusearch.yaml")
    assert cfg.enrich.vision_images is False
    assert cfg.enrich.vision_model == "claude-opus-4-8"


def test_template_documents_vision_model(tmp_path: Path) -> None:
    config.load(tmp_path / "docusearch.yaml")
    text = (tmp_path / "docusearch.yaml").read_text()
    assert "vision_model" in text


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


# -------------------------------------------------------------------- provider API


def test_make_vision_provider_gating(tmp_path: Path) -> None:
    cfg = config.load(tmp_path / "docusearch.yaml")
    assert make_vision_provider(cfg.enrich) is None  # off by default
    on = cfg.enrich.__class__(  # dataclass replace-lite
        preflight_sample=cfg.enrich.preflight_sample,
        ai_summaries=cfg.enrich.ai_summaries,
        vision_images=True,
        vision_model="claude-sonnet-5",
    )
    provider = make_vision_provider(on)
    assert isinstance(provider, AnthropicVisionProvider)
    assert provider.model_id == "claude-sonnet-5"


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
        # the insight is persisted + searchable
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
        assert len(prov.calls) == 1  # not re-called
        assert store.count_chunks() == chunks_after_first  # no duplicate enrichment chunk


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
        # seed a row but delete the staged original
        _seed_image(store, tmp_path, "0" * 64)
        (tmp_path / "images" / f"{'0' * 64}.png").unlink()
        result = enrich_images(store, FakeVision(), staging_dir=tmp_path)
        assert result.failed == 1
        assert result.errors and result.errors[0][0] == "0" * 64


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


# ------------------------------------------------------- real provider (stubbed)


class _StubMessages:
    def __init__(self, outer: _StubClient) -> None:
        self._outer = outer

    def create(self, **kwargs: Any) -> Any:
        self._outer.last = kwargs
        if self._outer.raises is not None:
            raise self._outer.raises
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._outer.reply)]
        )


class _StubClient:
    def __init__(self, reply: str, raises: Exception | None = None) -> None:
        self.reply = reply
        self.raises = raises
        self.last: dict[str, Any] = {}
        self.messages = _StubMessages(self)


def test_anthropic_provider_builds_request_and_parses() -> None:
    client = _StubClient(json.dumps({"text": "PA nWire SPI", "description": "an SPI block"}))
    prov = AnthropicVisionProvider("claude-opus-4-8", client=client)
    insight = prov.describe(PNG, media_type="image/png", caption="Fig 2", context="Setup")
    assert insight.text == "PA nWire SPI"
    assert insight.description == "an SPI block"
    assert insight.model == "claude-opus-4-8"
    # request shape: image block (base64) precedes the text block; structured output; no temperature
    sent = client.last
    assert sent["model"] == "claude-opus-4-8"
    assert "temperature" not in sent
    blocks = sent["messages"][0]["content"]
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["media_type"] == "image/png"
    assert blocks[1]["type"] == "text"
    assert sent["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_provider_wraps_api_error() -> None:
    client = _StubClient("", raises=RuntimeError("boom"))
    prov = AnthropicVisionProvider("claude-opus-4-8", client=client)
    with pytest.raises(VisionError):
        prov.describe(PNG, media_type="image/png")


def test_anthropic_provider_rejects_non_json() -> None:
    prov = AnthropicVisionProvider("claude-opus-4-8", client=_StubClient("not json at all"))
    with pytest.raises(VisionError):
        prov.describe(PNG, media_type="image/png")


def test_media_type_map_covers_common_raster() -> None:
    assert vision.MEDIA_TYPES["png"] == "image/png"
    assert vision.MEDIA_TYPES["jpg"] == "image/jpeg"
    assert "svg" not in vision.MEDIA_TYPES
