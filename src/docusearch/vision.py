"""Vision enrichment — cloud image OCR + description (enrich.vision_images, R-ING-6).

Retained images (R-ING-6) carry only alt/caption text, so a block diagram or a screenshot
of an instrument's test/debug display is invisible to search. When ``enrich.vision_images``
is on, ``docusearch vision`` sends each retained image **once** to a cloud vision model; the
returned OCR text + description are persisted (keyed by the image sha256) and inserted as a
searchable ``enrichment`` chunk.

**Determinism (R-SRCH-5) is by persistence, not temperature.** Opus 4.8 / Sonnet 5 reject
``temperature``/``top_p``/``top_k`` (400), so we can't pin the sampler to 0. Instead the model
is called only at enrichment time and the output is stored; search/ranking runs over the
stored text and never re-calls the API, so ranked results stay byte-identical across runs.

Reuse (R-REUSE-1): the official ``anthropic`` SDK ([vision] extra, lazy-imported) — it owns
the Messages-API request shape, retries, typed errors, and credential resolution (env key or
an ``ant auth login`` profile), so we don't hand-roll HTTP.

Public surface:
    VisionError
    ImageInsight               -- (text, description, model) result of one image
    VisionProvider (Protocol)  -- model_id; describe(image_bytes, ...) -> ImageInsight
    AnthropicVisionProvider    -- real provider (anthropic SDK); config-gated, off by default
    make_vision_provider(cfg)  -> VisionProvider | None   (None == vision off)
    enrich_images(store, provider, staging_dir=...) -> VisionResult
    MEDIA_TYPES                -- image extension -> media_type (the types the API accepts)
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .config import EnrichConfig
    from .store import Store

ProgressFn = Callable[[str, int, int], None]

# Raster formats the Anthropic Messages API accepts as image blocks. Vector (svg) and
# document (pdf) formats are not images to the vision endpoint — they're skipped.
MEDIA_TYPES: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}

# One shared, frozen prompt — kept byte-stable so the request is reproducible. It targets the
# vendor-doc image kinds Stephen flagged (block diagrams, instrument test/debug displays) but
# hardcodes no corpus tokens (R-PROC-6).
_PROMPT = (
    "You are extracting searchable text from a single technical-documentation image — "
    "typically a block diagram, schematic, wiring/pin diagram, or a screenshot of an "
    "instrument's test or debug display. Transcribe every label, signal name, pin, block "
    "title, numeric value, and caption you can read, verbatim, into `text`. In "
    "`description`, describe what the image shows and how its parts connect or relate, so an "
    "engineer could find it by searching the documentation. Use only what is visible in the "
    "image; do not guess or add facts that are not shown."
)

# Structured-output schema — Opus 4.8 / Sonnet 5 guarantee the response parses to this shape.
_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["text", "description"],
    "additionalProperties": False,
}


class VisionError(Exception):
    """A vision request failed or returned an unusable response."""


@dataclass(frozen=True)
class ImageInsight:
    """One image's vision result: transcribed ``text`` + ``description``, tagged by model."""

    text: str
    description: str
    model: str

    def searchable_text(self) -> str:
        """The description then the OCR text, blank parts dropped — the enrichment chunk body."""
        parts = [p.strip() for p in (self.description, self.text) if p.strip()]
        return "\n\n".join(parts)


@runtime_checkable
class VisionProvider(Protocol):
    """Anything that turns image bytes into an :class:`ImageInsight`, tagged by model."""

    @property
    def model_id(self) -> str: ...

    def describe(
        self,
        image_bytes: bytes,
        *,
        media_type: str,
        alt: str = "",
        caption: str = "",
        context: str = "",
    ) -> ImageInsight: ...


class AnthropicVisionProvider:
    """Describes images with an Anthropic vision model via the Messages API.

    The client loads lazily (the ``anthropic`` SDK is a [vision] extra) so importing
    docusearch never pulls it in. Credentials resolve from ``ANTHROPIC_API_KEY`` or an
    ``ant auth login`` profile — no key is read from config. No sampling params are sent
    (Opus 4.8 / Sonnet 5 reject them); reproducibility comes from persisting the result.
    """

    def __init__(self, model_id: str, *, client: Any = None, max_tokens: int = 1024) -> None:
        self._model_id = model_id
        self._client = client
        self._max_tokens = max_tokens

    @property
    def model_id(self) -> str:
        return self._model_id

    def _http(self) -> Any:
        if self._client is None:
            import anthropic  # lazy: [vision] extra only loaded when we actually enrich

            self._client = anthropic.Anthropic()
        return self._client

    def describe(
        self,
        image_bytes: bytes,
        *,
        media_type: str,
        alt: str = "",
        caption: str = "",
        context: str = "",
    ) -> ImageInsight:
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        hint = " ".join(p for p in (context, caption, alt) if p).strip()
        prompt = _PROMPT if not hint else f"{_PROMPT}\n\nNearby text (for context): {hint}"
        try:
            resp = self._http().messages.create(
                model=self._model_id,
                max_tokens=self._max_tokens,
                output_config={"format": {"type": "json_schema", "schema": _RESPONSE_SCHEMA}},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        except Exception as err:  # noqa: BLE001 - surface any SDK/transport error as one class
            raise VisionError(
                f"vision request to {self._model_id!r} failed: {type(err).__name__}: {err}"
            ) from err
        payload = _first_text(resp)
        try:
            data = json.loads(payload)
        except (ValueError, TypeError) as err:
            raise VisionError(
                f"vision model {self._model_id!r} returned non-JSON: {payload[:200]!r}"
            ) from err
        return ImageInsight(
            text=str(data.get("text", "")),
            description=str(data.get("description", "")),
            model=self._model_id,
        )


def _first_text(resp: Any) -> str:
    """The first text block of a Messages-API response (structured output lands here)."""
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            return str(block.text)
    return ""


def make_vision_provider(enrich_config: EnrichConfig) -> VisionProvider | None:
    """Build the configured vision provider, or ``None`` when ``vision_images`` is off."""
    if not enrich_config.vision_images:
        return None
    return AnthropicVisionProvider(enrich_config.vision_model)


@dataclass
class VisionResult:
    """Audit counts for one enrichment pass."""

    enriched: int = 0
    skipped: int = 0  # unsupported image type
    failed: int = 0  # missing original or a vision error
    errors: list[tuple[str, str]] = field(default_factory=list)  # (sha, message)


def enrich_images(
    store: Store,
    provider: VisionProvider,
    *,
    staging_dir: Path | str,
    limit: int | None = None,
    progress: ProgressFn | None = None,
) -> VisionResult:
    """Enrich every not-yet-enriched image: describe it, persist the result, and add a
    searchable ``enrichment`` chunk. Idempotent — already-enriched images are skipped.

    A per-image failure (missing original, vision error) is recorded and the pass continues,
    matching the operability contract (visible progress, one-line errors, never abort a batch).
    """
    images_dir = Path(staging_dir) / "images"
    todo = store.images_needing_vision(limit=limit)
    total = len(todo)
    result = VisionResult()
    for done, row in enumerate(todo, start=1):
        sha = str(row["sha256"])
        ext = str(row["ext"] or "").lower()
        media_type = MEDIA_TYPES.get(ext)
        if media_type is None:
            result.skipped += 1
            _tick(progress, done, total)
            continue
        path = images_dir / f"{sha}.{ext}"
        if not path.is_file():
            result.failed += 1
            result.errors.append((sha, "original image missing from staging"))
            _tick(progress, done, total)
            continue
        try:
            insight = provider.describe(
                path.read_bytes(),
                media_type=media_type,
                alt=str(row["alt"] or ""),
                caption=str(row["caption"] or ""),
                context=str(row["locator"] or ""),
            )
        except VisionError as err:
            result.failed += 1
            result.errors.append((sha, str(err)))
            _tick(progress, done, total)
            continue
        store.set_image_vision(sha, insight.text, insight.model)
        body = insight.searchable_text()
        if body:
            store.add_enrichment_chunk(int(row["doc_id"]), body, str(row["locator"] or ""))
        result.enriched += 1
        _tick(progress, done, total)
    return result


def _tick(progress: ProgressFn | None, done: int, total: int) -> None:
    if progress is not None:
        progress("vision", done, total)
