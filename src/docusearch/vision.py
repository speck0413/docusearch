"""Vision enrichment — image OCR + description (enrich.vision_images, R-ING-9).

Retained images (R-ING-6) carry only alt/caption text, so a block diagram or a screenshot
of an instrument's test/debug display is invisible to search. When ``enrich.vision_images``
is on, ``docusearch vision`` sends each retained image **once** to a vision model; the
returned OCR text + description are persisted (keyed by the image sha256) and inserted as a
searchable ``enrichment`` chunk.

Three interchangeable backends (``enrich.vision_provider``), so no single credential path is
required:

* ``claude-cli``  — shell out to the ``claude`` CLI. Uses the operator's existing Claude Code
  login, so **no API key** is needed (docusearch's natural companion path).
* ``anthropic``   — the Anthropic Messages API directly (needs ``ANTHROPIC_API_KEY`` or an
  ``ant auth login`` profile).
* ``local``       — a local multimodal model via ``transformers`` (offline; e.g. a small Gemma
  image-text model). Needs the ``[vision-local]`` extra.

**Determinism (R-SRCH-5) is by persistence, not sampling.** Whatever the backend, the model is
called only at enrichment time and the output is stored; search/ranking runs over the stored
text and never re-calls the model, so ranked results stay byte-identical across runs.

Public surface:
    VisionError
    ImageInsight               -- (text, description, model) result of one image
    VisionProvider (Protocol)  -- model_id; describe(image_path, ...) -> ImageInsight
    ClaudeCliVisionProvider    -- the `claude` CLI backend (no API key)
    AnthropicVisionProvider    -- the Anthropic Messages API backend
    LocalVisionProvider        -- a local transformers backend (offline)
    make_vision_provider(cfg)  -> VisionProvider | None   (None == vision off)
    enrich_images(store, provider, staging_dir=...) -> VisionResult
    MEDIA_TYPES                -- image extension -> media_type (the raster types we enrich)
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

# Raster formats every backend handles. Vector (svg) and document (pdf) formats aren't images
# to these models — they're skipped by enrich_images before any provider is called.
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

# Ask for exactly this shape; a JSON object embedded anywhere in the reply is parsed out.
_JSON_INSTRUCTION = (
    'Reply with ONLY a JSON object of the form {"text": "<transcription>", '
    '"description": "<description>"} and no other prose.'
)

# Structured-output schema for the Anthropic backend (Opus 4.8 / Sonnet 5 guarantee this shape).
_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "description": {"type": "string"}},
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
    """Anything that turns an image file into an :class:`ImageInsight`, tagged by model."""

    @property
    def model_id(self) -> str: ...

    def describe(
        self,
        image_path: Path,
        *,
        media_type: str,
        alt: str = "",
        caption: str = "",
        context: str = "",
    ) -> ImageInsight: ...


def _prompt_for(alt: str, caption: str, context: str) -> str:
    hint = " ".join(p for p in (context, caption, alt) if p).strip()
    base = _PROMPT if not hint else f"{_PROMPT}\n\nNearby text (for context): {hint}"
    return f"{base}\n\n{_JSON_INSTRUCTION}"


def _insight_from_json(raw: str, model: str) -> ImageInsight:
    """Parse ``{"text":..,"description":..}`` out of a model reply (tolerates surrounding prose
    or code fences by taking the outermost JSON object). One parser for all three backends."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise VisionError(f"vision output had no JSON object: {raw[:200]!r}")
    try:
        data = json.loads(raw[start : end + 1])
    except (ValueError, TypeError) as err:
        raise VisionError(f"vision output was not valid JSON: {raw[start : end + 1][:200]!r}") from err
    return ImageInsight(
        text=str(data.get("text", "")),
        description=str(data.get("description", "")),
        model=model,
    )


class ClaudeCliVisionProvider:
    """Describes images by shelling out to the ``claude`` CLI in headless mode (``-p``).

    Uses the operator's existing Claude Code authentication — **no API key**. The CLI reads the
    image with its ``Read`` tool (allow-listed) and returns a JSON object. ``runner`` is
    injectable for testing; by default it runs the real subprocess.
    """

    def __init__(
        self,
        model_id: str,
        *,
        runner: Callable[[list[str]], tuple[int, str, str]] | None = None,
        cli: str = "claude",
        timeout: float = 180.0,
    ) -> None:
        self._model_id = model_id
        self._runner = runner
        self._cli = cli
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return f"claude-cli:{self._model_id}"

    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        if self._runner is not None:
            return self._runner(argv)
        import subprocess  # lazy: only when actually enriching

        proc = subprocess.run(  # noqa: S603 - argv is built here, not from untrusted input
            argv, capture_output=True, text=True, timeout=self._timeout
        )
        return proc.returncode, proc.stdout, proc.stderr

    def describe(
        self,
        image_path: Path,
        *,
        media_type: str,
        alt: str = "",
        caption: str = "",
        context: str = "",
    ) -> ImageInsight:
        prompt = (
            f"{_prompt_for(alt, caption, context)}\n\n"
            f"The image to analyze is the local file:\n{image_path}\n"
            "Use the Read tool to view it before answering."
        )
        argv = [
            self._cli,
            "-p",
            prompt,
            "--model",
            self._model_id,
            "--allowedTools",
            "Read",
            "--output-format",
            "json",
        ]
        code, out, err = self._run(argv)
        if code != 0:
            raise VisionError(
                f"claude CLI vision failed (exit {code}): {(err or out).strip()[:200]}"
            )
        text = out.strip()
        try:  # `--output-format json` wraps the reply in a result envelope
            env = json.loads(text)
        except json.JSONDecodeError:
            env = None
        if isinstance(env, dict) and "result" in env:
            if env.get("is_error"):
                raise VisionError(f"claude CLI returned an error: {str(env['result'])[:200]}")
            text = str(env["result"])
        return _insight_from_json(text, self.model_id)


class AnthropicVisionProvider:
    """Describes images with an Anthropic vision model via the Messages API.

    The client loads lazily (the ``anthropic`` SDK is the [vision] extra). Credentials resolve
    from ``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile — no key is read from config. No
    sampling params are sent (Opus 4.8 / Sonnet 5 reject them); reproducibility comes from
    persisting the result.
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
        image_path: Path,
        *,
        media_type: str,
        alt: str = "",
        caption: str = "",
        context: str = "",
    ) -> ImageInsight:
        b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
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
                            {"type": "text", "text": _prompt_for(alt, caption, context)},
                        ],
                    }
                ],
            )
        except Exception as err:  # noqa: BLE001 - surface any SDK/transport error as one class
            raise VisionError(
                f"vision request to {self._model_id!r} failed: {type(err).__name__}: {err}"
            ) from err
        return _insight_from_json(_first_text(resp), self._model_id)


class LocalVisionProvider:
    """Describes images with a local ``transformers`` image-text-to-text model (offline).

    Aimed at a small multimodal model (e.g. a Gemma image-text model) so vision can run with no
    cloud call and no key. The pipeline loads lazily (heavy) and is injectable for testing.
    """

    def __init__(
        self, model_id: str, *, pipeline: Any = None, max_new_tokens: int = 512
    ) -> None:
        self._model_id = model_id
        self._pipeline = pipeline
        self._max_new_tokens = max_new_tokens

    @property
    def model_id(self) -> str:
        return self._model_id

    def _pipe(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline  # lazy: [vision-local] extra

            self._pipeline = pipeline("image-text-to-text", model=self._model_id)
        return self._pipeline

    def describe(
        self,
        image_path: Path,
        *,
        media_type: str,
        alt: str = "",
        caption: str = "",
        context: str = "",
    ) -> ImageInsight:
        try:
            from PIL import Image  # lazy: [vision-local] extra

            with Image.open(image_path) as opened:
                image = opened.convert("RGB")  # load fully so the file handle can close
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": _prompt_for(alt, caption, context)},
                    ],
                }
            ]
            out = self._pipe()(messages, max_new_tokens=self._max_new_tokens)
        except Exception as err:  # noqa: BLE001 - any load/inference failure -> one class
            raise VisionError(
                f"local vision model {self._model_id!r} failed: {type(err).__name__}: {err}"
            ) from err
        return _insight_from_json(_local_reply_text(out), self._model_id)


def _first_text(resp: Any) -> str:
    """The first text block of a Messages-API response (structured output lands here)."""
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            return str(block.text)
    return ""


def _local_reply_text(out: Any) -> str:
    """The assistant text from a transformers image-text-to-text result (shape varies by
    version): a list of dicts with ``generated_text`` that is either a plain string or a chat
    list whose last turn holds the reply."""
    item = out[0] if isinstance(out, list) and out else out
    gen = item.get("generated_text") if isinstance(item, dict) else item
    if isinstance(gen, list) and gen:
        content = gen[-1].get("content") if isinstance(gen[-1], dict) else gen[-1]
        return str(content)
    return str(gen)


def make_vision_provider(enrich_config: EnrichConfig) -> VisionProvider | None:
    """Build the configured vision provider, or ``None`` when ``vision_images`` is off."""
    if not enrich_config.vision_images:
        return None
    provider = enrich_config.vision_provider
    if provider == "claude-cli":
        return ClaudeCliVisionProvider(enrich_config.vision_model)
    if provider == "local":
        return LocalVisionProvider(enrich_config.vision_model)
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
                path,
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
