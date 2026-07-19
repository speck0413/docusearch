"""HTTP surface: FastAPI REST + (Phase 3b) MCP over streamable HTTP (§10, R-API-*).

Both the REST routes and the MCP tools are thin wrappers over ONE internal service
layer (``Service``) — route handlers carry no logic (§10). The model + vector index load
once and stay warm (R-PERF-3); the SQLite store is opened per request (WAL, thread-safe).

This module imports FastAPI/uvicorn, so it is only imported in server mode — `import
docusearch` stays light for standalone/client users (the package exposes ``serve`` lazily).

Public surface:
    Service              -- the internal service layer (search/documents/images/…)
    create_app(config)   -- the FastAPI application
    serve(config, ...)   -- run the app with uvicorn
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import citations, embed, report, runlog, search
from ._version import __version__
from .catalog import Catalog, open_federation
from .config import Config, SourceConfig, load
from .search import SearchHit
from .store import Store

_MEDIA_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "svg": "image/svg+xml",
    "webp": "image/webp",
    "bmp": "image/bmp",
}


def _approx_mb(dim: int) -> int:
    """Rough on-disk size of the embedding model, for `auto` negotiation (R-EMB-5)."""
    if dim >= 1024:
        return 1300
    if dim >= 768:
        return 440
    return 90


def _fits_i64(value: int) -> bool:
    """SQLite integer bound — an absurd id should 404, not raise OverflowError (500)."""
    return -(2**63) <= value < 2**63


class ModelMismatchError(Exception):
    """A query's embedding model doesn't match the index's — never mix (R-EMB-3, 409)."""

    def __init__(self, message: str, *, server_model: str, server_dim: int) -> None:
        super().__init__(message)
        self.server_model = server_model
        self.server_dim = server_dim


class Service:
    """The one place search/document/image/embedding logic lives (§10). Reused by REST
    and MCP so there is exactly one implementation per concept (R-REUSE-2)."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._provider: embed.EmbedProvider | None = None
        self._provider_loaded = False
        self._vector_index: search.VectorIndex | None = None
        self._vector_index_loaded = False

    # -- lazily-warmed model + index --------------------------------------------

    def _embed_provider(self) -> embed.EmbedProvider | None:
        """The server's own embedding provider (loaded once), or None if `model: none`."""
        if not self._provider_loaded:
            cfg = self.config
            if cfg.embed.model not in ("none", "auto"):
                try:
                    self._provider = embed.make_provider(cfg.embed)
                except NotImplementedError:
                    self._provider = None
            self._provider_loaded = True
        return self._provider

    def _vector_index_or_none(self, store: Store, dim: int) -> search.VectorIndex | None:
        if not self._vector_index_loaded:
            db = self.config.paths.db_path
            ann_path = Path(db).with_suffix(".hnsw") if db != ":memory:" else "__no_ann__"
            self._vector_index = search.VectorIndex.load(store, dim, ann_path)
            self._vector_index_loaded = True
        return self._vector_index

    # -- service methods (called by REST + MCP) ---------------------------------

    def health(self) -> dict[str, Any]:
        with Store.open(self.config.paths.db_path) as store:
            return {
                "version": __version__,
                "mode": self.config.mode,
                "documents": store.count_documents(),
                "chunks": store.count_chunks(),
                "embeddings": store.count_embeddings(),
                "images": store.count_images(),
                "relations": store.count_relations(),
                "embed_model": store.get_meta("embed_model") or "none",
                "embed_dim": int(store.get_meta("embed_dim") or 0),
            }

    def embed_info(self) -> dict[str, Any]:
        with Store.open(self.config.paths.db_path) as store:
            model = store.get_meta("embed_model") or "none"
            dim = int(store.get_meta("embed_dim") or 0)
        return {"model": model, "dim": dim, "approx_mb": _approx_mb(dim) if dim else 0}

    def search(
        self,
        query_texts: list[str],
        *,
        top_k: int | None = None,
        prefix: bool = False,
        roles: set[str] | None = None,
        bm25_only: bool | None = None,
        stores: list[str] | None = None,
        user: str | None = None,
        groups: set[str] | None = None,
    ) -> tuple[list[list[SearchHit]], str, str]:
        """Return (per-query results, embed_model_used, search_mode). Text queries only;
        pre-computed vectors + the 409 mismatch path arrive with the client (Phase 3b).

        When the config declares a ``federation:``, the query fans out across its member stores
        (R-TEST-3); ``stores`` scopes it to a named subset (e.g. ``["acme"]``). ``user`` + ``groups``
        (from the request's ``X-Docusearch-User`` / ``X-Docusearch-Groups`` headers) enforce store
        **access control**: a private store the requester isn't whitelisted for is invisible — in a
        federation it silently drops out; a private single store raises ``PermissionError``."""
        cfg = self.config
        k = top_k if top_k is not None else cfg.search.top_k_default
        grp = groups or set()
        if cfg.federation:
            accessible = {
                m.name for m in cfg.federation
                if load(Path(m.config)).access.permits(user=user, groups=grp)
            }
            if stores:  # a forbidden store looks 'unknown' — never reveal a private store exists
                unknown = [s for s in stores if s not in accessible]
                if unknown:
                    raise ValueError(
                        f"unknown store(s) {unknown}; available: {sorted(accessible)}"
                    )
                effective: list[str] = list(stores)
            else:
                effective = sorted(accessible)
            with open_federation(cfg, only=effective) as fed:
                fed_results = [
                    fed.search(q, top_k=k, prefix=prefix, roles=roles) for q in query_texts
                ]
            return fed_results, "(federation)", "federated"
        if not cfg.access.permits(user=user, groups=grp):
            raise PermissionError("access denied: this store is private")
        force_bm25 = cfg.search.bm25_only if bm25_only is None else bm25_only
        provider = None if force_bm25 else self._embed_provider()
        with Store.open(cfg.paths.db_path) as store:
            vector_index = None
            if provider is not None and store.count_embeddings() > 0:
                index_model = store.get_meta("embed_model")
                if index_model is not None and index_model != provider.model_id:
                    provider = None  # never mix embedding spaces (R-EMB-3)
                else:
                    vector_index = self._vector_index_or_none(store, provider.dim)
            results = search.search(
                store,
                query_texts,
                top_k=k,
                provider=provider,
                vector_index=vector_index,
                rrf_k=cfg.search.rrf_k,
                prefix=prefix,
                roles=roles,
                bm25_only=force_bm25,
            )
        # search() returns a list-of-lists for a sequence input
        result_lists: list[list[SearchHit]] = results  # type: ignore[assignment]
        model_used = provider.model_id if provider is not None else "none"
        mode = "hybrid" if (provider is not None and vector_index is not None) else "bm25"
        return result_lists, model_used, mode

    def list_stores(self) -> dict[str, Any]:
        """The document stores a query can target. In a federation, ``stores`` lists the member
        names a search can be scoped to (pass a subset as ``stores=[...]``); empty for single-store."""
        return {
            "federated": bool(self.config.federation),
            "stores": [m.name for m in self.config.federation],
        }

    def submit_feedback(
        self, *, user: str, text: str, doc_id: int | None = None,
        chunk_id: int | None = None, rating: int | None = None,
    ) -> dict[str, Any]:
        """Record a user's feedback as an append-only JSONL line under the config's ``tmp_dir``
        (``feedback/feedback.jsonl``) — lightweight, persistent, reviewable."""
        import json
        from datetime import UTC, datetime

        entry = {
            "ts": datetime.now(UTC).isoformat(), "user": user, "text": text,
            "doc_id": doc_id, "chunk_id": chunk_id, "rating": rating,
        }
        fb_dir = Path(self.config.paths.tmp_dir) / "feedback"
        fb_dir.mkdir(parents=True, exist_ok=True)
        with (fb_dir / "feedback.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        runlog.log("api.feedback", user=user, rating=rating)
        return {"recorded": True, **entry}

    def _target_config(self, store: str | None) -> Config:
        """Resolve which store to ingest into: a named federation member (vendor / internal / user
        / acme …) or, with no name, this config's own single store."""
        if store and self.config.federation:
            for member in self.config.federation:
                if member.name == store:
                    return load(Path(member.config))
            raise ValueError(
                f"unknown store {store!r}; available: {[m.name for m in self.config.federation]}"
            )
        return self.config

    def ingest_from_path(
        self,
        path: str | Path,
        *,
        store: str | None = None,
        label: str = "upload",
        uploaded_by: str = "",
        min_content_chars: int = 1,
    ) -> dict[str, Any]:
        """Ingest a **folder** or an uploaded **.zip/.tar.gz** (uncompressed into the target store's
        staging) as a labelled source, into the chosen store (R-ING write path). ``label`` tags the
        collection; ``uploaded_by`` records who added it. Returns the ingest counts."""
        target = self._target_config(store)
        src_path = Path(path)
        if src_path.is_dir():
            location = src_path
        elif _is_archive(src_path):
            location = Path(target.paths.staging_dir) / "uploads" / label
            location.mkdir(parents=True, exist_ok=True)
            _safe_extract(src_path, location)
        else:
            raise ValueError("provide a folder or a .zip / .tar.gz archive")
        source = SourceConfig(
            type="fs", name=label, version=uploaded_by, location=str(location),
            include=[], exclude=[], content_selector="", strip_selectors=[],
            min_content_chars=min_content_chars, audience=[],
        )
        result = Catalog(replace(target, sources=[source])).ingest()
        runlog.log("api.ingest", store=store or "default", label=label, docs=result.documents,
                   by=uploaded_by)
        return {
            "store": store or "default", "label": label, "uploaded_by": uploaded_by,
            "documents": result.documents, "chunks": result.chunks, "images": result.images,
        }

    def build_report(self, spec: dict[str, Any], *, base_url: str, fmt: str = "md") -> str:
        """Render a cited report from an answer ``spec`` (title/sections/evidence/provenance),
        verifying every citation against the evidence set (R-CIT-1) — the same renderer the CLI
        uses, so a given spec yields an identical report save for the reference links: a SERVED
        report links each reference to its HTTP ``/v1/documents`` URL (reachable by a remote MCP
        client), where the local CLI links to the original ``file://`` document. Raises
        ``citations.CitationError`` on a hallucinated citation."""
        cfg = self.config
        evidence = {(int(d), int(c)) for d, c in spec.get("evidence", [])}
        sources = list(spec.get("sources", [])) or [s.name for s in cfg.sources]
        return report.render_report(
            title=str(spec.get("title", "Report")),
            subtitle=str(spec.get("subtitle", "")),
            body=str(spec.get("body", "")),
            sections=spec.get("sections"),
            evidence=evidence,
            base_url=base_url,
            fmt=fmt,
            run_id=runlog.RUN_ID,
            audience=list(spec.get("audience", [])),
            embed_model=self.embed_info()["model"],
            sources=sources,
            images=list(spec.get("images", [])),
            embedded_images=report.evidence_images(cfg.paths.db_path, cfg.paths.staging_dir, evidence),
            request=str(spec.get("request", "")),
            requested_by=str(spec.get("requested_by", "")),
            model=str(spec.get("model", "")),
            classification=str(spec.get("classification", "Confidential — Acme")),
            # Rich reference labels (store — title — heading), but served /v1/documents HTTP links a
            # remote client can open — identical to the CLI report save for the link host.
            ref_targets=report.reference_targets(cfg.paths.db_path, evidence, base_url=base_url),
            trace=spec.get("trace"),
        )

    def search_vectors(
        self,
        query_vectors: list[list[float]],
        embed_model: str,
        *,
        top_k: int | None = None,
        roles: set[str] | None = None,
    ) -> tuple[list[list[SearchHit]], str, str]:
        """Rank against client-supplied vectors after verifying the model tag (R-EMB-3).

        Raises ModelMismatchError (-> HTTP 409) if the tag doesn't match the index model.
        """
        cfg = self.config
        k = top_k if top_k is not None else cfg.search.top_k_default
        with Store.open(cfg.paths.db_path) as store:
            server_model = store.get_meta("embed_model") or "none"
            server_dim = int(store.get_meta("embed_dim") or 0)
            if server_model == "none" or embed_model != server_model:
                raise ModelMismatchError(
                    f"query vectors were made with {embed_model!r} but the index uses "
                    f"{server_model!r}",
                    server_model=server_model,
                    server_dim=server_dim,
                )
            for vec in query_vectors:  # right model, wrong dim -> clean 400, not a crash
                if len(vec) != server_dim:
                    raise ValueError(
                        f"query vector has dim {len(vec)} but the index model uses {server_dim}"
                    )
            vector_index = self._vector_index_or_none(store, server_dim)
            assert vector_index is not None
            results = [
                search.vector_search(
                    store, np.asarray(vec, dtype=np.float32), vector_index, top_k=k, roles=roles
                )
                for vec in query_vectors
            ]
        return results, server_model, "vector"

    def embed_texts(self, texts: list[str]) -> dict[str, Any]:
        """Server-side embedding for the RemoteServerProvider client path (R-EMB-4)."""
        provider = self._embed_provider()
        if provider is None:
            raise ModelMismatchError(
                "this server has no embedding model (embed.model: none)",
                server_model="none",
                server_dim=0,
            )
        vectors = provider.embed(texts)
        return {"model": provider.model_id, "dim": provider.dim, "vectors": vectors.tolist()}

    def get_document(self, doc_id: int, *, chunk: int | None = None) -> dict[str, Any] | None:
        if not _fits_i64(doc_id):  # absurd id -> 404, not a sqlite OverflowError (500)
            return None
        with Store.open(self.config.paths.db_path) as store:
            doc = store.get_document(doc_id)
            if doc is None:
                return None
            chunks = [
                {
                    "id": int(c["id"]),
                    "ord": int(c["ord"]),
                    "kind": str(c["kind"]),
                    "locator": str(c["locator"] or ""),
                    "text": str(c["text"]),
                    "highlight": chunk is not None and int(c["id"]) == chunk,
                }
                for c in store.chunks_for_document(doc_id)
            ]
        return {
            "id": int(doc["id"]),
            "path": str(doc["path"]),
            "title": str(doc["title"] or ""),
            "fmt": str(doc["fmt"] or ""),
            "doc_id": str(doc["doc_id"] or ""),
            "audience": str(doc["audience"] or "[]"),
            "status": str(doc["status"] or ""),
            "chunks": chunks,
        }

    def document_path(self, doc_id: int) -> Path | None:
        if not _fits_i64(doc_id):
            return None
        with Store.open(self.config.paths.db_path) as store:
            doc = store.get_document(doc_id)
        if doc is None:
            return None
        path = Path(str(doc["path"]))
        return path if path.is_file() else None

    def relations(self, doc_id: int, direction: str = "out") -> list[dict[str, Any]]:
        """Linked / linking documents (R-ING-5 graph). direction: out | in | both."""
        out: list[dict[str, Any]] = []
        with Store.open(self.config.paths.db_path) as store:
            if direction in ("out", "both"):
                for r in store.relations_out(doc_id):
                    out.append(
                        {
                            "neighbor": r["dst_doc"],
                            "raw": r["dst_raw"],
                            "link_type": r["link_type"],
                            "direction": "out",
                        }
                    )
            if direction in ("in", "both"):
                for r in store.relations_in(doc_id):
                    out.append(
                        {
                            "neighbor": r["src_doc"],
                            "raw": r["dst_raw"],
                            "link_type": r["link_type"],
                            "direction": "in",
                        }
                    )
        return out

    def image(self, sha256: str) -> tuple[Path, str] | None:
        with Store.open(self.config.paths.db_path) as store:
            row = store.get_image(sha256)
        if row is None:
            return None
        ext = str(row["ext"] or "bin").lower()
        images_dir = (Path(self.config.paths.staging_dir) / "images").resolve()
        path = (images_dir / f"{sha256}.{ext}").resolve()
        # defence in depth: never serve a file resolved outside the images dir
        if not path.is_relative_to(images_dir) or not path.is_file():
            return None
        return path, _MEDIA_TYPES.get(ext, "application/octet-stream")


def _hit_dict(hit: SearchHit, base_url: str) -> dict[str, Any]:
    return {
        "doc_id": hit.doc_id,
        "chunk_id": hit.chunk_id,
        "title": hit.title,
        "path": hit.path,
        "fmt": hit.fmt,
        "locator": hit.locator,
        "snippet": hit.snippet,
        "score": hit.score,
        "kind": hit.kind,
        "images": hit.images,
        "citation": hit.citation,
        "url": f"{base_url}/v1/documents/{hit.doc_id}?chunk={hit.chunk_id}",
        "embed_model_used": hit.embed_model_used,
        "search_mode": hit.search_mode,
    }


class SearchRequest(BaseModel):
    query_texts: list[str] = []
    query_vectors: list[list[float]] | None = None
    embed_model: str | None = None
    top_k: int | None = None
    prefix: bool = False
    bm25_only: bool | None = None
    roles: list[str] | None = None
    stores: list[str] | None = None  # federation: scope to named member stores


class IngestRequest(BaseModel):
    path: str  # a server-side folder OR a .zip/.tar.gz archive to uncompress + ingest
    store: str | None = None  # target federation member (vendor/internal/user/…); default single
    label: str = "upload"  # collection tag for the added docs
    min_content_chars: int = 1


class FeedbackRequest(BaseModel):
    text: str  # the user's feedback
    doc_id: int | None = None  # optional: which document/result it's about
    chunk_id: int | None = None
    rating: int | None = None  # optional thumbs / 1-5


class EmbedRequest(BaseModel):
    texts: list[str] = []


class ReportRequest(BaseModel):
    title: str
    body: str = ""  # legacy single-body form; prefer `sections` cards
    sections: list[dict[str, str]] | None = None  # [{heading, kind, body}] card layout
    subtitle: str = ""
    evidence: list[list[int]] = []  # the [doc_id, chunk_id] pairs the report is built on
    fmt: str = "md"
    audience: list[str] = []
    sources: list[str] = []
    request: str = ""  # provenance: the verbatim ask this answers
    requested_by: str = ""
    model: str = ""  # which model generated it
    trace: dict[str, Any] | None = None  # generation log (not citation-verified)


def _mismatch_409(err: ModelMismatchError) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "error": "EMBED_MODEL_MISMATCH",
            "server_model": err.server_model,
            "server_dim": err.server_dim,
            "hint": "re-send as text (query_texts) or re-embed with the server's model",
        },
    )


def _public_base(config: Config) -> str:
    return f"http://localhost:{config.serve.port}"


def _request_identity(request: Request) -> tuple[str | None, set[str]]:
    """The requester's username + groups from the ``X-Docusearch-User`` / ``X-Docusearch-Groups``
    headers (comma-separated groups). Used to enforce private-store access — a request with no
    username can still see public stores, but no private ones."""
    user = request.headers.get("X-Docusearch-User") or None
    raw = request.headers.get("X-Docusearch-Groups") or ""
    groups = {g.strip() for g in raw.split(",") if g.strip()}
    return user, groups


def _is_archive(path: Path) -> bool:
    return path.suffix.lower() == ".zip" or path.name.lower().endswith((".tar.gz", ".tgz", ".tar"))


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract a ``.zip`` / ``.tar.gz`` into ``dest``, refusing any member whose path escapes
    ``dest`` (zip-slip / tar traversal). Windows-first (pathlib), stdlib only."""
    import tarfile
    import zipfile

    dest = dest.resolve()

    def _guard(name: str) -> None:
        if not (dest / name).resolve().is_relative_to(dest):
            raise ValueError(f"unsafe path in archive: {name!r}")

    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                _guard(name)
            zf.extractall(dest)
    else:  # tar / tar.gz
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                _guard(member.name)
            tf.extractall(dest, filter="data")  # py3.12+: refuse special files / abs paths


_MCP_HELP = """# docusearch — research + cited report (MCP)

Answer the user's question ONLY from this document catalog, then (optionally) render a cited
report. Discover the domain's terminology from the search results themselves — never rely on prior
knowledge of the domain.

## Tools
- `list_stores()` -> the document stores you can search. In a FEDERATION (e.g. python / rust /
  acme) you may scope any search to a subset by name. Call this first if the user names a store.
- `search_docs(queries, top_k=10, prefix=False, stores=None, bm25_only=False, roles=None)` ->
  per-query hits {doc_id, chunk_id, citation, title, path, locator, kind, snippet, score}. ALWAYS
  pass a LIST of query phrasings (batched). `stores=["acme"]` searches only those members; omit for
  all. `prefix` = partial-term matching; `bm25_only` = skip vectors; `roles` = cooperative filter.
- `get_document(doc_id, chunk=None)` -> full chunk text — use it to fill a card with real code / a
  full procedure, not just a snippet.
- `related_documents(doc_id, direction="both")` -> cross-referenced docs (follow leads).
- `catalog_stats()` -> counts + embedding model (sanity-check the catalog is populated).
- `build_report(spec, fmt="md")` -> a themed, cited report. VERIFIES every citation against your
  evidence and refuses hallucinated ones. `fmt` is "md" or "html".

## Ground rules
- Cite everything: each catalog claim ends with `[D:<doc_id>#<chunk_id>]`; general knowledge ends
  `[GK]`. Cite the exact (doc_id, chunk_id) the fact came from.
- Don't assume acronyms — e.g. "PA" might be *Protocol Aware*, not power amplifier. Let the
  retrieved documents define the terms. If the catalog doesn't cover something, say so plainly.
- Batch your searches: send all phrasings for a round in one `search_docs` call.

## Effort (the user picks): low | medium | high
- low: one `search_docs` call (3-4 phrasings); a short, direct, cited answer.
- medium: 6-8 phrasings; read the hits; one follow-up batch; a structured multi-card report.
- high: many phrasings over several rounds; `get_document` the strongest hits; `related_documents`
  to follow leads; keep going until new searches surface nothing new.

## Workflow
1. Discover + retrieve — plan phrasings for the effort level and `search_docs` them in one batched
   call; repeat per the level. `get_document` the strongest hits for full text.
2. Select evidence — the (doc_id, chunk_id) pairs whose text actually supports your answer.
3. `build_report` with `sections` cards (kind: overview | procedure | code | hardware | config |
   test-program | warning | reference), every catalog claim carrying its `[D:doc#chunk]` inline.
   Include a `trace` (searches run + reasoning). The builder links each reference to the original
   document automatically — you do not set references.

## build_report spec (JSON object)
{title, subtitle, request, requested_by, model, audience:[...],
 evidence:[[doc_id, chunk_id], ...], sections:[{heading, kind, body}, ...],
 trace:{prompt, queries:[...], retrieved:[...], reasoning}}
"""


def build_mcp(service: Service, config: Config) -> Any:
    """The central MCP server (§10, R-API-1). Registration is deliberately **minimal** — each tool
    carries a one-line description so connecting the server costs almost no context. An agent that
    actually needs docusearch calls ``help()`` first to pull the full research + report workflow
    (identical to the CLI skill). Tool names are a stable contract; ``serve`` mounts this over
    streamable HTTP at ``serve.mcp_path``."""
    from mcp.server.fastmcp import FastMCP

    # serve at the sub-app root so mounting at serve.mcp_path yields a clean path
    mcp: Any = FastMCP("docusearch", streamable_http_path="/")
    base = _public_base(config)

    @mcp.tool()
    def help() -> str:  # noqa: A001 - the tool name agents look for
        """Call FIRST — the full docusearch research + cited-report workflow, tools, and rules."""
        return _MCP_HELP

    @mcp.tool()
    def list_stores() -> dict[str, Any]:
        """The document stores you can search; in a federation, the member names to scope to."""
        return service.list_stores()

    @mcp.tool()
    def search_docs(
        queries: list[str],
        top_k: int = 10,
        prefix: bool = False,
        stores: list[str] | None = None,
        bm25_only: bool = False,
        roles: list[str] | None = None,
        user: str | None = None,
        groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search the catalog (batch: pass a LIST). `stores` scopes to federation members; `user`/
        `groups` gate access to private stores (forward the authenticated user)."""
        try:
            results, model_used, mode = service.search(
                queries, top_k=top_k, prefix=prefix, bm25_only=bm25_only,
                roles=set(roles) if roles else None, stores=stores,
                user=user, groups=set(groups) if groups else None,
            )
        except (ValueError, PermissionError) as err:
            return {"error": "ACCESS", "message": str(err), "results": []}
        return {
            "results": [[_hit_dict(h, base) for h in lst] for lst in results],
            "embed_model_used": model_used,
            "search_mode": mode,
        }

    @mcp.tool()
    def get_document(doc_id: int, chunk: int | None = None) -> dict[str, Any] | None:
        """Fetch a document's metadata + full chunk text by id."""
        return service.get_document(doc_id, chunk=chunk)

    @mcp.tool()
    def related_documents(doc_id: int, direction: str = "both") -> list[dict[str, Any]]:
        """Documents linked from / to this one (direction: out | in | both)."""
        return service.relations(doc_id, direction)

    @mcp.tool()
    def catalog_stats() -> dict[str, Any]:
        """Counts + embedding model for the catalog."""
        return service.health()

    @mcp.tool()
    def ingest_docs(
        path: str, user: str, store: str | None = None, label: str = "upload"
    ) -> dict[str, Any]:
        """Ingest a server-side folder or .zip/.tar.gz into `store`, labelled + attributed to `user`."""
        try:
            return service.ingest_from_path(path, store=store, label=label, uploaded_by=user)
        except (ValueError, FileNotFoundError) as err:
            return {"error": "INGEST", "message": str(err)}

    @mcp.tool()
    def submit_feedback(
        text: str, user: str, doc_id: int | None = None, chunk_id: int | None = None,
        rating: int | None = None,
    ) -> dict[str, Any]:
        """Record a user's feedback (attributed to `user`)."""
        return service.submit_feedback(
            user=user, text=text, doc_id=doc_id, chunk_id=chunk_id, rating=rating
        )

    @mcp.tool()
    def build_report(spec: dict[str, Any], fmt: str = "md") -> dict[str, Any]:
        """Render a cited report from an answer spec; verifies citations, refuses hallucinated ones."""
        try:
            rendered = service.build_report(spec, base_url=base, fmt=fmt)
        except citations.CitationError as err:
            return {"error": "HALLUCINATED_CITATION", "message": str(err)}
        return {"fmt": fmt, "report": rendered}

    return mcp


def create_app(config: Config) -> FastAPI:
    """Build the FastAPI app + MCP. All routes are thin wrappers over ``Service`` (§10)."""
    from contextlib import asynccontextmanager

    service = Service(config)
    mcp = build_mcp(service, config)
    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        # run the MCP session manager's lifespan alongside the app (streamable HTTP)
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    app = FastAPI(title="docusearch", version=__version__, lifespan=lifespan)

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/v1/embed-info")
    def embed_info() -> dict[str, Any]:
        return service.embed_info()

    @app.post("/v1/search")
    def search_route(req: SearchRequest, request: Request) -> dict[str, Any]:
        roles = set(req.roles) if req.roles else None
        user, groups = _request_identity(request)
        if req.query_vectors is not None:
            if not req.embed_model:
                raise HTTPException(status_code=400, detail="query_vectors require embed_model")
            try:
                results, model_used, mode = service.search_vectors(
                    req.query_vectors, req.embed_model, top_k=req.top_k, roles=roles
                )
            except ModelMismatchError as err:  # 409, recoverable (R-EMB-3)
                raise _mismatch_409(err) from err
            except ValueError as err:  # wrong-dimension vector -> clean 400
                raise HTTPException(status_code=400, detail=str(err)) from err
        else:
            try:
                results, model_used, mode = service.search(
                    req.query_texts,
                    top_k=req.top_k,
                    prefix=req.prefix,
                    roles=roles,
                    bm25_only=req.bm25_only,
                    stores=req.stores,
                    user=user,
                    groups=groups,
                )
            except ValueError as err:  # unknown federation store name -> clean 400
                raise HTTPException(status_code=400, detail=str(err)) from err
            except PermissionError as err:  # private store, requester not whitelisted -> 403
                raise HTTPException(status_code=403, detail=str(err)) from err
        base = str(request.base_url).rstrip("/")
        runlog.log("api.search", queries=len(results), mode=mode)
        return {
            "results": [[_hit_dict(h, base) for h in lst] for lst in results],
            "embed_model_used": model_used,
            "search_mode": mode,
            "run_id": runlog.RUN_ID,
        }

    @app.post("/v1/embed")
    def embed_route(req: EmbedRequest) -> dict[str, Any]:
        try:
            return service.embed_texts(req.texts)
        except ModelMismatchError as err:
            raise HTTPException(status_code=409, detail={"error": "NO_EMBED_MODEL"}) from err

    @app.get("/v1/documents/{doc_id}")
    def get_document(doc_id: int, chunk: int | None = None, download: int = 0) -> Any:
        if download:
            path = service.document_path(doc_id)
            if path is None:
                raise HTTPException(status_code=404, detail="document file not found")
            return FileResponse(path)
        doc = service.get_document(doc_id, chunk=chunk)
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        return doc

    @app.get("/v1/images/{sha256}")
    def get_image(sha256: str) -> FileResponse:
        img = service.image(sha256)
        if img is None:
            raise HTTPException(status_code=404, detail="image not found")
        path, media = img
        return FileResponse(path, media_type=media)

    @app.get("/v1/relations/{doc_id}")
    def relations(doc_id: int, direction: str = "both") -> list[dict[str, Any]]:
        return service.relations(doc_id, direction)

    @app.post("/v1/reports")
    def reports_route(req: ReportRequest, request: Request) -> dict[str, Any]:
        base = str(request.base_url).rstrip("/")
        spec: dict[str, Any] = {
            "title": req.title,
            "body": req.body,
            "sections": req.sections,
            "subtitle": req.subtitle,
            "evidence": req.evidence,
            "audience": req.audience,
            "sources": req.sources,
            "request": req.request,
            "requested_by": req.requested_by,
            "model": req.model,
            "trace": req.trace,
        }
        try:
            rendered = service.build_report(spec, base_url=base, fmt=req.fmt)
        except citations.CitationError as err:
            raise HTTPException(
                status_code=400,
                detail={"error": "HALLUCINATED_CITATION", "message": str(err)},
            ) from err
        runlog.log("api.report", fmt=req.fmt, evidence=len(req.evidence))
        return {"fmt": req.fmt, "report": rendered}

    def _require_user(request: Request) -> str:
        # A write always records who did it — the username must be supplied (R write-auth).
        user, _ = _request_identity(request)
        if not user:
            raise HTTPException(status_code=401, detail="X-Docusearch-User header required to ingest")
        return user

    @app.post("/v1/feedback")
    def feedback_route(req: FeedbackRequest, request: Request) -> dict[str, Any]:
        """Record user feedback (attributed to X-Docusearch-User)."""
        user = _require_user(request)
        return service.submit_feedback(
            user=user, text=req.text, doc_id=req.doc_id, chunk_id=req.chunk_id, rating=req.rating
        )

    @app.post("/v1/ingest")
    def ingest_route(req: IngestRequest, request: Request) -> dict[str, Any]:
        """Ingest a server-side folder or .zip/.tar.gz path into a store, labelled + attributed."""
        user = _require_user(request)
        try:
            return service.ingest_from_path(
                req.path, store=req.store, label=req.label, uploaded_by=user,
                min_content_chars=req.min_content_chars,
            )
        except (ValueError, FileNotFoundError) as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

    @app.post("/v1/ingest/upload")
    async def ingest_upload_route(
        request: Request,
        file: UploadFile = File(...),  # noqa: B008 - FastAPI dependency default
        store: str | None = Form(None),  # noqa: B008
        label: str = Form("upload"),  # noqa: B008
    ) -> dict[str, Any]:
        """Upload a .zip/.tar.gz (multipart), uncompress it server-side, and ingest it labelled."""
        user = _require_user(request)
        import tempfile

        suffix = Path(file.filename or "upload.zip").suffix or ".zip"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            archive = Path(tmp.name)
        try:
            return service.ingest_from_path(archive, store=store, label=label, uploaded_by=user)
        except (ValueError, FileNotFoundError) as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        finally:
            archive.unlink(missing_ok=True)

    # MCP over streamable HTTP, same service layer, at serve.mcp_path (R-API-1)
    app.mount(config.serve.mcp_path, mcp_app)
    return app


def serve(  # pragma: no cover - blocking uvicorn entry, exercised by manual/e2e run
    config: Config, *, host: str | None = None, port: int | None = None
) -> None:
    """Run the REST (and, Phase 3b, MCP) server with uvicorn."""
    import uvicorn

    runlog.configure(Path(config.paths.tmp_dir) / "logs", level=config.logging.level)
    runlog.log("serve.start", host=host or config.serve.host, port=port or config.serve.port)
    uvicorn.run(create_app(config), host=host or config.serve.host, port=port or config.serve.port)
