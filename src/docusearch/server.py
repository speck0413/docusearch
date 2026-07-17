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

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import citations, embed, report, runlog, search
from ._version import __version__
from .config import Config
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
    ) -> tuple[list[list[SearchHit]], str, str]:
        """Return (per-query results, embed_model_used, search_mode). Text queries only;
        pre-computed vectors + the 409 mismatch path arrive with the client (Phase 3b)."""
        cfg = self.config
        k = top_k if top_k is not None else cfg.search.top_k_default
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
        path = Path(self.config.paths.staging_dir) / "images" / f"{sha256}.{ext}"
        if not path.is_file():
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


class EmbedRequest(BaseModel):
    texts: list[str] = []


class ReportRequest(BaseModel):
    title: str
    body: str  # the claim text, each factual sentence ending in a [GK] or [D:...] tag
    evidence_chunk_ids: list[int] = []
    fmt: str = "md"
    audience: list[str] = []
    sources: list[str] = []


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


def build_mcp(service: Service, config: Config) -> Any:
    """The MCP server exposing the stable tool names over the same service layer (§10).

    Tool names are a stable contract agents depend on. `serve` mounts this over streamable
    HTTP at ``serve.mcp_path``. (annotate / list_versions / check_discrepancies /
    create_report arrive with their features in Phases 3b/5.)
    """
    from mcp.server.fastmcp import FastMCP

    # serve at the sub-app root so mounting at serve.mcp_path yields a clean path
    mcp: Any = FastMCP("docusearch", streamable_http_path="/")
    base = _public_base(config)

    @mcp.tool()
    def search_docs(queries: list[str], top_k: int = 10) -> dict[str, Any]:
        """Search the catalog (batch). Always pass a list of queries."""
        results, model_used, mode = service.search(queries, top_k=top_k)
        return {
            "results": [[_hit_dict(h, base) for h in lst] for lst in results],
            "embed_model_used": model_used,
            "search_mode": mode,
        }

    @mcp.tool()
    def get_document(doc_id: int, chunk: int | None = None) -> dict[str, Any] | None:
        """Fetch a document's metadata + chunks by id."""
        return service.get_document(doc_id, chunk=chunk)

    @mcp.tool()
    def related_documents(doc_id: int, direction: str = "both") -> list[dict[str, Any]]:
        """Documents linked from / to this one (direction: out | in | both)."""
        return service.relations(doc_id, direction)

    @mcp.tool()
    def catalog_stats() -> dict[str, Any]:
        """Counts + embedding model for the catalog."""
        return service.health()

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
        if req.query_vectors is not None:
            if not req.embed_model:
                raise HTTPException(status_code=400, detail="query_vectors require embed_model")
            try:
                results, model_used, mode = service.search_vectors(
                    req.query_vectors, req.embed_model, top_k=req.top_k, roles=roles
                )
            except ModelMismatchError as err:  # 409, recoverable (R-EMB-3)
                raise _mismatch_409(err) from err
        else:
            results, model_used, mode = service.search(
                req.query_texts,
                top_k=req.top_k,
                prefix=req.prefix,
                roles=roles,
                bm25_only=req.bm25_only,
            )
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
        info = service.embed_info()
        try:
            rendered = report.render_report(
                title=req.title,
                body=req.body,
                evidence_chunk_ids=set(req.evidence_chunk_ids),
                base_url=base,
                fmt=req.fmt,
                run_id=runlog.RUN_ID,
                audience=req.audience,
                embed_model=info["model"],
                sources=req.sources,
            )
        except citations.CitationError as err:
            raise HTTPException(
                status_code=400,
                detail={"error": "HALLUCINATED_CITATION", "message": str(err)},
            ) from err
        runlog.log("api.report", fmt=req.fmt, evidence=len(req.evidence_chunk_ids))
        return {"fmt": req.fmt, "report": rendered}

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
