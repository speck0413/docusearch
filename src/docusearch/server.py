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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import embed, runlog, search
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

    def _provider_or_none(self) -> embed.EmbedProvider | None:
        if not self._provider_loaded:
            cfg = self.config
            if cfg.embed.model != "none" and not cfg.search.bm25_only:
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
        provider = None if force_bm25 else self._provider_or_none()
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
    top_k: int | None = None
    prefix: bool = False
    bm25_only: bool | None = None
    roles: list[str] | None = None


def create_app(config: Config) -> FastAPI:
    """Build the FastAPI app. All routes are thin wrappers over ``Service`` (§10)."""
    service = Service(config)
    app = FastAPI(title="docusearch", version=__version__)

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/v1/embed-info")
    def embed_info() -> dict[str, Any]:
        return service.embed_info()

    @app.post("/v1/search")
    def search_route(req: SearchRequest, request: Request) -> dict[str, Any]:
        results, model_used, mode = service.search(
            req.query_texts,
            top_k=req.top_k,
            prefix=req.prefix,
            roles=set(req.roles) if req.roles else None,
            bm25_only=req.bm25_only,
        )
        base = str(request.base_url).rstrip("/")
        runlog.log("api.search", queries=len(req.query_texts), mode=mode)
        return {
            "results": [[_hit_dict(h, base) for h in lst] for lst in results],
            "embed_model_used": model_used,
            "search_mode": mode,
            "run_id": runlog.RUN_ID,
        }

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

    return app


def serve(  # pragma: no cover - blocking uvicorn entry, exercised by manual/e2e run
    config: Config, *, host: str | None = None, port: int | None = None
) -> None:
    """Run the REST (and, Phase 3b, MCP) server with uvicorn."""
    import uvicorn

    runlog.configure(Path(config.paths.tmp_dir) / "logs", level=config.logging.level)
    runlog.log("serve.start", host=host or config.serve.host, port=port or config.serve.port)
    uvicorn.run(create_app(config), host=host or config.serve.host, port=port or config.serve.port)
