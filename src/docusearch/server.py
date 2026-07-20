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

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import citations, embed, report, report_export, report_store, runlog, search
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

# Served for a generated report file. html/md render in the browser; the OOXML types are the
# official ones so a download lands with the right icon and opens in the right application.
_REPORT_MEDIA = {
    ".html": "text/html; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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


def _wafer_result(html: str) -> dict[str, Any]:
    """Wrap a wafer/mother-lot/trend page. Empty-state pages (:func:`wafer.empty_note`) also carry an
    ``empty``/``note`` so the CLI can fail with a one-line reason instead of writing a blank report;
    MCP/REST callers still get the graceful HTML either way."""
    from . import wafer

    note = wafer.empty_note(html)
    return {"html": html} if note is None else {"html": html, "empty": True, "note": note}


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
                "store_type": self.config.store_type,  # document | data — a client routes by this
                "documents": store.count_documents(),
                "chunks": store.count_chunks(),
                "embeddings": store.count_embeddings(),
                "images": store.count_images(),
                "relations": store.count_relations(),
                "stdf_results": store.count_stdf_results(),
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
        (R-TEST-3); ``stores`` scopes it to a named subset (e.g. ``["internal"]``). ``user`` + ``groups``
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
            # feedback-aware re-rank across members: each hit boosted by its member's tier + that
            # member's feedback for this user (feedback > internal > vendor), then re-sorted (Phase 8)
            fed_results = [self._rerank_federated(cfg, lst, user) for lst in fed_results]
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
            # feedback-aware re-rank (Phase 8): boost by source tier + this user's feedback
            result_lists = self._rerank_with_feedback(store, results, user)  # type: ignore[arg-type]
        model_used = provider.model_id if provider is not None else "none"
        mode = "hybrid" if (provider is not None and vector_index is not None) else "bm25"
        return result_lists, model_used, mode

    def _rerank_federated(
        self, cfg: Config, hits: list[SearchHit], user: str | None
    ) -> list[SearchHit]:
        """Re-rank a federation's merged hits by each hit's **member tier** (internal > vendor) and
        that member's feedback for the requesting user (feedback > internal > vendor), then re-sort
        (Phase 8). Each member's feedback lives in its own store; a no-op when weights are all 0."""
        rk = self.config.ranking
        if rk.internal_boost == 0 and rk.vendor_boost == 0 and rk.feedback_weight == 0:
            return hits
        member_tier = {m.name: m.tier for m in cfg.federation}
        boost = {"internal": rk.internal_boost, "vendor": rk.vendor_boost}
        needed = {h.store for h in hits if h.store}
        fb_by_member: dict[str, dict[int, int]] = {}
        for m in cfg.federation:
            if m.name in needed:
                with Store.open(load(Path(m.config)).paths.db_path) as db:
                    fb_by_member[m.name] = db.feedback_scores(author=user)
        scored = []
        for h in hits:
            tier = member_tier.get(h.store, "vendor")
            fb = fb_by_member.get(h.store, {}).get(h.doc_id, 0)
            delta = boost.get(tier, rk.vendor_boost) + rk.feedback_weight * fb
            scored.append((h.score + delta, h))
        scored.sort(key=lambda sh: sh[0], reverse=True)
        return [replace(h, score=round(sc, 6)) for sc, h in scored]

    def _rerank_with_feedback(
        self, store: Store, result_lists: list[list[SearchHit]], user: str | None
    ) -> list[list[SearchHit]]:
        """Adjust each hit's score by its **source tier** (internal > vendor) and the requesting
        user's **net feedback** on that document (feedback > internal > vendor), then re-sort
        (Phase 8, R-FB). A no-op when all ranking weights are 0. Single-store only for now — the
        federation path merges members before this and is a follow-up."""
        rk = self.config.ranking
        if rk.internal_boost == 0 and rk.vendor_boost == 0 and rk.feedback_weight == 0:
            return result_lists
        tier_of = {s.name: s.tier for s in self.config.sources}
        boost = {"internal": rk.internal_boost, "vendor": rk.vendor_boost}
        fb = store.feedback_scores(author=user)
        src = store.document_source_map(
            [h.doc_id for lst in result_lists for h in lst]
        )
        out: list[list[SearchHit]] = []
        for lst in result_lists:
            scored = []
            for h in lst:
                tier = tier_of.get(src.get(h.doc_id, ""), "vendor")
                delta = boost.get(tier, rk.vendor_boost) + rk.feedback_weight * fb.get(h.doc_id, 0)
                scored.append((h.score + delta, h))
            scored.sort(key=lambda sh: sh[0], reverse=True)
            out.append([replace(h, score=round(sc, 6)) for sc, h in scored])
        return out

    def check_read_access(self, user: str | None, groups: set[str] | None) -> None:
        """Raise ``PermissionError`` if a SINGLE private store may not be read by this requester —
        the gate every read endpoint (get_document, images, relations, health, …) calls so a private
        store's content/metadata is never reachable by a caller who couldn't search it (red-team H1).
        A federation's own store is empty; its members are gated per-member at search/list time."""
        if not self.config.federation and not self.config.access.permits(
            user=user, groups=groups or set()
        ):
            raise PermissionError("access denied: this store is private")

    def list_stores(self, *, user: str | None = None, groups: set[str] | None = None) -> dict[str, Any]:
        """The document stores a query can target. In a federation, lists only the member names the
        requester may access (a private member the caller isn't whitelisted for is omitted — its
        existence isn't leaked, red-team H2). Empty for a single store."""
        grp = groups or set()
        names = [
            m.name for m in self.config.federation
            if load(Path(m.config)).access.permits(user=user, groups=grp)
        ]
        return {"federated": bool(self.config.federation), "stores": names}

    def submit_feedback(
        self, *, user: str, text: str, doc_id: int | None = None,
        chunk_id: int | None = None, rating: int | None = None, make_global: bool = False,
        store: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """Record a user's feedback in the store (Phase 8). It is **private to ``user``** by default;
        ``make_global`` promotes it to everyone. A ``rating`` (-1/0/+1) with a ``doc_id`` target makes
        it ranking-eligible. In a federation, ``store`` names the member the target doc belongs to so
        the feedback lands where that member's re-rank reads it. Also mirrored to JSONL for review.

        Writing is gated on the target store's access policy (red-team #H1) — a caller who couldn't
        search the store cannot write feedback into it — and absurd ids are refused (red-team #M1)."""
        import json

        if (doc_id is not None and not _fits_i64(doc_id)) or (
            chunk_id is not None and not _fits_i64(chunk_id)
        ):
            raise ValueError("doc_id/chunk_id out of range")
        target = self._target_config(store)
        if not target.access.permits(user=user or None, groups=groups or set()):
            raise PermissionError(
                f"access denied: cannot submit feedback to store {store or 'default'!r}")
        scope = "global" if make_global else "user"
        target_db = target.paths.db_path
        with Store.open(target_db) as db:
            fb_id = db.add_feedback(
                author=user, scope=scope, text=text, doc_id=doc_id, chunk_id=chunk_id, rating=rating,
            )
        entry = {
            "id": fb_id, "user": user, "scope": scope, "text": text,
            "doc_id": doc_id, "chunk_id": chunk_id, "rating": rating,
        }
        fb_dir = Path(self.config.paths.tmp_dir) / "feedback"
        fb_dir.mkdir(parents=True, exist_ok=True)
        with (fb_dir / "feedback.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        runlog.log("api.feedback", user=user, scope=scope, rating=rating)
        return {"recorded": True, **entry}

    def discrepancies(
        self, *, store: str | None = None, persist: bool = False,
        user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """Discrepancy scan for one store (§17 Phase 5) — duplicate active docs + conflict
        candidates. Read-gated for a private store; ``persist`` records ``discrepancy`` flags."""
        from .catalog import Catalog

        cfg = self._target_config(store)
        if not cfg.access.permits(user=user, groups=groups or set()):
            raise PermissionError("access denied: this store is private")
        report = Catalog(cfg).check_discrepancies(persist=persist)
        # enrich with paths + the source's authority TIER so a cross-tier disagreement (internal vs
        # vendor saying near-identical-but-possibly-conflicting things) is flagged with which tier
        # wins (feedback > internal > vendor, R-FB) — the "log which system says what" ask (#63).
        with Store.open(cfg.paths.db_path) as store_db:
            meta = {
                int(r["id"]): (str(r["path"] or ""), str(r["source"] or ""))
                for r in store_db._conn.execute("SELECT id, path, source FROM documents")  # noqa: SLF001
            }
        tier_of = {s.name: s.tier for s in cfg.sources}
        conflicts, cross_tier = _annotate_conflict_tiers(report.conflict_candidates, meta, tier_of)
        if cross_tier:
            runlog.log("api.discrepancy.cross_tier", store=store or "default", count=cross_tier)
        return {
            "duplicate_actives": [
                {
                    "content_hash": g.content_hash,
                    "docs": [{"doc_id": d, "path": p} for d, p in g.docs],
                }
                for g in report.duplicate_actives
            ],
            "conflict_candidates": conflicts,
            "cross_tier_conflicts": cross_tier,
            "persisted": persist,
        }

    def _target_config(self, store: str | None) -> Config:
        """Resolve which store to ingest into: a named federation member (vendor / internal / user
        / internal …) or, with no name, this config's own single store. A named store errors if there is
        no federation or the name is unknown — never a silent misroute to the default (red-team M4)."""
        if not store:
            return self.config
        if not self.config.federation:
            raise ValueError(f"no 'federation:' configured; cannot target store {store!r}")
        for member in self.config.federation:
            if member.name == store:
                return load(Path(member.config))
        raise ValueError(
            f"unknown store {store!r}; available: {[m.name for m in self.config.federation]}"
        )

    def ingest_from_path(
        self,
        path: str | Path,
        *,
        store: str | None = None,
        label: str = "upload",
        uploaded_by: str = "",
        groups: set[str] | None = None,
        min_content_chars: int = 1,
        insertion: str = "",
    ) -> dict[str, Any]:
        """Ingest a **folder** or an uploaded **.zip/.tar.gz** (uncompressed into the target store's
        staging) as a labelled source, into the chosen store (R-ING write path). ``label`` tags the
        collection; ``uploaded_by`` records who added it. Writing to a **private** store requires the
        uploader be whitelisted for it (red-team H3). A server-side folder path must sit under the
        target store's ``inbound`` staging area — no arbitrary-filesystem read (red-team H4).
        Returns the ingest counts."""
        target = self._target_config(store)
        if not target.access.permits(user=uploaded_by or None, groups=groups or set()):
            raise PermissionError(f"access denied: cannot write to store {store or 'default'!r}")
        # `label` (the upload SKU) is untrusted and is used as a path component below — a SKU with a
        # path separator or `..` would let the extract escape the staging tree (red-team phase6b H1).
        if "/" in label or "\\" in label or ".." in Path(label).parts:
            raise ValueError(
                f"invalid SKU/label {label!r}: must not contain path separators or '..'"
            )
        inbound = (Path(target.paths.staging_dir) / "inbound").resolve()
        src_path = Path(path)
        if src_path.is_dir():
            if not src_path.resolve().is_relative_to(inbound):
                raise ValueError(
                    f"folder ingest is confined to the store's inbound dir ({inbound}); "
                    "place files there or use the upload endpoint"
                )
            location = src_path
        elif _is_archive(src_path):
            uploads = (Path(target.paths.staging_dir) / "uploads").resolve()
            location = uploads / label
            if not location.resolve().is_relative_to(uploads):  # defence in depth beyond the check above
                raise ValueError(f"invalid SKU/label {label!r}: escapes the store's uploads dir")
            location.mkdir(parents=True, exist_ok=True)
            _safe_extract(src_path, location)
        else:
            raise ValueError("provide a folder or a .zip / .tar.gz archive")
        source = SourceConfig(
            type="fs", name=label, version=uploaded_by, location=str(location),
            include=[], exclude=[], content_selector="", strip_selectors=[],
            min_content_chars=min_content_chars, audience=[], insertion=insertion,
        )
        result = Catalog(replace(target, sources=[source])).ingest()
        runlog.log("api.ingest", store=store or "default", label=label, docs=result.documents,
                   by=uploaded_by)
        return {
            "store": store or "default", "label": label, "uploaded_by": uploaded_by,
            "documents": result.documents, "chunks": result.chunks, "images": result.images,
        }

    def build_report_file(
        self, spec: dict[str, Any], *, base_url: str, fmt: str = "md"
    ) -> dict[str, Any]:
        """Render a report, write it under ``tmp_dir/reports/``, and return a clickable URL.

        All six formats come back the same way, so an agent's terminal step never depends on the
        format it was configured with. ``md``/``html`` also carry the text inline (small, and the
        agent may want to quote it); the binary formats do not — a base64'd pptx through the
        model's context is exactly the cost the compact search payload exists to avoid.

        The citation guard runs on every path: ``render_report`` and ``export_report`` each verify
        the body against the evidence set first, so a hallucinated citation is refused in a pptx
        exactly as in HTML (R-CIT-1)."""
        fmt = (fmt or "md").lower()
        assets = str(report_store.reports_dir(self.config.paths.tmp_dir) / "assets")
        rendered = self.build_report(
            spec, base_url=base_url, fmt=fmt, asset_dir=assets if fmt == "md" else ""
        )
        name = report_store.filename(str(spec.get("title", "report")), runlog.RUN_ID, fmt)
        path = report_store.write(self.config.paths.tmp_dir, name, rendered)
        report_store.sweep(self.config.paths.tmp_dir, self.config.reports.retain_days)
        out: dict[str, Any] = {
            "fmt": fmt,
            "filename": path.name,
            "url": f"{base_url}/v1/reports/{path.name}",
            "path": str(path),
            "bytes": path.stat().st_size,
        }
        if isinstance(rendered, str):
            out["report"] = rendered
        return out

    def build_report(
        self, spec: dict[str, Any], *, base_url: str, fmt: str = "md", asset_dir: str = ""
    ) -> str | bytes:
        """Render a cited report from an answer ``spec`` (title/sections/evidence/provenance),
        verifying every citation against the evidence set (R-CIT-1) — the same renderer the CLI
        uses, so a given spec yields an identical report save for the reference links: a SERVED
        report links each reference to its HTTP ``/v1/documents`` URL (reachable by a remote MCP
        client), where the local CLI links to the original ``file://`` document. Raises
        ``citations.CitationError`` on a hallucinated citation."""
        cfg = self.config
        fmt = (fmt or "md").lower()
        evidence = {(int(d), int(c)) for d, c in spec.get("evidence", [])}
        sources = list(spec.get("sources", [])) or [s.name for s in cfg.sources]
        # Every image the spec references, from its sections and the top level alike.
        shas: list[str] = [str(x) for x in spec.get("images", []) or []]
        for sec in spec.get("sections") or []:
            if isinstance(sec, dict):
                shas += [str(x) for x in (sec.get("images") or [])]

        if fmt in report_export.EXPORT_FORMATS:
            # Resolve to files so a deck/document can SHOW the diagram, not just describe it.
            # Anything unresolvable is skipped, never an error.
            figure_map: dict[str, tuple[str, str]] = {}
            order = report.number_figures(report._norm_sections(spec.get("sections"), ""),
                                          [str(x) for x in spec.get("images", []) or []])
            for sha in dict.fromkeys(shas):
                found = self.image(sha)
                if found is None:
                    continue
                with Store.open(cfg.paths.db_path) as _db:
                    row = _db.get_image(sha)
                caption = str(row["caption"] or row["alt"] or "") if row is not None else ""
                # one numbering for every format: the source's own "Figure 1" is stripped
                figure_map[sha] = (str(found[0]),
                                   report.figure_label(order.get(sha, len(figure_map) + 1), caption))
            # PDF is built from the MARKDOWN rendering — a document flow, not the web layout.
            # Its figures are written beside the report so the print step can load them.
            markdown = ""
            if fmt == "pdf":
                # NO asset_dir here on purpose. The PDF is printed with set_content(), which has
                # no base URL, so a relative "assets/x.png" can never resolve — every figure came
                # out broken. Inline data URIs are what chromium can actually load.
                markdown = str(self.build_report(spec, base_url=base_url, fmt="md"))
            return report_export.export_report(
                title=str(spec.get("title", "Report")),
                subtitle=str(spec.get("subtitle", "")),
                body=str(spec.get("body", "")),
                sections=spec.get("sections"),
                evidence=evidence,
                fmt=fmt,
                request=str(spec.get("request", "")),
                requested_by=str(spec.get("requested_by", "")),
                model=str(spec.get("model", "")),
                classification=str(spec.get("classification", "Confidential")),
                ref_targets=report.reference_targets(cfg.paths.db_path, evidence, base_url=base_url),
                markdown=markdown,
                pptx_template=cfg.reports.pptx_template,
                figure_map=figure_map,
            )
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
            classification=str(spec.get("classification", "Confidential")),
            # Rich reference labels (store — title — heading), but served /v1/documents HTTP links a
            # remote client can open — identical to the CLI report save for the link host.
            ref_targets=report.reference_targets(cfg.paths.db_path, evidence, base_url=base_url),
            trace=spec.get("trace"),
            # the requester may ask for a different look; the config value is only the default
            theme=str(spec.get("theme", "") or cfg.reports.theme),
            # a section's figures render inside it; inlined so the report stays self-contained
            figure_srcs=report.figure_sources(
                cfg.paths.db_path, cfg.paths.staging_dir, shas, base_url=base_url
            ),
            asset_dir=asset_dir,
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

    def _db_for_read(
        self, store: str | None, user: str | None, groups: set[str] | None
    ) -> str:
        """The db_path to read from, with an access gate: route to a named federation member (or
        this single store) and deny a private store the requester isn't whitelisted for. Raises
        ValueError (unknown store) / PermissionError (denied). Used by every by-id read so a
        federated citation resolves to the RIGHT member (red-team H6) and never leaks (H1)."""
        cfg = self._target_config(store)
        if not cfg.access.permits(user=user, groups=groups or set()):
            raise PermissionError("access denied: this store is private")
        return cfg.paths.db_path

    def get_document(
        self, doc_id: int, *, chunk: int | None = None, store: str | None = None,
        user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any] | None:
        if not _fits_i64(doc_id):  # absurd id -> 404, not a sqlite OverflowError (500)
            return None
        with Store.open(self._db_for_read(store, user, groups)) as store_db:
            doc = store_db.get_document(doc_id)
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
                for c in store_db.chunks_for_document(doc_id)
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

    def document_path(
        self, doc_id: int, *, store: str | None = None, user: str | None = None,
        groups: set[str] | None = None,
    ) -> Path | None:
        if not _fits_i64(doc_id):
            return None
        with Store.open(self._db_for_read(store, user, groups)) as store_db:
            doc = store_db.get_document(doc_id)
        if doc is None:
            return None
        path = Path(str(doc["path"]))
        return path if path.is_file() else None

    def _stdf_run(
        self, doc_id: int, *, store: str | None, user: str | None, groups: set[str] | None
    ) -> Any:
        """Resolve an STDF document id to its parsed run (access-gated via document_path)."""
        from . import stdf

        path = self.document_path(doc_id, store=store, user=user, groups=groups)
        if path is None:
            raise ValueError(f"no readable STDF document with id {doc_id}")
        try:
            return stdf.parse_stdf_tests(path.read_bytes(), scope=self.config.stdf.cond_scope)
        except Exception as exc:  # noqa: BLE001 - a non-STDF doc in a mixed store must fail clean (H2)
            raise ValueError(
                f"document {doc_id} ({path.name}) is not a readable STDF file: {type(exc).__name__}"
            ) from exc

    def _backend(self, backend: str) -> str:
        return backend or self.config.stdf.plot_backend

    def list_stdf_documents(
        self, *, glob: str = "", sku: str = "", store: str | None = None,
        user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """The STDF file catalog for a store, optionally narrowed by ``sku`` (the part SKU/name the
        file was filed under — its ``source`` label) and/or a ``glob`` on the file path/name
        (``lotZ_*``, ``*run2.stdf``). Returns the matching documents plus every distinct SKU present,
        so ``stdf ls`` can show the buckets to upload into. Access-gated for a private store."""
        with Store.open(self._db_for_read(store, user, groups)) as db:
            rows = db.stdf_documents()
        docs, skus = [], set()
        for r in rows:
            doc_sku = str(r["sku"] or "")
            skus.add(doc_sku)
            path = str(r["path"])
            if sku and doc_sku != sku:
                continue
            if glob and not _match_glob(path, glob):
                continue
            docs.append({
                "doc_id": int(r["doc_id"]), "path": path, "title": str(r["title"] or ""),
                "sku": doc_sku, "lot": str(r["lot"] or ""),
                "insertions": str(r["insertions"] or ""), "status": str(r["status"] or ""),
                "tests": int(r["n_results"]), "parts": int(r["n_parts"]),
            })
        return {"documents": docs, "skus": sorted(s for s in skus if s)}

    def upload_archive(
        self, *, data: bytes, filename: str, sku: str, insertion: str = "",
        store: str | None = None, user: str = "", groups: set[str] | None = None,
        max_bytes: int = 512 * 1024 * 1024,
    ) -> dict[str, Any]:
        """Receive an uploaded ``.zip``/``.tar.gz`` bundle (bytes) and ingest it into ``store`` under
        the part **SKU** (``sku`` becomes the ``source`` label — the STDF equivalent of a document
        category). ``sku`` is required so a file is never filed into an unnamed bucket. ``insertion``
        (WS1 / WS1-RT / FT …) is the operator's insertion label for these files — passed through so
        the yield engine separates first-pass from retest correctly instead of guessing (the operator).
        Bytes are written to the store's own ``uploads`` staging and extracted through the same
        traversal-safe path as a server-side archive (red-team H4). Refuses an over-cap payload."""
        if not sku.strip():
            raise ValueError("a part SKU/name is required — pick which bucket to upload into")
        if len(data) > max_bytes:
            raise ValueError(
                f"upload is {len(data) / 1e6:.0f} MB; over the {max_bytes / 1e6:.0f} MB cap"
            )
        if not _is_archive(Path(filename)):
            raise ValueError("upload must be a .zip or .tar.gz bundle")
        target = self._target_config(store)
        if not target.access.permits(user=user or None, groups=groups or set()):
            raise PermissionError(f"access denied: cannot write to store {store or 'default'!r}")
        uploads = Path(target.paths.staging_dir) / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        tmp = uploads / f"_incoming-{runlog.RUN_ID}{_archive_suffix(filename)}"
        tmp.write_bytes(data)
        try:
            return self.ingest_from_path(
                tmp, store=store, label=sku, uploaded_by=user, groups=groups, insertion=insertion,
            )
        finally:
            tmp.unlink(missing_ok=True)

    def stdf_plot(
        self, doc_id: int, test_num: int, *, kind: str = "histogram", backend: str = "",
        store: str | None = None, user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """Distribution plot + stats for one test in an STDF document (R-STDF-2)."""
        from . import stdf_analytics

        run = self._stdf_run(doc_id, store=store, user=user, groups=groups)
        html = stdf_analytics.plot_test_html(run, test_num, kind=kind, backend=self._backend(backend))
        return {"html": html, "backend": self._backend(backend)}

    def stdf_audit(
        self, doc_a: int, doc_b: int, *, backend: str = "",
        store: str | None = None, user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """Drill-down audit comparing two STDF documents (yield/alignment/condition-diff/per-test)."""
        from . import stdf_analytics

        ra = self._stdf_run(doc_a, store=store, user=user, groups=groups)
        rb = self._stdf_run(doc_b, store=store, user=user, groups=groups)
        html = stdf_analytics.audit_report_html(
            ra, rb, backend=self._backend(backend), label_a=f"doc {doc_a}", label_b=f"doc {doc_b}"
        )
        return {"html": html}

    def stdf_site_compare(
        self, doc_id: int, test_num: int, *, backend: str = "",
        store: str | None = None, user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """Site-to-site box comparison of one test in an STDF document."""
        from . import stdf_analytics

        run = self._stdf_run(doc_id, store=store, user=user, groups=groups)
        return {"html": stdf_analytics.site_compare_html(run, test_num, backend=self._backend(backend))}

    def stdf_trend(
        self, doc_ids: list[int], test_num: int, *, stat: str = "mean", backend: str = "",
        store: str | None = None, user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """Long-run trend of a test's ``stat`` across an ordered list of STDF documents."""
        from . import stdf_analytics

        if not doc_ids:  # empty list would IndexError in trend_html (red-team phase6b #4)
            raise ValueError("stdf_trend needs at least one document id")
        runs = [
            (f"doc {d}", self._stdf_run(d, store=store, user=user, groups=groups)) for d in doc_ids
        ]
        html = stdf_analytics.trend_html(runs, test_num, stat=stat, backend=self._backend(backend))
        return {"html": html}

    def wafer_map(
        self, doc_id: int, *, wafer_id: str = "", color_by: str = "pass", test_num: int = 0,
        store: str | None = None, user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """A wafer map from an STDF document's parts: coloured by pass/fail or soft bin, or — when
        ``test_num`` is given — a **parametric** (WAT-style) map coloured by that test's value."""
        from . import wafer

        run = self._stdf_run(doc_id, store=store, user=user, groups=groups)
        if test_num:
            html = wafer.param_wafer_map_html(run.tests, run.parts, test_num, wafer_id=wafer_id)
        else:
            html = wafer.wafer_map_html(run.parts, wafer_id=wafer_id, color_by=color_by)
        return _wafer_result(html)

    def mother_lot(
        self, doc_id: int, *, backend: str = "", store: str | None = None,
        user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """Every wafer's yield across the lot in one STDF document (the mother-lot view)."""
        from . import wafer

        run = self._stdf_run(doc_id, store=store, user=user, groups=groups)
        return _wafer_result(wafer.mother_lot_html(run.parts, backend=self._backend(backend)))

    def production_trend(
        self, doc_ids: list[int], *, backend: str = "", store: str | None = None,
        user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """Long-term yield trend across an ordered list of STDF documents (one point per lot/date)."""
        from . import wafer

        if not doc_ids:
            raise ValueError("production_trend needs at least one document id")
        lots = []
        for d in doc_ids:
            run = self._stdf_run(d, store=store, user=user, groups=groups)
            lots.append((run.lot_id or f"doc {d}", run.parts))
        return _wafer_result(wafer.production_trend_html(lots, backend=self._backend(backend)))

    def plot_data(
        self, *, kind: str, series: Any = None, x: Any = None, y: Any = None,
        title: str = "", xlabel: str = "", ylabel: str = "", backend: str = "",
    ) -> dict[str, Any]:
        """General plot of caller-supplied data (any engine): the agent charts a column from a text
        or Excel file for embedding in a report. Returns a self-contained HTML fragment."""
        from . import analytics

        html = analytics.render_plot(
            kind, series=series, x=x, y=y, title=title, xlabel=xlabel, ylabel=ylabel,
            backend=self._backend(backend),
        )
        return {"html": html, "backend": self._backend(backend)}

    # -- structured STDF data (non-AI: a thin web UI queries these directly) -------

    def stdf_data_tests(
        self, *, store: str | None = None, user: str | None = None, groups: set[str] | None = None
    ) -> dict[str, Any]:
        """The distinct tests in a data store — a web UI's test picker, no AI involved."""
        with Store.open(self._db_for_read(store, user, groups)) as db:
            tests = [
                {"test_num": r["test_num"], "test_txt": r["test_txt"], "n": r["n"]}
                for r in db.stdf_test_list()
            ]
        return {"tests": tests}

    def stdf_data_results(
        self, *, test_num: int | None = None, insertion: str | None = None,
        store: str | None = None, user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """Numeric results (optionally filtered) as plain JSON — the data behind a plot, queryable
        without AI so a thin web UI can chart it directly."""
        with Store.open(self._db_for_read(store, user, groups)) as db:
            rows = db.stdf_results_query(test_num=test_num, insertion=insertion)
        return {
            "results": [
                {
                    "test_num": r["test_num"], "test_txt": r["test_txt"], "result": r["result"],
                    "units": r["units"], "head": r["head"], "site": r["site"],
                    "part_id": r["part_id"], "insertion": r["insertion"], "passed": bool(r["passed"]),
                }
                for r in rows
            ]
        }

    def stdf_data_yield(
        self, *, part_key: str = "", store: str | None = None,
        user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """First-pass + final yield per insertion, computed from the structured parts table."""
        from . import stdf as stdf_mod
        from . import stdf_analytics

        with Store.open(self._db_for_read(store, user, groups)) as db:
            rows = db.stdf_parts_all()
        parts = [
            stdf_mod.StdfPart(
                lot_id=str(r["lot"] or ""), sublot_id=str(r["sublot"] or ""),
                wafer_id=str(r["wafer"] or ""), x=r["x"], y=r["y"], part_id=str(r["part_id"] or ""),
                head=int(r["head"] or 0), site=int(r["site"] or 0), hard_bin=int(r["hard_bin"] or 0),
                soft_bin=int(r["soft_bin"] or 0), passed=bool(r["passed"]),
                insertion=str(r["insertion"] or ""),
            )
            for r in rows
        ]
        pk = stdf_analytics.parse_part_key(part_key or self.config.stdf.part_key)
        return {"insertions": stdf_analytics.insertion_yield(parts, part_key=pk)}

    # -- generic columnar data (Phase 10): any CSV/table, queried + plotted the same way -----

    def data_columns(
        self, *, store: str | None = None, user: str | None = None, groups: set[str] | None = None
    ) -> dict[str, Any]:
        """Every numeric column in a data store (id, dataset, name, kind, units, limits, n) — the
        catalog a caller (agent, thin web UI, script) lists to pick something to query or plot."""
        with Store.open(self._db_for_read(store, user, groups)) as db:
            cols = [
                {"id": int(r["id"]), "dataset": str(r["dataset"] or ""), "name": str(r["name"] or ""),
                 "kind": str(r["kind"] or ""), "units": str(r["units"] or ""),
                 "lo": r["lo"], "hi": r["hi"], "n": int(r["n"] or 0)}
                for r in db.data_columns()
            ]
        return {"columns": cols}

    def data_values(
        self, column_id: int, *, store: str | None = None,
        user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """One column's values (+ group per row) as plain JSON — the data behind a plot/query, no AI."""
        if not _fits_i64(column_id):  # absurd id -> a clean ValueError (404 / DATA error), not an
            raise ValueError(f"no data column with id {column_id}")  # OverflowError (red-team #H2)
        with Store.open(self._db_for_read(store, user, groups)) as db:
            rows = db.data_values(column_id=column_id)
        return {"values": [{"row": int(r["row_idx"]), "value": r["value"], "group": r["grp"]}
                           for r in rows]}

    def data_plot(
        self, column_id: int, *, kind: str = "histogram", backend: str = "", by_group: bool = False,
        store: str | None = None, user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """Plot one stored data column with the general engine (histogram/whisker/quantile/…): red
        limit lines for lo/hi, capability stats, and — with ``by_group`` — one series per group
        (e.g. site). Works on ANY table's column, not just STDF."""
        from . import analytics

        with Store.open(self._db_for_read(store, user, groups)) as db:
            meta = next((c for c in db.data_columns() if int(c["id"]) == column_id), None)
            if meta is None:
                raise ValueError(f"no data column with id {column_id}")
            rows = db.data_values(column_id=column_id)
        # a NULL value (e.g. a NaN written via the public Store API) must not crash float() (#M1)
        vals = [float(r["value"]) for r in rows if r["value"] is not None]
        name, lo, hi = str(meta["name"]), meta["lo"], meta["hi"]
        vlines = [float(x) for x in (lo, hi) if x is not None]
        be = self._backend(backend)
        if by_group and any(r["grp"] for r in rows):
            grouped: dict[str, list[float]] = {}
            for r in rows:
                if r["value"] is not None:
                    grouped.setdefault(str(r["grp"]), []).append(float(r["value"]))
            html = analytics.render_plot(
                kind if kind != "histogram" else "whisker",
                series=[(g, v) for g, v in sorted(grouped.items())],
                title=f"{name} by group", ylabel=name, backend=be, vlines=vlines,
            )
        else:
            html = analytics.render_plot(
                kind, y=vals, series=None, title=f"{name} distribution", xlabel=name,
                ylabel="count", backend=be, vlines=vlines,
            )
        return {
            "html": html, "column": name, "n": len(vals),
            "stats": analytics.summary_stats(vals),
            "capability": analytics.capability(vals, lo, hi),
        }

    def list_code(
        self, *, language: str | None = None, kind: str | None = None, name_like: str | None = None,
        doc_id: int | None = None, limit: int = 100000,
        store: str | None = None, user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """Symbols in a **code store** (functions/classes/methods/…), optionally filtered by language,
        kind, name glob (SQL LIKE, e.g. ``open%``), or document — the catalog the ``code`` CLI/MCP
        lists and an agent browses to find a snippet to code against. Gated for a private store."""
        if doc_id is not None and not _fits_i64(doc_id):  # absurd id -> no symbols, not OverflowError
            return {"symbols": [], "count": 0}
        with Store.open(self._db_for_read(store, user, groups)) as db:
            rows = db.code_symbols_query(language=language, kind=kind, name_like=name_like,
                                         doc_id=doc_id, limit=limit)
        symbols = [
            {"qualname": str(r["qualname"]), "name": str(r["name"]), "kind": str(r["kind"]),
             "language": str(r["language"]), "signature": str(r["signature"] or ""),
             "docstring": str(r["docstring"] or ""), "parent": str(r["parent"] or ""),
             "path": str(r["path"] or ""), "start_line": int(r["start_line"] or 0),
             "end_line": int(r["end_line"] or 0)}
            for r in rows
        ]
        return {"symbols": symbols, "count": len(symbols)}

    def code_style_guide(
        self, *, language: str | None = None, store: str | None = None,
        user: str | None = None, groups: set[str] | None = None,
    ) -> dict[str, Any]:
        """Derive the house **style guide** from a code store's symbols (naming conventions, docstring
        coverage, typing discipline, structure) and render a themed report. One language or all."""
        from . import code_index, code_style

        with Store.open(self._db_for_read(store, user, groups)) as db:
            rows = db.code_symbols_query(language=language)
        symbols = [code_index.symbol_from_row(r) for r in rows]
        guides = ([code_style.derive_style(symbols, language)] if language
                  else code_style.derive_all(symbols))
        return {"html": code_style.style_guide_html(guides),
                "languages": [g.language for g in guides if g.counts]}

    def relations(
        self, doc_id: int, direction: str = "both", *, depth: int = 1,
        store: str | None = None, user: str | None = None, groups: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Cross-referenced documents over the relations graph (R-ING-5, N-hop §17). direction:
        out | in | both; ``depth`` walks N hops. Each row carries the neighbor's id, path, title,
        shortest hop count, direction, and (for direct hops) link_type."""
        if direction not in ("out", "in", "both"):
            raise ValueError(f"direction must be out|in|both, got {direction!r}")
        if not _fits_i64(doc_id):  # absurd id -> no neighbours, not a sqlite OverflowError (M2)
            return []
        with Store.open(self._db_for_read(store, user, groups)) as store_db:
            rows = store_db.related_documents(doc_id, direction, depth=depth)
        # keep the legacy `neighbor` key (agents depend on it) alongside the richer fields
        return [{**r, "neighbor": r["doc_id"]} for r in rows]

    def image(
        self, sha256: str, *, store: str | None = None, user: str | None = None,
        groups: set[str] | None = None,
    ) -> tuple[Path, str] | None:
        cfg = self._target_config(store)
        if not cfg.access.permits(user=user, groups=groups or set()):
            raise PermissionError("access denied: this store is private")
        with Store.open(cfg.paths.db_path) as store_db:
            row = store_db.get_image(sha256)
        if row is None:
            return None
        ext = str(row["ext"] or "bin").lower()
        images_dir = (Path(cfg.paths.staging_dir) / "images").resolve()
        path = (images_dir / f"{sha256}.{ext}").resolve()
        # defence in depth: never serve a file resolved outside the images dir
        if not path.is_relative_to(images_dir) or not path.is_file():
            return None
        return path, _MEDIA_TYPES.get(ext, "application/octet-stream")


# `search_docs` replies are read by a model through a tool-output cap, so the shape is optimised
# for density: a flat list of self-describing hits cost ~0.65 KB each and a normal batched search
# (8 queries x top_k=10) overran the cap at 52 KB, which spilled the reply to a temp file and burned
# the agent's run parsing it. The cost was duplication, not content — 80 hits spanned 16 documents.
# Three things collapse it, with no information lost and nothing truncated:
#   * per-document title/path/fmt are stated once in `documents`, not on every hit;
#   * hits are rows under a declared `hit_fields` header, so no key name repeats per hit;
#   * anything derivable is stated once — the URL prefix, the embed model, the search mode.
# `cite` leads every row rather than a (doc_id, chunk_id) pair: it is the exact string the model
# must emit, so it cannot be mis-paired the way two positional integers can (R-CIT-1).
_MCP_BATCH_QUERIES = 4  # above this many queries in one call…
_MCP_BATCH_TOP_K = 5  # …top_k clamps to this: a wide batch trades depth for breadth


def _doc_key(hit: SearchHit) -> str:
    """Key into the ``documents`` map. Federated members have independent doc_id sequences, so a
    bare id would collide across stores and mis-attribute a title; the store qualifies it."""
    return f"{hit.store}:{hit.doc_id}" if hit.store else str(hit.doc_id)


def _search_payload(
    results: Sequence[Sequence[SearchHit]], base_url: str, model_used: str, mode: str,
) -> dict[str, Any]:
    """The `search_docs` reply: ranked rows plus a document table to join them against.

    ``score`` is dropped — rows are already ranked best-first, so it is rank stated twice — and so
    is ``images``, which the search path never populates."""
    federated = any(hit.store for lst in results for hit in lst)
    documents: dict[str, list[str]] = {}
    for lst in results:
        for hit in lst:
            documents.setdefault(_doc_key(hit), [hit.title, hit.path, hit.fmt])

    # A hit whose section holds a figure carries it, so an agent can SHOW the diagram rather than
    # only describe it. Captions are stated once in an `images` map (the same image recurs across
    # hits), so the model can judge relevance without fetching anything.
    with_images = any(hit.images for lst in results for hit in lst)
    captions: dict[str, str] = {}
    for lst in results:
        for hit in lst:
            for sha, text in hit.image_captions.items():
                captions.setdefault(sha, text)

    def row(hit: SearchHit) -> list[Any]:
        cells: list[Any] = [hit.citation, _doc_key(hit), hit.locator, hit.kind, hit.snippet]
        if with_images:
            cells.append(hit.images)
        return cells if federated else cells[:1] + cells[2:]

    fields = ["cite", "doc", "locator", "kind", "snippet"] + (["img"] if with_images else [])
    return {
        "hit_fields": fields if federated else [f for f in fields if f != "doc"],
        "results": [[row(hit) for hit in lst] for lst in results],
        "doc_fields": ["title", "path", "fmt"],
        "documents": documents,
        "reading": (
            "Each row is one hit, ranked best-first, with the columns named by `hit_fields`. "
            "`cite` is the citation to quote verbatim as [D:doc#chunk]; join a row to its document "
            + ("via its `doc` column" if federated else "on the doc part of `cite`")
            + " -> `documents[key]`, whose columns are named by `doc_fields`. "
            "Full text: get_document(doc_id). Open in a browser: url_base + doc_id."
            + (
                " A row's `img` lists figures in that chunk's section: `images[sha]` is the "
                "caption, and GET img_base + sha returns the file — fetch one when a diagram "
                "would say it better, and pass the shas as the report spec's `images` to show "
                "them. Cite the row's `cite` as usual."
                if with_images
                else ""
            )
        ),
        "url_base": f"{base_url}/v1/documents/",
        **({"img_base": f"{base_url}/v1/images/", "images": captions} if with_images else {}),
        "embed_model_used": model_used,
        "search_mode": mode,
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
    rating: int | None = None  # optional thumbs (-1/0/+1)
    make_global: bool = False  # promote from private (default) to everyone
    store: str | None = None  # federation member the target doc belongs to


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


_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar")
_MAX_EXTRACT_BYTES = 2 * 1024**3  # 2 GB decompressed cap — zip-bomb guard (red-team M5)


_TIER_RANK = {"internal": 2, "vendor": 1}  # feedback outranks both via the separate feedback signal


def _annotate_conflict_tiers(
    conflicts: Any, meta: dict[int, tuple[str, str]], tier_of: dict[str, str]
) -> tuple[list[dict[str, Any]], int]:
    """Tag each conflict-candidate pair with each doc's authority **tier** and, when the two sides
    are **cross-tier** (a disagreement between e.g. internal and vendor), the ``authoritative_doc``
    (the higher tier wins — feedback > internal > vendor). Returns (rows, cross_tier_count) — the
    "log which system says what" output for #63."""
    def tier(doc: int) -> str:
        return tier_of.get(meta.get(doc, ("", ""))[1], "vendor")

    rows: list[dict[str, Any]] = []
    cross = 0
    for p in conflicts:
        ta, tb = tier(p.doc_a), tier(p.doc_b)
        is_cross = ta != tb
        cross += int(is_cross)
        # only claim an authoritative doc when one tier actually OUTRANKS the other — two docs whose
        # tiers are both unrecognized (rank 0) have no clear authority (red-team #L1).
        ra, rb = _TIER_RANK.get(ta, 0), _TIER_RANK.get(tb, 0)
        authoritative = None
        if is_cross and ra != rb:
            authoritative = p.doc_a if ra > rb else p.doc_b
        rows.append({
            "chunk_a": p.chunk_a, "chunk_b": p.chunk_b, "doc_a": p.doc_a, "doc_b": p.doc_b,
            "doc_a_path": meta.get(p.doc_a, ("", ""))[0],
            "doc_b_path": meta.get(p.doc_b, ("", ""))[0],
            "doc_a_tier": ta, "doc_b_tier": tb, "cross_tier": is_cross,
            "authoritative_doc": authoritative, "similarity": p.similarity,
        })
    return rows, cross


def _match_glob(path: str, pattern: str) -> bool:
    """Case-insensitive glob (``*``/``?``/``[seq]``) against an STDF document's path — matches on the
    **basename** (``lotZ_run2.stdf``) or the full posix path, so ``*run2*`` and ``ate/lot*/*`` both
    work. Path separators are normalised so a stored Windows path still matches a posix pattern."""
    import fnmatch
    from pathlib import PurePosixPath

    # lower-case both sides so the match is case-insensitive on every OS (fnmatch is only
    # case-insensitive on Windows otherwise — red-team phase6b #6)
    norm = path.replace("\\", "/").lower()
    name = PurePosixPath(norm).name
    pat = pattern.replace("\\", "/").lower()
    return fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(norm, pat)


def _is_archive(path: Path) -> bool:
    return path.suffix.lower() == ".zip" or path.name.lower().endswith(_ARCHIVE_SUFFIXES)


def _archive_suffix(filename: str) -> str:
    """The full archive extension of a filename, so a ``.tar.gz`` upload keeps both parts (a bare
    ``Path.suffix`` would give ``.gz`` and later fail the archive check — red-team M3)."""
    low = filename.lower()
    for suf in _ARCHIVE_SUFFIXES:
        if low.endswith(suf):
            return suf
    return Path(filename).suffix or ".zip"


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract a ``.zip`` / ``.tar.gz`` into ``dest``, refusing any member whose path escapes
    ``dest`` (zip-slip / tar traversal) and any archive that decompresses past the size cap. ANY
    archive problem (malformed, oversize, unsafe member) raises ``ValueError`` so callers return a
    clean 400, never a 500 (red-team M2). Windows-first (pathlib), stdlib only."""
    import tarfile
    import zipfile

    dest = dest.resolve()

    def _guard(name: str) -> None:
        if not (dest / name).resolve().is_relative_to(dest):
            raise ValueError(f"unsafe path in archive: {name!r}")

    try:
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive) as zf:
                if sum(i.file_size for i in zf.infolist()) > _MAX_EXTRACT_BYTES:
                    raise ValueError("archive too large decompressed")
                for name in zf.namelist():
                    _guard(name)
                zf.extractall(dest)
        else:  # tar / tar.gz
            with tarfile.open(archive) as tf:
                members = tf.getmembers()
                if sum(m.size for m in members) > _MAX_EXTRACT_BYTES:
                    raise ValueError("archive too large decompressed")
                for member in members:
                    _guard(member.name)
                tf.extractall(dest, filter="data")  # py3.12+: refuse special files / abs paths
    except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError) as err:
        raise ValueError(f"could not read archive: {err}") from err


_MCP_HELP = """# docusearch — research + cited report (MCP)

Answer the user's question ONLY from this document catalog, then (optionally) render a cited
report. Discover the domain's terminology from the search results themselves — never rely on prior
knowledge of the domain.

## Tools
- `list_stores()` -> the document stores you can search. In a FEDERATION (e.g. python / rust /
  internal) you may scope any search to a subset by name. Call this first if the user names a store.
- `search_docs(queries, top_k=10, prefix=False, stores=None, bm25_only=False, roles=None)` ->
  ALWAYS pass a LIST of query phrasings (batched). The reply is a table, stated once rather than
  repeated per hit — read it like this:
    * `results[i]` = the rows for query i, ranked best-first. Each row's columns are named by
      `hit_fields` (normally `cite, locator, kind, snippet`).
    * `cite` is the citation — quote it verbatim as `[D:doc#chunk]`, never rebuild it by hand.
    * A row's document: take the doc part of `cite` (`D:12#5` -> `12`) and look up
      `documents["12"]`, whose columns are named by `doc_fields` (`title, path, fmt`). In a
      federation each row also has a `doc` column — use that as the key instead, because member
      stores number their documents independently.
    * Full chunk text: `get_document(doc_id)`. Browser link: `url_base` + doc_id.
    * FIGURES: a row's `img` lists the shas of images in that chunk's section (present whether or
      not vision ran). `images[sha]` is the caption; `GET img_base + sha` fetches the file. Pass
      shas as the report spec's `images` to show them in the output.
  Batches over 4 queries clamp top_k to 5. `stores=["internal"]` searches only those members; omit
  for all. `prefix` = partial-term matching; `bm25_only` = skip vectors; `roles` = cooperative
  filter.
- `get_document(doc_id, chunk=None)` -> full chunk text — use it to fill a card with real code / a
  full procedure, not just a snippet.
- `related_documents(doc_id, direction="both", depth=1)` -> cross-referenced docs (follow leads);
  `depth` walks N hops, each result carries its shortest `hops`.
- `catalog_stats()` -> counts + embedding model (sanity-check the catalog is populated).
- `report_format(fmt="")` -> CALL BEFORE WRITING. How to author for the target format, plus the
  operator's configured default. Content must be shaped for its output: a deck needs short
  bullets, a spreadsheet one fact per row, a document takes prose. The renderer lays out what you
  give it — it cannot invent structure you did not write, so a document-shaped section becomes a
  wall of text on a slide.
- `build_report(spec, fmt=...)` -> a themed, cited report SAVED on the server; returns
  {fmt, filename, url, bytes}. Give the user the `url`. VERIFIES every citation against your
  evidence and refuses hallucinated ones. `fmt`: md | html | html-slide | pdf | docx | pptx |
  xlsx.

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
0. `report_format()` — learn the target format and how to write for it BEFORE you draft. If the
   user named a format ("make me a PowerPoint"), that WINS over the configured default.
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
    def list_stores(user: str | None = None, groups: list[str] | None = None) -> dict[str, Any]:
        """The document stores you can search; in a federation, only the member names YOU may access
        (pass user/groups). A private member you're not whitelisted for is omitted."""
        return service.list_stores(user=user, groups=set(groups) if groups else None)

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
        """Search the catalog (batch: pass a LIST). Hits carry `doc_id`/`chunk_id`/`citation`/
        `locator`/`snippet`; title+path live once per doc in `documents`. `stores` scopes to
        federation members; `user`/`groups` gate access to private stores (forward the
        authenticated user)."""
        if len(queries) > _MCP_BATCH_QUERIES:  # wide batch -> less depth, so the reply still fits
            top_k = min(top_k, _MCP_BATCH_TOP_K)
        try:
            results, model_used, mode = service.search(
                queries, top_k=top_k, prefix=prefix, bm25_only=bm25_only,
                roles=set(roles) if roles else None, stores=stores,
                user=user, groups=set(groups) if groups else None,
            )
        except (ValueError, PermissionError) as err:
            return {"error": "ACCESS", "message": str(err), "results": []}
        return _search_payload(results, base, model_used, mode)

    @mcp.tool()
    def get_document(
        doc_id: int, chunk: int | None = None, store: str | None = None,
        user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Fetch a document's metadata + full chunk text by id (pass `store` for a federated
        citation). `user`/`groups` gate a private store."""
        try:
            return service.get_document(
                doc_id, chunk=chunk, store=store, user=user,
                groups=set(groups) if groups else None,
            )
        except (PermissionError, ValueError) as err:
            return {"error": "ACCESS", "message": str(err)}

    @mcp.tool()
    def related_documents(
        doc_id: int, direction: str = "both", depth: int = 1, store: str | None = None,
        user: str | None = None, groups: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Documents cross-referenced from / to this one over the relations graph. direction:
        out (this doc links to) | in (links to this doc) | both; `depth` walks N hops (each row
        carries its shortest `hops`). Gated for a private store."""
        try:
            return service.relations(
                doc_id, direction, depth=depth, store=store, user=user,
                groups=set(groups) if groups else None,
            )
        except (PermissionError, ValueError):
            return []

    @mcp.tool()
    def check_discrepancies(
        store: str | None = None, persist: bool = False,
        user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Scan a store for duplicate active documents + high-similarity conflict candidates
        (chunk pairs across docs that say near-identical things). `persist` records discrepancy
        flags. Gated for a private store."""
        try:
            return service.discrepancies(
                store=store, persist=persist, user=user,
                groups=set(groups) if groups else None,
            )
        except (PermissionError, ValueError) as err:
            return {"error": "ACCESS", "message": str(err)}

    @mcp.tool()
    def catalog_stats(user: str | None = None, groups: list[str] | None = None) -> dict[str, Any]:
        """Counts + embedding model for the catalog (gated for a private store)."""
        try:
            service.check_read_access(user, set(groups) if groups else None)
        except PermissionError as err:
            return {"error": "ACCESS", "message": str(err)}
        return service.health()

    def _grp(groups: list[str] | None) -> set[str] | None:
        return set(groups) if groups else None

    @mcp.tool()
    def list_stdf(
        glob: str = "", sku: str = "", store: str | None = None,
        user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """List the STDF data files in a store, optionally narrowed by `sku` (the part SKU/name they
        were filed under) and/or a `glob` on the file name/path. Returns each file's doc_id, SKU, lot,
        insertions, and test/part counts, plus every SKU bucket present. Gated for a private store."""
        try:
            return service.list_stdf_documents(
                glob=glob, sku=sku, store=store, user=user, groups=_grp(groups)
            )
        except (PermissionError, ValueError) as err:
            return {"error": "STDF", "message": str(err), "documents": []}

    @mcp.tool()
    def upload_archive(
        filename: str, data_b64: str, sku: str, insertion: str = "", store: str | None = None,
        user: str = "", groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Upload a base64 `.zip`/`.tar.gz` bundle (e.g. a set of STDF files) and ingest it into
        `store` under the part **`sku`** (required — the STDF equivalent of a document category; it
        becomes the files' `source` label). `insertion` is the operator's insertion label
        (WS1 / WS1-RT / FT …) for these files, so yield separates first-pass from retest correctly.
        Traversal-safe extraction; refuses an oversized payload. Writing a private store requires
        `user`/`groups` be whitelisted."""
        import base64
        import binascii

        try:
            data = base64.b64decode(data_b64, validate=True)
        except (binascii.Error, ValueError):
            return {"error": "UPLOAD", "message": "data_b64 is not valid base64"}
        try:
            return service.upload_archive(
                data=data, filename=filename, sku=sku, insertion=insertion, store=store,
                user=user, groups=_grp(groups),
            )
        except (ValueError, FileNotFoundError, PermissionError) as err:
            return {"error": "UPLOAD", "message": str(err)}

    @mcp.tool()
    def list_data(
        store: str | None = None, user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """List the numeric columns in a **data store** (any ingested CSV/table, not just STDF): each
        column's id, dataset, name, units, spec limits, and count. Pick a column id to query
        (`data_values`) or plot (`data_plot`). Gated for a private store."""
        try:
            return service.data_columns(store=store, user=user, groups=_grp(groups))
        except (PermissionError, ValueError) as err:
            return {"error": "DATA", "message": str(err), "columns": []}

    @mcp.tool()
    def data_values(
        column_id: int, store: str | None = None,
        user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Pull one data column's values (+ per-row group) as plain JSON — the raw data behind a plot
        or your own analysis, from any ingested table."""
        try:
            return service.data_values(column_id, store=store, user=user, groups=_grp(groups))
        except (PermissionError, ValueError) as err:
            return {"error": "DATA", "message": str(err), "values": []}

    @mcp.tool()
    def data_plot(
        column_id: int, kind: str = "histogram", backend: str = "", by_group: bool = False,
        store: str | None = None, user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Plot one stored data column (any CSV/table) with the general engine. kind: histogram |
        whisker | quantile | qq | xy | linear. `by_group` draws one series per group (e.g. site).
        Red lines mark the column's lo/hi limits; returns a self-contained HTML fragment + stats."""
        try:
            return service.data_plot(column_id, kind=kind, backend=backend, by_group=by_group,
                                     store=store, user=user, groups=_grp(groups))
        except (PermissionError, ValueError) as err:
            return {"error": "DATA", "message": str(err)}

    @mcp.tool()
    def list_code(
        language: str | None = None, kind: str | None = None, name_like: str | None = None,
        doc_id: int | None = None, store: str | None = None,
        user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """List symbols in a **code store** (functions/classes/methods/…) with their qualified name,
        kind, language, signature, docstring, and line span. Filter by `language`, `kind`, a `name_like`
        glob (SQL LIKE, e.g. `open%`), or `doc_id`. The catalog to browse for a snippet to code
        against; pair with `search_docs` to find one by intent. Gated for a private store."""
        try:
            return service.list_code(language=language, kind=kind, name_like=name_like,
                                     doc_id=doc_id, store=store, user=user, groups=_grp(groups))
        except (PermissionError, ValueError) as err:
            return {"error": "CODE", "message": str(err), "symbols": [], "count": 0}

    @mcp.tool()
    def code_styleguide(
        language: str | None = None, store: str | None = None,
        user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Derive the house **style guide** from a code store: dominant naming convention per symbol
        group, docstring coverage, typing discipline, and structure — the conventions to follow when
        writing new code that fits the repo. One `language` or all. Returns a self-contained HTML
        report + the languages covered."""
        try:
            return service.code_style_guide(language=language, store=store, user=user,
                                            groups=_grp(groups))
        except (PermissionError, ValueError) as err:
            return {"error": "CODE", "message": str(err)}

    @mcp.tool()
    def wafer_map(
        doc_id: int, wafer_id: str = "", color_by: str = "pass", test_num: int = 0,
        store: str | None = None, user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """A **wafer map** for an STDF document: a die grid at each part's (x,y), coloured by
        `pass` (default) or `softbin`. Pass `test_num` for a **parametric** (WAT-style) map coloured
        by that test's measured value. `wafer_id` picks the wafer (else the first). Returns HTML."""
        try:
            return service.wafer_map(doc_id, wafer_id=wafer_id, color_by=color_by, test_num=test_num,
                                     store=store, user=user, groups=_grp(groups))
        except (PermissionError, ValueError) as err:
            return {"error": "WAFER", "message": str(err)}

    @mcp.tool()
    def mother_lot(
        doc_id: int, backend: str = "", store: str | None = None,
        user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """The **mother-lot** view for an STDF document: every wafer's yield across the lot + the
        pooled lot yield, flagging the lowest wafer. Returns an HTML report."""
        try:
            return service.mother_lot(doc_id, backend=backend, store=store, user=user,
                                      groups=_grp(groups))
        except (PermissionError, ValueError) as err:
            return {"error": "WAFER", "message": str(err)}

    @mcp.tool()
    def production_trend(
        doc_ids: list[int], backend: str = "", store: str | None = None,
        user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Long-term **production yield trend** across an ordered list of STDF documents (one point
        per lot/date) — drift detection over time. Returns an HTML report."""
        try:
            return service.production_trend(doc_ids, backend=backend, store=store, user=user,
                                            groups=_grp(groups))
        except (PermissionError, ValueError) as err:
            return {"error": "WAFER", "message": str(err)}

    @mcp.tool()
    def stdf_plot(
        doc_id: int, test_num: int, kind: str = "histogram", backend: str = "",
        store: str | None = None, user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Plot one test's distribution from an STDF data document. kind: histogram | whisker |
        quantile | qq | xy | linear. backend: matplotlib (PNG) | plotly (interactive); default from
        config. Returns a self-contained HTML fragment + stats — embed it in a report."""
        try:
            return service.stdf_plot(
                doc_id, test_num, kind=kind, backend=backend, store=store, user=user,
                groups=_grp(groups),
            )
        except (PermissionError, ValueError) as err:
            return {"error": "STDF", "message": str(err)}

    @mcp.tool()
    def stdf_audit(
        doc_a: int, doc_b: int, backend: str = "", store: str | None = None,
        user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare two STDF data documents: yield delta, test alignment, condition diff, and a
        collapsible per-test Q-Q drill-down. Returns a navigable HTML report to dig through."""
        try:
            return service.stdf_audit(
                doc_a, doc_b, backend=backend, store=store, user=user, groups=_grp(groups)
            )
        except (PermissionError, ValueError) as err:
            return {"error": "STDF", "message": str(err)}

    @mcp.tool()
    def stdf_site_compare(
        doc_id: int, test_num: int, backend: str = "", store: str | None = None,
        user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Site-to-site distribution comparison of one test in an STDF document."""
        try:
            return service.stdf_site_compare(
                doc_id, test_num, backend=backend, store=store, user=user, groups=_grp(groups)
            )
        except (PermissionError, ValueError) as err:
            return {"error": "STDF", "message": str(err)}

    @mcp.tool()
    def stdf_trend(
        doc_ids: list[int], test_num: int, stat: str = "mean", backend: str = "",
        store: str | None = None, user: str | None = None, groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Trend a test's stat (mean/median/std/min/max) across an ordered list of STDF documents —
        long-run drift detection across loop runs."""
        try:
            return service.stdf_trend(
                doc_ids, test_num, stat=stat, backend=backend, store=store, user=user,
                groups=_grp(groups),
            )
        except (PermissionError, ValueError) as err:
            return {"error": "STDF", "message": str(err)}

    @mcp.tool()
    def plot_data(
        kind: str, series: list | None = None, x: list | None = None, y: list | None = None,  # type: ignore[type-arg]
        title: str = "", xlabel: str = "", ylabel: str = "", backend: str = "",
    ) -> dict[str, Any]:
        """General-purpose plot of data YOU supply (any engine): chart a column pulled from a text
        or Excel document for embedding in a report. kind: histogram | whisker | quantile | qq | xy
        | linear. `y` (or `series` of [name, values]) for distributions; `x`+`y` for xy/linear."""
        try:
            tuples = [(str(s[0]), list(s[1])) for s in series] if series else None
            return service.plot_data(
                kind=kind, series=tuples, x=x, y=y, title=title, xlabel=xlabel,
                ylabel=ylabel, backend=backend,
            )
        except (ValueError, TypeError, IndexError) as err:
            return {"error": "PLOT", "message": str(err)}

    @mcp.tool()
    def ingest_docs(
        path: str, user: str, store: str | None = None, label: str = "upload",
        groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Ingest a server-side folder or .zip/.tar.gz into `store`, labelled + attributed to `user`.
        Writing a private store requires `user`/`groups` be whitelisted for it."""
        try:
            return service.ingest_from_path(
                path, store=store, label=label, uploaded_by=user,
                groups=set(groups) if groups else None,
            )
        except (ValueError, FileNotFoundError, PermissionError) as err:
            return {"error": "INGEST", "message": str(err)}

    @mcp.tool()
    def submit_feedback(
        text: str, user: str, doc_id: int | None = None, chunk_id: int | None = None,
        rating: int | None = None, make_global: bool = False, store: str | None = None,
        groups: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record a user's feedback (attributed to `user`). Private to that user by default; set
        `make_global=true` to share with everyone. A `rating` (-1/0/+1) on a `doc_id` nudges ranking.
        In a federation, pass the `store` the doc came from so the feedback lands in that member.
        Gated on the target store's access policy — you can only give feedback where you may search."""
        try:
            return service.submit_feedback(
                user=user, text=text, doc_id=doc_id, chunk_id=chunk_id, rating=rating,
                make_global=make_global, store=store, groups=_grp(groups),
            )
        except (ValueError, PermissionError) as err:
            return {"error": "FEEDBACK", "message": str(err)}

    @mcp.tool()
    def report_format(fmt: str = "") -> dict[str, Any]:
        """Call BEFORE writing a report: how to author content for the target format. A deck
        needs short bullets, a spreadsheet needs one fact per row, a document takes prose — the
        renderer cannot invent structure the author did not write. Omit `fmt` for all formats.

        PRECEDENCE: if the requester named a format ("make me a PowerPoint", "as a spreadsheet"),
        THAT wins. `configured_default` applies only when they did not say."""
        note = (
            "If the requester named a format, use it — configured_default applies only when "
            "they did not. Map plain words: PowerPoint/deck/slides -> pptx, Word/doc -> docx, "
            "spreadsheet/Excel -> xlsx, web page -> html, browsable deck -> html-slide. Same for "
            "the look: set spec['theme'] if they ask for one, else the configured theme is used."
        )
        themes = {"available": sorted(report.THEMES), "configured": config.reports.theme}
        if fmt:
            return {"fmt": fmt, "guidance": report_export.guidance(fmt),
                    "configured_default": config.reports.default_format,
                    "themes": themes, "precedence": note}
        return {"formats": report_export.FORMAT_GUIDANCE,
                "configured_default": config.reports.default_format,
                "themes": themes, "precedence": note}

    @mcp.tool()
    def build_report(spec: dict[str, Any], fmt: str = "md") -> dict[str, Any]:
        """Render a cited report and SAVE it on the server; returns {fmt, filename, url, bytes}
        — give the user the `url`, it is a direct link to the file. fmt: md | html | pdf | docx |
        pptx | xlsx (all six write a file). md/html also return the text as `report`. Verifies
        every citation and refuses hallucinated ones."""
        try:
            out = service.build_report_file(spec, base_url=base, fmt=fmt)
        except citations.CitationError as err:
            return {"error": "HALLUCINATED_CITATION", "message": str(err)}
        except (report_export.ExportDependencyError, ValueError) as err:
            return {"error": "EXPORT", "message": str(err)}
        out["authored_for"] = report_export.guidance(fmt)
        return out

    return mcp


def create_app(config: Config) -> FastAPI:
    """Build the FastAPI app + MCP. All routes are thin wrappers over ``Service`` (§10)."""
    from contextlib import asynccontextmanager

    service = Service(config)
    mcp = build_mcp(service, config)
    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        # age out reports left by earlier runs before serving anything (reports.retain_days)
        swept = report_store.sweep(config.paths.tmp_dir, config.reports.retain_days)
        if swept:
            runlog.log("serve.reports_swept", count=swept)
        # run the MCP session manager's lifespan alongside the app (streamable HTTP)
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    app = FastAPI(title="docusearch", version=__version__, lifespan=lifespan)

    def _gate_read(request: Request) -> tuple[str | None, set[str]]:
        """Extract the requester + enforce read access to a private single store (403 if denied) —
        the guard every read route runs so private content/metadata never leaks to a non-whitelisted
        caller (red-team H1/M1)."""
        user, groups = _request_identity(request)
        try:
            service.check_read_access(user, groups)
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err
        return user, groups

    @app.get("/v1/health")
    def health(request: Request) -> dict[str, Any]:
        _gate_read(request)
        return service.health()

    @app.get("/v1/embed-info")
    def embed_info(request: Request) -> dict[str, Any]:
        _gate_read(request)
        return service.embed_info()

    @app.post("/v1/search")
    def search_route(req: SearchRequest, request: Request) -> dict[str, Any]:
        roles = set(req.roles) if req.roles else None
        user, groups = _request_identity(request)
        if req.query_vectors is not None:
            try:
                service.check_read_access(user, groups)  # private single store -> 403 (H1)
            except PermissionError as err:
                raise HTTPException(status_code=403, detail=str(err)) from err
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
        # one shape for REST and MCP — the gate has to prove the two paths agree (R-API-1)
        return {**_search_payload(results, base, model_used, mode), "run_id": runlog.RUN_ID}

    @app.post("/v1/embed")
    def embed_route(req: EmbedRequest) -> dict[str, Any]:
        try:
            return service.embed_texts(req.texts)
        except ModelMismatchError as err:
            raise HTTPException(status_code=409, detail={"error": "NO_EMBED_MODEL"}) from err

    def _read_identity(request: Request) -> tuple[str | None, set[str]]:
        return _request_identity(request)  # the by-id read methods gate per-store themselves

    @app.get("/v1/documents/{doc_id}")
    def get_document(
        doc_id: int, request: Request, chunk: int | None = None, download: int = 0,
        store: str | None = None,
    ) -> Any:
        # A federated citation carries ?store=<member>; the method routes there and gates access.
        user, groups = _read_identity(request)
        try:
            if download:
                path = service.document_path(doc_id, store=store, user=user, groups=groups)
                if path is None:
                    raise HTTPException(status_code=404, detail="document file not found")
                return FileResponse(path)
            doc = service.get_document(doc_id, chunk=chunk, store=store, user=user, groups=groups)
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        return doc

    @app.get("/v1/images/{sha256}")
    def get_image(sha256: str, request: Request, store: str | None = None) -> FileResponse:
        user, groups = _read_identity(request)
        try:
            img = service.image(sha256, store=store, user=user, groups=groups)
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        if img is None:
            raise HTTPException(status_code=404, detail="image not found")
        path, media = img
        return FileResponse(path, media_type=media)

    @app.get("/v1/relations/{doc_id}")
    def relations(
        doc_id: int, request: Request, direction: str = "both", depth: int = 1,
        store: str | None = None,
    ) -> list[dict[str, Any]]:
        user, groups = _read_identity(request)
        try:
            return service.relations(
                doc_id, direction, depth=depth, store=store, user=user, groups=groups
            )
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

    @app.get("/v1/data/stdf/tests")
    def data_tests(request: Request, store: str | None = None) -> dict[str, Any]:
        user, groups = _read_identity(request)
        try:
            return service.stdf_data_tests(store=store, user=user, groups=groups)
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err

    @app.get("/v1/data/stdf/results")
    def data_results(
        request: Request, test_num: int | None = None, insertion: str | None = None,
        store: str | None = None,
    ) -> dict[str, Any]:
        user, groups = _read_identity(request)
        try:
            return service.stdf_data_results(
                test_num=test_num, insertion=insertion, store=store, user=user, groups=groups
            )
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err

    @app.get("/v1/data/stdf/yield")
    def data_yield(request: Request, part_key: str = "", store: str | None = None) -> dict[str, Any]:
        user, groups = _read_identity(request)
        try:
            return service.stdf_data_yield(part_key=part_key, store=store, user=user, groups=groups)
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err

    @app.get("/v1/data/columns")
    def data_columns(request: Request, store: str | None = None) -> dict[str, Any]:
        """Numeric columns in a data store (any CSV/table) — a thin web UI's column picker, no AI."""
        user, groups = _read_identity(request)
        try:
            return service.data_columns(store=store, user=user, groups=groups)
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err

    @app.get("/v1/data/columns/{column_id}/values")
    def data_column_values(
        column_id: int, request: Request, store: str | None = None
    ) -> dict[str, Any]:
        user, groups = _read_identity(request)
        try:
            return service.data_values(column_id, store=store, user=user, groups=groups)
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err
        except ValueError as err:
            raise HTTPException(status_code=404, detail=str(err)) from err

    @app.get("/v1/data/columns/{column_id}/plot")
    def data_column_plot(
        column_id: int, request: Request, kind: str = "histogram", backend: str = "",
        by_group: bool = False, store: str | None = None,
    ) -> dict[str, Any]:
        user, groups = _read_identity(request)
        try:
            return service.data_plot(column_id, kind=kind, backend=backend, by_group=by_group,
                                     store=store, user=user, groups=groups)
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err
        except ValueError as err:
            raise HTTPException(status_code=404, detail=str(err)) from err

    @app.post("/v1/data/plot")
    def data_plot(req: dict[str, Any]) -> dict[str, Any]:
        """Render a plot from posted data — a thin web UI calls this directly, no AI. Body:
        {kind, y|series|x+y, title, backend}."""
        try:
            series = req.get("series")
            tuples = [(str(s[0]), list(s[1])) for s in series] if series else None
            return service.plot_data(
                kind=str(req.get("kind", "histogram")), series=tuples, x=req.get("x"),
                y=req.get("y"), title=str(req.get("title", "")), backend=str(req.get("backend", "")),
            )
        except (ValueError, TypeError, IndexError, KeyError) as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

    @app.get("/v1/discrepancies")
    def discrepancies_route(
        request: Request, store: str | None = None, persist: bool = False
    ) -> dict[str, Any]:
        user, groups = _read_identity(request)
        try:
            return service.discrepancies(
                store=store, persist=persist, user=user, groups=groups
            )
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

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
            out = service.build_report_file(spec, base_url=base, fmt=req.fmt)
        except citations.CitationError as err:
            raise HTTPException(
                status_code=400,
                detail={"error": "HALLUCINATED_CITATION", "message": str(err)},
            ) from err
        except report_export.ExportDependencyError as err:
            raise HTTPException(status_code=501, detail=str(err)) from err
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        runlog.log("api.report", fmt=req.fmt, evidence=len(req.evidence))
        return out

    @app.get("/v1/reports/{name}")
    def get_report(name: str) -> FileResponse:
        """Serve a generated report file — the `url` build_report hands back."""
        path = report_store.resolve(config.paths.tmp_dir, name)
        if path is None:
            raise HTTPException(status_code=404, detail="report not found (it may have aged out)")
        return FileResponse(path, media_type=_REPORT_MEDIA.get(path.suffix.lower(),
                                                               "application/octet-stream"))

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
        _, req_groups = _request_identity(request)
        try:
            return service.submit_feedback(
                user=user, text=req.text, doc_id=req.doc_id, chunk_id=req.chunk_id,
                rating=req.rating, make_global=req.make_global, store=req.store, groups=req_groups,
            )
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

    @app.post("/v1/ingest")
    def ingest_route(req: IngestRequest, request: Request) -> dict[str, Any]:
        """Ingest a server-side folder (confined to the store's inbound dir) or a .zip/.tar.gz path
        into a store, labelled + attributed. Writing a private store requires being whitelisted."""
        user, groups = _request_identity(request)
        if not user:
            raise HTTPException(status_code=401, detail="X-Docusearch-User header required to ingest")
        try:
            return service.ingest_from_path(
                req.path, store=req.store, label=req.label, uploaded_by=user, groups=groups,
                min_content_chars=req.min_content_chars,
            )
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err
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
        user, groups = _request_identity(request)
        if not user:
            raise HTTPException(status_code=401, detail="X-Docusearch-User header required to ingest")
        import tempfile

        suffix = _archive_suffix(file.filename or "upload.zip")  # keep .tar.gz whole (M3)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            archive = Path(tmp.name)
        try:
            return service.ingest_from_path(
                archive, store=store, label=label, uploaded_by=user, groups=groups
            )
        except PermissionError as err:
            raise HTTPException(status_code=403, detail=str(err)) from err
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
