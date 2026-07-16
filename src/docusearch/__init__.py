"""docusearch — enterprise documentation catalog (local BM25 + optional hybrid search).

Public API (R-ARCH-2): this package deliberately exposes a *thin* surface. Only the
names in ``__all__`` are supported; everything else under ``docusearch.*`` is internal
and may change without notice.

The surface grows as phases land — ``Catalog`` (ingest/search) and ``serve`` (the
HTTP/MCP server) are added to ``__all__`` when Phases 1 and 3 implement them. Adding
empty stubs now would be dead code (R-PROC-7), so the Phase-0 surface is:

    from docusearch import Config, __version__
"""

from __future__ import annotations

from ._version import __version__
from .config import Config

__all__ = ["Config", "__version__"]
