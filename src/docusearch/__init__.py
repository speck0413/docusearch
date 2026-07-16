"""docusearch — enterprise documentation catalog (local BM25 + optional hybrid search).

Public API (R-ARCH-2): this package deliberately exposes a *thin* surface. Only the
names in ``__all__`` are supported; everything else under ``docusearch.*`` is internal
and may change without notice.

The surface grows as phases land — ``serve`` (the HTTP/MCP server) joins ``__all__``
when Phase 3 implements it. Adding empty stubs earlier would be dead code (R-PROC-7).
Current surface:

    from docusearch import Catalog, Config, __version__
"""

from __future__ import annotations

from ._version import __version__
from .catalog import Catalog
from .config import Config

__all__ = ["Catalog", "Config", "__version__"]
