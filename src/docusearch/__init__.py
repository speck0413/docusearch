"""docusearch — enterprise documentation catalog (local BM25 + optional hybrid search).

Public API (R-ARCH-2): this package deliberately exposes a *thin* surface. Only the
names re-exported here are supported; everything else under ``docusearch.*`` is
internal and may change without notice.

    from docusearch import Catalog, Config, serve, __version__
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
