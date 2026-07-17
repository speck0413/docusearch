"""docusearch — enterprise documentation catalog (local BM25 + optional hybrid search).

Public API (R-ARCH-2): this package deliberately exposes a *thin* surface. Only the
names in ``__all__`` are supported; everything else under ``docusearch.*`` is internal
and may change without notice.

``serve`` is exposed lazily (via ``__getattr__``) so ``import docusearch`` never pulls
in FastAPI/uvicorn for standalone or client-only users.

    from docusearch import Catalog, Config, serve, __version__
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._version import __version__
from .catalog import Catalog
from .config import Config

if TYPE_CHECKING:
    from .server import serve

__all__ = ["Catalog", "Config", "__version__", "serve"]


def __getattr__(name: str) -> Any:
    # Lazy import so the heavy server stack loads only when `serve` is actually used.
    if name == "serve":
        from .server import serve

        return serve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
