"""The public API stays thin (R-ARCH-2).

Only sanctioned names are re-exported; internal modules must not leak into the
package's ``__all__``. This guards against the surface quietly widening.
"""

from __future__ import annotations

import docusearch
from docusearch import Catalog, Config, __version__


def test_version_is_a_string() -> None:
    assert isinstance(__version__, str)
    assert __version__ == docusearch.__version__


def test_config_is_the_public_config() -> None:
    assert Config is docusearch.config.Config


def test_catalog_is_the_public_catalog() -> None:
    assert Catalog is docusearch.catalog.Catalog


def test_all_is_thin_and_hides_internals() -> None:
    assert set(docusearch.__all__) == {"Catalog", "Config", "serve", "__version__"}
    for internal in ("store", "runlog", "cli", "config", "ingest", "search", "server"):
        assert internal not in docusearch.__all__


def test_serve_is_lazily_exported() -> None:
    # `import docusearch` must not require FastAPI; serve resolves on access.
    assert callable(docusearch.serve)
