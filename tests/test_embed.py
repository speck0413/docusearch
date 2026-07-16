"""Embedding provider + provenance tests (§8, R-EMB-1/2/7).

Model-dependent tests are marked ``model`` (need torch + a downloaded model) and
suppress third-party warning noise; the rest run fast and offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from docusearch import embed
from docusearch.config import EmbedConfig

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _embed_config(model: str, device: str = "cpu") -> EmbedConfig:
    return EmbedConfig(
        model=model, device=device, batch_size=16, auto_max_mb=200, trust_remote_code=False
    )


def test_make_provider_none_is_bm25_only() -> None:
    assert embed.make_provider(_embed_config("none")) is None


def test_make_provider_auto_is_deferred_to_phase3() -> None:
    with pytest.raises(NotImplementedError, match="Phase 3"):
        embed.make_provider(_embed_config("auto"))


def test_api_provider_is_a_stub() -> None:
    with pytest.raises(NotImplementedError, match="R-EMB-7"):
        embed.ApiProvider()


def test_to_from_blob_roundtrip() -> None:
    vec = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    assert np.array_equal(embed.from_blob(embed.to_blob(vec)), vec)


def test_model_id_available_without_loading() -> None:
    # provenance tag needs no download (R-EMB-2)
    assert embed.LocalProvider(MODEL).model_id == MODEL


# --- real model (torch); skips gracefully if unavailable ---------------------


@pytest.fixture(scope="module")
def local_provider():  # type: ignore[no-untyped-def]
    pytest.importorskip("sentence_transformers")
    import warnings

    prov = embed.LocalProvider(MODEL, device="cpu", batch_size=16)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = prov.dim  # triggers load/download
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"embedding model unavailable: {err}")
    return prov


@pytest.mark.model
@pytest.mark.filterwarnings("ignore")
def test_local_provider_dim(local_provider) -> None:  # type: ignore[no-untyped-def]
    assert local_provider.dim == 384


@pytest.mark.model
@pytest.mark.filterwarnings("ignore")
def test_local_provider_embed_shape_and_normalized(local_provider) -> None:  # type: ignore[no-untyped-def]
    vecs = local_provider.embed(["SPI timing configuration", "watchdog timer setup"])
    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)  # L2-normalized


@pytest.mark.model
@pytest.mark.filterwarnings("ignore")
def test_local_provider_is_deterministic(local_provider) -> None:  # type: ignore[no-untyped-def]
    a = local_provider.embed(["deterministic embedding please"])
    b = local_provider.embed(["deterministic embedding please"])
    assert np.array_equal(a, b)  # temperature-0 / eval-mode determinism (R-SRCH-5)


@pytest.mark.model
@pytest.mark.filterwarnings("ignore")
def test_local_provider_empty_input(local_provider) -> None:  # type: ignore[no-untyped-def]
    vecs = local_provider.embed([])
    assert vecs.shape == (0, 384)


@pytest.mark.model
@pytest.mark.filterwarnings("ignore")
def test_make_provider_builds_local(local_provider) -> None:  # type: ignore[no-untyped-def]
    prov = embed.make_provider(_embed_config(MODEL))
    assert prov is not None
    assert prov.model_id == MODEL
    assert prov.dim == 384
