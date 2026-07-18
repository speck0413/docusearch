"""Embedding provider + provenance tests (§8, R-EMB-1/2/7).

Model-dependent tests are marked ``model`` (need torch + a downloaded model) and
suppress third-party warning noise; the rest run fast and offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from docusearch import embed
from docusearch.config import EmbedConfig
from docusearch.config import load as cfg_load

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


def test_choose_auto_strategy() -> None:
    assert embed.choose_auto_strategy({"model": "none"}, 200) == "text"
    assert embed.choose_auto_strategy({"model": "m", "approx_mb": 90}, 200) == "local"
    assert embed.choose_auto_strategy({"model": "m", "approx_mb": 1300}, 200) == "text"


def test_best_device_prefers_cuda_then_mps_then_cpu() -> None:
    # pure preference order (R-EMB-1): discrete GPU > Apple Metal > cpu
    assert embed._best_device(has_cuda=True, has_mps=True) == "cuda"
    assert embed._best_device(has_cuda=True, has_mps=False) == "cuda"
    assert embed._best_device(has_cuda=False, has_mps=True) == "mps"
    assert embed._best_device(has_cuda=False, has_mps=False) == "cpu"


def test_detect_device_falls_back_to_cpu_without_torch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import builtins

    real_import = builtins.__import__

    def _no_torch(name: str, *a: object, **k: object) -> object:
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_torch)
    assert embed._detect_device() == "cpu"  # a probe failure must never abort a run


def test_make_provider_passes_device_through() -> None:
    # cpu/cuda/mps/auto reach the provider unchanged; 'auto' is resolved lazily at load
    for device in ("cpu", "cuda", "mps", "auto"):
        prov = embed.make_provider(_embed_config(MODEL, device=device))
        assert isinstance(prov, embed.LocalProvider)
        assert prov._device == device


def _asgi_client(config: object):  # type: ignore[no-untyped-def]
    import warnings

    from docusearch.server import create_app

    with warnings.catch_warnings():  # sync httpx client that runs the ASGI app in-process
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient

        return TestClient(create_app(config), base_url="http://server")  # type: ignore[arg-type]


def test_remote_provider_raises_when_server_has_no_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from docusearch import ingest
    from docusearch.store import Store

    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.html").write_text("<body><p>timing content for indexing here</p></body>", "utf-8")
    config_path = tmp_path / "docusearch.yaml"
    config_path.write_text(
        f'paths:\n  db_path: "{(tmp_path / "c.db").as_posix()}"\n'
        f'  staging_dir: "{(tmp_path / "s").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    config = cfg_load(config_path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
    hc = _asgi_client(config)
    provider = embed.RemoteServerProvider("http://server", http_client=hc)
    with pytest.raises(embed.EmbedError):  # server has no model -> 409 -> EmbedError
        provider.embed(["hello world"])


@pytest.mark.model
@pytest.mark.filterwarnings("ignore")
def test_remote_provider_embeds_via_server(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from docusearch import ingest
    from docusearch.store import Store

    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.html").write_text("<body><p>timing content for indexing here</p></body>", "utf-8")
    config_path = tmp_path / "docusearch.yaml"
    config_path.write_text(
        f'paths:\n  db_path: "{(tmp_path / "c.db").as_posix()}"\n'
        f'  staging_dir: "{(tmp_path / "s").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 5\n'
        f'embed:\n  model: "{MODEL}"\n  device: cpu\n',
        encoding="utf-8",
    )
    config = cfg_load(config_path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
    provider = embed.RemoteServerProvider("http://server", http_client=_asgi_client(config))
    assert provider.model_id == MODEL and provider.dim == 384
    vecs = provider.embed(["how does the clock work", "peripheral bus"])
    assert vecs.shape == (2, 384)


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
