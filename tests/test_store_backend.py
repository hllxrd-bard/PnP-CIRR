import copy
from pathlib import Path

import pytest

from cir.config import DEFAULT_CONFIG, AppConfig
from cir.service_store import FETCH_ENDPOINT, SEARCH_ENDPOINT, ServiceStore
from cir.store_base import FrameStore, StoreBackendError, build_store


def make_config(**milvus_overrides) -> AppConfig:
    data = copy.deepcopy(DEFAULT_CONFIG)
    data["milvus"].update(milvus_overrides)
    return AppConfig(data=data, source_path=Path("config.yaml"))


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class FakeHttpClient:
    """Stands in for httpx.Client so the store can be driven without a service."""

    def __init__(self, paths=(SEARCH_ENDPOINT, FETCH_ENDPOINT), healthy=True):
        self._paths = {name: {} for name in paths}
        self._healthy = healthy
        self.posts = []
        self.responses = {}

    def get(self, url):
        if url == "/health":
            return FakeResponse(200 if self._healthy else 503, {"status": "healthy"})
        if url == "/openapi.json":
            return FakeResponse(200, {"paths": self._paths})
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, json=None):
        self.posts.append((url, json))
        return FakeResponse(200, self.responses.get(url, {}))

    def close(self):
        pass


def build_service_store(monkeypatch, client) -> ServiceStore:
    monkeypatch.setattr("cir.service_store.httpx.Client", lambda **kwargs: client)
    return ServiceStore(make_config(backend="service"))


# -- factory ---------------------------------------------------------------


def test_build_store_rejects_unknown_backend():
    with pytest.raises(StoreBackendError, match="Unknown milvus.backend"):
        build_store(make_config(backend="postgres"))


def test_build_store_selects_service_backend(monkeypatch):
    store = build_service_store(monkeypatch, FakeHttpClient())
    assert isinstance(store, ServiceStore)


def test_default_backend_is_direct():
    assert DEFAULT_CONFIG["milvus"]["backend"] == "direct"


def test_service_store_satisfies_frame_store_protocol(monkeypatch):
    store = build_service_store(monkeypatch, FakeHttpClient())
    assert isinstance(store, FrameStore)


def test_milvus_store_satisfies_frame_store_protocol():
    # Structural check only -- constructing MilvusStore would need a live Milvus.
    from cir.milvus_store import MilvusStore

    for name in ("describe", "get_by_id", "get_by_video_frame", "fetch_entities", "search"):
        assert callable(getattr(MilvusStore, name))


# -- capability probe ------------------------------------------------------


def test_missing_vector_endpoints_fail_fast_and_name_them(monkeypatch):
    client = FakeHttpClient(paths=("/v1/search/text", "/v1/search/image"))
    with pytest.raises(StoreBackendError) as excinfo:
        build_service_store(monkeypatch, client)
    message = str(excinfo.value)
    assert SEARCH_ENDPOINT in message
    assert FETCH_ENDPOINT in message
    assert "milvus.backend: direct" in message


def test_partial_endpoint_support_still_fails(monkeypatch):
    client = FakeHttpClient(paths=(SEARCH_ENDPOINT,))
    with pytest.raises(StoreBackendError) as excinfo:
        build_service_store(monkeypatch, client)
    assert FETCH_ENDPOINT in str(excinfo.value)


def test_unhealthy_service_fails_fast(monkeypatch):
    with pytest.raises(StoreBackendError, match="unhealthy"):
        build_service_store(monkeypatch, FakeHttpClient(healthy=False))


def test_missing_base_url_is_rejected():
    config = make_config(backend="service", service={"base_url": ""})
    with pytest.raises(StoreBackendError, match="base_url is required"):
        build_store(config)


# -- store operations ------------------------------------------------------


def test_search_normalizes_hits_to_pipeline_shape(monkeypatch):
    client = FakeHttpClient()
    client.responses[SEARCH_ENDPOINT] = {
        "results": [
            [
                {"id": 11, "distance": 0.9, "entity": {"video_name": "L30_V091"}},
                {"id": 12, "score": 0.8, "entity": {"video_name": "L30_V083"}},
            ]
        ]
    }
    store = build_service_store(monkeypatch, client)

    results = store.search([[0.1] * 4], limit=2, expression="frame_id > 0")

    assert len(results) == 1
    first, second = results[0]
    assert (first["id"], first["distance"]) == (11, 0.9)
    # A hit carrying "score" instead of "distance" is still normalized.
    assert (second["id"], second["distance"]) == (12, 0.8)
    # The id is mirrored into the entity, matching MilvusStore.
    assert first["entity"]["id"] == 11

    url, payload = client.posts[-1]
    assert url == SEARCH_ENDPOINT
    assert payload["top_k"] == 2
    assert payload["expr"] == "frame_id > 0"
    assert payload["anns_field"] == "image_embedding"


def test_fetch_entities_deduplicates_and_skips_empty(monkeypatch):
    client = FakeHttpClient()
    client.responses[FETCH_ENDPOINT] = {"entities": [{"id": 5}]}
    store = build_service_store(monkeypatch, client)

    assert store.fetch_entities([]) == []
    assert not client.posts  # no request for an empty id list

    store.fetch_entities([5, 5, 5])
    assert client.posts[-1][1]["ids"] == [5]


def test_get_by_id_returns_none_when_absent(monkeypatch):
    client = FakeHttpClient()
    client.responses[FETCH_ENDPOINT] = {"entities": []}
    store = build_service_store(monkeypatch, client)
    assert store.get_by_id(999) is None


def test_get_by_video_frame_sends_quoted_filter(monkeypatch):
    client = FakeHttpClient()
    client.responses[FETCH_ENDPOINT] = {"entities": [{"id": 7}]}
    store = build_service_store(monkeypatch, client)

    entity = store.get_by_video_frame("L30_V091", "frame_059")

    assert entity == {"id": 7}
    payload = client.posts[-1][1]
    assert payload["limit"] == 1
    assert 'video_name == "L30_V091"' in payload["filter"]
    assert 'frame_name == "frame_059"' in payload["filter"]


def test_describe_reports_configured_embedding_dimension(monkeypatch):
    store = build_service_store(monkeypatch, FakeHttpClient())
    fields = {field["name"]: field for field in store.describe()["fields"]}
    assert fields["image_embedding"]["params"]["dim"] == 1024


def test_non_200_response_raises_with_status(monkeypatch):
    client = FakeHttpClient()
    store = build_service_store(monkeypatch, client)
    client.post = lambda url, json=None: FakeResponse(500, {}, "boom")

    with pytest.raises(StoreBackendError, match=r"\[500\]"):
        store.fetch_entities([1])
