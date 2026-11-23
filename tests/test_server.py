"""HTTP API: port 0'da gerçek bir sunucu, FakeBackend ile."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from miras.server import make_server


@pytest.fixture
def call(assistant):
    server = make_server(assistant, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path: str, body: dict | bytes | None = None) -> tuple[int, dict]:
        data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(base + path, data=data, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    yield call
    server.shutdown()
    server.server_close()


def test_health_reports_models_and_index(call):
    status, data = call("/health")
    assert status == 200 and data["status"] == "ok"
    assert data["chat_model"] == "fake-chat" and data["docs"] == 2 and data["chunks"] == 3


def test_ask_returns_the_same_shape_as_ask_json(call, backend):
    backend.reply = "Efes İzmir'dedir [K1]."
    status, data = call("/ask", {"question": "Efes nerede?"})
    assert status == 200 and data["answer"] == "Efes İzmir'dedir [K1]." and data["refused"] is False
    assert data["sources"][0]["filename"] == "efes.md" and data["sources"][0]["cited"] is True
    status, data = call("/ask", {"question": "Pizza hamuru tarifi nedir?"})
    assert data["refusal_reason"] == "low_similarity" and len(backend.chat_calls) == 1


def test_ask_can_be_limited_to_some_documents(call, backend):
    status, data = call("/ask", {"question": "Göbeklitepe nerede?", "docs": ["Ephesos"]})
    assert status == 200 and data["docs"] == ["efes.md"]
    assert {s["filename"] for s in data["sources"]} == {"efes.md"}
    status, data = call("/ask", {"question": "Troya nerede?", "docs": ["Troya"]})
    assert status == 400 and "'Troya' indekste yok" in data["error"]


def test_search_shows_chunks_and_gate_without_chat_model(call, backend):
    status, data = call("/search", {"question": "Göbeklitepe mi Efes mi daha eski?"})
    assert status == 200 and backend.chat_calls == []
    assert data["mentioned"] == ["Efes", "Göbeklitepe"] and data["passes_threshold"] is True
    assert {h["filename"] for h in data["hits"]} == {"efes.md", "gobeklitepe.md"}
    assert data["hits"][0]["label"] == "K1" and data["best_score"] == data["hits"][0]["score"]
    data = call("/search", {"question": "Pizza hamuru tarifi nedir?"})[1]
    assert data["passes_threshold"] is False and data["best_score"] < data["min_similarity"]


def test_docs_lists_the_index(call):
    status, data = call("/docs")
    assert status == 200
    assert [d["filename"] for d in data["docs"]] == ["efes.md", "gobeklitepe.md"]
    assert data["docs"][0]["aliases"] == ["Ephesos", "Ephesus"] and data["docs"][1]["chunks"] == 2


def test_bad_requests_get_a_json_error(call):
    assert call("/ask", b"{bozuk")[0] == 400
    status, data = call("/ask", {"question": "  "})
    assert status == 400 and "'question'" in data["error"]
    assert call("/ask", b"[1, 2]")[0] == 400
    assert call("/ask", {"question": "Efes?", "docs": "efes.md"})[0] == 400
    status, data = call("/yok")
    assert status == 404 and data["error"] == "GET /yok yok."
