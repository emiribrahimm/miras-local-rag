"""Yerel HTTP API: başka araçlar da asistanı `ask --json` ile aynı biçimde kullanabilsin.

Yalnızca standart kütüphane. Model ve SQLite bağlantısı tek olduğu için asistana giden
istekler bir kilitle sırayla işlenir; uzun bir cevap üretilirken diğerleri bekler.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .export import answer_json
from .pipeline import Assistant


class RequestError(Exception):
    """İstemcinin düzeltebileceği hata; mesajıyla birlikte JSON olarak döner."""

    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def _question(body: dict) -> str:
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise RequestError(HTTPStatus.BAD_REQUEST, "'question' alanı boş olmayan bir metin olmalı.")
    return question


def _docs(assistant: Assistant, body: dict) -> frozenset[str] | None:
    names = body.get("docs") or []
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise RequestError(HTTPStatus.BAD_REQUEST, "'docs' alanı metin listesi olmalı.")
    if not names:
        return None
    try:
        return assistant.retriever.resolve_docs(names)
    except ValueError as err:
        raise RequestError(HTTPStatus.BAD_REQUEST, str(err)) from err


class Api:
    """Uç noktalar: her biri istek gövdesini (GET'te boş) alıp JSON'a çevrilecek sözlük döner."""

    def __init__(self, assistant: Assistant) -> None:
        self.assistant = assistant
        self.lock = threading.Lock()
        self.routes: dict[tuple[str, str], Callable[[dict], dict]] = {
            ("GET", "/health"): self.health,
            ("POST", "/ask"): self.ask,
        }

    def health(self, _body: dict) -> dict:
        a = self.assistant
        return {"status": "ok", "chat_model": a.backend.chat_model, "embed_model": a.backend.embed_model,
                "docs": len({c.filename for c in a.retriever.chunks}), "chunks": len(a.retriever.chunks),
                "top_k": a.settings.top_k, "min_similarity": a.settings.min_similarity}

    def ask(self, body: dict) -> dict:
        question = _question(body)
        with self.lock:
            docs = _docs(self.assistant, body)
            answer = self.assistant.ask(question, refresh=bool(body.get("refresh")), docs=docs)
        return answer_json(question, answer, self.assistant.settings)


def make_server(assistant: Assistant, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Henüz dinlemeye başlamamış sunucu; `serve_forever()` ile çalışır. `port=0` boş bir port seçer."""
    api = Api(assistant)

    class Handler(BaseHTTPRequestHandler):
        server_version = "miras"

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass  # her isteği stderr'e basmasın; hatalar zaten JSON olarak dönüyor

        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def _handle(self, method: str) -> None:
            try:
                route = api.routes.get((method, self.path.split("?", 1)[0]))
                if route is None:
                    raise RequestError(HTTPStatus.NOT_FOUND, f"{method} {self.path} yok.")
                self._send(HTTPStatus.OK, route(self._body() if method == "POST" else {}))
            except RequestError as err:
                self._send(err.status, {"error": str(err)})

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as err:
                raise RequestError(HTTPStatus.BAD_REQUEST, f"Gövde geçerli bir JSON değil: {err}") from err
            if not isinstance(body, dict):
                raise RequestError(HTTPStatus.BAD_REQUEST, "Gövde bir JSON nesnesi olmalı.")
            return body

        def _send(self, status: HTTPStatus, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer((host, port), Handler)
