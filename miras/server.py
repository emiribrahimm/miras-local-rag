"""Yerel HTTP API: başka araçlar da asistanı `ask --json` ile aynı biçimde kullanabilsin.

Yalnızca standart kütüphane. Model ve SQLite bağlantısı tek olduğu için asistana giden
istekler bir kilitle sırayla işlenir; uzun bir cevap üretilirken diğerleri bekler.
"""

from __future__ import annotations

import hmac
import json
import threading
import time
from collections.abc import Callable, Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .backends import BackendError
from .export import answer_json
from .pipeline import Assistant

# Soru en fazla 1000 karakter; bundan büyük gövde okunmadan reddedilir.
MAX_BODY_BYTES = 64 * 1024


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


def _topic(assistant: Assistant, body: dict) -> tuple[str, ...]:
    """Önceki cevabın `topic` alanı; yalnızca indeksteki alan başlıkları kabul edilir."""
    topic = body.get("topic") or []
    titles = {c.title for c in assistant.retriever.chunks}
    if not isinstance(topic, list) or not all(isinstance(t, str) and t in titles for t in topic):
        raise RequestError(HTTPStatus.BAD_REQUEST,
                           "'topic' önceki cevabın topic alanı olmalı (indeksteki alan başlıkları).")
    return tuple(topic)


class Api:
    """Uç noktalar: her biri istek gövdesini (GET'te boş) alıp JSON'a çevrilecek sözlük döner."""

    def __init__(self, assistant: Assistant) -> None:
        self.assistant = assistant
        self.lock = threading.Lock()
        # Akış isteyen uç noktalar sözlük yerine satır satır yazılacak sözlükler üretir.
        self.routes: dict[tuple[str, str], Callable[[dict], dict | Iterator[dict]]] = {
            ("GET", "/health"): self.health,
            ("GET", "/docs"): self.docs,
            ("POST", "/ask"): self.ask,
            ("POST", "/search"): self.search,
        }

    def health(self, _body: dict) -> dict:
        a = self.assistant
        return {"status": "ok", "chat_model": a.backend.chat_model, "embed_model": a.backend.embed_model,
                "docs": len({c.filename for c in a.retriever.chunks}), "chunks": len(a.retriever.chunks),
                "top_k": a.settings.top_k, "min_similarity": a.settings.min_similarity}

    def docs(self, _body: dict) -> dict:
        """İndeksteki dokümanlar; `docs` alanına verilebilecek dosya adları ve takma adlar."""
        found: dict[str, dict] = {}
        for c in self.assistant.retriever.chunks:
            doc = found.setdefault(c.filename, {"filename": c.filename, "title": c.title,
                                                "aliases": list(c.aliases), "source_url": c.source_url,
                                                "chunks": 0})
            doc["chunks"] += 1
        return {"docs": sorted(found.values(), key=lambda d: (d["title"], d["filename"]))}

    def search(self, body: dict) -> dict:
        """Modele gidecek parçalar ve kapının kararı; sohbet modeli çağrılmaz (`search` komutu gibi)."""
        question = _question(body)
        with self.lock:
            docs = _docs(self.assistant, body)
            prepared = self.assistant.prepare(question, docs=docs)
            mentioned = self.assistant.retriever.mentioned_titles(prepared.question)
        return {"question": prepared.question, "mentioned": sorted(mentioned), "docs": list(prepared.docs),
                "best_score": round(max(h.score for h in prepared.hits), 3),
                "min_similarity": self.assistant.settings.min_similarity,
                "passes_threshold": prepared.refusal_reason is None,
                "hits": [{"label": f"K{i}", "filename": h.chunk.filename, "section": h.chunk.section,
                          "score": round(h.score, 3), "text": h.chunk.text}
                         for i, h in enumerate(prepared.hits, start=1)],
                "retrieval_seconds": round(prepared.retrieval_seconds, 3)}

    def ask(self, body: dict) -> dict | Iterator[dict]:
        """Sunucu sohbet tutmaz: takip sorusu için son cevabın `topic` alanı geri gönderilir.

        `"stream": true` ise cevap NDJSON olarak gelir: parça başına `{"delta": "..."}`, sonda
        doğrulanmış cevapla `{"done": true, ...}`. Hatalı istekler akış başlamadan reddedilir.
        """
        question = _question(body)
        refresh = bool(body.get("refresh"))
        with self.lock:
            docs = _docs(self.assistant, body)
            prepared = self.assistant.prepare(question, docs=docs, topic=_topic(self.assistant, body))
        if body.get("stream"):
            return self._stream(question, prepared, refresh)
        with self.lock:
            started = time.perf_counter()
            raw = "".join(self.assistant.generate(prepared, refresh=refresh))
            answer = self.assistant.finalize(prepared, raw, time.perf_counter() - started)
        return answer_json(question, answer, self.assistant.settings)

    def _stream(self, question: str, prepared, refresh: bool) -> Iterator[dict]:
        with self.lock:
            started = time.perf_counter()
            pieces = []
            for piece in self.assistant.generate(prepared, refresh=refresh):
                pieces.append(piece)
                yield {"delta": piece}
            answer = self.assistant.finalize(prepared, "".join(pieces), time.perf_counter() - started)
        yield {"done": True, **answer_json(question, answer, self.assistant.settings)}


# Token verilse de açık kalan uç nokta; izleme araçları kimlik bilgisi olmadan bakabilsin.
PUBLIC_PATHS = {"/health"}


def make_server(assistant: Assistant, host: str = "127.0.0.1", port: int = 8765,
                token: str | None = None) -> ThreadingHTTPServer:
    """Henüz dinlemeye başlamamış sunucu; `serve_forever()` ile çalışır. `port=0` boş bir port seçer.

    `token` verilirse `/health` dışındaki istekler `Authorization: Bearer <token>` ister.
    """
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
            path = self.path.split("?", 1)[0]
            try:
                if token and path not in PUBLIC_PATHS and not self._authorized():
                    raise RequestError(HTTPStatus.UNAUTHORIZED, "Geçerli bir 'Authorization: Bearer' "
                                                                "başlığı gerekli.")
                route = api.routes.get((method, path))
                if route is None:
                    if any(p == path for _, p in api.routes):
                        raise RequestError(HTTPStatus.METHOD_NOT_ALLOWED, f"{path} için {method} kullanılmaz.")
                    raise RequestError(HTTPStatus.NOT_FOUND, f"{method} {path} yok.")
                result = route(self._body() if method == "POST" else {})
                if isinstance(result, dict):
                    self._send(HTTPStatus.OK, result)
                else:
                    self._send_lines(result)
                    return
            except RequestError as err:
                self._send(err.status, {"error": str(err)})
            except BackendError as err:
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(err)})
            except Exception as err:  # sunucu tek bir bozuk istek yüzünden düşmesin
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(err).__name__}: {err}"})

        def _authorized(self) -> bool:
            given = self.headers.get("Authorization", "")
            # Sabit süreli karşılaştırma: token karakter karakter tahmin edilemesin.
            return hmac.compare_digest(given.encode("utf-8"), f"Bearer {token}".encode("utf-8"))

        def _body(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise RequestError(HTTPStatus.BAD_REQUEST, "Content-Length geçersiz.") from None
            if length > MAX_BODY_BYTES:
                # Gövde okunmadığı için bağlantı açık kalırsa sıradaki istek bozulur.
                self.close_connection = True
                raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                                   f"Gövde en fazla {MAX_BODY_BYTES} bayt olabilir.")
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as err:
                raise RequestError(HTTPStatus.BAD_REQUEST, f"Gövde geçerli bir JSON değil: {err}") from err
            if not isinstance(body, dict):
                raise RequestError(HTTPStatus.BAD_REQUEST, "Gövde bir JSON nesnesi olmalı.")
            return body

        def _send_lines(self, lines: Iterator[dict]) -> None:
            """NDJSON akışı; uzunluk baştan bilinmediği için bağlantı sonunda kapanır."""
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.end_headers()
            self.close_connection = True
            try:
                for line in lines:
                    self.wfile.write(json.dumps(line, ensure_ascii=False).encode("utf-8") + b"\n")
                    self.wfile.flush()
            except BackendError as err:
                # Başlıklar gitti; hata son satır olarak bildirilir.
                self.wfile.write(json.dumps({"error": str(err)}, ensure_ascii=False).encode("utf-8") + b"\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # istemci akışı yarıda bıraktı; üretilen kısım önbelleğe yazılmaz
            finally:
                close = getattr(lines, "close", None)
                if close:
                    close()  # yarıda kalan üreteç kilidi bıraksın

        def _send(self, status: HTTPStatus, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            if status == HTTPStatus.UNAUTHORIZED:
                self.send_header("WWW-Authenticate", 'Bearer realm="miras"')
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer((host, port), Handler)
