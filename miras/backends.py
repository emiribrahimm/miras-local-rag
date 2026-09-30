"""Model arka uçları.

Uygulamanın geri kalanı yalnızca `Backend` arayüzünü bilir:
  * `FoundryBackend` gerçek modelleri Microsoft Foundry Local ile cihazda çalıştırır.
  * `FakeBackend` testler içindir; model indirmeden, deterministik çalışır.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable, Iterator
from typing import Protocol

import numpy as np

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}

# Qwen3-Embedding sorgularda görev talimatı bekler; dokümanlar öneksiz gömülür.
QUERY_INSTRUCTION = "Given a question, retrieve relevant passages that answer the question"


class BackendError(RuntimeError):
    """Model yüklenemediğinde ya da çıkarım başarısız olduğunda fırlatılır."""


class Backend(Protocol):
    embed_model: str
    chat_model: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def stream_chat(self, messages: list[Message], *, max_tokens: int, temperature: float,
                    seed: int) -> Iterator[str]: ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)


class FoundryBackend:
    """Foundry Local SDK (2.x) üzerinden süreç içi çıkarım; ağ çağrısı yapmaz."""

    EMBED_BATCH = 16

    def __init__(self, chat_model: str, embed_model: str, chat_device: str = "cpu",
                 on_status: Callable[[str], None] | None = None) -> None:
        if chat_device not in ("cpu", "gpu"):
            raise BackendError(f"Geçersiz cihaz '{chat_device}' (cpu ya da gpu olmalı).")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self._devices = {chat_model: chat_device, embed_model: "cpu"}
        self._on_status = on_status or (lambda _msg: None)
        # Yerel çalışma zamanı aynı anda tek istek işler; Streamlit çok iş parçacıklıdır.
        self._lock = threading.RLock()
        self._models: dict[str, object] = {}
        self._embed_session = None

    def _load(self, alias: str):
        """Modeli istenen cihaz varyantıyla hazırlar (gerekirse indirir, belleğe yükler)."""
        if alias in self._models:
            return self._models[alias]
        try:
            from foundry_local_sdk import Configuration, FoundryLocalManager
        except ImportError as err:  # pragma: no cover - kurulum hatası
            raise BackendError("foundry-local-sdk kurulu değil: `uv sync` çalıştırın.") from err
        try:
            # initialize() süreç başına bir kez çağrılabilir (Streamlit yeniden çalıştırmaları).
            if FoundryLocalManager.instance is None:
                FoundryLocalManager.initialize(Configuration(app_name="miras-rag"))
            manager = FoundryLocalManager.instance
            device = self._devices[alias]
            if device == "gpu":
                # GPU varyantları ancak WebGPU yürütme sağlayıcısı kaydedilince katalogda görünür.
                self._on_status("GPU yürütme sağlayıcısı hazırlanıyor...")
                manager.download_and_register_eps()
            model = manager.catalog.get_model(alias)
            if model is None:
                raise BackendError(f"'{alias}' Foundry Local kataloğunda bulunamadı.")
            # Varsayılan varyant makineye göre değişir; cihazı açıkça seçmek davranışı sabitler.
            variant = next((v for v in model.variants if f"-generic-{device}" in v.id), None)
            if variant is None:
                raise BackendError(f"'{alias}' için {device} varyantı yok: "
                                   f"{[v.id for v in model.variants]}")
            model.select_variant(variant)
            if not model.is_cached:
                self._on_status(f"{alias} indiriliyor (yalnızca ilk çalıştırmada)...")
                model.download()
            if not model.is_loaded:
                self._on_status(f"{alias} belleğe yükleniyor...")
                model.load()
        except BackendError:
            raise
        except Exception as err:
            raise BackendError(f"'{alias}' modeli hazırlanamadı: {err}") from err
        self._models[alias] = model
        return model

    def _embed(self, texts: list[str]) -> np.ndarray:
        from foundry_local_sdk import EmbeddingsSession, Request, TextItem

        with self._lock:
            if self._embed_session is None:
                self._embed_session = EmbeddingsSession(self._load(self.embed_model))
            rows: list[np.ndarray] = []
            for start in range(0, len(texts), self.EMBED_BATCH):
                request = Request()
                for text in texts[start:start + self.EMBED_BATCH]:
                    request.add_item(TextItem(text))
                try:
                    with self._embed_session.process_request(request) as response:
                        rows += [np.frombuffer(item.data, dtype=np.float32).copy() for item in response]
                except Exception as err:
                    raise BackendError(f"Embedding üretilemedi: {err}") from err
        vectors = np.stack(rows)
        if not np.isfinite(vectors).all():
            raise BackendError("Embedding modeli geçersiz (NaN/Inf) değer üretti.")
        return _normalize(vectors)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"])[0]

    def stream_chat(self, messages: list[Message], *, max_tokens: int, temperature: float,
                    seed: int) -> Iterator[str]:
        from foundry_local_sdk import (ChatSession, MessageItem, MessageRole, Request,
                                       RequestOptions, SearchOptions, TextItem, TextItemType)

        roles = {"system": MessageRole.SYSTEM, "user": MessageRole.USER,
                 "assistant": MessageRole.ASSISTANT}
        with self._lock:
            model = self._load(self.chat_model)
            try:
                # Her soru için yeni oturum: RAG cevabı önceki turlara bağlı olmamalı.
                with ChatSession(model) as session:
                    session.set_options(RequestOptions(search=SearchOptions(
                        temperature=temperature, max_output_tokens=max_tokens, seed=seed)))
                    session.set_streaming(True)
                    request = Request()
                    for message in messages:
                        request.add_item(MessageItem(roles[message["role"]], message["content"]))
                    with session.process_streaming_request(request) as stream:
                        for item in stream:
                            # Akıl yürütme ("thinking") parçaları kullanıcıya gösterilmez.
                            if isinstance(item, TextItem) and item.type == TextItemType.DEFAULT:
                                yield item.text
            except BackendError:
                raise
            except Exception as err:
                raise BackendError(f"Model cevap üretemedi: {err}") from err


class FakeBackend:
    """Testler için: kelime özetlerinden vektör üretir, sabit bir cevap döndürür."""

    embed_model = "fake-hash-embedding"
    chat_model = "fake-chat"
    DIM = 4096

    def __init__(self, reply: str = "Sahte cevap [K1].") -> None:
        self.reply = reply
        self.chat_calls: list[list[Message]] = []

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.DIM, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            # Türkçe ekleri kabaca yok saymak için kelimenin ilk 5 harfi kullanılır.
            digest = hashlib.md5(word[:5].encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "little") % self.DIM] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def stream_chat(self, messages: list[Message], *, max_tokens: int, temperature: float,
                    seed: int) -> Iterator[str]:
        self.chat_calls.append(messages)
        yield from re.findall(r"\S+\s*", self.reply)
