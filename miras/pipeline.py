"""RAG hattı: soruyu göm -> ilgili parçaları getir -> bağlamla modele sor."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .backends import Backend, Message
from .config import Settings
from .highlight import supporting_sentences
from .prompts import (ABSTAIN_SENTENCE, EMPTY_QUESTION_MESSAGE, build_messages, cited_indices,
                      is_abstention, strip_invalid_citations)
from .retrieval import _ASCII, Hit, Retriever, _fold
from .store import Store

MAX_QUESTION_CHARS = 1000

# Önceki konuya açıkça gönderme yapan kelimeler ("orası", "oradaki", "onun", "hangisi").
# Türkçe karaktersiz yazılır, soru da öyle katlanarak aranır. "oran", "oranı", "anı" gibi
# kelimeler eşleşmesin diye ekler tek tek sayılır; "oranın" iki anlamlı olduğu için yok.
_REFERS_BACK = re.compile(
    r"(?<!\w)(?:"
    r"(?:ora|bura)(?:si|sina|sini|sinin|sinda|sindaki|sindan|da|daki|dan|ya|yi|lari|larda|lardaki)"
    r"|(?:on|bun|sun)(?:un|u|a|da|dan|lar|lari|larin)"
    r"|hangisi\w*|kendisi\w*"
    r")(?!\w)")


def refers_back(question: str) -> bool:
    """Soru önceki konuya bir gönderme kelimesiyle mi bağlanıyor ("Orası ne zaman yapılmış?").

    "Peki" ve "ya" tek başına sayılmaz; "Peki Türk kahvesi nasıl pişirilir?" konu dışı kalmalı.
    """
    return _REFERS_BACK.search(_fold(question).translate(_ASCII)) is not None


@dataclass
class Prepared:
    """Arama adımının sonucu; model çağrılmadan önceki durum."""
    question: str
    hits: list[Hit] = field(default_factory=list)
    # None ise model çağrılır; aksi halde sabit cevap döner.
    refusal_reason: str | None = None  # "empty" | "low_similarity"
    retrieval_seconds: float = 0.0
    # Önceki turdan taşınan alan başlıkları; boş değilse `question` bunlarla bağlamlandırılmıştır.
    carried: tuple[str, ...] = ()
    cached: bool = False  # cevap model yerine önbellekten geldiyse `generate` doğru yapar


@dataclass
class Answer:
    text: str
    refused: bool
    refusal_reason: str | None  # "empty" | "low_similarity" | "model"
    hits: list[Hit]             # getirilen tüm parçalar (skorlarıyla)
    cited: list[int]            # cevapta atıf yapılan parçaların `hits` içindeki indeksleri
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    carried: tuple[str, ...] = ()  # bu soruya önceki turdan taşınan konu
    topic: tuple[str, ...] = ()    # bu turun konusu; sonraki takip sorusuna taşınır
    cached: bool = False           # model çağrılmadı, ham cevap önbellekten geldi
    # Atıf yapılan parça indeksi -> parçada cevabı destekleyen cümleler (vurgulamak için).
    support: dict[int, list[str]] = field(default_factory=dict)

    @property
    def sources(self) -> list[Hit]:
        return [self.hits[i] for i in self.cited]


class Assistant:
    def __init__(self, backend: Backend, retriever: Retriever, settings: Settings,
                 cache: Store | None = None) -> None:
        self.backend = backend
        self.retriever = retriever
        self.settings = settings
        # Verilirse ham model cevapları bu veritabanında önbelleğe alınır (settings.answer_cache).
        self.cache = cache

    def prepare(self, question: str, previous: Answer | None = None) -> Prepared:
        question = " ".join(question.split())[:MAX_QUESTION_CHARS]
        if not question:
            return Prepared(question, refusal_reason="empty")
        started = time.perf_counter()
        vector = self.backend.embed_query(question)
        # Takip sorusu: soruda hiçbir alan adı geçmiyorsa önceki turun konusu eklenir
        # ("Peki ne zaman yapılmış?" -> "Selimiye Camii hakkında: Peki ne zaman yapılmış?").
        # Bu metin hem aramada hem de modele giden soruda kullanılır. Soru tek başına hiçbir
        # parçaya benzemiyorsa (konu dışı) konu taşınmaz: alan adı onu kapıdan geçirmesin.
        # "Orası", "onun" gibi açık bir gönderme varsa benzerliğe bakılmaz; kısa sorular
        # embedding'de zayıf kalıp tabanın altına düşebiliyor.
        carried: tuple[str, ...] = ()
        if (previous and previous.topic and not self.retriever.mentioned_titles(question)
                and (refers_back(question) or self.retriever.best_score(vector)
                     >= self.settings.followup_min_similarity)):
            carried = previous.topic
            question = f"{', '.join(carried)} hakkında: {question}"
            vector = self.backend.embed_query(question)
        hits = self.retriever.search(vector, self.settings.top_k, question, self.settings.bm25_weight)
        prepared = Prepared(question, hits, retrieval_seconds=time.perf_counter() - started,
                            carried=carried)
        # Benzerlik kapısı: ilgili parça yoksa modeli hiç çağırma, uydurma riskini sıfırla.
        if max(hit.score for hit in hits) < self.settings.min_similarity:
            prepared.refusal_reason = "low_similarity"
        return prepared

    def generate(self, prepared: Prepared) -> Iterator[str]:
        """Cevabı parça parça üretir (akış). Reddedilen sorularda sabit metni verir."""
        if prepared.refusal_reason == "empty":
            yield EMPTY_QUESTION_MESSAGE
        elif prepared.refusal_reason == "low_similarity":
            yield ABSTAIN_SENTENCE
        else:
            messages = build_messages(prepared.question, prepared.hits)
            cache = self.cache if self.settings.answer_cache else None
            key = self._cache_key(messages) if cache else ""
            cached = cache.cached_answer(key) if cache else None
            if cached is not None:
                prepared.cached = True
                yield cached
                return
            pieces = []
            for piece in self.backend.stream_chat(
                    messages, max_tokens=self.settings.max_output_tokens,
                    temperature=self.settings.temperature, seed=self.settings.seed):
                pieces.append(piece)
                yield piece
            # Yalnızca sonuna kadar okunmuş akış kaydedilir; yarıda kesilen cevap önbelleğe girmez.
            if cache and "".join(pieces).strip():
                cache.cache_answer(key, "".join(pieces))

    def _cache_key(self, messages: list[Message]) -> str:
        """Prompt (sistem mesajı, parçalar, soru), model ve üretim ayarlarının özeti.

        Doküman, prompt ya da model değişince anahtar da değişir; ayrıca geçersizleştirme gerekmez.
        """
        s = self.settings
        payload = [self.backend.chat_model, s.chat_device, s.max_output_tokens, s.temperature,
                   s.seed, messages]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()

    def finalize(self, prepared: Prepared, raw_text: str, generation_seconds: float = 0.0) -> Answer:
        """Ham model çıktısını doğrular: reddetme tespiti ve kaynak etiketlerinin kontrolü."""
        # Örneklerdeki "CEVAP:" kalıbını tekrarlayan modeller için önek temizlenir.
        text = re.sub(r"^\s*CEVAP\s*:\s*", "", raw_text).strip()
        reason = prepared.refusal_reason
        cited: list[int] = []
        if reason is None:
            if not text or is_abstention(text):
                # Reddetme cümlesine eklenmiş açıklama/atıf varsa atılır.
                text, reason = ABSTAIN_SENTENCE, "model"
            else:
                text = strip_invalid_citations(text, len(prepared.hits))
                cited = cited_indices(text, len(prepared.hits))
        return Answer(text, reason is not None, reason, prepared.hits, cited,
                      prepared.retrieval_seconds, generation_seconds,
                      carried=prepared.carried, topic=self._topic(prepared, cited),
                      cached=prepared.cached,
                      support=supporting_sentences(text, prepared.hits, cited) if cited else {})

    def _topic(self, prepared: Prepared, cited: list[int]) -> tuple[str, ...]:
        """Soruda adı geçen alanlar; hiç ad geçmiyorsa cevabın atıf yaptığı tek alan."""
        titles = self.retriever.mentioned_titles(prepared.question) if prepared.question else set()
        if not titles:
            cited_titles = {prepared.hits[i].chunk.title for i in cited}
            if len(cited_titles) == 1:
                titles = cited_titles
        return tuple(sorted(titles))

    def ask(self, question: str, previous: Answer | None = None) -> Answer:
        prepared = self.prepare(question, previous)
        started = time.perf_counter()
        raw = "".join(self.generate(prepared))
        return self.finalize(prepared, raw, time.perf_counter() - started)
