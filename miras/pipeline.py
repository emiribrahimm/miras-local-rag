"""RAG hattı: soruyu göm -> ilgili parçaları getir -> bağlamla modele sor."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .backends import Backend
from .config import Settings
from .prompts import (ABSTAIN_SENTENCE, EMPTY_QUESTION_MESSAGE, build_messages, cited_indices,
                      is_abstention, strip_invalid_citations)
from .retrieval import Hit, Retriever

MAX_QUESTION_CHARS = 1000


@dataclass
class Prepared:
    """Arama adımının sonucu; model çağrılmadan önceki durum."""
    question: str
    hits: list[Hit] = field(default_factory=list)
    # None ise model çağrılır; aksi halde sabit cevap döner.
    refusal_reason: str | None = None  # "empty" | "low_similarity"
    retrieval_seconds: float = 0.0


@dataclass
class Answer:
    text: str
    refused: bool
    refusal_reason: str | None  # "empty" | "low_similarity" | "model"
    hits: list[Hit]             # getirilen tüm parçalar (skorlarıyla)
    cited: list[int]            # cevapta atıf yapılan parçaların `hits` içindeki indeksleri
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0

    @property
    def sources(self) -> list[Hit]:
        return [self.hits[i] for i in self.cited]


class Assistant:
    def __init__(self, backend: Backend, retriever: Retriever, settings: Settings) -> None:
        self.backend = backend
        self.retriever = retriever
        self.settings = settings

    def prepare(self, question: str) -> Prepared:
        question = " ".join(question.split())[:MAX_QUESTION_CHARS]
        if not question:
            return Prepared(question, refusal_reason="empty")
        started = time.perf_counter()
        hits = self.retriever.search(self.backend.embed_query(question), self.settings.top_k, question)
        prepared = Prepared(question, hits, retrieval_seconds=time.perf_counter() - started)
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
            yield from self.backend.stream_chat(
                build_messages(prepared.question, prepared.hits),
                max_tokens=self.settings.max_output_tokens,
                temperature=self.settings.temperature, seed=self.settings.seed)

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
                      prepared.retrieval_seconds, generation_seconds)

    def ask(self, question: str) -> Answer:
        prepared = self.prepare(question)
        started = time.perf_counter()
        raw = "".join(self.generate(prepared))
        return self.finalize(prepared, raw, time.perf_counter() - started)
