"""Vektör araması: tüm parçalarla kosinüs benzerliği, en iyi K sonuç."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .store import Store, StoredChunk


class IndexError_(RuntimeError):
    """İndeks boş ya da başka bir embedding modeliyle oluşturulmuş."""


# Soruda bir alanın adı geçiyorsa o dokümanın parçalarına eklenen sıralama puanı.
TITLE_BOOST = 0.1


@dataclass(frozen=True)
class Hit:
    chunk: StoredChunk
    score: float  # ham kosinüs benzerliği (kapı eşiği bununla karşılaştırılır)


def _fold(text: str) -> str:
    return text.replace("İ", "i").replace("I", "ı").lower()


def title_key(title: str) -> str:
    """"Divriği Ulu Camii ve Darüşşifası" -> "divriği"; kullanıcılar alanı çoğunlukla ilk kelimesiyle anar."""
    return _fold(title).split()[0]


class Retriever:
    """Vektörleri bir kez belleğe alır; her sorgu tek bir matris çarpımıdır.

    Birkaç yüz parça için kaba kuvvet arama milisaniyenin altındadır. Binlerce
    dokümana çıkılırsa sqlite-vec gibi bir vektör eklentisine geçmek gerekir.
    """

    def __init__(self, store: Store, embed_model: str) -> None:
        indexed_with = store.get_meta("embed_model")
        self.chunks, self.matrix = store.load_all()
        if not self.chunks:
            raise IndexError_("İndeks boş. Önce `python main.py ingest` çalıştırın.")
        if indexed_with != embed_model:
            raise IndexError_(
                f"İndeks '{indexed_with}' ile oluşturulmuş, şu anki model '{embed_model}'. "
                "`python main.py ingest --rebuild` çalıştırın.")
        titles = [c.title for c in self.chunks]
        self._title_keys = {t: title_key(t) for t in set(titles)}
        self._titles = np.array(titles, dtype=object)

    def _title_bonus(self, query: str) -> np.ndarray:
        """Soruda geçen alan adlarına ait parçalara TITLE_BOOST verir.

        "Ani" gibi kısa adlar embedding'de zayıf temsil ediliyor; bu, sorunun hangi
        alanla ilgili olduğunu açıkça söylediği durumda sıralamayı düzeltir. Özel
        isimlere ek kesme işaretiyle geldiği için ("Ani'deki") ad sonrası harf gelmesi
        eşleşme sayılmaz; böylece "aniden" gibi kelimeler "Ani"yi tetiklemez.
        """
        folded = _fold(query)
        matched = {title for title, key in self._title_keys.items()
                   if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", folded)}
        return np.where(np.isin(self._titles, list(matched)), TITLE_BOOST, 0.0)

    def search(self, query_vector: np.ndarray, top_k: int, query_text: str = "") -> list[Hit]:
        # Vektörler normalize saklandığı için iç çarpım = kosinüs benzerliği.
        scores = self.matrix @ query_vector
        ranking = scores + self._title_bonus(query_text) if query_text else scores
        top_k = min(top_k, len(scores))
        best = np.argpartition(-ranking, top_k - 1)[:top_k]
        best = best[np.argsort(-ranking[best])]
        return [Hit(self.chunks[i], float(scores[i])) for i in best]
