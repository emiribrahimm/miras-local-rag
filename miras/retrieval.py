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


_ASCII = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def title_key(title: str) -> str:
    """"Divriği Ulu Camii ve Darüşşifası" -> "divriği"; kullanıcılar alanı çoğunlukla ilk kelimesiyle anar."""
    return _fold(title).split()[0]


def title_keys(title: str, aliases: tuple[str, ...] = ()) -> set[str]:
    """Bir alanı anan tüm biçimler: başlığın ilk kelimesi ve takma adlar.

    Her ad Türkçe karaktersiz haliyle de eklenir ("gobeklitepe"). Sorgunun kendisi
    katlanmaz; böylece "anı" gibi Türkçe kelimeler "Ani"ye eşleşmez.
    """
    names = [title_key(title)] + [_fold(a) for a in aliases]
    return {form for name in names for form in (name, name.translate(_ASCII))}


# Yazım hatası toleransı bu uzunluktan kısa adlara uygulanmaz: "Ani" -> "anı",
# "İlion" -> "ilin" gibi gündelik kelimelerle karışır.
FUZZY_MIN_CHARS = 6


def edit_distance(a: str, b: str) -> int:
    """Ekleme, silme, değiştirme ve yan yana iki harfin yer değiştirmesi birer adım sayılır."""
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]


def fuzzy_mentions(words: list[str], key: str) -> bool:
    """Sorgudaki kelimelerden biri (çok kelimeli adlarda ardışık kelimeler) ada en fazla
    bir harf uzaklıktaysa doğru: "göbeklitpe", "selimye", "gobekli tpe"."""
    key = key.translate(_ASCII)
    if len(key) < FUZZY_MIN_CHARS:
        return False
    n = len(key.split())
    return any(edit_distance(" ".join(words[i:i + n]), key) <= 1
               for i in range(len(words) - n + 1))


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
        self._title_keys = {c.title: title_keys(c.title, c.aliases) for c in self.chunks}
        self._titles = np.array(titles, dtype=object)

    def mentioned_titles(self, query: str) -> set[str]:
        """Soruda adı geçen alanların başlıkları.

        Özel isimlere ek kesme işaretiyle geldiği için ("Ani'deki") ad sonrası harf gelmesi
        eşleşme sayılmaz; böylece "aniden" gibi kelimeler "Ani"yi tetiklemez. Takma adlar
        ("Truva" -> Troya) ve Türkçe karaktersiz yazımlar ("Gobeklitepe") da eşleşir.
        Birebir eşleşmeyen uzun adlarda tek harflik yazım hataları da kabul edilir.
        """
        folded = _fold(query)
        words = re.findall(r"\w+", folded.translate(_ASCII))
        return {title for title, keys in self._title_keys.items()
                if any(re.search(rf"(?<!\w){re.escape(key)}(?!\w)", folded) for key in keys)
                or any(fuzzy_mentions(words, key) for key in keys)}

    def _title_bonus(self, query: str) -> np.ndarray:
        """Soruda geçen alan adlarına ait parçalara TITLE_BOOST verir.

        "Ani" gibi kısa adlar embedding'de zayıf temsil ediliyor; bu, sorunun hangi
        alanla ilgili olduğunu açıkça söylediği durumda sıralamayı düzeltir.
        """
        matched = self.mentioned_titles(query)
        return np.where(np.isin(self._titles, list(matched)), TITLE_BOOST, 0.0)

    def search(self, query_vector: np.ndarray, top_k: int, query_text: str = "") -> list[Hit]:
        # Vektörler normalize saklandığı için iç çarpım = kosinüs benzerliği.
        scores = self.matrix @ query_vector
        ranking = scores + self._title_bonus(query_text) if query_text else scores
        top_k = min(top_k, len(scores))
        best = np.argpartition(-ranking, top_k - 1)[:top_k]
        best = best[np.argsort(-ranking[best])]
        return [Hit(self.chunks[i], float(scores[i])) for i in best]
