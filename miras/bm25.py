"""Anahtar kelime araması: parçalar üzerinde BM25, hibrit sıralama için."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# Türkçe eklere karşı kelimenin ilk 5 harfi kök sayılır ("Selimiye'nin", "Selimiye'de" -> "selim").
# Kaba ama Türkçe bilgi erişiminde iyi bilinen, sözlük gerektirmeyen bir yöntem.
PREFIX = 5

_ASCII = str.maketrans("çğıöşüâîû", "cgiosuaiu")

# Sorularda sık geçen ama içerik taşımayan kelimeler; dokümanlarda nadir oldukları için
# atılmazlarsa yüksek IDF alıp sıralamayı bozarlar.
STOPWORDS = {
    "hangi", "hangisi", "hangisidir", "kim", "kimdir", "kimin", "nedir", "neden", "nerede",
    "neresi", "nasil", "kac", "kacinci", "ne", "mi", "mu", "peki", "ya", "ve", "ile", "bir",
    "bu", "da", "de", "ki", "hakkinda", "icin", "gibi", "midir", "oradaki", "orada",
}


def tokenize(text: str) -> list[str]:
    folded = text.replace("İ", "i").replace("I", "ı").lower().translate(_ASCII)
    # Kesme işaretinden sonraki ek ("Ani'deki") ayrı kelime sayılmaz.
    return [word[:PREFIX] for word in re.findall(r"(\w+)(?:['’]\w+)?", folded)
            if word not in STOPWORDS and len(word) > 1]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        tokenized = [tokenize(d) for d in docs]
        self.size = len(docs)
        lengths = np.array([len(t) for t in tokenized], dtype=np.float32)
        avg = float(lengths.mean()) if self.size and lengths.mean() > 0 else 1.0
        self._norm = k1 * (1 - b + b * lengths / avg)
        # terim -> (parça indeksleri, terim sıklıkları)
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, tokens in enumerate(tokenized):
            for term, tf in Counter(tokens).items():
                ids, tfs = postings.setdefault(term, ([], []))
                ids.append(i)
                tfs.append(tf)
        self._postings = {t: (np.array(ids), np.array(tfs, dtype=np.float32))
                          for t, (ids, tfs) in postings.items()}

    def scores(self, query: str) -> np.ndarray:
        """Her parça için 0–1 aralığına ölçeklenmiş BM25 skoru (en iyi parça 1)."""
        result = np.zeros(self.size, dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self._postings:
                continue
            ids, tfs = self._postings[term]
            idf = math.log(1 + (self.size - len(ids) + 0.5) / (len(ids) + 0.5))
            result[ids] += idf * tfs * (self.k1 + 1) / (tfs + self._norm[ids])
        best = result.max(initial=0.0)
        return result / best if best > 0 else result
