"""Atıf yapılan parçada cevabı destekleyen cümleyi bulur (model çağrısı yok).

Cevabın her cümlesi içindeki `[Kn]` etiketiyle bir parçaya bağlanır; parçanın cümleleri
arasından o cümleyle en çok kelime kökü paylaşan seçilir. Parçalar 900–1400 karakter olduğu
için kullanıcı bilginin nereden geldiğini bütün parçayı okumadan görebilsin diye.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .bm25 import tokenize
from .prompts import cited_indices
from .retrieval import Hit

# Bir cümlenin seçilmesi için gereken en düşük puan: iki ortak kök ya da bir ortak sayı.
MIN_SCORE = 2
MAX_PER_CHUNK = 2

# Nokta + boşluk + büyük harf/rakam cümle sınırıdır; "18. yüzyılda" gibi küçük harfle devam
# eden sıra sayıları bölünmez.
_BOUNDARY = re.compile(r"[.!?…]+[\"'”’)]*\s+(?=[\"'“(]?[A-ZÇĞİÖŞÜ0-9])")
_ROMAN = re.compile(r"[IVXLCDM]+")
# Ardından büyük harf gelse de cümleyi bitirmeyen kısaltmalar ("MÖ 62", "Dr. Schmidt").
_ABBREVIATIONS = {"mö", "ms", "m.ö", "m.s", "dr", "prof", "doç", "vb", "vs", "bkz", "st", "yy", "örn", "no", "hz"}
_TAGS = re.compile(r"\[[^\]]*\]")


def split_sentences(text: str) -> list[str]:
    """Metni cümlelere böler; her cümle metnin birebir bir parçasıdır (vurgulamak için)."""
    sentences: list[str] = []
    for paragraph in text.splitlines():
        start = 0
        for match in _BOUNDARY.finditer(paragraph):
            words = paragraph[start:match.start()].split()
            last = words[-1].lstrip("(\"'“") if words else ""
            if match.group().startswith(".") and (
                    _ROMAN.fullmatch(last) or last.lower() in _ABBREVIATIONS
                    or (len(last) == 1 and last.isalpha())):
                continue  # "II. Claudius", "M.Ö. 62", "J. Mellaart"
            sentences.append(paragraph[start:match.end()].strip())
            start = match.end()
        if paragraph[start:].strip():
            sentences.append(paragraph[start:].strip())
    return sentences


def _claims(answer_text: str, hit_count: int) -> list[tuple[str, list[int]]]:
    """Cevap cümleleri ve atıf yaptıkları parçalar. Yalnızca etiketten oluşan cümle
    ("... girdi. [K1]") bir öncekine aittir."""
    claims: list[tuple[str, list[int]]] = []
    for sentence in split_sentences(answer_text):
        targets = cited_indices(sentence, hit_count)
        text = _TAGS.sub("", sentence)
        if not tokenize(text) and claims:
            claims[-1][1].extend(t for t in targets if t not in claims[-1][1])
        elif targets:
            claims.append((text, targets))
    return claims


def _score(claim: set[str], sentence: str, ignored: set[str]) -> int:
    # Sayılar (yıl, yükseklik, kapasite) en güçlü kanıt: iki puan.
    return sum(2 if any(c.isdigit() for c in term) else 1
               for term in claim & set(tokenize(sentence)) - ignored)


def supporting_sentences(answer_text: str, hits: list[Hit], cited: list[int]) -> dict[int, list[str]]:
    """Atıf yapılan her parça için (indeks -> cümleler) cevabı destekleyen cümleler.

    Bölüm yolundaki kelimeler ("Göbeklitepe › Koruma") sayılmaz; parçanın neredeyse her
    cümlesinde geçtikleri için ayırt edici değiller. Yeterince örtüşen cümle yoksa parça boş kalır.
    """
    support: dict[int, list[str]] = {}
    claims = _claims(answer_text, len(hits))
    for i in cited:
        chunk = hits[i].chunk
        sentences = split_sentences(chunk.text)
        ignored = set(tokenize(chunk.section))
        chosen: set[int] = set()
        for text, targets in claims:
            if i not in targets:
                continue
            terms = set(tokenize(text))
            scores = [_score(terms, s, ignored) for s in sentences]
            best = max(range(len(sentences)), key=lambda j: scores[j], default=None)
            if best is not None and scores[best] >= MIN_SCORE:
                chosen.add(best)
        if chosen:
            support[i] = [sentences[j] for j in sorted(chosen)[:MAX_PER_CHUNK]]
    return support


def emphasize(text: str, sentences: list[str], wrap: Callable[[str], str]) -> str:
    """Metinde her cümlenin ilk geçtiği yeri `wrap` ile sarar."""
    for sentence in sentences:
        text = text.replace(sentence, wrap(sentence), 1)
    return text
