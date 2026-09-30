"""Markdown dokümanlarını başlık yapısını koruyarak parçalara böler."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    title: str
    source_url: str
    body: str


@dataclass(frozen=True)
class Chunk:
    section: str  # ör. "Göbeklitepe › Mimari › Dikilitaşlar"
    text: str

    @property
    def embedding_text(self) -> str:
        # Başlık yolu da gömülür: "Tarihçe" gibi bölümler hangi alana ait olduğunu taşısın.
        return f"{self.section}\n{self.text}"


def parse_document(raw: str, fallback_title: str) -> Document:
    """Baştaki '---' ile çevrili üst bilgiyi (title, source) ayırır."""
    meta: dict[str, str] = {}
    body = raw
    match = re.match(r"---\n(.*?)\n---\n", raw, flags=re.DOTALL)
    if match:
        body = raw[match.end():]
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    # Üst bilgi yoksa başlık ilk '# ...' satırından, o da yoksa dosya adından alınır.
    h1 = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
    title = meta.get("title") or (h1.group(1).strip() if h1 else fallback_title)
    return Document(title, meta.get("source", ""), body.strip())


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Üst sınırı aşan tek paragrafı cümle sınırlarından böler."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sentence) > max_chars:  # noktasız çok uzun metin
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_document(doc: Document, target_chars: int, max_chars: int) -> list[Chunk]:
    """Her bölümün paragraflarını hedef boyuta kadar birleştirir; bölümler karışmaz."""
    chunks: list[Chunk] = []
    path: list[str] = [doc.title]
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            chunks.append(Chunk(" › ".join(path), "\n\n".join(buffer)))
            buffer.clear()

    for block in re.split(r"\n\s*\n", doc.body):
        block = block.strip()
        if not block:
            continue
        heading = re.fullmatch(r"(#{1,6})\s+(.+)", block)
        if heading:
            flush()
            level = len(heading.group(1))
            if level == 1:
                path = [doc.title]
            else:
                path = path[:level - 1] + [heading.group(2).strip()]
            continue
        for piece in _split_long(block, max_chars):
            if buffer and sum(map(len, buffer)) + len(piece) > target_chars:
                flush()
            buffer.append(piece)
    flush()
    return chunks
