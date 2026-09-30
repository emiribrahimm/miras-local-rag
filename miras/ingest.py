"""Veri alma hattı: dokümanları oku -> parçala -> embedding üret -> SQLite'a yaz."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .backends import Backend
from .chunking import chunk_document, parse_document
from .config import Settings
from .store import Store


@dataclass
class IngestReport:
    added: list[str]
    unchanged: list[str]
    removed: list[str]
    chunks_embedded: int


def ingest(store: Store, backend: Backend, settings: Settings, *, rebuild: bool = False,
           on_progress: Callable[[str], None] = lambda _msg: None) -> IngestReport:
    """Yalnızca değişen dosyaları yeniden işler (içerik özetine göre)."""
    # Farklı bir embedding modeliyle üretilmiş vektörler karşılaştırılamaz: baştan kur.
    if rebuild or store.get_meta("embed_model") not in (None, backend.embed_model):
        store.clear()

    files = sorted(p for p in settings.docs_dir.iterdir() if p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise FileNotFoundError(f"{settings.docs_dir} içinde .md/.txt doküman yok.")

    known = store.fingerprints()
    report = IngestReport([], [], [], 0)
    for path in files:
        raw = path.read_text(encoding="utf-8")
        fingerprint = hashlib.sha256(
            f"{settings.chunk_target_chars}|{settings.chunk_max_chars}|{raw}".encode("utf-8")
        ).hexdigest()
        if known.get(path.name) == fingerprint:
            report.unchanged.append(path.name)
            continue
        doc = parse_document(raw, fallback_title=path.stem)
        chunks = chunk_document(doc, settings.chunk_target_chars, settings.chunk_max_chars)
        if not chunks:
            continue
        on_progress(f"{path.name}: {len(chunks)} parça gömülüyor...")
        embeddings = backend.embed_documents([c.embedding_text for c in chunks])
        store.add_document(path.name, doc.title, doc.source_url, fingerprint,
                           [c.section for c in chunks], [c.text for c in chunks], embeddings)
        report.added.append(path.name)
        report.chunks_embedded += len(chunks)

    for filename in set(known) - {p.name for p in files}:
        store.delete_document(filename)
        report.removed.append(filename)

    store.set_meta("embed_model", backend.embed_model)
    return report
